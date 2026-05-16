"""
Compliance Checker for Vision_S8.

Validates product images against marketplace requirements.
"""

import json
import time
from typing import Any

from PIL import Image

from ..config import settings
from ..models.schemas import (
    ComplianceCheck,
    ComplianceResult,
    ComplianceViolation,
    ComplianceWarning,
    Platform,
)
from ..services.gemini_service import GeminiService, get_gemini_service
from ..services.image_service import ImageService, get_image_service
from ..utils.logger import get_logger

logger = get_logger("vision_s8.compliance_checker")


class ComplianceChecker:
    """Engine for checking image compliance with marketplace requirements."""

    def __init__(
        self,
        gemini_service: GeminiService | None = None,
        image_service: ImageService | None = None,
    ):
        """Initialize the checker."""
        self._gemini = gemini_service or get_gemini_service()
        self._image = image_service or get_image_service()
        self._platform_rules = self._load_platform_rules()

    def _load_platform_rules(self) -> dict[str, Any]:
        """Load platform rules from JSON file."""
        rules_path = settings.platform_rules_path
        if rules_path.exists():
            with open(rules_path, "r") as f:
                data = json.load(f)
                return data.get("platforms", {})
        return {}

    async def check(
        self,
        image: Image.Image,
        platform: Platform | str,
    ) -> tuple[ComplianceResult, int]:
        """
        Check image compliance for a specific platform.

        Args:
            image: PIL Image to check
            platform: Target platform

        Returns:
            Tuple of (ComplianceResult, processing_time_ms)
        """
        start_time = time.time()

        platform_str = platform.value if isinstance(platform, Platform) else platform.lower()
        logger.info(f"Checking compliance for {platform_str}")

        # Get platform rules
        rules = self._platform_rules.get(platform_str, {})
        requirements = rules.get("requirements", {})

        # Get image info
        image_info = self._image.get_image_info(image)

        # Run technical checks
        checks = self._run_technical_checks(image, image_info, requirements)

        # Get AI compliance assessment for visual checks
        ai_result = await self._gemini.check_compliance(image, platform_str)

        # Merge AI checks with technical checks
        ai_checks = ai_result.get("checks", {})
        for check_name, check_data in ai_checks.items():
            if check_name not in checks:
                checks[check_name] = ComplianceCheck(
                    passed=check_data.get("passed", True),
                    detected=check_data.get("detected"),
                    requirement=check_data.get("requirement"),
                    issue=check_data.get("issue"),
                )

        # Compile violations and warnings
        violations = self._extract_violations(checks, ai_result)
        warnings = self._extract_warnings(ai_result)

        # Calculate overall compliance
        critical_checks = ["dimensions", "file_size", "aspect_ratio"]
        critical_passed = all(
            checks.get(c, ComplianceCheck(passed=True)).passed
            for c in critical_checks if c in checks
        )
        all_passed = all(c.passed for c in checks.values())

        compliance_score = (sum(1 for c in checks.values() if c.passed) / len(checks) * 100) if checks else 0

        # Determine auto-fixable issues
        auto_fixable = []
        manual_required = []
        for v in violations:
            if v.rule.lower() in ["dimensions", "aspect_ratio", "background"]:
                auto_fixable.append(v.rule)
            else:
                manual_required.append(v.rule)

        result = ComplianceResult(
            platform=platform_str,
            overall_compliance=all_passed,
            compliance_score=round(compliance_score, 1),
            checks=checks,
            violations=violations,
            warnings=warnings,
            auto_fixable=auto_fixable,
            manual_required=manual_required,
            summary=ai_result.get("summary", self._generate_summary(all_passed, len(violations))),
        )

        processing_time = int((time.time() - start_time) * 1000)
        logger.info(f"Compliance check completed in {processing_time}ms. Score: {compliance_score:.1f}%")

        return result, processing_time

    def _run_technical_checks(
        self,
        image: Image.Image,
        image_info: dict[str, Any],
        requirements: dict[str, Any],
    ) -> dict[str, ComplianceCheck]:
        """Run technical compliance checks."""
        checks = {}
        width, height = image.size

        # Dimensions check
        min_w = requirements.get("min_width", 0)
        min_h = requirements.get("min_height", 0)
        max_w = requirements.get("max_width", float("inf"))
        max_h = requirements.get("max_height", float("inf"))

        dim_passed = min_w <= width <= max_w and min_h <= height <= max_h
        checks["dimensions"] = ComplianceCheck(
            passed=dim_passed,
            detected=f"{width}x{height}",
            requirement=f"Min: {min_w}x{min_h}, Max: {max_w}x{max_h}",
            issue=None if dim_passed else f"Dimensions {width}x{height} outside allowed range",
        )

        # Aspect ratio check
        required_ratios = requirements.get("aspect_ratios", [])
        if required_ratios:
            actual_ratio = width / height
            ratio_passed = False
            for ratio_str in required_ratios:
                w, h = map(int, ratio_str.split(":"))
                if abs(actual_ratio - w / h) < 0.1:
                    ratio_passed = True
                    break

            checks["aspect_ratio"] = ComplianceCheck(
                passed=ratio_passed,
                detected=f"{width}:{height}",
                requirement=f"One of: {', '.join(required_ratios)}",
                issue=None if ratio_passed else "Aspect ratio does not match requirements",
            )

        # Background check
        bg_requirement = requirements.get("background")
        if bg_requirement:
            has_white = image_info.get("has_white_background", False)
            bg_passed = has_white if bg_requirement == "pure_white" else True

            checks["background"] = ComplianceCheck(
                passed=bg_passed,
                detected="White" if has_white else "Non-white",
                requirement=bg_requirement,
                issue=None if bg_passed else f"Background should be {bg_requirement}",
            )

        # Image quality check (based on resolution)
        megapixels = image_info.get("megapixels", 0)
        quality_passed = megapixels >= 1  # At least 1MP for reasonable quality

        checks["image_quality"] = ComplianceCheck(
            passed=quality_passed,
            detected=f"{megapixels}MP",
            requirement="At least 1MP",
            issue=None if quality_passed else "Image resolution too low",
        )

        return checks

    def _extract_violations(
        self,
        checks: dict[str, ComplianceCheck],
        ai_result: dict[str, Any],
    ) -> list[ComplianceViolation]:
        """Extract violations from checks and AI result."""
        violations = []

        # From technical checks
        for check_name, check in checks.items():
            if not check.passed and check.issue:
                severity = "critical" if check_name in ["dimensions", "file_size"] else "major"
                violations.append(ComplianceViolation(
                    rule=check_name.replace("_", " ").title(),
                    severity=severity,
                    description=check.issue,
                    fix=f"Adjust {check_name.replace('_', ' ')} to meet requirements",
                ))

        # From AI result
        for v in ai_result.get("violations", []):
            violations.append(ComplianceViolation(
                rule=v.get("rule", "Unknown"),
                severity=v.get("severity", "minor"),
                description=v.get("description", ""),
                fix=v.get("fix", ""),
            ))

        return violations

    def _extract_warnings(self, ai_result: dict[str, Any]) -> list[ComplianceWarning]:
        """Extract warnings from AI result."""
        warnings = []
        for w in ai_result.get("warnings", []):
            warnings.append(ComplianceWarning(
                aspect=w.get("aspect", "Unknown"),
                message=w.get("message", ""),
                recommendation=w.get("recommendation", ""),
            ))
        return warnings

    def _generate_summary(self, passed: bool, violation_count: int) -> str:
        """Generate a compliance summary."""
        if passed:
            return "Image meets all compliance requirements and is ready for listing."
        elif violation_count == 1:
            return "Image has 1 compliance issue that needs to be addressed before listing."
        else:
            return f"Image has {violation_count} compliance issues that need to be addressed before listing."

    async def quick_check(self, image: Image.Image, platform: Platform | str) -> bool:
        """
        Quick pass/fail compliance check.

        Args:
            image: PIL Image
            platform: Target platform

        Returns:
            True if image passes compliance
        """
        result, _ = await self.check(image, platform)
        return result.overall_compliance
