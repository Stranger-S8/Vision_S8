"""
Platform Optimizer for Vision_S8.

Optimizes images for specific e-commerce platforms.
"""

import json
import time
from pathlib import Path
from typing import Any

from PIL import Image

from ..config import settings
from ..models.schemas import Platform, PlatformOptimizeResult, PlatformRequirement
from ..services.gemini_service import GeminiService, get_gemini_service
from ..services.image_service import ImageService, get_image_service
from ..utils.logger import get_logger

logger = get_logger("vision_s8.platform_optimizer")


class PlatformOptimizer:
    """Engine for platform-specific image optimization."""

    def __init__(
        self,
        gemini_service: GeminiService | None = None,
        image_service: ImageService | None = None,
    ):
        """Initialize the optimizer."""
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

    async def optimize(
        self,
        image: Image.Image,
        platform: Platform | str,
    ) -> tuple[PlatformOptimizeResult, int]:
        """
        Analyze and optimize image for a specific platform.

        Args:
            image: PIL Image to optimize
            platform: Target platform

        Returns:
            Tuple of (PlatformOptimizeResult, processing_time_ms)
        """
        start_time = time.time()

        platform_str = platform.value if isinstance(platform, Platform) else platform.lower()
        logger.info(f"Optimizing image for {platform_str}")

        # Get platform rules
        rules = self._platform_rules.get(platform_str, {})
        requirements_spec = rules.get("requirements", {})

        # Get image info
        image_info = self._image.get_image_info(image)

        # Check technical requirements
        requirements = self._check_requirements(image, image_info, requirements_spec)

        # Get AI assessment for subjective criteria
        compliance_result = await self._gemini.check_compliance(image, platform_str)

        # Calculate platform score
        passed_count = sum(1 for r in requirements if r.passed)
        platform_score = (passed_count / len(requirements)) * 10 if requirements else 5

        # Adjust based on AI assessment
        ai_score = compliance_result.get("compliance_score", 50) / 10
        platform_score = (platform_score + ai_score) / 2

        # Determine if ready to list
        critical_requirements = [r for r in requirements if "dimension" in r.name.lower() or "size" in r.name.lower()]
        critical_passed = all(r.passed for r in critical_requirements)
        overall_compliance = passed_count >= len(requirements) * 0.8 and critical_passed

        result = PlatformOptimizeResult(
            platform=platform_str,
            platform_score=round(platform_score, 1),
            overall_compliance=overall_compliance,
            requirements=requirements,
            best_practices=rules.get("best_practices", []),
            recommendations=self._generate_recommendations(requirements, rules),
            ready_to_list=overall_compliance and platform_score >= 7,
        )

        processing_time = int((time.time() - start_time) * 1000)
        logger.info(f"Platform optimization completed in {processing_time}ms. Score: {platform_score:.1f}")

        return result, processing_time

    def _check_requirements(
        self,
        image: Image.Image,
        image_info: dict[str, Any],
        requirements: dict[str, Any],
    ) -> list[PlatformRequirement]:
        """Check image against platform requirements."""
        results = []
        width, height = image.size

        # Check minimum dimensions
        min_width = requirements.get("min_width", 0)
        min_height = requirements.get("min_height", 0)
        if min_width or min_height:
            passed = width >= min_width and height >= min_height
            results.append(PlatformRequirement(
                name="Minimum Dimensions",
                required=f"{min_width}x{min_height}px",
                actual=f"{width}x{height}px",
                passed=passed,
                message=None if passed else f"Image is too small. Minimum: {min_width}x{min_height}px",
            ))

        # Check maximum dimensions
        max_width = requirements.get("max_width")
        max_height = requirements.get("max_height")
        if max_width and max_height:
            passed = width <= max_width and height <= max_height
            results.append(PlatformRequirement(
                name="Maximum Dimensions",
                required=f"{max_width}x{max_height}px max",
                actual=f"{width}x{height}px",
                passed=passed,
                message=None if passed else f"Image is too large. Maximum: {max_width}x{max_height}px",
            ))

        # Check aspect ratio
        required_ratios = requirements.get("aspect_ratios", [])
        if required_ratios:
            actual_ratio = round(width / height, 2)
            ratio_matched = False
            for ratio_str in required_ratios:
                w, h = map(int, ratio_str.split(":"))
                target_ratio = round(w / h, 2)
                if abs(actual_ratio - target_ratio) < 0.1:  # 10% tolerance
                    ratio_matched = True
                    break

            results.append(PlatformRequirement(
                name="Aspect Ratio",
                required=", ".join(required_ratios),
                actual=f"{width}:{height} ({actual_ratio})",
                passed=ratio_matched,
                message=None if ratio_matched else f"Aspect ratio should be one of: {', '.join(required_ratios)}",
            ))

        # Check background (based on image analysis)
        required_bg = requirements.get("background")
        if required_bg:
            has_white = image_info.get("has_white_background", False)
            if required_bg == "pure_white":
                passed = has_white
                results.append(PlatformRequirement(
                    name="Background",
                    required="Pure white (RGB 255,255,255)",
                    actual="White" if has_white else "Not white",
                    passed=passed,
                    message=None if passed else "Background should be pure white",
                ))

        # Check if square (for platforms that require it)
        if "1:1" in required_ratios:
            is_square = image_info.get("is_square", False)
            results.append(PlatformRequirement(
                name="Square Format",
                required="1:1 (square)",
                actual="Square" if is_square else "Not square",
                passed=is_square,
                message=None if is_square else "Image should be square (1:1 aspect ratio)",
            ))

        return results

    def _generate_recommendations(
        self,
        requirements: list[PlatformRequirement],
        rules: dict[str, Any],
    ) -> list[str]:
        """Generate actionable recommendations based on failed requirements."""
        recommendations = []

        for req in requirements:
            if not req.passed and req.message:
                recommendations.append(req.message)

        # Add general best practices if few specific recommendations
        if len(recommendations) < 3:
            best_practices = rules.get("best_practices", [])
            for practice in best_practices[:3 - len(recommendations)]:
                if practice not in recommendations:
                    recommendations.append(practice)

        return recommendations

    def get_supported_platforms(self) -> list[str]:
        """Get list of supported platforms."""
        return list(self._platform_rules.keys())
