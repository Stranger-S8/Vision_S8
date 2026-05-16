"""
Core Product Auditor for Vision_S8.

Performs comprehensive analysis of product images using both
traditional Computer Vision algorithms and AI (Gemini) analysis.
"""

import time
from typing import Any

from PIL import Image

from ..models.schemas import AuditResult, Improvement, Priority, ScoreBreakdown, TechnicalAnalysis
from ..services.gemini_service import GeminiService, get_gemini_service
from ..services.image_service import ImageService, get_image_service
from ..utils.logger import get_logger
from .cv_analyzer import CVAnalyzer, get_cv_analyzer

logger = get_logger("vision_s8.auditor")


class ProductAuditor:
    """
    Core engine for auditing product images.
    
    Uses a HYBRID approach combining:
    1. Traditional Computer Vision (OpenCV) for technical metrics
    2. AI (Gemini) for subjective quality assessment
    """

    def __init__(
        self,
        gemini_service: GeminiService | None = None,
        image_service: ImageService | None = None,
        cv_analyzer: CVAnalyzer | None = None,
    ):
        """Initialize the auditor."""
        self._gemini = gemini_service or get_gemini_service()
        self._image = image_service or get_image_service()
        self._cv = cv_analyzer or get_cv_analyzer()

    async def audit(self, image: Image.Image, use_ai: bool = True) -> tuple[AuditResult, int]:
        """
        Perform a comprehensive audit on a product image.

        Uses HYBRID analysis:
        1. Computer Vision algorithms for objective technical metrics
        2. AI (Gemini) for subjective quality assessment

        Args:
            image: PIL Image to audit
            use_ai: Whether to include AI analysis (default True)

        Returns:
            Tuple of (AuditResult, processing_time_ms)
        """
        start_time = time.time()

        logger.info("Starting HYBRID product image audit (CV + AI)")

        # Step 1: Computer Vision Analysis (always run)
        logger.info("Running Computer Vision analysis...")
        cv_analysis = self._cv.analyze(image)
        cv_quality = self._cv.get_quality_score(cv_analysis)
        logger.info(f"CV Analysis complete. Technical score: {cv_quality['overall_score']}")

        # Step 2: AI Analysis (optional but recommended)
        if use_ai:
            logger.info("Running AI (Gemini) analysis...")
            raw_result = await self._gemini.audit_image(image)
        else:
            raw_result = {}

        # Get technical image info
        image_info = self._image.get_image_info(image)

        # Merge CV and AI results
        result = self._parse_audit_result(raw_result, image_info, cv_analysis, cv_quality)

        processing_time = int((time.time() - start_time) * 1000)
        logger.info(f"HYBRID Audit completed in {processing_time}ms. Final Score: {result.overall_score}")

        return result, processing_time
    
    def audit_cv_only(self, image: Image.Image) -> tuple[dict[str, Any], int]:
        """
        Perform CV-only analysis (no AI required).
        
        Useful for quick technical analysis without API calls.

        Args:
            image: PIL Image to audit

        Returns:
            Tuple of (cv_analysis_dict, processing_time_ms)
        """
        start_time = time.time()
        
        logger.info("Running CV-only audit...")
        cv_analysis = self._cv.analyze(image)
        cv_quality = self._cv.get_quality_score(cv_analysis)
        
        # Add quality score to analysis
        cv_analysis["quality_score"] = cv_quality
        
        processing_time = int((time.time() - start_time) * 1000)
        logger.info(f"CV-only audit completed in {processing_time}ms. Score: {cv_quality['overall_score']}")
        
        return cv_analysis, processing_time

    def _parse_audit_result(
        self,
        raw_result: dict[str, Any],
        image_info: dict[str, Any],
        cv_analysis: dict[str, Any] | None = None,
        cv_quality: dict[str, Any] | None = None,
    ) -> AuditResult:
        """
        Parse and merge CV + AI results into structured format.
        
        CV scores are used for technical metrics (sharpness, brightness, etc.)
        AI scores are used for subjective metrics (composition, professionalism)
        """
        # Extract AI scores with defaults
        raw_scores = raw_result.get("scores", {})
        
        # Merge with CV scores when available
        cv = cv_analysis or {}
        # HYBRID scoring: Use CV metrics where available, AI for subjective
        scores = ScoreBreakdown(
            # CV-based scores (objective/technical)
            lighting=self._clamp_score(
                cv.get("brightness", {}).get("score") or raw_scores.get("lighting", 5)
            ),
            sharpness=self._clamp_score(
                cv.get("sharpness", {}).get("score") or raw_scores.get("sharpness", 5)
            ),
            background=self._clamp_score(
                cv.get("background_analysis", {}).get("uniformity_score") or raw_scores.get("background", 5)
            ),
            color_accuracy=self._clamp_score(
                cv.get("histogram_analysis", {}).get("exposure_score") or raw_scores.get("color_accuracy", 5)
            ),
            # AI-based scores (subjective)
            composition=self._clamp_score(raw_scores.get("composition", 5)),
            product_focus=self._clamp_score(raw_scores.get("product_focus", 5)),
            professionalism=self._clamp_score(raw_scores.get("professionalism", 5)),
        )

        # Extract technical analysis - prefer CV data
        raw_tech = raw_result.get("technical_analysis", {})
        technical = TechnicalAnalysis(
            estimated_lighting_type=raw_tech.get("estimated_lighting_type") or 
                cv.get("brightness", {}).get("level", "unknown"),
            background_type=raw_tech.get("background_type") or 
                cv.get("background_analysis", {}).get("type", "unknown"),
            product_visibility=raw_tech.get("product_visibility", "unknown"),
            image_quality=cv_quality.get("quality_tier", "unknown") if cv_quality else 
                raw_tech.get("image_quality", "unknown"),
        )

        # Parse improvements
        improvements = []
        for imp in raw_result.get("improvements", []):
            priority_str = imp.get("priority", "medium").lower()
            priority = Priority.HIGH if priority_str == "high" else (
                Priority.LOW if priority_str == "low" else Priority.MEDIUM
            )
            improvements.append(Improvement(
                issue=imp.get("issue", "Unknown issue"),
                priority=priority,
                suggestion=imp.get("suggestion", "No suggestion provided"),
            ))

        # Calculate overall score - blend CV and AI scores
        ai_overall = raw_result.get("overall_score")
        cv_overall = cv_quality.get("overall_score") if cv_quality else None
        
        if ai_overall is not None and cv_overall is not None:
            # Weighted blend: 40% CV (technical), 60% AI (perceptual)
            overall = (cv_overall * 0.4) + (ai_overall * 0.6)
        elif cv_overall is not None:
            overall = cv_overall
        elif ai_overall is not None:
            overall = ai_overall
        else:
            # Calculate from component scores
            overall = sum([
                scores.lighting,
                scores.composition,
                scores.background,
                scores.product_focus,
                scores.color_accuracy,
                scores.sharpness,
                scores.professionalism,
            ]) / 7
        overall = self._clamp_score(overall)

        return AuditResult(
            overall_score=round(overall, 1),
            scores=scores,
            technical_analysis=technical,
            strengths=raw_result.get("strengths", []),
            improvements=improvements,
            professional_assessment=raw_result.get(
                "professional_assessment",
                "Assessment not available."
            ),
            competitor_comparison=raw_result.get(
                "competitor_comparison",
                "Comparison not available."
            ),
        )

    @staticmethod
    def _clamp_score(score: float | int | None) -> float:
        """Clamp score to valid range."""
        if score is None:
            return 5.0
        return max(0.0, min(10.0, float(score)))

    async def quick_score(self, image: Image.Image) -> float:
        """
        Get a quick overall score without full analysis.

        Args:
            image: PIL Image

        Returns:
            Overall score (0-10)
        """
        result, _ = await self.audit(image)
        return result.overall_score
