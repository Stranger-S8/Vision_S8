"""
A/B Test Predictor for Vision_S8.

Predicts performance of product image variants.
"""

import time
from typing import Any

from PIL import Image

from ..models.schemas import ABTestResult, VariantRanking, VariantScore
from ..services.gemini_service import GeminiService, get_gemini_service
from ..utils.logger import get_logger

logger = get_logger("vision_s8.ab_predictor")


class ABTestPredictor:
    """Engine for predicting A/B test performance of image variants."""

    def __init__(self, gemini_service: GeminiService | None = None):
        """Initialize the predictor."""
        self._gemini = gemini_service or get_gemini_service()

    async def predict(
        self,
        variant_images: list[Image.Image],
    ) -> tuple[ABTestResult, int]:
        """
        Predict A/B test performance for image variants.

        Args:
            variant_images: List of product image variants (same product, different shots)

        Returns:
            Tuple of (ABTestResult, processing_time_ms)
        """
        if len(variant_images) < 2:
            raise ValueError("At least 2 variants required for A/B test prediction")

        start_time = time.time()

        logger.info(f"Predicting A/B test for {len(variant_images)} variants")

        # Get AI prediction
        raw_result = await self._gemini.predict_ab_test(variant_images)

        # Parse into structured format
        result = self._parse_ab_result(raw_result)

        processing_time = int((time.time() - start_time) * 1000)
        winner = result.ranking[0].variant_index if result.ranking else "N/A"
        logger.info(f"A/B prediction completed in {processing_time}ms. Winner: Variant {winner}")

        return result, processing_time

    def _parse_ab_result(self, raw: dict[str, Any]) -> ABTestResult:
        """Parse raw AI result into structured format."""
        # Parse variant scores
        variants = []
        for v in raw.get("variants", []):
            variants.append(VariantScore(
                index=v.get("index", 0),
                predicted_ctr_score=self._clamp_score(v.get("predicted_ctr_score", 50), 100),
                predicted_conversion_score=self._clamp_score(v.get("predicted_conversion_score", 50), 100),
                overall_effectiveness=self._clamp_score(v.get("overall_effectiveness", 5), 10),
                strengths=v.get("strengths", []),
                weaknesses=v.get("weaknesses", []),
            ))

        # Parse rankings
        rankings = []
        for r in raw.get("ranking", []):
            rankings.append(VariantRanking(
                rank=r.get("rank", 0),
                variant_index=r.get("variant_index", 0),
                confidence=r.get("confidence", "medium"),
                expected_performance_lift=r.get("expected_performance_lift", "0%"),
            ))

        # Sort rankings by rank
        rankings.sort(key=lambda x: x.rank)

        return ABTestResult(
            variants=variants,
            ranking=rankings,
            winner_analysis=raw.get("winner_analysis", {}),
            testing_recommendations=raw.get("testing_recommendations", {}),
            detailed_comparison=raw.get("detailed_comparison", {}),
            summary=raw.get("summary", ""),
        )

    @staticmethod
    def _clamp_score(value: float | int | None, max_val: float) -> float:
        """Clamp score to valid range."""
        if value is None:
            return max_val / 2
        return max(0, min(max_val, float(value)))

    async def get_winner(self, variant_images: list[Image.Image]) -> int:
        """
        Get the predicted winning variant index.

        Args:
            variant_images: List of variant images

        Returns:
            Index of predicted winning variant (1-based)
        """
        result, _ = await self.predict(variant_images)
        if result.ranking:
            return result.ranking[0].variant_index
        return 1
