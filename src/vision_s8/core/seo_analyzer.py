"""
SEO Analyzer for Vision_S8.

Generates SEO-optimized content for product images.
"""

import time
from typing import Any

from PIL import Image

from ..models.schemas import AltTextSuggestions, Keywords, ProductDetection, SEOResult
from ..services.gemini_service import GeminiService, get_gemini_service
from ..utils.logger import get_logger

logger = get_logger("vision_s8.seo_analyzer")


class SEOAnalyzer:
    """Engine for analyzing and generating SEO content for product images."""

    def __init__(self, gemini_service: GeminiService | None = None):
        """Initialize the analyzer."""
        self._gemini = gemini_service or get_gemini_service()

    async def analyze(self, image: Image.Image) -> tuple[SEOResult, int]:
        """
        Analyze image and generate SEO-optimized content.

        Args:
            image: PIL Image to analyze

        Returns:
            Tuple of (SEOResult, processing_time_ms)
        """
        start_time = time.time()

        logger.info("Analyzing image for SEO optimization")

        # Get AI SEO analysis
        raw_result = await self._gemini.analyze_seo(image)

        # Parse into structured format
        result = self._parse_seo_result(raw_result)

        processing_time = int((time.time() - start_time) * 1000)
        logger.info(f"SEO analysis completed in {processing_time}ms. Score: {result.seo_score}")

        return result, processing_time

    def _parse_seo_result(self, raw: dict[str, Any]) -> SEOResult:
        """Parse raw AI result into structured format."""
        # Parse product detection
        detection_raw = raw.get("product_detection", {})
        product_detection = ProductDetection(
            detected_product=detection_raw.get("detected_product", "Unknown product"),
            category=detection_raw.get("category", "Unknown"),
            subcategory=detection_raw.get("subcategory", "Unknown"),
            detected_attributes=detection_raw.get("detected_attributes", []),
        )

        # Parse alt text suggestions
        alt_raw = raw.get("alt_text", {})
        alt_text = AltTextSuggestions(
            primary=alt_raw.get("primary", "Product image"),
            short=alt_raw.get("short", "Product"),
            detailed=alt_raw.get("detailed", "Detailed product image"),
        )

        # Parse keywords
        keywords_raw = raw.get("keywords", {})
        keywords = Keywords(
            primary=keywords_raw.get("primary", []),
            secondary=keywords_raw.get("secondary", []),
            long_tail=keywords_raw.get("long_tail", []),
        )

        return SEOResult(
            product_detection=product_detection,
            alt_text=alt_text,
            filename_suggestions=raw.get("filename_suggestions", []),
            title_suggestions=raw.get("title_suggestions", []),
            meta_description=raw.get("meta_description", ""),
            keywords=keywords,
            schema_markup_suggestions=raw.get("schema_markup_suggestions", {}),
            accessibility=raw.get("accessibility", {}),
            platform_specific=raw.get("platform_specific", {}),
            seo_score=raw.get("seo_score", 5),
            optimization_tips=raw.get("optimization_tips", []),
        )

    async def generate_alt_text(self, image: Image.Image) -> str:
        """
        Generate optimized alt text for an image.

        Args:
            image: PIL Image

        Returns:
            Primary alt text string
        """
        result, _ = await self.analyze(image)
        return result.alt_text.primary

    async def generate_filename(self, image: Image.Image) -> str:
        """
        Generate SEO-friendly filename for an image.

        Args:
            image: PIL Image

        Returns:
            Suggested filename
        """
        result, _ = await self.analyze(image)
        if result.filename_suggestions:
            return result.filename_suggestions[0]
        return "product-image.jpg"
