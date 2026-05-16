"""
Image Comparator for Vision_S8.

Handles comparison of product images against competitors.
"""

import time
from typing import Any

from PIL import Image

from ..models.schemas import AspectComparison, CompareResult, CompetitorScore
from ..services.gemini_service import GeminiService, get_gemini_service
from ..services.image_service import ImageService, get_image_service
from ..services.scraper_service import ScraperService, get_scraper_service
from ..utils.logger import get_logger

logger = get_logger("vision_s8.comparator")


class ImageComparator:
    """Engine for comparing product images against competitors."""

    def __init__(
        self,
        gemini_service: GeminiService | None = None,
        image_service: ImageService | None = None,
        scraper_service: ScraperService | None = None,
    ):
        """Initialize the comparator."""
        self._gemini = gemini_service or get_gemini_service()
        self._image = image_service or get_image_service()
        self._scraper = scraper_service or get_scraper_service()

    async def compare_with_images(
        self,
        main_image: Image.Image,
        competitor_images: list[Image.Image],
    ) -> tuple[CompareResult, int]:
        """
        Compare main image against provided competitor images.

        Args:
            main_image: The user's product image
            competitor_images: List of competitor images

        Returns:
            Tuple of (CompareResult, processing_time_ms)
        """
        start_time = time.time()

        logger.info(f"Comparing image against {len(competitor_images)} competitors")

        # Get AI comparison
        raw_result = await self._gemini.compare_images(main_image, competitor_images)

        # Parse into structured format
        result = self._parse_compare_result(raw_result)

        processing_time = int((time.time() - start_time) * 1000)
        logger.info(f"Comparison completed in {processing_time}ms. Position: {result.competitive_position}")

        return result, processing_time

    async def compare_with_urls(
        self,
        main_image: Image.Image,
        competitor_urls: list[str],
        max_images_per_url: int = 1,
    ) -> tuple[CompareResult, list[str], int]:
        """
        Compare main image against images scraped from URLs.

        Args:
            main_image: The user's product image
            competitor_urls: List of competitor product page URLs
            max_images_per_url: Max images to fetch per URL

        Returns:
            Tuple of (CompareResult, list of scraped image URLs, processing_time_ms)
        """
        start_time = time.time()

        logger.info(f"Scraping competitor images from {len(competitor_urls)} URLs")

        # Scrape competitor images
        scraped = await self._scraper.scrape_multiple_urls(
            competitor_urls,
            max_images_per_url=max_images_per_url,
        )

        if not scraped:
            raise ValueError("No competitor images could be fetched from provided URLs")

        competitor_images = [item["image"] for item in scraped]
        scraped_urls = [item["url"] for item in scraped]

        # Perform comparison
        raw_result = await self._gemini.compare_images(main_image, competitor_images)
        result = self._parse_compare_result(raw_result, sources=scraped_urls)

        processing_time = int((time.time() - start_time) * 1000)
        logger.info(f"URL comparison completed in {processing_time}ms")

        return result, scraped_urls, processing_time

    def _parse_compare_result(
        self,
        raw: dict[str, Any],
        sources: list[str] | None = None,
    ) -> CompareResult:
        """Parse raw AI result into structured format."""
        # Parse competitor scores
        competitor_scores = []
        for i, score_data in enumerate(raw.get("competitor_scores", [])):
            source = sources[i] if sources and i < len(sources) else None
            competitor_scores.append(CompetitorScore(
                index=score_data.get("index", i + 1),
                score=score_data.get("score", 0),
                source=source,
            ))

        # Parse aspect comparisons
        aspects = {}
        for aspect_name, aspect_data in raw.get("comparison_aspects", {}).items():
            aspects[aspect_name] = AspectComparison(
                main=aspect_data.get("main", 0),
                competitor_average=aspect_data.get("competitor_average", 0),
                gap=aspect_data.get("gap", "even"),
                gap_magnitude=aspect_data.get("gap_magnitude", "none"),
            )

        return CompareResult(
            main_image_score=raw.get("main_image_score", 0),
            competitor_scores=competitor_scores,
            comparison_aspects=aspects,
            competitive_position=raw.get("competitive_position", "unknown"),
            key_differentiators=raw.get("key_differentiators", []),
            critical_gaps=raw.get("critical_gaps", []),
            action_plan=raw.get("action_plan", []),
            summary=raw.get("summary", ""),
        )

    async def quick_compare(
        self,
        main_image: Image.Image,
        competitor_images: list[Image.Image],
    ) -> str:
        """
        Get a quick competitive position assessment.

        Args:
            main_image: The user's product image
            competitor_images: List of competitor images

        Returns:
            Competitive position string
        """
        result, _ = await self.compare_with_images(main_image, competitor_images)
        return result.competitive_position
