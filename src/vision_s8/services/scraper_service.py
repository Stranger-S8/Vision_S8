"""
Web scraping service for Vision_S8.

Handles fetching competitor product images from URLs and marketplaces.
"""

import asyncio
import re
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from PIL import Image

from ..config import settings
from ..utils.logger import get_logger

logger = get_logger("vision_s8.scraper")


class ScraperService:
    """Service for scraping product images from websites."""

    # Common image selectors for different platforms
    PLATFORM_SELECTORS = {
        "amazon": [
            "#landingImage",
            "#imgBlkFront",
            ".a-dynamic-image",
            "#main-image-container img",
            ".imgTagWrapper img",
        ],
        "ebay": [
            "#icImg",
            ".ux-image-magnify__image--original",
            ".s-item__image img",
        ],
        "etsy": [
            ".listing-page-image-carousel img",
            ".carousel-image img",
            "[data-carousel-pane] img",
        ],
        "shopify": [
            ".product__media img",
            ".product-single__photo img",
            ".product-featured-img",
            "[data-product-featured-image]",
        ],
        "walmart": [
            "[data-testid='hero-image'] img",
            ".hover-zoom-hero-image img",
            ".prod-hero-image img",
        ],
        "generic": [
            'meta[property="og:image"]',
            ".product-image img",
            ".main-image img",
            "#product-image",
            ".gallery img",
            '[class*="product"] img',
            '[class*="main-image"] img',
        ],
    }

    # User agent to mimic a real browser
    USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    def __init__(self):
        """Initialize the scraper service."""
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                headers={
                    "User-Agent": self.USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _detect_platform(self, url: str) -> str:
        """
        Detect the platform from a URL.

        Args:
            url: Website URL

        Returns:
            Platform name or "generic"
        """
        domain = urlparse(url).netloc.lower()

        platform_domains = {
            "amazon": ["amazon.com", "amazon.co.uk", "amazon.de", "amazon.ca"],
            "ebay": ["ebay.com", "ebay.co.uk", "ebay.de"],
            "etsy": ["etsy.com"],
            "walmart": ["walmart.com"],
            "shopify": [],  # Detected by content
        }

        for platform, domains in platform_domains.items():
            for d in domains:
                if d in domain:
                    return platform

        return "generic"

    def _extract_image_urls(
        self,
        soup: BeautifulSoup,
        base_url: str,
        platform: str,
    ) -> list[str]:
        """
        Extract product image URLs from parsed HTML.

        Args:
            soup: BeautifulSoup object
            base_url: Base URL for resolving relative URLs
            platform: Detected platform

        Returns:
            List of image URLs
        """
        image_urls = []
        selectors = self.PLATFORM_SELECTORS.get(platform, [])
        selectors.extend(self.PLATFORM_SELECTORS["generic"])

        for selector in selectors:
            elements = soup.select(selector)
            for elem in elements:
                url = None

                # Handle meta tags
                if elem.name == "meta":
                    url = elem.get("content")
                # Handle img tags
                elif elem.name == "img":
                    # Try various attributes
                    url = elem.get("data-src") or elem.get("data-lazy-src") or elem.get("src")
                    # Also check srcset for high-res images
                    srcset = elem.get("srcset") or elem.get("data-srcset")
                    if srcset:
                        # Parse srcset and get highest resolution
                        srcset_urls = self._parse_srcset(srcset)
                        if srcset_urls:
                            url = srcset_urls[-1]  # Usually highest res is last

                if url:
                    # Resolve relative URLs
                    if not url.startswith(("http://", "https://", "//")):
                        url = urljoin(base_url, url)
                    elif url.startswith("//"):
                        url = "https:" + url

                    # Filter out tiny images (icons, thumbnails)
                    if self._is_likely_product_image(url):
                        image_urls.append(url)

        # Remove duplicates while preserving order
        seen = set()
        unique_urls = []
        for url in image_urls:
            normalized = self._normalize_url(url)
            if normalized not in seen:
                seen.add(normalized)
                unique_urls.append(url)

        return unique_urls

    def _parse_srcset(self, srcset: str) -> list[str]:
        """Parse srcset attribute and return URLs sorted by size."""
        urls_with_size = []
        parts = srcset.split(",")

        for part in parts:
            part = part.strip()
            match = re.match(r"(\S+)(?:\s+(\d+)w)?", part)
            if match:
                url = match.group(1)
                size = int(match.group(2)) if match.group(2) else 0
                urls_with_size.append((url, size))

        # Sort by size and return URLs
        urls_with_size.sort(key=lambda x: x[1])
        return [url for url, _ in urls_with_size]

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication."""
        # Remove common query parameters
        parsed = urlparse(url)
        return f"{parsed.netloc}{parsed.path}".lower()

    def _is_likely_product_image(self, url: str) -> bool:
        """Check if URL is likely a product image (not icon/logo)."""
        url_lower = url.lower()

        # Skip common non-product patterns
        skip_patterns = [
            "logo", "icon", "sprite", "button", "banner",
            "1x1", "pixel", "tracking", "spacer", "blank",
            ".svg", ".gif",
        ]

        return not any(pattern in url_lower for pattern in skip_patterns)

    async def fetch_page(self, url: str) -> str:
        """
        Fetch a webpage.

        Args:
            url: Page URL

        Returns:
            HTML content
        """
        client = await self._get_client()
        response = await client.get(url)
        response.raise_for_status()
        return response.text

    async def fetch_image(self, url: str) -> Image.Image:
        """
        Fetch an image from URL.

        Args:
            url: Image URL

        Returns:
            PIL Image object
        """
        client = await self._get_client()
        response = await client.get(url)
        response.raise_for_status()

        from io import BytesIO
        image = Image.open(BytesIO(response.content))

        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")

        return image

    async def scrape_product_images(
        self,
        url: str,
        max_images: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Scrape product images from a URL.

        Args:
            url: Product page URL
            max_images: Maximum number of images to fetch

        Returns:
            List of dicts with image data and metadata
        """
        logger.info(f"Scraping images from: {url}")

        try:
            html = await self.fetch_page(url)
            soup = BeautifulSoup(html, "html.parser")
            platform = self._detect_platform(url)

            image_urls = self._extract_image_urls(soup, url, platform)
            logger.debug(f"Found {len(image_urls)} potential images")

            results = []
            for img_url in image_urls[:max_images]:
                try:
                    image = await self.fetch_image(img_url)
                    results.append({
                        "url": img_url,
                        "image": image,
                        "source": url,
                        "platform": platform,
                        "size": image.size,
                    })
                    logger.debug(f"Fetched image: {img_url}")
                except Exception as e:
                    logger.warning(f"Failed to fetch image {img_url}: {e}")
                    continue

            return results

        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            raise

    async def scrape_multiple_urls(
        self,
        urls: list[str],
        max_images_per_url: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Scrape images from multiple URLs concurrently.

        Args:
            urls: List of product page URLs
            max_images_per_url: Max images per URL

        Returns:
            Combined list of image results
        """
        tasks = [
            self.scrape_product_images(url, max_images_per_url)
            for url in urls
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_images = []
        for result in results:
            if isinstance(result, list):
                all_images.extend(result)
            elif isinstance(result, Exception):
                logger.warning(f"Scrape task failed: {result}")

        return all_images


# Singleton instance
_scraper_service: ScraperService | None = None


def get_scraper_service() -> ScraperService:
    """Get the scraper service singleton."""
    global _scraper_service
    if _scraper_service is None:
        _scraper_service = ScraperService()
    return _scraper_service
