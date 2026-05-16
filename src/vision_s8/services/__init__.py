"""External service integrations.

This module intentionally avoids eager imports so optional native dependencies
do not fail package import in restricted environments.
"""

from importlib import import_module
from typing import Any

__all__ = ["GeminiService", "ImageService", "ScraperService"]


def __getattr__(name: str) -> Any:
    """Lazily resolve service classes on first access."""
    if name == "GeminiService":
        return import_module(".gemini_service", __name__).GeminiService
    if name == "ImageService":
        return import_module(".image_service", __name__).ImageService
    if name == "ScraperService":
        return import_module(".scraper_service", __name__).ScraperService
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
