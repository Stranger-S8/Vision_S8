"""
API Dependencies for Vision_S8.

Provides shared dependencies for FastAPI routes.
"""

import time
import uuid
from typing import Annotated

from fastapi import Depends, HTTPException, Request, UploadFile, status
from PIL import Image

from ..config import settings
from ..core.auditor import ProductAuditor
from ..core.comparator import ImageComparator
from ..core.enhancer import ImageEnhancer
from ..core.ab_predictor import ABTestPredictor
from ..core.platform_optimizer import PlatformOptimizer
from ..core.seo_analyzer import SEOAnalyzer
from ..core.compliance_checker import ComplianceChecker
from ..models.database import AsyncSession, get_session
from ..services.gemini_service import GeminiService, get_gemini_service
from ..services.image_service import ImageService, get_image_service
from ..utils.file_handler import FileHandler, get_file_handler
from ..utils.logger import get_request_logger


async def get_db() -> AsyncSession:
    """Get database session."""
    async for session in get_session():
        yield session


async def get_request_id(request: Request) -> str:
    """Get or generate request ID."""
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = str(uuid.uuid4())
    return request_id


async def log_request(request: Request, request_id: str = Depends(get_request_id)):
    """Log incoming request."""
    logger = get_request_logger(request_id)
    logger.info(f"{request.method} {request.url.path}")
    request.state.request_id = request_id
    request.state.start_time = time.time()
    return logger


def get_auditor() -> ProductAuditor:
    """Get product auditor instance."""
    return ProductAuditor()


def get_enhancer() -> ImageEnhancer:
    """Get image enhancer instance."""
    return ImageEnhancer()


def get_comparator() -> ImageComparator:
    """Get image comparator instance."""
    return ImageComparator()


def get_ab_predictor() -> ABTestPredictor:
    """Get A/B test predictor instance."""
    return ABTestPredictor()


def get_platform_optimizer() -> PlatformOptimizer:
    """Get platform optimizer instance."""
    return PlatformOptimizer()


def get_seo_analyzer() -> SEOAnalyzer:
    """Get SEO analyzer instance."""
    return SEOAnalyzer()


def get_compliance_checker() -> ComplianceChecker:
    """Get compliance checker instance."""
    return ComplianceChecker()


async def validate_image_upload(file: UploadFile) -> tuple[bytes, str]:
    """
    Validate and read an uploaded image file.

    Args:
        file: Uploaded file

    Returns:
        Tuple of (file_content, filename)

    Raises:
        HTTPException: If validation fails
    """
    file_handler = get_file_handler()

    # Read file content
    content = await file.read()

    # Validate
    is_valid, error = file_handler.validate_file(file.filename or "unknown", len(content))
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error,
        )

    return content, file.filename or "uploaded_image.jpg"


async def load_image_from_upload(file: UploadFile) -> Image.Image:
    """
    Load a PIL Image from an uploaded file.

    Args:
        file: Uploaded file

    Returns:
        PIL Image object
    """
    content, _ = await validate_image_upload(file)
    image_service = get_image_service()
    return image_service.load_image_from_bytes(content)


async def load_multiple_images(files: list[UploadFile]) -> list[Image.Image]:
    """
    Load multiple PIL Images from uploaded files.

    Args:
        files: List of uploaded files

    Returns:
        List of PIL Image objects
    """
    images = []
    for file in files:
        image = await load_image_from_upload(file)
        images.append(image)
    return images


# Type aliases for dependency injection
DbSession = Annotated[AsyncSession, Depends(get_db)]
RequestId = Annotated[str, Depends(get_request_id)]
Auditor = Annotated[ProductAuditor, Depends(get_auditor)]
Enhancer = Annotated[ImageEnhancer, Depends(get_enhancer)]
Comparator = Annotated[ImageComparator, Depends(get_comparator)]
ABPredictor = Annotated[ABTestPredictor, Depends(get_ab_predictor)]
Optimizer = Annotated[PlatformOptimizer, Depends(get_platform_optimizer)]
SEOAnalyzerDep = Annotated[SEOAnalyzer, Depends(get_seo_analyzer)]
ComplianceCheckerDep = Annotated[ComplianceChecker, Depends(get_compliance_checker)]
