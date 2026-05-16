"""
SEO endpoint for Vision_S8.

Handles image SEO analysis and content generation.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from ..dependencies import DbSession, SEOAnalyzerDep, load_image_from_upload
from ...models.schemas import APIResponse, SEOResponse

router = APIRouter(prefix="/seo", tags=["SEO"])


@router.post("", response_model=APIResponse)
async def analyze_seo(
    file: Annotated[UploadFile, File(description="Product image to analyze")],
    analyzer: SEOAnalyzerDep,
    db: DbSession,
) -> APIResponse:
    """
    Analyze a product image and generate SEO-optimized content.

    Generates:
    - Alt text suggestions (primary, short, detailed)
    - SEO-friendly filename suggestions
    - Title recommendations
    - Meta description
    - Keyword extraction (primary, secondary, long-tail)
    - Schema markup suggestions
    - Platform-specific content (Pinterest, Instagram, Google Shopping)
    - Accessibility descriptions

    Use this to optimize your product images for search engines.
    """
    # Load image
    image = await load_image_from_upload(file)

    # Perform SEO analysis
    result, processing_time = await analyzer.analyze(image)

    # Generate audit ID
    audit_id = str(uuid.uuid4())

    # Save to database
    from ...models.database import Audit
    audit_record = Audit(
        id=audit_id,
        image_path=file.filename or "uploaded_image",
        audit_type="seo",
        score=result.seo_score,
        result_json=result.model_dump_json(),
        processing_time_ms=processing_time,
    )
    db.add(audit_record)

    # Build response
    response_data = SEOResponse(
        audit_id=audit_id,
        image_filename=file.filename or "uploaded_image",
        result=result,
        processing_time_ms=processing_time,
    )

    return APIResponse(
        success=True,
        data=response_data.model_dump(),
        meta={
            "processing_time_ms": processing_time,
            "detected_product": result.product_detection.detected_product,
        },
    )


@router.post("/alt-text")
async def generate_alt_text(
    file: Annotated[UploadFile, File(description="Product image")],
    analyzer: SEOAnalyzerDep,
) -> APIResponse:
    """
    Quick endpoint to generate just the alt text for an image.

    Returns primary, short, and detailed alt text suggestions.
    """
    image = await load_image_from_upload(file)
    result, _ = await analyzer.analyze(image)

    return APIResponse(
        success=True,
        data={
            "primary": result.alt_text.primary,
            "short": result.alt_text.short,
            "detailed": result.alt_text.detailed,
        },
    )


@router.post("/filename")
async def suggest_filename(
    file: Annotated[UploadFile, File(description="Product image")],
    analyzer: SEOAnalyzerDep,
) -> APIResponse:
    """
    Quick endpoint to generate SEO-friendly filename suggestions.
    """
    image = await load_image_from_upload(file)
    result, _ = await analyzer.analyze(image)

    return APIResponse(
        success=True,
        data={
            "suggestions": result.filename_suggestions,
            "current_filename": file.filename,
        },
    )
