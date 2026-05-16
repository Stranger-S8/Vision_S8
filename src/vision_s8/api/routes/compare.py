"""
Comparison endpoint for Vision_S8.

Handles competitor image comparison.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile
from pydantic import HttpUrl

from ..dependencies import Comparator, DbSession, load_image_from_upload, load_multiple_images
from ...models.schemas import APIResponse, CompareResponse, ComparisonMode

router = APIRouter(prefix="/compare", tags=["Comparison"])


@router.post("", response_model=APIResponse)
async def compare_images(
    main_image: Annotated[UploadFile, File(description="Your product image")],
    comparator: Comparator,
    db: DbSession,
    competitor_images: Annotated[
        list[UploadFile] | None,
        File(description="Competitor images (for manual mode)")
    ] = None,
    competitor_urls: Annotated[
        str | None,
        Form(description="Comma-separated competitor URLs (for URL mode)")
    ] = None,
    mode: Annotated[
        ComparisonMode,
        Form(description="Comparison mode: manual, url, or marketplace")
    ] = ComparisonMode.MANUAL,
) -> APIResponse:
    """
    Compare your product image against competitor images.

    Modes:
    - **manual**: Upload competitor images directly
    - **url**: Scrape competitor images from provided URLs
    - **marketplace**: (Coming soon) Search marketplace for similar products

    Returns competitive analysis with scores, gaps, and recommendations.
    """
    # Load main image
    main = await load_image_from_upload(main_image)

    if mode == ComparisonMode.MANUAL:
        if not competitor_images:
            return APIResponse(
                success=False,
                message="Competitor images required for manual mode",
            )

        # Load competitor images
        competitors = await load_multiple_images(competitor_images)

        # Perform comparison
        result, processing_time = await comparator.compare_with_images(main, competitors)
        competitor_sources = [f.filename for f in competitor_images]

    elif mode == ComparisonMode.URL:
        if not competitor_urls:
            return APIResponse(
                success=False,
                message="Competitor URLs required for URL mode",
            )

        # Parse URLs
        urls = [url.strip() for url in competitor_urls.split(",") if url.strip()]
        if not urls:
            return APIResponse(
                success=False,
                message="At least one valid URL required",
            )

        # Scrape and compare
        result, scraped_urls, processing_time = await comparator.compare_with_urls(main, urls)
        competitor_sources = scraped_urls

    else:  # MARKETPLACE
        return APIResponse(
            success=False,
            message="Marketplace mode coming soon",
        )

    # Generate audit ID
    audit_id = str(uuid.uuid4())

    # Save to database
    from ...models.database import Audit
    audit_record = Audit(
        id=audit_id,
        image_path=main_image.filename or "main_image",
        audit_type="compare",
        score=result.main_image_score,
        result_json=result.model_dump_json(),
        processing_time_ms=processing_time,
    )
    db.add(audit_record)

    # Build response
    response_data = CompareResponse(
        audit_id=audit_id,
        main_image=main_image.filename or "main_image",
        competitor_images=competitor_sources,
        result=result,
        processing_time_ms=processing_time,
    )

    return APIResponse(
        success=True,
        data=response_data.model_dump(),
        meta={
            "processing_time_ms": processing_time,
            "mode": mode.value,
            "competitor_count": len(competitor_sources),
        },
    )
