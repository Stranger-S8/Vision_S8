"""
Compliance endpoint for Vision_S8.

Handles marketplace compliance checking.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Path, UploadFile

from ..dependencies import ComplianceCheckerDep, DbSession, load_image_from_upload
from ...models.schemas import APIResponse, ComplianceResponse, Platform

router = APIRouter(prefix="/compliance", tags=["Compliance"])


@router.post("/{platform}", response_model=APIResponse)
async def check_compliance(
    file: Annotated[UploadFile, File(description="Product image to check")],
    platform: Annotated[Platform, Path(description="Target marketplace platform")],
    checker: ComplianceCheckerDep,
    db: DbSession,
) -> APIResponse:
    """
    Check if a product image meets marketplace requirements.

    Supported platforms:
    - amazon
    - shopify
    - instagram
    - ebay
    - etsy
    - walmart

    Validates:
    - Image dimensions (min/max)
    - File size limits
    - Aspect ratio requirements
    - Background requirements
    - Watermark detection
    - Text overlay detection
    - Border/frame detection
    - Product focus/coverage

    Returns pass/fail status with specific violations and auto-fix suggestions.
    """
    # Load image
    image = await load_image_from_upload(file)

    # Perform compliance check
    result, processing_time = await checker.check(image, platform)

    # Generate audit ID
    audit_id = str(uuid.uuid4())

    # Save to database
    from ...models.database import Audit
    audit_record = Audit(
        id=audit_id,
        image_path=file.filename or "uploaded_image",
        audit_type="compliance",
        platform=platform.value,
        score=result.compliance_score,
        result_json=result.model_dump_json(),
        processing_time_ms=processing_time,
    )
    db.add(audit_record)

    # Build response
    response_data = ComplianceResponse(
        audit_id=audit_id,
        image_filename=file.filename or "uploaded_image",
        platform=platform.value,
        result=result,
        processing_time_ms=processing_time,
    )

    return APIResponse(
        success=True,
        data=response_data.model_dump(),
        message="Compliant" if result.overall_compliance else f"{len(result.violations)} issues found",
        meta={
            "processing_time_ms": processing_time,
            "platform": platform.value,
            "compliant": result.overall_compliance,
            "violation_count": len(result.violations),
        },
    )


@router.get("/platforms")
async def list_platforms() -> APIResponse:
    """
    List all supported marketplace platforms.
    """
    return APIResponse(
        success=True,
        data={
            "platforms": [p.value for p in Platform],
            "descriptions": {
                "amazon": "Amazon Marketplace - Strict white background requirements",
                "shopify": "Shopify Stores - Flexible requirements",
                "instagram": "Instagram Shopping - Square format preferred",
                "ebay": "eBay Marketplace - White/light background preferred",
                "etsy": "Etsy Marketplace - Lifestyle images welcome",
                "walmart": "Walmart Marketplace - Similar to Amazon",
            },
        },
    )


@router.post("/quick/{platform}")
async def quick_compliance_check(
    file: Annotated[UploadFile, File(description="Product image")],
    platform: Annotated[Platform, Path(description="Target platform")],
    checker: ComplianceCheckerDep,
) -> APIResponse:
    """
    Quick pass/fail compliance check without detailed analysis.
    """
    image = await load_image_from_upload(file)
    is_compliant = await checker.quick_check(image, platform)

    return APIResponse(
        success=True,
        data={
            "platform": platform.value,
            "compliant": is_compliant,
            "message": "Ready to list" if is_compliant else "Does not meet requirements",
        },
    )
