"""
Enhancement endpoint for Vision_S8.

Handles image enhancement analysis and auto-enhancement.
"""

import io
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import StreamingResponse

from ..dependencies import DbSession, Enhancer, load_image_from_upload
from ...models.schemas import APIResponse, EnhanceResponse
from ...models.database import Audit, EnhancedImage
from ...services.image_service import get_image_service
from ...utils.file_handler import get_file_handler

router = APIRouter(prefix="/enhance", tags=["Enhancement"])


@router.post("", response_model=APIResponse)
async def enhance_image(
    file: Annotated[UploadFile, File(description="Product image to enhance")],
    enhancer: Enhancer,
    db: DbSession,
    apply_enhancements: Annotated[bool, Form(description="Apply recommended enhancements")] = True,
) -> APIResponse:
    """
    Analyze an image and optionally apply AI-recommended enhancements.

    Enhancement capabilities:
    - Background removal/replacement
    - Lighting adjustment
    - Color correction
    - Auto-cropping
    - Sharpening
    - Shadow addition

    Set `apply_enhancements=true` to automatically apply recommended improvements.
    """
    # Load image
    image = await load_image_from_upload(file)
    image_service = get_image_service()
    file_handler = get_file_handler()

    # Perform enhancement analysis
    if apply_enhancements:
        enhanced_image, result, processing_time = await enhancer.enhance(image)

        # Calculate scores
        from ...core.auditor import ProductAuditor
        auditor = ProductAuditor()
        original_result, _ = await auditor.audit(image)
        enhanced_result, _ = await auditor.audit(enhanced_image)

        score_before = original_result.overall_score
        score_after = enhanced_result.overall_score

        # Save enhanced image
        enhanced_buffer = io.BytesIO()
        enhanced_image.save(enhanced_buffer, format="JPEG", quality=95)
        enhanced_buffer.seek(0)

        enhanced_filename = f"enhanced_{uuid.uuid4().hex[:8]}.jpg"
        enhanced_path = await file_handler.save_output(
            enhanced_buffer.read(),
            enhanced_filename,
            prefix="enhanced",
        )

        enhanced_url = file_handler.get_output_url(enhanced_path)
    else:
        result, processing_time = await enhancer.analyze(image)
        enhanced_url = None
        score_before = result.current_assessment.get("overall_quality", 0)
        score_after = result.expected_score_after

    # Generate audit ID
    audit_id = str(uuid.uuid4())

    # Save to database
    audit_record = Audit(
        id=audit_id,
        image_path=file.filename or "uploaded_image",
        audit_type="enhance",
        score=score_after if apply_enhancements else score_before,
        result_json=result.model_dump_json(),
        processing_time_ms=processing_time,
    )
    db.add(audit_record)

    # Build response
    response_data = EnhanceResponse(
        audit_id=audit_id,
        original_image=file.filename or "uploaded_image",
        enhanced_image=enhanced_url,
        analysis=result,
        score_before=score_before,
        score_after=score_after if apply_enhancements else None,
        processing_time_ms=processing_time,
    )

    return APIResponse(
        success=True,
        data=response_data.model_dump(),
        message="Enhancements applied successfully" if apply_enhancements else "Enhancement analysis complete",
        meta={
            "processing_time_ms": processing_time,
            "enhancements_applied": apply_enhancements,
        },
    )


@router.post("/download")
async def enhance_and_download(
    file: Annotated[UploadFile, File(description="Product image to enhance")],
    enhancer: Enhancer,
) -> StreamingResponse:
    """
    Enhance an image and return the enhanced version directly.

    Returns the enhanced image file for download.
    """
    # Load and enhance
    image = await load_image_from_upload(file)
    enhanced_image, _, _ = await enhancer.enhance(image)

    # Convert to bytes
    buffer = io.BytesIO()
    enhanced_image.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)

    filename = f"enhanced_{file.filename}" if file.filename else "enhanced_image.jpg"

    return StreamingResponse(
        buffer,
        media_type="image/jpeg",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )
