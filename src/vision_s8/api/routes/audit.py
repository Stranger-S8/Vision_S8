"""
Audit endpoint for Vision_S8.

Handles single image quality audits using HYBRID analysis:
- Computer Vision (OpenCV) for technical metrics
- AI (Gemini) for subjective quality assessment
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ..dependencies import Auditor, DbSession, load_image_from_upload
from ...models.schemas import APIResponse, AuditResponse
from ...models.database import Audit
from ...services.image_service import get_image_service
from ...core.cv_analyzer import get_cv_analyzer

router = APIRouter(prefix="/audit", tags=["Audit"])


@router.post("", response_model=APIResponse)
async def audit_image(
    file: Annotated[UploadFile, File(description="Product image to audit")],
    auditor: Auditor,
    db: DbSession,
    use_ai: Annotated[bool, Form(description="Include AI analysis (requires API key)")] = True,
) -> APIResponse:
    """
    Perform a comprehensive HYBRID quality audit on a product image.

    **Computer Vision Analysis (OpenCV):**
    - Sharpness detection (Laplacian variance)
    - Brightness & exposure analysis
    - Contrast measurement
    - Color analysis & dominant colors
    - Background uniformity detection
    - Edge density calculation
    - Noise estimation
    - Blur detection
    - Symmetry analysis
    - Composition analysis

    **AI Analysis (Gemini):**
    - Subjective quality assessment
    - Professional composition evaluation
    - Improvement suggestions
    - Competitor comparison insights

    Set `use_ai=false` for CV-only analysis (no API key required).
    """
    # Load image
    image = await load_image_from_upload(file)

    # Perform HYBRID audit
    result, processing_time = await auditor.audit(image, use_ai=use_ai)

    # Generate audit ID
    audit_id = str(uuid.uuid4())

    # Save to database
    audit_record = Audit(
        id=audit_id,
        image_path=file.filename or "uploaded_image",
        audit_type="hybrid" if use_ai else "cv_only",
        score=result.overall_score,
        result_json=result.model_dump_json(),
        processing_time_ms=processing_time,
    )
    db.add(audit_record)

    # Build response
    response_data = AuditResponse(
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
            "analysis_type": "hybrid" if use_ai else "cv_only",
            "model_used": "gemini-2.5-flash" if use_ai else "opencv",
        },
    )


@router.post("/cv", response_model=APIResponse)
async def cv_only_audit(
    file: Annotated[UploadFile, File(description="Product image to audit")],
    db: DbSession,
) -> APIResponse:
    """
    Perform Computer Vision ONLY analysis (no AI/API key required).

    Uses OpenCV algorithms to analyze:
    - **Sharpness**: Laplacian variance method
    - **Brightness**: LAB color space analysis
    - **Contrast**: Michelson & RMS contrast
    - **Color Analysis**: HSV analysis, dominant colors via k-means
    - **Background**: Corner sampling, uniformity detection
    - **Edge Density**: Canny edge detection
    - **Noise Level**: Laplacian-based estimation
    - **Blur Detection**: FFT frequency analysis
    - **Symmetry**: Horizontal & vertical symmetry
    - **Composition**: Subject centering, rule of thirds
    - **Histogram**: Exposure & dynamic range analysis

    This is a **FREE** analysis that doesn't require an API key.
    """
    import time
    import json

    # Load image
    image = await load_image_from_upload(file)

    # Get CV analyzer
    cv_analyzer = get_cv_analyzer()

    # Perform CV analysis
    start_time = time.time()
    cv_analysis = cv_analyzer.analyze(image)
    quality_score = cv_analyzer.get_quality_score(cv_analysis)
    processing_time = int((time.time() - start_time) * 1000)

    # Generate audit ID
    audit_id = str(uuid.uuid4())

    # Save to database
    audit_record = Audit(
        id=audit_id,
        image_path=file.filename or "uploaded_image",
        audit_type="cv_only",
        score=quality_score["overall_score"],
        result_json=json.dumps(cv_analysis),
        processing_time_ms=processing_time,
    )
    db.add(audit_record)

    return APIResponse(
        success=True,
        data={
            "audit_id": audit_id,
            "image_filename": file.filename or "uploaded_image",
            "quality_score": quality_score,
            "analysis": cv_analysis,
            "processing_time_ms": processing_time,
        },
        message=f"CV Analysis complete. Quality: {quality_score['quality_tier']}",
        meta={
            "processing_time_ms": processing_time,
            "analysis_type": "computer_vision_only",
            "algorithms_used": [
                "laplacian_sharpness",
                "lab_brightness",
                "canny_edge_detection",
                "kmeans_color_clustering",
                "fft_blur_detection",
                "histogram_analysis",
            ],
        },
    )


@router.get("/{audit_id}", response_model=APIResponse)
async def get_audit(audit_id: str, db: DbSession) -> APIResponse:
    """
    Retrieve a previous audit result by ID.

    Args:
        audit_id: The unique audit identifier
    """
    from sqlalchemy import select

    result = await db.execute(select(Audit).where(Audit.id == audit_id))
    audit = result.scalar_one_or_none()

    if not audit:
        return APIResponse(
            success=False,
            message=f"Audit not found: {audit_id}",
        )

    import json
    return APIResponse(
        success=True,
        data={
            "audit_id": audit.id,
            "image_path": audit.image_path,
            "score": audit.score,
            "result": json.loads(audit.result_json) if audit.result_json else None,
            "created_at": audit.created_at.isoformat(),
        },
    )
