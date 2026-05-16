"""
A/B Test endpoint for Vision_S8.

Handles A/B test performance prediction for image variants.
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from ..dependencies import ABPredictor, DbSession, load_multiple_images
from ...models.schemas import ABTestResponse, APIResponse

router = APIRouter(prefix="/ab-test", tags=["A/B Testing"])


@router.post("", response_model=APIResponse)
async def predict_ab_test(
    variants: Annotated[
        list[UploadFile],
        File(description="Image variants of the same product (2 or more)")
    ],
    predictor: ABPredictor,
    db: DbSession,
) -> APIResponse:
    """
    Predict A/B test performance for product image variants.

    Upload 2 or more variants of the same product image to receive:
    - CTR (Click-Through Rate) predictions
    - Conversion score predictions
    - Ranked recommendations
    - Detailed comparison analysis

    Use this to choose the best image before running actual A/B tests.
    """
    if len(variants) < 2:
        return APIResponse(
            success=False,
            message="At least 2 image variants required for A/B test prediction",
        )

    # Load all variant images
    images = await load_multiple_images(variants)

    # Perform prediction
    result, processing_time = await predictor.predict(images)

    # Generate audit ID
    audit_id = str(uuid.uuid4())

    # Save to database
    from ...models.database import Audit
    winner_idx = result.ranking[0].variant_index if result.ranking else 1
    winner_score = next(
        (v.overall_effectiveness for v in result.variants if v.index == winner_idx),
        0
    )

    audit_record = Audit(
        id=audit_id,
        image_path=f"ab_test_{len(variants)}_variants",
        audit_type="ab_test",
        score=winner_score,
        result_json=result.model_dump_json(),
        processing_time_ms=processing_time,
    )
    db.add(audit_record)

    # Build response
    response_data = ABTestResponse(
        audit_id=audit_id,
        variant_count=len(variants),
        result=result,
        processing_time_ms=processing_time,
    )

    return APIResponse(
        success=True,
        data=response_data.model_dump(),
        message=f"Variant {winner_idx} predicted to perform best",
        meta={
            "processing_time_ms": processing_time,
            "variant_count": len(variants),
            "predicted_winner": winner_idx,
        },
    )
