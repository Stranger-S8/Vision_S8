"""
Batch processing endpoint for Vision_S8.

Handles bulk image processing with progress tracking.
"""

import asyncio
import json
import uuid
import zipfile
from datetime import datetime
from io import BytesIO
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse

from ..dependencies import DbSession, load_image_from_upload
from ...models.schemas import (
    APIResponse,
    AuditType,
    BatchItemStatus,
    BatchJobCreate,
    BatchJobDetail,
    BatchJobStatus,
    JobStatus,
    Platform,
)
from ...models.database import Audit, BatchItem, BatchJob, get_session_factory
from ...core.auditor import ProductAuditor
from ...services.image_service import get_image_service
from ...utils.report_generator import get_report_generator

router = APIRouter(prefix="/batch", tags=["Batch Processing"])


async def process_batch_job(batch_id: str, images_data: list[tuple[bytes, str]], audit_type: str, platform: str | None):
    """Background task to process a batch job."""
    session_factory = get_session_factory()
    auditor = ProductAuditor()
    image_service = get_image_service()

    async with session_factory() as db:
        # Update job status to processing
        from sqlalchemy import select, update
        await db.execute(
            update(BatchJob)
            .where(BatchJob.id == batch_id)
            .values(status="processing", started_at=datetime.utcnow())
        )
        await db.commit()

        # Process each image
        results = []
        processed = 0
        failed = 0

        for content, filename in images_data:
            item_id = str(uuid.uuid4())

            try:
                # Load image
                image = image_service.load_image_from_bytes(content)

                # Perform audit
                result, processing_time = await auditor.audit(image)

                # Save audit record
                audit_record = Audit(
                    id=str(uuid.uuid4()),
                    image_path=filename,
                    audit_type=audit_type,
                    platform=platform,
                    score=result.overall_score,
                    result_json=result.model_dump_json(),
                    processing_time_ms=processing_time,
                )
                db.add(audit_record)

                # Save batch item
                batch_item = BatchItem(
                    id=item_id,
                    batch_id=batch_id,
                    image_path=filename,
                    original_filename=filename,
                    status="completed",
                    score=result.overall_score,
                    result_json=result.model_dump_json(),
                    processing_time_ms=processing_time,
                    audit_id=audit_record.id,
                    processed_at=datetime.utcnow(),
                )
                db.add(batch_item)

                results.append({
                    "filename": filename,
                    "overall_score": result.overall_score,
                    "scores": result.scores.model_dump(),
                })

                processed += 1

            except Exception as e:
                # Save failed item
                batch_item = BatchItem(
                    id=item_id,
                    batch_id=batch_id,
                    image_path=filename,
                    original_filename=filename,
                    status="failed",
                    error_message=str(e),
                    processed_at=datetime.utcnow(),
                )
                db.add(batch_item)
                failed += 1

            # Update progress
            await db.execute(
                update(BatchJob)
                .where(BatchJob.id == batch_id)
                .values(processed_images=processed, failed_images=failed)
            )
            await db.commit()

        # Generate report
        if results:
            report_gen = get_report_generator()
            report_path = report_gen.generate_excel(results, filename=f"batch_{batch_id[:8]}.xlsx")
            report_path_str = str(report_path)
        else:
            report_path_str = None

        # Update job as completed
        await db.execute(
            update(BatchJob)
            .where(BatchJob.id == batch_id)
            .values(
                status="completed",
                completed_at=datetime.utcnow(),
                report_path=report_path_str,
            )
        )
        await db.commit()


@router.post("/upload", response_model=APIResponse)
async def create_batch_job(
    files: Annotated[list[UploadFile], File(description="Images to process")],
    background_tasks: BackgroundTasks,
    db: DbSession,
    name: Annotated[str | None, Form(description="Batch job name")] = None,
    audit_type: Annotated[AuditType, Form(description="Type of audit")] = AuditType.SINGLE,
    platform: Annotated[Platform | None, Form(description="Target platform")] = None,
) -> APIResponse:
    """
    Create a new batch processing job.

    Upload multiple images or a ZIP file to process them in bulk.
    The job runs in the background with progress tracking.

    Returns a job ID to track progress.
    """
    if not files:
        return APIResponse(
            success=False,
            message="No files provided",
        )

    # Read all files
    images_data: list[tuple[bytes, str]] = []
    image_service = get_image_service()

    for file in files:
        content = await file.read()
        filename = file.filename or f"image_{len(images_data) + 1}"

        # Handle ZIP files
        if filename.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(BytesIO(content)) as zf:
                    for name in zf.namelist():
                        if image_service.is_supported_format(name):
                            img_content = zf.read(name)
                            images_data.append((img_content, name))
            except zipfile.BadZipFile:
                return APIResponse(
                    success=False,
                    message="Invalid ZIP file",
                )
        elif image_service.is_supported_format(filename):
            images_data.append((content, filename))

    if not images_data:
        return APIResponse(
            success=False,
            message="No valid images found",
        )

    # Create batch job
    batch_id = str(uuid.uuid4())
    batch_job = BatchJob(
        id=batch_id,
        name=name or f"Batch {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        status="pending",
        total_images=len(images_data),
        audit_type=audit_type.value,
        platform=platform.value if platform else None,
    )
    db.add(batch_job)

    # Start background processing
    background_tasks.add_task(
        process_batch_job,
        batch_id,
        images_data,
        audit_type.value,
        platform.value if platform else None,
    )

    return APIResponse(
        success=True,
        data={
            "batch_id": batch_id,
            "name": batch_job.name,
            "total_images": len(images_data),
            "status": "pending",
        },
        message=f"Batch job created with {len(images_data)} images",
    )


@router.get("/{batch_id}", response_model=APIResponse)
async def get_batch_status(batch_id: str, db: DbSession) -> APIResponse:
    """
    Get the status of a batch processing job.
    """
    from sqlalchemy import select

    result = await db.execute(select(BatchJob).where(BatchJob.id == batch_id))
    job = result.scalar_one_or_none()

    if not job:
        return APIResponse(
            success=False,
            message=f"Batch job not found: {batch_id}",
        )

    status = BatchJobStatus(
        id=job.id,
        name=job.name,
        status=JobStatus(job.status),
        audit_type=job.audit_type,
        platform=job.platform,
        total_images=job.total_images,
        processed_images=job.processed_images,
        failed_images=job.failed_images,
        progress_percent=job.progress_percent,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        report_path=job.report_path,
    )

    return APIResponse(
        success=True,
        data=status.model_dump(),
    )


@router.get("/{batch_id}/items", response_model=APIResponse)
async def get_batch_items(batch_id: str, db: DbSession) -> APIResponse:
    """
    Get all items in a batch job with their results.
    """
    from sqlalchemy import select

    result = await db.execute(
        select(BatchItem).where(BatchItem.batch_id == batch_id)
    )
    items = result.scalars().all()

    item_statuses = [
        BatchItemStatus(
            id=item.id,
            filename=item.original_filename or item.image_path,
            status=item.status,
            score=item.score,
            error=item.error_message,
            processed_at=item.processed_at,
        )
        for item in items
    ]

    return APIResponse(
        success=True,
        data={"items": [s.model_dump() for s in item_statuses]},
    )


@router.get("/{batch_id}/report")
async def download_batch_report(batch_id: str, db: DbSession):
    """
    Download the generated report for a completed batch job.
    """
    from sqlalchemy import select

    result = await db.execute(select(BatchJob).where(BatchJob.id == batch_id))
    job = result.scalar_one_or_none()

    if not job:
        return APIResponse(
            success=False,
            message=f"Batch job not found: {batch_id}",
        )

    if job.status != "completed":
        return APIResponse(
            success=False,
            message="Batch job not completed yet",
        )

    if not job.report_path:
        return APIResponse(
            success=False,
            message="No report available",
        )

    return FileResponse(
        job.report_path,
        filename=f"vision_s8_batch_{batch_id[:8]}.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@router.get("", response_model=APIResponse)
async def list_batch_jobs(db: DbSession) -> APIResponse:
    """
    List all batch jobs.
    """
    from sqlalchemy import select

    result = await db.execute(select(BatchJob).order_by(BatchJob.created_at.desc()).limit(50))
    jobs = result.scalars().all()

    job_list = [
        {
            "id": job.id,
            "name": job.name,
            "status": job.status,
            "total_images": job.total_images,
            "processed_images": job.processed_images,
            "progress_percent": job.progress_percent,
            "created_at": job.created_at.isoformat(),
        }
        for job in jobs
    ]

    return APIResponse(
        success=True,
        data={"jobs": job_list},
    )
