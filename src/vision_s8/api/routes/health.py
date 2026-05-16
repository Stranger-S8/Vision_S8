"""
Health check endpoint for Vision_S8.
"""

from datetime import datetime

from fastapi import APIRouter, Depends

from ...config import settings
from ...models.schemas import HealthStatus
from ...services.gemini_service import get_gemini_service
from ...models.database import get_session

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthStatus)
async def health_check() -> HealthStatus:
    """
    Check the health status of the API.

    Returns service status, version, and connectivity checks.
    """
    # Check Gemini API
    gemini_ok = False
    try:
        gemini = get_gemini_service()
        gemini_ok = await gemini.verify_connection()
    except Exception:
        pass

    # Check database
    db_ok = False
    try:
        async for session in get_session():
            await session.execute("SELECT 1")
            db_ok = True
            break
    except Exception:
        pass

    return HealthStatus(
        status="healthy" if gemini_ok and db_ok else "degraded",
        version="0.1.0",
        gemini_api=gemini_ok,
        database=db_ok,
        timestamp=datetime.utcnow(),
    )


@router.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Vision_S8 API",
        "version": "0.1.0",
        "description": "AI-Powered Product Image Auditing & Optimization",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
