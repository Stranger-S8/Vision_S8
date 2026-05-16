"""API route modules."""

from .health import router as health_router
from .audit import router as audit_router
from .enhance import router as enhance_router
from .batch import router as batch_router
from .compare import router as compare_router
from .ab_test import router as ab_test_router
from .seo import router as seo_router
from .compliance import router as compliance_router

__all__ = [
    "health_router",
    "audit_router",
    "enhance_router",
    "batch_router",
    "compare_router",
    "ab_test_router",
    "seo_router",
    "compliance_router",
]
