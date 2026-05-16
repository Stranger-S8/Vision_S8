"""
Vision_S8 - AI-Powered Product Image Auditing & Optimization API

Main FastAPI application entry point.
"""

import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import settings
from .models.database import close_database, init_database
from .models.schemas import ErrorResponse
from .utils.logger import setup_logger, get_logger
from .api.routes import (
    health_router,
    audit_router,
    enhance_router,
    batch_router,
    compare_router,
    ab_test_router,
    seo_router,
    compliance_router,
)


# Setup logging
logger = setup_logger(
    name="vision_s8",
    level=settings.log_level,
    log_file=settings.log_file,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Vision_S8 API...")
    logger.info(f"Debug mode: {settings.debug}")

    # Initialize database
    await init_database(settings.database_url)
    logger.info("Database initialized")

    # Ensure directories exist
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    yield

    # Shutdown
    logger.info("Shutting down Vision_S8 API...")
    await close_database()


# Create FastAPI app
app = FastAPI(
    title="Vision_S8 API",
    description="""
# Vision_S8 - AI-Powered Product Image Auditing & Optimization

A comprehensive API for e-commerce sellers to analyze, compare, and enhance product images using Google Gemini AI.

## Features

- 📸 **Image Audit** - Get quality scores and improvement suggestions
- ✨ **Auto-Enhancement** - AI-powered background removal, lighting, and color correction
- 🔍 **Competitor Comparison** - Compare your images against competitors
- 📊 **A/B Test Predictor** - Predict which image variant will perform best
- 🎯 **Platform Optimization** - Optimize for Amazon, Shopify, Instagram, etc.
- 🔎 **SEO Analyzer** - Generate alt text, filenames, and keywords
- ✅ **Compliance Checker** - Validate against marketplace requirements
- 📦 **Batch Processing** - Process entire catalogs with progress tracking

## Quick Start

1. Upload an image to `/api/v1/audit` to get a quality analysis
2. Use `/api/v1/enhance` to get improvement recommendations
3. Check platform compliance with `/api/v1/compliance/{platform}`

    """,
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware - Configure for production!
# For local use (Etsy digital product), localhost is fine
# For web deployment, restrict to your specific domains
allowed_origins = settings.cors_origins if hasattr(settings, 'cors_origins') else [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# ============================================================================
# Rate Limiting Middleware (Simple in-memory implementation)
# ============================================================================
class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: dict[str, list[datetime]] = defaultdict(list)
    
    def is_allowed(self, client_ip: str) -> bool:
        """Check if a request from this IP is allowed."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean old requests
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if req_time > minute_ago
        ]
        
        # Check limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return False
        
        # Record this request
        self.requests[client_ip].append(now)
        return True
    
    def get_remaining(self, client_ip: str) -> int:
        """Get remaining requests for this IP."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        recent = [r for r in self.requests[client_ip] if r > minute_ago]
        return max(0, self.requests_per_minute - len(recent))


# Initialize rate limiter
rate_limiter = RateLimiter(
    requests_per_minute=getattr(settings, 'rate_limit_rpm', 60)
)


# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Enforce rate limiting per client IP."""
    # Skip rate limiting for health checks
    if request.url.path in ["/api/v1/health", "/api/v1/", "/docs", "/redoc", "/openapi.json"]:
        return await call_next(request)
    
    client_ip = request.client.host if request.client else "unknown"
    
    if not rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "success": False,
                "error": "Rate limit exceeded. Please wait before making more requests.",
                "error_code": "RATE_LIMIT_EXCEEDED",
                "retry_after_seconds": 60,
            },
            headers={"Retry-After": "60"},
        )
    
    response = await call_next(request)
    
    # Add rate limit headers
    remaining = rate_limiter.get_remaining(client_ip)
    response.headers["X-RateLimit-Limit"] = str(rate_limiter.requests_per_minute)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    
    return response


# ============================================================================
# API Key Authentication Middleware (Optional - enable in .env)
# ============================================================================
@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """
    Optional API key authentication.
    
    Enable by setting API_KEY_ENABLED=true and API_KEY=your-secret-key in .env
    Clients must then provide X-API-Key header with requests.
    """
    # Skip if API key auth is disabled
    if not getattr(settings, 'api_key_enabled', False):
        return await call_next(request)
    
    # Skip for docs and health endpoints
    public_paths = ["/docs", "/redoc", "/openapi.json", "/api/v1/health", "/api/v1/"]
    if request.url.path in public_paths:
        return await call_next(request)
    
    # Check for API key
    api_key = request.headers.get("X-API-Key")
    expected_key = getattr(settings, 'api_key', None)
    
    if not api_key:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={
                "success": False,
                "error": "API key required. Provide X-API-Key header.",
                "error_code": "MISSING_API_KEY",
            },
        )
    
    if api_key != expected_key:
        logger.warning(f"Invalid API key attempt from {request.client.host if request.client else 'unknown'}")
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "success": False,
                "error": "Invalid API key.",
                "error_code": "INVALID_API_KEY",
            },
        )
    
    return await call_next(request)


# Request timing middleware
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
    return response


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error=str(exc),
            error_code="VALIDATION_ERROR",
        ).model_dump(),
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    """Handle file not found errors."""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=ErrorResponse(
            error=str(exc),
            error_code="FILE_NOT_FOUND",
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="An unexpected error occurred",
            error_code="INTERNAL_ERROR",
            details={"type": type(exc).__name__} if settings.debug else None,
        ).model_dump(),
    )


# Mount static files for outputs
output_path = Path(settings.output_dir)
if output_path.exists():
    app.mount("/outputs", StaticFiles(directory=str(output_path)), name="outputs")

# Include routers
app.include_router(health_router, prefix="/api/v1")
app.include_router(audit_router, prefix="/api/v1")
app.include_router(enhance_router, prefix="/api/v1")
app.include_router(batch_router, prefix="/api/v1")
app.include_router(compare_router, prefix="/api/v1")
app.include_router(ab_test_router, prefix="/api/v1")
app.include_router(seo_router, prefix="/api/v1")
app.include_router(compliance_router, prefix="/api/v1")


def run():
    """Run the application using uvicorn."""
    uvicorn.run(
        "vision_s8.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    run()
