"""
Pydantic schemas for API request/response validation.

Defines all the data models used in the Vision_S8 API.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


# ============== Enums ==============

class AuditType(str, Enum):
    """Types of audits available."""
    SINGLE = "single"
    COMPARE = "compare"
    ENHANCE = "enhance"
    AB_TEST = "ab_test"
    SEO = "seo"
    COMPLIANCE = "compliance"
    PLATFORM = "platform"


class Platform(str, Enum):
    """Supported e-commerce platforms."""
    AMAZON = "amazon"
    SHOPIFY = "shopify"
    INSTAGRAM = "instagram"
    EBAY = "ebay"
    ETSY = "etsy"
    WALMART = "walmart"


class JobStatus(str, Enum):
    """Batch job status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Priority(str, Enum):
    """Priority levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ComparisonMode(str, Enum):
    """Comparison modes."""
    MANUAL = "manual"
    URL = "url"
    MARKETPLACE = "marketplace"


# ============== Base Response ==============

class APIResponse(BaseModel):
    """Standard API response wrapper."""
    success: bool = True
    data: Any = None
    message: str | None = None
    meta: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    error: str
    error_code: str | None = None
    details: dict[str, Any] | None = None


# ============== Audit Schemas ==============

class ScoreBreakdown(BaseModel):
    """Detailed score breakdown."""
    lighting: float = Field(..., ge=0, le=10)
    composition: float = Field(..., ge=0, le=10)
    background: float = Field(..., ge=0, le=10)
    product_focus: float = Field(..., ge=0, le=10)
    color_accuracy: float = Field(..., ge=0, le=10)
    sharpness: float = Field(..., ge=0, le=10)
    professionalism: float = Field(..., ge=0, le=10)


class TechnicalAnalysis(BaseModel):
    """Technical image analysis."""
    estimated_lighting_type: str
    background_type: str
    product_visibility: str
    image_quality: str


class Improvement(BaseModel):
    """Improvement suggestion."""
    issue: str
    priority: Priority
    suggestion: str


class AuditResult(BaseModel):
    """Single image audit result."""
    overall_score: float = Field(..., ge=0, le=10)
    scores: ScoreBreakdown
    technical_analysis: TechnicalAnalysis
    strengths: list[str]
    improvements: list[Improvement]
    professional_assessment: str
    competitor_comparison: str


class AuditResponse(BaseModel):
    """Audit endpoint response."""
    audit_id: str
    image_filename: str
    result: AuditResult
    processing_time_ms: int


# ============== Enhancement Schemas ==============

class LightingAdjustment(BaseModel):
    """Lighting adjustment parameters."""
    brightness: int = Field(default=0, ge=-100, le=100)
    contrast: int = Field(default=0, ge=-100, le=100)
    highlights: int = Field(default=0, ge=-100, le=100)
    shadows: int = Field(default=0, ge=-100, le=100)


class ColorCorrection(BaseModel):
    """Color correction parameters."""
    saturation: int = Field(default=0, ge=-100, le=100)
    vibrance: int = Field(default=0, ge=-100, le=100)
    temperature: int = Field(default=0, ge=-100, le=100)
    tint: int = Field(default=0, ge=-100, le=100)


class EnhancementRecommendation(BaseModel):
    """Individual enhancement recommendation."""
    recommended: bool
    reason: str


class BackgroundEnhancement(EnhancementRecommendation):
    """Background removal/replacement recommendation."""
    suggested_background: str | None = None


class LightingEnhancement(EnhancementRecommendation):
    """Lighting enhancement recommendation."""
    adjustments: LightingAdjustment | None = None


class ColorEnhancement(EnhancementRecommendation):
    """Color correction recommendation."""
    adjustments: ColorCorrection | None = None


class CroppingEnhancement(EnhancementRecommendation):
    """Cropping recommendation."""
    suggested_aspect_ratio: str | None = None
    crop_focus: str | None = None


class SharpeningEnhancement(EnhancementRecommendation):
    """Sharpening recommendation."""
    intensity: str | None = None


class ShadowEnhancement(EnhancementRecommendation):
    """Shadow addition recommendation."""
    shadow_type: str | None = None
    intensity: str | None = None


class EnhancementPlan(BaseModel):
    """Full enhancement plan."""
    background_removal: BackgroundEnhancement
    lighting_adjustment: LightingEnhancement
    color_correction: ColorEnhancement
    cropping: CroppingEnhancement
    sharpening: SharpeningEnhancement
    shadow_addition: ShadowEnhancement


class EnhanceResult(BaseModel):
    """Enhancement analysis result."""
    current_assessment: dict[str, Any]
    enhancements: EnhancementPlan
    enhancement_order: list[str]
    expected_score_after: float
    professional_notes: str


class EnhanceResponse(BaseModel):
    """Enhancement endpoint response."""
    audit_id: str
    original_image: str
    enhanced_image: str | None = None
    analysis: EnhanceResult
    score_before: float
    score_after: float | None = None
    processing_time_ms: int


# ============== Comparison Schemas ==============

class CompareRequest(BaseModel):
    """Comparison request for URL/marketplace modes."""
    mode: ComparisonMode
    urls: list[HttpUrl] | None = None
    marketplace_query: str | None = None
    platform: Platform | None = None


class CompetitorScore(BaseModel):
    """Score for a competitor image."""
    index: int
    score: float
    source: str | None = None


class AspectComparison(BaseModel):
    """Comparison for a single aspect."""
    main: float
    competitor_average: float
    gap: str  # ahead/behind/even
    gap_magnitude: str  # significant/moderate/slight/none


class CompareResult(BaseModel):
    """Comparison result."""
    main_image_score: float
    competitor_scores: list[CompetitorScore]
    comparison_aspects: dict[str, AspectComparison]
    competitive_position: str
    key_differentiators: list[str]
    critical_gaps: list[dict[str, Any]]
    action_plan: list[dict[str, Any]]
    summary: str


class CompareResponse(BaseModel):
    """Comparison endpoint response."""
    audit_id: str
    main_image: str
    competitor_images: list[str]
    result: CompareResult
    processing_time_ms: int


# ============== A/B Test Schemas ==============

class VariantScore(BaseModel):
    """Score for a single variant."""
    index: int
    predicted_ctr_score: float = Field(..., ge=0, le=100)
    predicted_conversion_score: float = Field(..., ge=0, le=100)
    overall_effectiveness: float = Field(..., ge=0, le=10)
    strengths: list[str]
    weaknesses: list[str]


class VariantRanking(BaseModel):
    """Ranking entry for a variant."""
    rank: int
    variant_index: int
    confidence: str
    expected_performance_lift: str


class ABTestResult(BaseModel):
    """A/B test prediction result."""
    variants: list[VariantScore]
    ranking: list[VariantRanking]
    winner_analysis: dict[str, Any]
    testing_recommendations: dict[str, Any]
    detailed_comparison: dict[str, str]
    summary: str


class ABTestResponse(BaseModel):
    """A/B test endpoint response."""
    audit_id: str
    variant_count: int
    result: ABTestResult
    processing_time_ms: int


# ============== SEO Schemas ==============

class ProductDetection(BaseModel):
    """Detected product information."""
    detected_product: str
    category: str
    subcategory: str
    detected_attributes: list[str]


class AltTextSuggestions(BaseModel):
    """Alt text suggestions."""
    primary: str
    short: str
    detailed: str


class Keywords(BaseModel):
    """Keyword suggestions."""
    primary: list[str]
    secondary: list[str]
    long_tail: list[str]


class SEOResult(BaseModel):
    """SEO analysis result."""
    product_detection: ProductDetection
    alt_text: AltTextSuggestions
    filename_suggestions: list[str]
    title_suggestions: list[str]
    meta_description: str
    keywords: Keywords
    schema_markup_suggestions: dict[str, str | None]
    accessibility: dict[str, Any]
    platform_specific: dict[str, str]
    seo_score: float
    optimization_tips: list[str]


class SEOResponse(BaseModel):
    """SEO endpoint response."""
    audit_id: str
    image_filename: str
    result: SEOResult
    processing_time_ms: int


# ============== Compliance Schemas ==============

class ComplianceCheck(BaseModel):
    """Single compliance check result."""
    passed: bool
    detected: str | None = None
    requirement: str | None = None
    issue: str | None = None


class ComplianceViolation(BaseModel):
    """Compliance violation."""
    rule: str
    severity: str  # critical/major/minor
    description: str
    fix: str


class ComplianceWarning(BaseModel):
    """Compliance warning."""
    aspect: str
    message: str
    recommendation: str


class ComplianceResult(BaseModel):
    """Compliance check result."""
    platform: str
    overall_compliance: bool
    compliance_score: float = Field(..., ge=0, le=100)
    checks: dict[str, ComplianceCheck]
    violations: list[ComplianceViolation]
    warnings: list[ComplianceWarning]
    auto_fixable: list[str]
    manual_required: list[str]
    summary: str


class ComplianceResponse(BaseModel):
    """Compliance endpoint response."""
    audit_id: str
    image_filename: str
    platform: str
    result: ComplianceResult
    processing_time_ms: int


# ============== Batch Schemas ==============

class BatchJobCreate(BaseModel):
    """Request to create a batch job."""
    name: str | None = None
    audit_type: AuditType = AuditType.SINGLE
    platform: Platform | None = None


class BatchJobStatus(BaseModel):
    """Batch job status response."""
    id: str
    name: str | None
    status: JobStatus
    audit_type: str
    platform: str | None
    total_images: int
    processed_images: int
    failed_images: int
    progress_percent: float
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    report_path: str | None


class BatchItemStatus(BaseModel):
    """Individual batch item status."""
    id: str
    filename: str
    status: str
    score: float | None
    error: str | None
    processed_at: datetime | None


class BatchJobDetail(BatchJobStatus):
    """Detailed batch job with items."""
    items: list[BatchItemStatus]


# ============== Platform Optimization Schemas ==============

class PlatformRequirement(BaseModel):
    """Platform requirement check."""
    name: str
    required: Any
    actual: Any
    passed: bool
    message: str | None = None


class PlatformOptimizeResult(BaseModel):
    """Platform optimization result."""
    platform: str
    platform_score: float = Field(..., ge=0, le=10)
    overall_compliance: bool
    requirements: list[PlatformRequirement]
    best_practices: list[str]
    recommendations: list[str]
    ready_to_list: bool


class PlatformOptimizeResponse(BaseModel):
    """Platform optimization endpoint response."""
    audit_id: str
    image_filename: str
    result: PlatformOptimizeResult
    processing_time_ms: int


# ============== Health Check ==============

class HealthStatus(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str
    gemini_api: bool
    database: bool
    timestamp: datetime
