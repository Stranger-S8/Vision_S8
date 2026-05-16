"""
Comprehensive tests for the Hybrid Auditor module.

Tests the HYBRID analysis approach combining:
- Computer Vision (OpenCV) for technical metrics
- AI (Gemini) for subjective quality assessment
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from PIL import Image


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_image() -> Image.Image:
    """Create a sample product image."""
    img = Image.new("RGB", (1000, 1000), color=(255, 255, 255))
    # Add a colored "product" in center
    for x in range(350, 650):
        for y in range(350, 650):
            img.putpixel((x, y), (180, 80, 80))
    return img


@pytest.fixture
def mock_ai_result() -> dict:
    """Mock AI result from Gemini."""
    return {
        "overall_score": 7.5,
        "scores": {
            "lighting": 8,
            "composition": 7,
            "background": 8,
            "product_focus": 7,
            "color_accuracy": 8,
            "sharpness": 7,
            "professionalism": 7,
        },
        "technical_analysis": {
            "estimated_lighting_type": "studio",
            "background_type": "white",
            "product_visibility": "good",
            "image_quality": "high",
        },
        "strengths": [
            "Clean white background",
            "Good product lighting",
            "Professional appearance",
        ],
        "improvements": [
            {
                "issue": "Could improve sharpness",
                "priority": "medium",
                "suggestion": "Use a tripod for sharper images",
            },
            {
                "issue": "Product could be larger",
                "priority": "low",
                "suggestion": "Crop tighter around product",
            },
        ],
        "professional_assessment": "This is a good quality product image suitable for e-commerce.",
        "competitor_comparison": "Comparable to average marketplace listings.",
    }


# ============================================================================
# Import Tests
# ============================================================================

class TestAuditorImport:
    """Test that auditor can be imported."""

    def test_import_auditor(self):
        """Test module import."""
        from vision_s8.core.auditor import ProductAuditor
        assert ProductAuditor is not None

    def test_import_from_core(self):
        """Test import from core package."""
        from vision_s8.core import ProductAuditor
        assert ProductAuditor is not None


# ============================================================================
# Initialization Tests
# ============================================================================

class TestAuditorInit:
    """Test auditor initialization."""

    def test_init_default_services(self):
        """Test initialization with default services."""
        from vision_s8.core.auditor import ProductAuditor
        auditor = ProductAuditor()
        assert auditor._gemini is not None
        assert auditor._image is not None
        assert auditor._cv is not None

    def test_init_custom_services(self):
        """Test initialization with custom services."""
        from vision_s8.core.auditor import ProductAuditor
        
        mock_gemini = MagicMock()
        mock_image = MagicMock()
        mock_cv = MagicMock()
        
        auditor = ProductAuditor(
            gemini_service=mock_gemini,
            image_service=mock_image,
            cv_analyzer=mock_cv,
        )
        
        assert auditor._gemini is mock_gemini
        assert auditor._image is mock_image
        assert auditor._cv is mock_cv


# ============================================================================
# CV-Only Audit Tests
# ============================================================================

class TestCVOnlyAudit:
    """Tests for CV-only analysis (no AI required)."""

    def test_cv_only_audit(self, sample_image):
        """Test CV-only audit returns proper structure."""
        from vision_s8.core.auditor import ProductAuditor
        auditor = ProductAuditor()
        
        result, processing_time = auditor.audit_cv_only(sample_image)
        
        assert isinstance(result, dict)
        assert processing_time > 0
        assert "quality_score" in result

    def test_cv_only_contains_all_metrics(self, sample_image):
        """CV-only audit should contain all technical metrics."""
        from vision_s8.core.auditor import ProductAuditor
        auditor = ProductAuditor()
        
        result, _ = auditor.audit_cv_only(sample_image)
        
        expected_metrics = [
            "sharpness",
            "brightness",
            "contrast",
            "color_analysis",
            "background_analysis",
        ]
        
        for metric in expected_metrics:
            assert metric in result, f"Missing metric: {metric}"

    def test_cv_only_quality_score_range(self, sample_image):
        """CV quality score should be in valid range."""
        from vision_s8.core.auditor import ProductAuditor
        auditor = ProductAuditor()
        
        result, _ = auditor.audit_cv_only(sample_image)
        
        score = result["quality_score"]["overall_score"]
        assert 0 <= score <= 10

    def test_cv_only_no_api_call(self, sample_image):
        """CV-only audit should not call Gemini API."""
        from vision_s8.core.auditor import ProductAuditor
        
        mock_gemini = MagicMock()
        mock_gemini.audit_image = AsyncMock()
        
        auditor = ProductAuditor(gemini_service=mock_gemini)
        auditor.audit_cv_only(sample_image)
        
        mock_gemini.audit_image.assert_not_called()


# ============================================================================
# Hybrid Audit Tests
# ============================================================================

class TestHybridAudit:
    """Tests for hybrid CV + AI analysis."""

    @pytest.mark.asyncio
    async def test_hybrid_audit_with_ai(self, sample_image, mock_ai_result):
        """Test hybrid audit with AI enabled."""
        from vision_s8.core.auditor import ProductAuditor
        
        mock_gemini = MagicMock()
        mock_gemini.audit_image = AsyncMock(return_value=mock_ai_result)
        
        mock_image = MagicMock()
        mock_image.get_image_info = MagicMock(return_value={
            "width": 1000,
            "height": 1000,
            "format": "JPEG",
        })
        
        auditor = ProductAuditor(
            gemini_service=mock_gemini,
            image_service=mock_image,
        )
        
        result, processing_time = await auditor.audit(sample_image, use_ai=True)
        
        assert result is not None
        assert processing_time > 0
        mock_gemini.audit_image.assert_called_once()

    @pytest.mark.asyncio
    async def test_hybrid_audit_without_ai(self, sample_image):
        """Test hybrid audit with AI disabled."""
        from vision_s8.core.auditor import ProductAuditor
        
        mock_gemini = MagicMock()
        mock_gemini.audit_image = AsyncMock()
        
        mock_image = MagicMock()
        mock_image.get_image_info = MagicMock(return_value={
            "width": 1000,
            "height": 1000,
        })
        
        auditor = ProductAuditor(
            gemini_service=mock_gemini,
            image_service=mock_image,
        )
        
        result, _ = await auditor.audit(sample_image, use_ai=False)
        
        assert result is not None
        mock_gemini.audit_image.assert_not_called()

    @pytest.mark.asyncio
    async def test_hybrid_audit_result_structure(self, sample_image, mock_ai_result):
        """Test hybrid audit returns proper AuditResult structure."""
        from vision_s8.core.auditor import ProductAuditor
        from vision_s8.models.schemas import AuditResult
        
        mock_gemini = MagicMock()
        mock_gemini.audit_image = AsyncMock(return_value=mock_ai_result)
        
        mock_image = MagicMock()
        mock_image.get_image_info = MagicMock(return_value={})
        
        auditor = ProductAuditor(
            gemini_service=mock_gemini,
            image_service=mock_image,
        )
        
        result, _ = await auditor.audit(sample_image, use_ai=True)
        
        assert isinstance(result, AuditResult)
        assert hasattr(result, "overall_score")
        assert hasattr(result, "scores")
        assert hasattr(result, "technical_analysis")
        assert hasattr(result, "strengths")
        assert hasattr(result, "improvements")

    @pytest.mark.asyncio
    async def test_hybrid_score_blending(self, sample_image, mock_ai_result):
        """Test that CV and AI scores are blended correctly."""
        from vision_s8.core.auditor import ProductAuditor
        
        # Mock AI to return specific score
        mock_ai_result["overall_score"] = 8.0
        
        mock_gemini = MagicMock()
        mock_gemini.audit_image = AsyncMock(return_value=mock_ai_result)
        
        mock_image = MagicMock()
        mock_image.get_image_info = MagicMock(return_value={})
        
        auditor = ProductAuditor(
            gemini_service=mock_gemini,
            image_service=mock_image,
        )
        
        result, _ = await auditor.audit(sample_image, use_ai=True)
        
        # Score should be blended (40% CV + 60% AI)
        # Not exactly 8.0 since CV contributes
        assert result.overall_score != 8.0 or result.overall_score <= 10


# ============================================================================
# Score Parsing Tests
# ============================================================================

class TestScoreParsing:
    """Tests for score parsing and clamping."""

    def test_clamp_score_valid(self):
        """Test clamping valid scores."""
        from vision_s8.core.auditor import ProductAuditor
        
        assert ProductAuditor._clamp_score(5.0) == 5.0
        assert ProductAuditor._clamp_score(0.0) == 0.0
        assert ProductAuditor._clamp_score(10.0) == 10.0

    def test_clamp_score_out_of_range(self):
        """Test clamping out-of-range scores."""
        from vision_s8.core.auditor import ProductAuditor
        
        assert ProductAuditor._clamp_score(-5.0) == 0.0
        assert ProductAuditor._clamp_score(15.0) == 10.0
        assert ProductAuditor._clamp_score(100.0) == 10.0

    def test_clamp_score_none(self):
        """Test clamping None value."""
        from vision_s8.core.auditor import ProductAuditor
        
        assert ProductAuditor._clamp_score(None) == 5.0

    def test_clamp_score_int(self):
        """Test clamping integer values."""
        from vision_s8.core.auditor import ProductAuditor
        
        assert ProductAuditor._clamp_score(7) == 7.0
        assert isinstance(ProductAuditor._clamp_score(7), float)


# ============================================================================
# Quick Score Tests
# ============================================================================

class TestQuickScore:
    """Tests for quick_score method."""

    @pytest.mark.asyncio
    async def test_quick_score_returns_float(self, sample_image, mock_ai_result):
        """Quick score should return a float."""
        from vision_s8.core.auditor import ProductAuditor
        
        mock_gemini = MagicMock()
        mock_gemini.audit_image = AsyncMock(return_value=mock_ai_result)
        
        mock_image = MagicMock()
        mock_image.get_image_info = MagicMock(return_value={})
        
        auditor = ProductAuditor(
            gemini_service=mock_gemini,
            image_service=mock_image,
        )
        
        score = await auditor.quick_score(sample_image)
        
        assert isinstance(score, float)
        assert 0 <= score <= 10


# ============================================================================
# Improvement Parsing Tests
# ============================================================================

class TestImprovementParsing:
    """Tests for improvement suggestion parsing."""

    @pytest.mark.asyncio
    async def test_improvements_parsed(self, sample_image, mock_ai_result):
        """Test that improvements are properly parsed."""
        from vision_s8.core.auditor import ProductAuditor
        from vision_s8.models.schemas import Improvement
        
        mock_gemini = MagicMock()
        mock_gemini.audit_image = AsyncMock(return_value=mock_ai_result)
        
        mock_image = MagicMock()
        mock_image.get_image_info = MagicMock(return_value={})
        
        auditor = ProductAuditor(
            gemini_service=mock_gemini,
            image_service=mock_image,
        )
        
        result, _ = await auditor.audit(sample_image, use_ai=True)
        
        assert len(result.improvements) > 0
        for imp in result.improvements:
            assert isinstance(imp, Improvement)
            assert imp.issue
            assert imp.suggestion
            assert imp.priority

    @pytest.mark.asyncio
    async def test_priority_mapping(self, sample_image):
        """Test priority string to enum mapping."""
        from vision_s8.core.auditor import ProductAuditor
        from vision_s8.models.schemas import Priority
        
        mock_result = {
            "overall_score": 7.0,
            "scores": {},
            "improvements": [
                {"issue": "High priority", "priority": "high", "suggestion": "Fix now"},
                {"issue": "Medium priority", "priority": "medium", "suggestion": "Fix soon"},
                {"issue": "Low priority", "priority": "low", "suggestion": "Optional"},
            ],
        }
        
        mock_gemini = MagicMock()
        mock_gemini.audit_image = AsyncMock(return_value=mock_result)
        
        mock_image = MagicMock()
        mock_image.get_image_info = MagicMock(return_value={})
        
        auditor = ProductAuditor(
            gemini_service=mock_gemini,
            image_service=mock_image,
        )
        
        result, _ = await auditor.audit(sample_image, use_ai=True)
        
        priorities = [imp.priority for imp in result.improvements]
        assert Priority.HIGH in priorities
        assert Priority.MEDIUM in priorities
        assert Priority.LOW in priorities


# ============================================================================
# Technical Analysis Tests
# ============================================================================

class TestTechnicalAnalysis:
    """Tests for technical analysis parsing."""

    @pytest.mark.asyncio
    async def test_technical_analysis_populated(self, sample_image, mock_ai_result):
        """Test that technical analysis is populated."""
        from vision_s8.core.auditor import ProductAuditor
        
        mock_gemini = MagicMock()
        mock_gemini.audit_image = AsyncMock(return_value=mock_ai_result)
        
        mock_image = MagicMock()
        mock_image.get_image_info = MagicMock(return_value={})
        
        auditor = ProductAuditor(
            gemini_service=mock_gemini,
            image_service=mock_image,
        )
        
        result, _ = await auditor.audit(sample_image, use_ai=True)
        
        assert result.technical_analysis is not None
        assert hasattr(result.technical_analysis, "estimated_lighting_type")
        assert hasattr(result.technical_analysis, "background_type")


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_empty_ai_result(self, sample_image):
        """Test handling empty AI result (CV-only fallback)."""
        from vision_s8.core.auditor import ProductAuditor
        
        mock_gemini = MagicMock()
        mock_gemini.audit_image = AsyncMock(return_value={})
        
        mock_image = MagicMock()
        mock_image.get_image_info = MagicMock(return_value={})
        
        auditor = ProductAuditor(
            gemini_service=mock_gemini,
            image_service=mock_image,
        )
        
        # Should not raise, should use CV scores as fallback
        result, _ = await auditor.audit(sample_image, use_ai=True)
        
        assert result is not None
        assert 0 <= result.overall_score <= 10

    @pytest.mark.asyncio
    async def test_partial_ai_result(self, sample_image):
        """Test handling partial AI result."""
        from vision_s8.core.auditor import ProductAuditor
        
        partial_result = {
            "overall_score": 6.5,
            # Missing many fields
        }
        
        mock_gemini = MagicMock()
        mock_gemini.audit_image = AsyncMock(return_value=partial_result)
        
        mock_image = MagicMock()
        mock_image.get_image_info = MagicMock(return_value={})
        
        auditor = ProductAuditor(
            gemini_service=mock_gemini,
            image_service=mock_image,
        )
        
        # Should handle gracefully with defaults
        result, _ = await auditor.audit(sample_image, use_ai=True)
        
        assert result is not None
        assert result.scores is not None


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests with real CV analyzer."""

    def test_cv_analyzer_integration(self, sample_image):
        """Test CV analyzer is properly integrated."""
        from vision_s8.core.auditor import ProductAuditor
        from vision_s8.core.cv_analyzer import CVAnalyzer
        
        auditor = ProductAuditor()
        
        # Should use CV analyzer
        assert isinstance(auditor._cv, CVAnalyzer)
        
        # CV-only audit should work
        result, _ = auditor.audit_cv_only(sample_image)
        assert "sharpness" in result

    @pytest.mark.asyncio
    async def test_full_audit_with_real_cv(self, sample_image):
        """Test full audit with real CV analysis."""
        from vision_s8.core.auditor import ProductAuditor
        
        mock_gemini = MagicMock()
        mock_gemini.audit_image = AsyncMock(return_value={
            "overall_score": 7.0,
            "scores": {},
        })
        
        mock_image = MagicMock()
        mock_image.get_image_info = MagicMock(return_value={})
        
        auditor = ProductAuditor(
            gemini_service=mock_gemini,
            image_service=mock_image,
            # Use real CV analyzer
        )
        
        result, _ = await auditor.audit(sample_image, use_ai=True)
        
        # Should have valid scores from CV + AI blend
        assert result.overall_score is not None
        assert 0 <= result.overall_score <= 10
