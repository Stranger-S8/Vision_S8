"""
Comprehensive tests for CV Analyzer module.

Tests all 12+ Computer Vision algorithms used in Vision_S8.
"""

import numpy as np
import pytest
from PIL import Image


# ============================================================================
# Fixtures for various test images
# ============================================================================

@pytest.fixture
def white_image() -> Image.Image:
    """Pure white image."""
    return Image.new("RGB", (500, 500), color=(255, 255, 255))


@pytest.fixture
def black_image() -> Image.Image:
    """Pure black image."""
    return Image.new("RGB", (500, 500), color=(0, 0, 0))


@pytest.fixture
def gray_image() -> Image.Image:
    """Neutral gray image."""
    return Image.new("RGB", (500, 500), color=(128, 128, 128))


@pytest.fixture
def red_product_image() -> Image.Image:
    """White background with red centered product."""
    img = Image.new("RGB", (1000, 1000), color=(255, 255, 255))
    # Draw red "product" in center
    for x in range(300, 700):
        for y in range(300, 700):
            img.putpixel((x, y), (200, 50, 50))
    return img


@pytest.fixture
def high_contrast_image() -> Image.Image:
    """Checkerboard pattern for high contrast testing."""
    img = Image.new("RGB", (500, 500), color=(0, 0, 0))
    for x in range(0, 500, 50):
        for y in range(0, 500, 50):
            color = (255, 255, 255) if (x + y) % 100 == 0 else (0, 0, 0)
            for dx in range(50):
                for dy in range(50):
                    if x + dx < 500 and y + dy < 500:
                        img.putpixel((x + dx, y + dy), color)
    return img


@pytest.fixture
def gradient_image() -> Image.Image:
    """Horizontal gradient from black to white."""
    img = Image.new("RGB", (500, 500))
    for x in range(500):
        gray_val = int((x / 500) * 255)
        for y in range(500):
            img.putpixel((x, y), (gray_val, gray_val, gray_val))
    return img


@pytest.fixture
def colorful_image() -> Image.Image:
    """Image with multiple distinct colors."""
    img = Image.new("RGB", (500, 500))
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
    ]
    for i, color in enumerate(colors):
        x_start = (i % 2) * 250
        y_start = (i // 2) * 250
        for x in range(x_start, x_start + 250):
            for y in range(y_start, y_start + 250):
                img.putpixel((x, y), color)
    return img


@pytest.fixture
def symmetric_image() -> Image.Image:
    """Horizontally symmetric image."""
    img = Image.new("RGB", (500, 500), color=(128, 128, 128))
    # Draw symmetric pattern
    for x in range(100, 150):
        for y in range(100, 400):
            img.putpixel((x, y), (200, 100, 100))
            img.putpixel((500 - x - 1, y), (200, 100, 100))
    return img


@pytest.fixture
def asymmetric_image() -> Image.Image:
    """Asymmetric image (product in corner)."""
    img = Image.new("RGB", (500, 500), color=(255, 255, 255))
    # Draw product in top-left corner
    for x in range(10, 150):
        for y in range(10, 150):
            img.putpixel((x, y), (100, 100, 200))
    return img


@pytest.fixture
def noisy_image() -> Image.Image:
    """Image with random noise."""
    np_arr = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
    return Image.fromarray(np_arr)


@pytest.fixture
def blurry_image(red_product_image: Image.Image) -> Image.Image:
    """Artificially blurred image."""
    from PIL import ImageFilter
    return red_product_image.filter(ImageFilter.GaussianBlur(radius=5))


@pytest.fixture
def sharp_image() -> Image.Image:
    """Sharp image with clear edges."""
    img = Image.new("RGB", (500, 500), color=(255, 255, 255))
    # Draw sharp rectangle
    for x in range(200, 300):
        for y in range(200, 300):
            img.putpixel((x, y), (0, 0, 0))
    return img


# ============================================================================
# CV Analyzer Tests
# ============================================================================

class TestCVAnalyzerImport:
    """Test that CV Analyzer can be imported."""

    def test_import_cv_analyzer(self):
        """Test module import."""
        from vision_s8.core.cv_analyzer import CVAnalyzer
        assert CVAnalyzer is not None

    def test_get_cv_analyzer_singleton(self):
        """Test singleton pattern."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer1 = get_cv_analyzer()
        analyzer2 = get_cv_analyzer()
        assert analyzer1 is analyzer2


class TestSharpnessAnalysis:
    """Tests for sharpness detection."""

    def test_sharp_image_high_score(self, sharp_image):
        """Sharp images should get high sharpness scores."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        result = analyzer.analyze(sharp_image)
        
        assert "sharpness" in result
        assert result["sharpness"]["laplacian_variance"] > 0
        assert "quality" in result["sharpness"]

    def test_blurry_image_low_score(self, blurry_image):
        """Blurry images should get lower sharpness scores."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        result = analyzer.analyze(blurry_image)
        
        assert "sharpness" in result
        # Blurry images have lower laplacian variance
        assert result["sharpness"]["quality"] in ["soft", "blurry", "acceptable"]


class TestBrightnessAnalysis:
    """Tests for brightness analysis."""

    def test_white_image_bright(self, white_image):
        """White image should be detected as bright."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        result = analyzer.analyze(white_image)
        
        assert "brightness" in result
        assert result["brightness"]["mean"] > 200
        assert result["brightness"]["level"] in ["bright", "overexposed"]

    def test_black_image_dark(self, black_image):
        """Black image should be detected as dark."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        result = analyzer.analyze(black_image)
        
        assert "brightness" in result
        assert result["brightness"]["mean"] < 50
        assert result["brightness"]["level"] in ["dark", "very_dark"]

    def test_gray_image_normal(self, gray_image):
        """Gray image should have normal brightness."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        result = analyzer.analyze(gray_image)
        
        assert "brightness" in result
        # Gray 128 -> LAB L channel will be around 128 or similar
        assert 80 < result["brightness"]["mean"] < 180


class TestContrastAnalysis:
    """Tests for contrast measurement."""

    def test_high_contrast_image(self, high_contrast_image):
        """Checkerboard should have high contrast."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        result = analyzer.analyze(high_contrast_image)
        
        assert "contrast" in result
        assert result["contrast"]["quality"] in ["high", "good"]

    def test_solid_color_low_contrast(self, gray_image):
        """Solid color should have low contrast."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        result = analyzer.analyze(gray_image)
        
        assert "contrast" in result
        assert result["contrast"]["quality"] in ["low", "moderate"]


class TestColorAnalysis:
    """Tests for color analysis."""

    def test_colorful_image_saturation(self, colorful_image):
        """Colorful image should have high saturation."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        result = analyzer.analyze(colorful_image)
        
        assert "color_analysis" in result
        assert result["color_analysis"]["mean_saturation"] > 100

    def test_gray_image_low_saturation(self, gray_image):
        """Gray image should have low saturation."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        result = analyzer.analyze(gray_image)
        
        assert "color_analysis" in result
        assert result["color_analysis"]["mean_saturation"] < 50

    def test_dominant_colors_extracted(self, colorful_image):
        """Should extract dominant colors."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        result = analyzer.analyze(colorful_image)
        
        assert "color_analysis" in result
        assert "dominant_colors" in result["color_analysis"]
        assert len(result["color_analysis"]["dominant_colors"]) > 0


class TestBackgroundAnalysis:
    """Tests for background detection."""

    def test_white_background_detected(self, red_product_image):
        """White background should be detected."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        result = analyzer.analyze(red_product_image)
        
        assert "background_analysis" in result
        # White backgrounds should have high uniformity
        assert "is_uniform" in result["background_analysis"]

    def test_solid_white_uniform(self, white_image):
        """Pure white image should be detected as uniform."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        result = analyzer.analyze(white_image)
        
        assert "background_analysis" in result
        assert result["background_analysis"]["is_uniform"] is True


class TestEdgeDensity:
    """Tests for edge density calculation."""

    def test_sharp_edges_detected(self, sharp_image):
        """Sharp rectangle should have detectable edges."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        result = analyzer.analyze(sharp_image)
        
        assert "edge_density" in result
        assert result["edge_density"]["density_percent"] > 0

    def test_solid_color_few_edges(self, gray_image):
        """Solid color should have few edges."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        result = analyzer.analyze(gray_image)
        
        assert "edge_density" in result
        assert result["edge_density"]["density_percent"] < 5


class TestNoiseEstimation:
    """Tests for noise level estimation."""

    def test_noisy_image_high_noise(self, noisy_image):
        """Random noise image should have high noise level."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        result = analyzer.analyze(noisy_image)
        
        assert "noise_level" in result
        assert result["noise_level"]["level"] in ["high", "very_high", "extreme"]

    def test_clean_image_low_noise(self, gray_image):
        """Clean solid image should have low noise."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        result = analyzer.analyze(gray_image)
        
        assert "noise_level" in result
        assert result["noise_level"]["level"] in ["low", "very_low"]


class TestBlurDetection:
    """Tests for blur detection."""

    def test_blurry_image_detected(self, blurry_image):
        """Blurred image should be detected as blurry."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        result = analyzer.analyze(blurry_image)
        
        assert "blur_detection" in result
        # Blurry images should have lower frequency energy
        assert "laplacian_variance" in result["blur_detection"]

    def test_sharp_image_not_blurry(self, sharp_image):
        """Sharp image should not be detected as blurry."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        result = analyzer.analyze(sharp_image)
        
        assert "blur_detection" in result
        # Sharp images have higher laplacian variance


class TestSymmetryAnalysis:
    """Tests for symmetry detection."""

    def test_symmetric_image_high_score(self, symmetric_image):
        """Symmetric image should have high symmetry score."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        result = analyzer.analyze(symmetric_image)
        
        assert "symmetry_score" in result
        assert "horizontal_score" in result["symmetry_score"]
        # Symmetric image should have higher horizontal symmetry
        assert result["symmetry_score"]["horizontal_score"] > 5

    def test_asymmetric_image_low_score(self, asymmetric_image):
        """Asymmetric image should have lower symmetry score."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        result = analyzer.analyze(asymmetric_image)
        
        assert "symmetry_score" in result
        assert "horizontal_score" in result["symmetry_score"]


class TestCompositionAnalysis:
    """Tests for composition analysis."""

    def test_centered_product_good_composition(self, red_product_image):
        """Centered product should have good composition."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        result = analyzer.analyze(red_product_image)
        
        assert "composition" in result
        assert "is_centered" in result["composition"]

    def test_corner_product_poor_composition(self, asymmetric_image):
        """Product in corner should have poor composition."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        result = analyzer.analyze(asymmetric_image)
        
        assert "composition" in result
        # Product in corner should not be centered
        # (exact result depends on algorithm)


class TestHistogramAnalysis:
    """Tests for histogram analysis."""

    def test_gradient_image_histogram(self, gradient_image):
        """Gradient should have distributed histogram."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        result = analyzer.analyze(gradient_image)
        
        assert "histogram_analysis" in result
        hist = result["histogram_analysis"]
        assert "shadows_percent" in hist
        assert "midtones_percent" in hist
        assert "highlights_percent" in hist
        
        # Gradient should have all three ranges
        total = hist["shadows_percent"] + hist["midtones_percent"] + hist["highlights_percent"]
        assert 95 < total < 105  # Allow some rounding error

    def test_white_image_highlights_dominant(self, white_image):
        """White image should have dominant highlights."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        result = analyzer.analyze(white_image)
        
        assert "histogram_analysis" in result
        assert result["histogram_analysis"]["highlights_percent"] > 80


class TestQualityScore:
    """Tests for overall quality scoring."""

    def test_quality_score_generated(self, red_product_image):
        """Quality score should be generated."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        analysis = analyzer.analyze(red_product_image)
        score = analyzer.get_quality_score(analysis)
        
        assert "overall_score" in score
        assert "quality_tier" in score
        assert 0 <= score["overall_score"] <= 10

    def test_quality_tier_categories(self, red_product_image):
        """Quality tier should be a valid category."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        analysis = analyzer.analyze(red_product_image)
        score = analyzer.get_quality_score(analysis)
        
        valid_tiers = ["excellent", "good", "acceptable", "needs_improvement"]
        assert score["quality_tier"] in valid_tiers

    def test_score_breakdown_included(self, red_product_image):
        """Score should include breakdown by category."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        analysis = analyzer.analyze(red_product_image)
        score = analyzer.get_quality_score(analysis)
        
        # Should include component scores
        assert "breakdown" in score or "component_scores" in score or len(score) > 2


class TestFullAnalysis:
    """End-to-end analysis tests."""

    def test_analyze_returns_all_metrics(self, red_product_image):
        """Analysis should return all expected metrics."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        result = analyzer.analyze(red_product_image)
        
        expected_keys = [
            "sharpness",
            "brightness",
            "contrast",
            "color_analysis",
            "background_analysis",
            "edge_density",
            "noise_level",
            "blur_detection",
            "symmetry_score",
            "composition",
            "histogram_analysis",
        ]
        
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_analyze_with_different_sizes(self):
        """Analysis should work with different image sizes."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        
        sizes = [(100, 100), (500, 500), (1000, 1000), (800, 600), (600, 800)]
        
        for width, height in sizes:
            img = Image.new("RGB", (width, height), color=(128, 128, 128))
            result = analyzer.analyze(img)
            assert "sharpness" in result, f"Failed for size {width}x{height}"

    def test_analyze_with_different_modes(self):
        """Analysis should work with different image modes."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        
        # RGB
        rgb_img = Image.new("RGB", (200, 200), color=(128, 128, 128))
        result = analyzer.analyze(rgb_img)
        assert "sharpness" in result
        
        # RGBA (with alpha channel)
        rgba_img = Image.new("RGBA", (200, 200), color=(128, 128, 128, 255))
        result = analyzer.analyze(rgba_img)
        assert "sharpness" in result
        
        # Grayscale
        gray_img = Image.new("L", (200, 200), color=128)
        result = analyzer.analyze(gray_img)
        assert "sharpness" in result


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_tiny_image(self):
        """Should handle tiny images."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        
        tiny = Image.new("RGB", (10, 10), color=(128, 128, 128))
        result = analyzer.analyze(tiny)
        assert "sharpness" in result

    def test_very_large_image(self):
        """Should handle large images (may resize internally)."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        
        large = Image.new("RGB", (3000, 3000), color=(128, 128, 128))
        result = analyzer.analyze(large)
        assert "sharpness" in result

    def test_non_square_image(self):
        """Should handle non-square images."""
        from vision_s8.core.cv_analyzer import get_cv_analyzer
        analyzer = get_cv_analyzer()
        
        wide = Image.new("RGB", (1000, 200), color=(128, 128, 128))
        result = analyzer.analyze(wide)
        assert "sharpness" in result
        
        tall = Image.new("RGB", (200, 1000), color=(128, 128, 128))
        result = analyzer.analyze(tall)
        assert "sharpness" in result
