"""
Tests for Image Service module.

Tests image loading, processing, and information extraction.
"""

import io
import pytest
from PIL import Image


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def rgb_image() -> Image.Image:
    """Create RGB test image."""
    return Image.new("RGB", (800, 600), color=(128, 128, 128))


@pytest.fixture
def rgba_image() -> Image.Image:
    """Create RGBA test image with transparency."""
    img = Image.new("RGBA", (500, 500), color=(255, 255, 255, 255))
    # Add transparent region
    for x in range(100, 200):
        for y in range(100, 200):
            img.putpixel((x, y), (255, 0, 0, 128))
    return img


@pytest.fixture
def grayscale_image() -> Image.Image:
    """Create grayscale test image."""
    return Image.new("L", (400, 400), color=128)


# ============================================================================
# Import Tests
# ============================================================================

class TestImageServiceImport:
    """Test that image service can be imported."""

    def test_import_image_service(self):
        """Test module import."""
        from vision_s8.services.image_service import ImageService
        assert ImageService is not None

    def test_get_image_service_singleton(self):
        """Test singleton pattern."""
        from vision_s8.services.image_service import get_image_service
        service1 = get_image_service()
        service2 = get_image_service()
        assert service1 is service2


# ============================================================================
# Image Info Tests
# ============================================================================

class TestImageInfo:
    """Tests for get_image_info method."""

    def test_rgb_image_info(self, rgb_image):
        """Test getting info from RGB image."""
        from vision_s8.services.image_service import get_image_service
        service = get_image_service()
        
        info = service.get_image_info(rgb_image)
        
        assert info["width"] == 800
        assert info["height"] == 600
        assert info["mode"] == "RGB"
        assert "aspect_ratio_decimal" in info

    def test_rgba_image_info(self, rgba_image):
        """Test getting info from RGBA image."""
        from vision_s8.services.image_service import get_image_service
        service = get_image_service()
        
        info = service.get_image_info(rgba_image)
        
        assert info["width"] == 500
        assert info["height"] == 500
        assert info["mode"] == "RGBA"

    def test_image_info_megapixels(self, rgb_image):
        """Test megapixels calculation."""
        from vision_s8.services.image_service import get_image_service
        service = get_image_service()
        
        info = service.get_image_info(rgb_image)
        
        expected_mp = (800 * 600) / 1_000_000
        assert info["megapixels"] == pytest.approx(expected_mp, rel=0.01)


# ============================================================================
# Image Loading Tests
# ============================================================================

class TestImageLoading:
    """Tests for image loading functionality."""

    def test_load_from_bytes_jpeg(self, rgb_image):
        """Test loading image from JPEG bytes."""
        from vision_s8.services.image_service import get_image_service
        service = get_image_service()
        
        # Save to bytes
        buffer = io.BytesIO()
        rgb_image.save(buffer, format="JPEG")
        buffer.seek(0)
        
        loaded = service.load_image_from_bytes(buffer.read())
        
        assert loaded is not None
        assert loaded.size == (800, 600)

    def test_load_from_bytes_png(self, rgba_image):
        """Test loading image from PNG bytes."""
        from vision_s8.services.image_service import get_image_service
        service = get_image_service()
        
        buffer = io.BytesIO()
        rgba_image.save(buffer, format="PNG")
        buffer.seek(0)
        
        loaded = service.load_image_from_bytes(buffer.read())
        
        assert loaded is not None
        # PNG preserves RGBA
        assert loaded.mode in ["RGBA", "RGB"]

    def test_load_invalid_bytes(self):
        """Test loading invalid bytes raises error."""
        from vision_s8.services.image_service import get_image_service
        service = get_image_service()
        
        with pytest.raises(Exception):
            service.load_image_from_bytes(b"not an image")


# ============================================================================
# Image Saving Tests
# ============================================================================

class TestImageSaving:
    """Tests for image saving functionality."""

    def test_save_as_jpeg(self, rgb_image, tmp_path):
        """Test saving image as JPEG."""
        from vision_s8.services.image_service import get_image_service
        service = get_image_service()
        
        output_path = service.save_image(rgb_image, "test.jpg", output_dir=tmp_path)
        
        assert output_path.exists()
        
        # Verify can be loaded back
        loaded = Image.open(output_path)
        assert loaded.size == (800, 600)

    def test_save_as_png(self, rgba_image, tmp_path):
        """Test saving image as PNG."""
        from vision_s8.services.image_service import get_image_service
        service = get_image_service()
        
        output_path = service.save_image(rgba_image, "test.png", output_dir=tmp_path)
        
        assert output_path.exists()


# ============================================================================
# Image Enhancement Tests
# ============================================================================

class TestImageEnhancement:
    """Tests for image enhancement utilities."""

    def test_adjust_lighting_brightness(self, rgb_image):
        """Test brightness adjustment."""
        from vision_s8.services.image_service import get_image_service
        service = get_image_service()
        
        brightened = service.adjust_lighting(rgb_image, brightness=50)
        
        assert brightened is not None
        assert brightened.size == rgb_image.size

    def test_adjust_lighting_contrast(self, rgb_image):
        """Test contrast adjustment."""
        from vision_s8.services.image_service import get_image_service
        service = get_image_service()
        
        adjusted = service.adjust_lighting(rgb_image, contrast=20)
        
        assert adjusted is not None
        assert adjusted.size == rgb_image.size

    def test_adjust_colors_saturation(self, rgb_image):
        """Test saturation adjustment."""
        from vision_s8.services.image_service import get_image_service
        service = get_image_service()
        
        saturated = service.adjust_colors(rgb_image, saturation=30)
        
        assert saturated is not None

    def test_adjust_colors_temperature(self, rgb_image):
        """Test temperature (warm/cool) adjustment."""
        from vision_s8.services.image_service import get_image_service
        service = get_image_service()
        
        warm = service.adjust_colors(rgb_image, temperature=30)
        cool = service.adjust_colors(rgb_image, temperature=-30)
        
        assert warm is not None
        assert cool is not None


# ============================================================================
# Crop Tests
# ============================================================================

class TestImageCropping:
    """Tests for image cropping functionality."""

    def test_auto_crop(self):
        """Test auto crop with product detection."""
        from vision_s8.services.image_service import get_image_service
        service = get_image_service()
        
        # Create image with "product" in center
        img = Image.new("RGB", (1000, 1000), color=(255, 255, 255))
        for x in range(300, 700):
            for y in range(300, 700):
                img.putpixel((x, y), (100, 100, 100))
        
        cropped = service.auto_crop(img)
        
        # Should crop to focus on the product
        assert cropped is not None
        assert cropped.width <= img.width
        assert cropped.height <= img.height

    def test_auto_crop_with_aspect_ratio(self):
        """Test auto crop with target aspect ratio."""
        from vision_s8.services.image_service import get_image_service
        service = get_image_service()
        
        img = Image.new("RGB", (1000, 800), color=(255, 255, 255))
        for x in range(200, 800):
            for y in range(200, 600):
                img.putpixel((x, y), (100, 100, 100))
        
        cropped = service.auto_crop(img, aspect_ratio="1:1")
        
        # Should be square or close to it
        assert cropped is not None


# ============================================================================
# Platform Resize Tests
# ============================================================================

class TestPlatformResize:
    """Tests for platform-specific resizing."""

    def test_resize_for_amazon(self):
        """Test resizing for Amazon requirements."""
        from vision_s8.services.image_service import get_image_service
        service = get_image_service()
        
        # Small image
        small = Image.new("RGB", (500, 500), color=(128, 128, 128))
        
        resized = service.resize_for_platform(small, "amazon")
        
        # Should be at least 1000x1000 for Amazon
        assert resized.width >= 1000
        assert resized.height >= 1000

    def test_resize_for_etsy(self):
        """Test resizing for Etsy requirements."""
        from vision_s8.services.image_service import get_image_service
        service = get_image_service()
        
        small = Image.new("RGB", (1000, 1000), color=(128, 128, 128))
        
        resized = service.resize_for_platform(small, "etsy")
        
        # Should be at least 2000x2000 for Etsy
        assert resized.width >= 2000
        assert resized.height >= 2000

    def test_no_resize_if_large_enough(self):
        """Test no resize if image meets requirements."""
        from vision_s8.services.image_service import get_image_service
        service = get_image_service()
        
        large = Image.new("RGB", (2000, 2000), color=(128, 128, 128))
        
        resized = service.resize_for_platform(large, "amazon")
        
        # Should not change size
        assert resized.size == (2000, 2000)


# ============================================================================
# Shadow Tests
# ============================================================================

class TestShadow:
    """Tests for shadow effects."""

    def test_add_drop_shadow(self, rgba_image):
        """Test adding drop shadow."""
        from vision_s8.services.image_service import get_image_service
        service = get_image_service()
        
        with_shadow = service.add_shadow(rgba_image, shadow_type="drop")
        
        # Shadow adds to dimensions
        assert with_shadow is not None
        assert with_shadow.width >= rgba_image.width

    def test_add_reflection_shadow(self, rgba_image):
        """Test adding reflection shadow."""
        from vision_s8.services.image_service import get_image_service
        service = get_image_service()
        
        with_reflection = service.add_shadow(rgba_image, shadow_type="reflection")
        
        assert with_reflection is not None
        # Reflection adds to height
        assert with_reflection.height > rgba_image.height


# ============================================================================
# Sharpening Tests
# ============================================================================

class TestSharpening:
    """Tests for image sharpening."""

    def test_sharpen_light(self, rgb_image):
        """Test light sharpening."""
        from vision_s8.services.image_service import get_image_service
        service = get_image_service()
        
        sharpened = service.sharpen(rgb_image, intensity="light")
        
        assert sharpened is not None
        assert sharpened.size == rgb_image.size

    def test_sharpen_strong(self, rgb_image):
        """Test strong sharpening."""
        from vision_s8.services.image_service import get_image_service
        service = get_image_service()
        
        sharpened = service.sharpen(rgb_image, intensity="strong")
        
        assert sharpened is not None


# ============================================================================
# Format Validation Tests
# ============================================================================

class TestFormatValidation:
    """Tests for format validation."""

    def test_supported_formats(self):
        """Test supported format detection."""
        from vision_s8.services.image_service import ImageService
        
        assert ImageService.is_supported_format("test.jpg") is True
        assert ImageService.is_supported_format("test.jpeg") is True
        assert ImageService.is_supported_format("test.png") is True
        assert ImageService.is_supported_format("test.webp") is True
        
    def test_unsupported_formats(self):
        """Test unsupported format detection."""
        from vision_s8.services.image_service import ImageService
        
        assert ImageService.is_supported_format("test.txt") is False
        assert ImageService.is_supported_format("test.pdf") is False
        assert ImageService.is_supported_format("test.exe") is False


# ============================================================================
# Hash Calculation Tests
# ============================================================================

class TestHashCalculation:
    """Tests for image hash calculation."""

    def test_same_image_same_hash(self, rgb_image):
        """Same image should produce same hash."""
        from vision_s8.services.image_service import ImageService
        
        hash1 = ImageService.calculate_hash(rgb_image)
        hash2 = ImageService.calculate_hash(rgb_image)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

    def test_different_images_different_hash(self, rgb_image, rgba_image):
        """Different images should produce different hashes."""
        from vision_s8.services.image_service import ImageService
        
        hash1 = ImageService.calculate_hash(rgb_image)
        hash2 = ImageService.calculate_hash(rgba_image)
        
        assert hash1 != hash2
