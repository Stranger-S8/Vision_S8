"""
Image processing service for Vision_S8.

Handles image loading, manipulation, enhancement, and analysis using PIL, OpenCV, and rembg.
"""

import hashlib
import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from ..config import settings
from ..utils.logger import get_logger

logger = get_logger("vision_s8.image")


class ImageService:
    """Service for image processing operations."""

    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".tiff", ".bmp"}

    def __init__(self):
        """Initialize the image service."""
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure upload and output directories exist."""
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        settings.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def is_supported_format(filename: str) -> bool:
        """Check if a file has a supported image format."""
        return Path(filename).suffix.lower() in ImageService.SUPPORTED_FORMATS

    @staticmethod
    def calculate_hash(image: Image.Image) -> str:
        """Calculate SHA256 hash of an image."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return hashlib.sha256(buffer.getvalue()).hexdigest()

    def load_image(self, path: str | Path) -> Image.Image:
        """
        Load an image from file.

        Args:
            path: Path to the image file

        Returns:
            PIL Image object
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = Image.open(path)
        # Convert to RGB if necessary (handles RGBA, P mode, etc.)
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")

        logger.debug(f"Loaded image: {path} ({image.size})")
        return image

    def load_image_from_bytes(self, data: bytes) -> Image.Image:
        """
        Load an image from bytes.

        Args:
            data: Image data as bytes

        Returns:
            PIL Image object
        """
        image = Image.open(io.BytesIO(data))
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGB")
        return image

    def save_image(
        self,
        image: Image.Image,
        filename: str,
        output_dir: Path | None = None,
        quality: int = 95,
    ) -> Path:
        """
        Save an image to file.

        Args:
            image: PIL Image to save
            filename: Output filename
            output_dir: Output directory (default: settings.output_dir)
            quality: JPEG quality (1-100)

        Returns:
            Path to saved file
        """
        output_dir = output_dir or settings.output_dir
        output_path = output_dir / filename

        # Determine format from extension
        ext = Path(filename).suffix.lower()
        if ext in (".jpg", ".jpeg"):
            image = image.convert("RGB")  # JPEG doesn't support alpha
            image.save(output_path, "JPEG", quality=quality, optimize=True)
        elif ext == ".png":
            image.save(output_path, "PNG", optimize=True)
        elif ext == ".webp":
            image.save(output_path, "WEBP", quality=quality)
        else:
            image.save(output_path)

        logger.debug(f"Saved image: {output_path}")
        return output_path

    def get_image_info(self, image: Image.Image) -> dict[str, Any]:
        """
        Get detailed information about an image.

        Args:
            image: PIL Image

        Returns:
            Dictionary with image metadata
        """
        width, height = image.size

        # Convert to numpy for additional analysis
        np_image = np.array(image.convert("RGB"))

        # Calculate basic statistics
        brightness = np.mean(np_image)
        contrast = np.std(np_image)

        # Detect if image has white background (simplified)
        corners = [
            np_image[0, 0],
            np_image[0, -1],
            np_image[-1, 0],
            np_image[-1, -1],
        ]
        avg_corner = np.mean(corners, axis=0)
        has_white_bg = np.all(avg_corner > 240)

        return {
            "width": width,
            "height": height,
            "aspect_ratio": f"{width}:{height}",
            "aspect_ratio_decimal": round(width / height, 2),
            "mode": image.mode,
            "format": image.format,
            "megapixels": round((width * height) / 1_000_000, 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "has_white_background": has_white_bg,
            "is_square": abs(width - height) < 10,
        }

    def remove_background(
        self,
        image: Image.Image,
        background_color: tuple[int, int, int] = (255, 255, 255),
    ) -> Image.Image:
        """
        Remove background from an image.

        Args:
            image: Input image
            background_color: RGB color for new background

        Returns:
            Image with background removed/replaced
        """
        logger.debug("Removing background from image")

        try:
            from rembg import remove as remove_background
        except ImportError as exc:
            raise RuntimeError(
                "Background removal is unavailable in this environment because "
                "the optional 'rembg' dependency could not be loaded."
            ) from exc

        # Use rembg to remove background
        result = remove_background(image)

        # If a background color is specified, composite onto it
        if background_color:
            background = Image.new("RGBA", result.size, (*background_color, 255))
            background.paste(result, mask=result.split()[-1])
            result = background.convert("RGB")

        return result

    def adjust_lighting(
        self,
        image: Image.Image,
        brightness: int = 0,
        contrast: int = 0,
        highlights: int = 0,
        shadows: int = 0,
    ) -> Image.Image:
        """
        Adjust image lighting.

        Args:
            image: Input image
            brightness: Brightness adjustment (-100 to 100)
            contrast: Contrast adjustment (-100 to 100)
            highlights: Highlights adjustment (-100 to 100)
            shadows: Shadows adjustment (-100 to 100)

        Returns:
            Adjusted image
        """
        result = image.convert("RGB")

        # Apply brightness
        if brightness != 0:
            factor = 1 + (brightness / 100)
            enhancer = ImageEnhance.Brightness(result)
            result = enhancer.enhance(factor)

        # Apply contrast
        if contrast != 0:
            factor = 1 + (contrast / 100)
            enhancer = ImageEnhance.Contrast(result)
            result = enhancer.enhance(factor)

        # Apply highlights/shadows using curves (simplified using numpy)
        if highlights != 0 or shadows != 0:
            np_image = np.array(result).astype(np.float32)

            if highlights != 0:
                # Adjust highlights (bright areas)
                highlight_mask = np_image > 128
                adjustment = highlights / 100 * 50
                np_image = np.where(highlight_mask, np_image + adjustment, np_image)

            if shadows != 0:
                # Adjust shadows (dark areas)
                shadow_mask = np_image < 128
                adjustment = shadows / 100 * 50
                np_image = np.where(shadow_mask, np_image + adjustment, np_image)

            np_image = np.clip(np_image, 0, 255).astype(np.uint8)
            result = Image.fromarray(np_image)

        return result

    def adjust_colors(
        self,
        image: Image.Image,
        saturation: int = 0,
        vibrance: int = 0,
        temperature: int = 0,
        tint: int = 0,
    ) -> Image.Image:
        """
        Adjust image colors.

        Args:
            image: Input image
            saturation: Saturation adjustment (-100 to 100)
            vibrance: Vibrance adjustment (-100 to 100)
            temperature: Temperature adjustment (-100 to 100, warm/cool)
            tint: Tint adjustment (-100 to 100, green/magenta)

        Returns:
            Color-adjusted image
        """
        result = image.convert("RGB")

        # Apply saturation
        if saturation != 0:
            factor = 1 + (saturation / 100)
            enhancer = ImageEnhance.Color(result)
            result = enhancer.enhance(factor)

        # Apply vibrance (selective saturation for less saturated colors)
        if vibrance != 0:
            np_image = np.array(result).astype(np.float32)
            hsv = cv2.cvtColor(np_image.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)

            # Increase saturation more for less saturated pixels
            saturation_channel = hsv[:, :, 1]
            mask = 1 - (saturation_channel / 255)  # Inverse saturation
            adjustment = mask * (vibrance / 100) * 50
            hsv[:, :, 1] = np.clip(saturation_channel + adjustment, 0, 255)

            np_image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            result = Image.fromarray(np_image)

        # Apply temperature (warm/cool shift)
        if temperature != 0:
            np_image = np.array(result).astype(np.float32)
            # Warm: increase red, decrease blue
            # Cool: decrease red, increase blue
            shift = temperature / 100 * 30
            np_image[:, :, 0] = np.clip(np_image[:, :, 0] + shift, 0, 255)  # Red
            np_image[:, :, 2] = np.clip(np_image[:, :, 2] - shift, 0, 255)  # Blue
            result = Image.fromarray(np_image.astype(np.uint8))

        # Apply tint (green/magenta shift)
        if tint != 0:
            np_image = np.array(result).astype(np.float32)
            shift = tint / 100 * 20
            np_image[:, :, 1] = np.clip(np_image[:, :, 1] - shift, 0, 255)  # Green
            result = Image.fromarray(np_image.astype(np.uint8))

        return result

    def auto_crop(
        self,
        image: Image.Image,
        aspect_ratio: str | None = None,
        padding_percent: float = 5,
    ) -> Image.Image:
        """
        Auto-crop image to focus on the product.

        Args:
            image: Input image
            aspect_ratio: Target aspect ratio (e.g., "1:1", "4:3")
            padding_percent: Padding around detected product

        Returns:
            Cropped image
        """
        np_image = np.array(image.convert("RGB"))

        # Convert to grayscale and find edges
        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return image

        # Get bounding box of all contours
        all_points = np.vstack(contours)
        x, y, w, h = cv2.boundingRect(all_points)

        # Add padding
        pad_x = int(w * padding_percent / 100)
        pad_y = int(h * padding_percent / 100)
        x = max(0, x - pad_x)
        y = max(0, y - pad_y)
        w = min(np_image.shape[1] - x, w + 2 * pad_x)
        h = min(np_image.shape[0] - y, h + 2 * pad_y)

        # Crop
        cropped = image.crop((x, y, x + w, y + h))

        # Adjust to target aspect ratio if specified
        if aspect_ratio:
            target_w, target_h = map(int, aspect_ratio.split(":"))
            target_ratio = target_w / target_h
            current_ratio = cropped.width / cropped.height

            if current_ratio > target_ratio:
                # Too wide, crop width
                new_width = int(cropped.height * target_ratio)
                offset = (cropped.width - new_width) // 2
                cropped = cropped.crop((offset, 0, offset + new_width, cropped.height))
            else:
                # Too tall, crop height
                new_height = int(cropped.width / target_ratio)
                offset = (cropped.height - new_height) // 2
                cropped = cropped.crop((0, offset, cropped.width, offset + new_height))

        return cropped

    def add_shadow(
        self,
        image: Image.Image,
        shadow_type: str = "drop",
        intensity: str = "medium",
    ) -> Image.Image:
        """
        Add a shadow to the image.

        Args:
            image: Input image (should have transparent or white background)
            shadow_type: Type of shadow ("drop", "natural", "reflection")
            intensity: Shadow intensity ("subtle", "medium", "pronounced")

        Returns:
            Image with shadow added
        """
        # Ensure RGBA mode
        if image.mode != "RGBA":
            image = image.convert("RGBA")

        # Shadow parameters based on intensity
        intensity_map = {
            "subtle": {"opacity": 0.2, "blur": 5, "offset": 3},
            "medium": {"opacity": 0.4, "blur": 10, "offset": 7},
            "pronounced": {"opacity": 0.6, "blur": 15, "offset": 12},
        }
        params = intensity_map.get(intensity, intensity_map["medium"])

        # Create shadow layer
        shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))

        if shadow_type == "drop":
            # Simple drop shadow
            alpha = image.split()[-1]
            shadow_alpha = alpha.filter(ImageFilter.GaussianBlur(params["blur"]))
            shadow.putalpha(shadow_alpha)

            # Create new image with shadow offset
            result = Image.new("RGBA", (image.width + params["offset"] * 2, image.height + params["offset"] * 2), (255, 255, 255, 255))
            result.paste(shadow, (params["offset"], params["offset"]), shadow_alpha)
            result.paste(image, (0, 0), image)

        elif shadow_type == "reflection":
            # Reflection shadow (mirrored, fading)
            reflection = image.transpose(Image.FLIP_TOP_BOTTOM)
            reflection = reflection.resize((image.width, image.height // 3))

            # Create gradient mask for fade effect
            gradient = Image.new("L", reflection.size)
            for y in range(reflection.height):
                alpha = int(255 * (1 - y / reflection.height) * params["opacity"])
                for x in range(reflection.width):
                    gradient.putpixel((x, y), alpha)

            reflection.putalpha(gradient)

            # Combine original and reflection
            result = Image.new("RGBA", (image.width, image.height + reflection.height), (255, 255, 255, 255))
            result.paste(image, (0, 0), image)
            result.paste(reflection, (0, image.height), reflection)

        else:  # natural
            # Natural contact shadow
            alpha = image.split()[-1]
            # Squash shadow vertically
            shadow_height = int(image.height * 0.1)
            shadow_alpha = alpha.resize((image.width, shadow_height))
            shadow_alpha = shadow_alpha.filter(ImageFilter.GaussianBlur(params["blur"]))

            result = Image.new("RGBA", (image.width, image.height + shadow_height // 2), (255, 255, 255, 255))
            shadow_layer = Image.new("RGBA", (image.width, shadow_height), (0, 0, 0, int(255 * params["opacity"])))
            result.paste(shadow_layer, (0, image.height - shadow_height // 2), shadow_alpha)
            result.paste(image, (0, 0), image)

        return result

    def sharpen(self, image: Image.Image, intensity: str = "medium") -> Image.Image:
        """
        Sharpen an image.

        Args:
            image: Input image
            intensity: Sharpening intensity ("light", "medium", "strong")

        Returns:
            Sharpened image
        """
        intensity_map = {
            "light": 1.2,
            "medium": 1.5,
            "strong": 2.0,
        }
        factor = intensity_map.get(intensity, 1.5)

        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)

    def resize_for_platform(
        self,
        image: Image.Image,
        platform: str,
        maintain_aspect: bool = True,
    ) -> Image.Image:
        """
        Resize image to meet platform requirements.

        Args:
            image: Input image
            platform: Target platform
            maintain_aspect: Whether to maintain aspect ratio

        Returns:
            Resized image
        """
        # Platform minimum sizes
        platform_sizes = {
            "amazon": (1000, 1000),
            "shopify": (800, 800),
            "instagram": (1080, 1080),
            "ebay": (500, 500),
            "etsy": (2000, 2000),
            "walmart": (1000, 1000),
        }

        min_size = platform_sizes.get(platform.lower(), (1000, 1000))

        if image.width >= min_size[0] and image.height >= min_size[1]:
            return image

        if maintain_aspect:
            # Scale to meet minimum while maintaining aspect ratio
            scale = max(min_size[0] / image.width, min_size[1] / image.height)
            new_size = (int(image.width * scale), int(image.height * scale))
        else:
            new_size = min_size

        return image.resize(new_size, Image.Resampling.LANCZOS)


# Singleton instance
_image_service: ImageService | None = None


def get_image_service() -> ImageService:
    """Get the image service singleton."""
    global _image_service
    if _image_service is None:
        _image_service = ImageService()
    return _image_service
