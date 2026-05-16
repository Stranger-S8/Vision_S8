"""
Image Enhancer for Vision_S8.

Handles AI-powered enhancement recommendations and automatic image improvements.
"""

import time
from typing import Any

from PIL import Image

from ..models.schemas import (
    BackgroundEnhancement,
    ColorCorrection,
    ColorEnhancement,
    CroppingEnhancement,
    EnhancementPlan,
    EnhanceResult,
    LightingAdjustment,
    LightingEnhancement,
    ShadowEnhancement,
    SharpeningEnhancement,
)
from ..services.gemini_service import GeminiService, get_gemini_service
from ..services.image_service import ImageService, get_image_service
from ..utils.logger import get_logger

logger = get_logger("vision_s8.enhancer")


class ImageEnhancer:
    """Engine for analyzing and enhancing product images."""

    def __init__(
        self,
        gemini_service: GeminiService | None = None,
        image_service: ImageService | None = None,
    ):
        """Initialize the enhancer."""
        self._gemini = gemini_service or get_gemini_service()
        self._image = image_service or get_image_service()

    async def analyze(self, image: Image.Image) -> tuple[EnhanceResult, int]:
        """
        Analyze an image and get enhancement recommendations.

        Args:
            image: PIL Image to analyze

        Returns:
            Tuple of (EnhanceResult, processing_time_ms)
        """
        start_time = time.time()

        logger.info("Analyzing image for enhancement recommendations")

        # Get AI enhancement plan
        raw_result = await self._gemini.get_enhancement_plan(image)

        # Parse into structured format
        result = self._parse_enhancement_result(raw_result)

        processing_time = int((time.time() - start_time) * 1000)
        logger.info(f"Enhancement analysis completed in {processing_time}ms")

        return result, processing_time

    async def enhance(
        self,
        image: Image.Image,
        apply_all: bool = True,
        enhancements: list[str] | None = None,
    ) -> tuple[Image.Image, EnhanceResult, int]:
        """
        Analyze and apply enhancements to an image.

        Args:
            image: PIL Image to enhance
            apply_all: Apply all recommended enhancements
            enhancements: Specific enhancements to apply (if not apply_all)

        Returns:
            Tuple of (enhanced_image, analysis_result, processing_time_ms)
        """
        start_time = time.time()

        # Get analysis
        result, _ = await self.analyze(image)

        # Determine which enhancements to apply
        if apply_all:
            to_apply = [e for e in result.enhancement_order if self._is_recommended(result, e)]
        else:
            to_apply = enhancements or []

        # Apply enhancements in order
        enhanced = image.copy()
        for enhancement_name in to_apply:
            enhanced = self._apply_enhancement(enhanced, enhancement_name, result)

        processing_time = int((time.time() - start_time) * 1000)
        logger.info(f"Applied {len(to_apply)} enhancements in {processing_time}ms")

        return enhanced, result, processing_time

    def _parse_enhancement_result(self, raw: dict[str, Any]) -> EnhanceResult:
        """Parse raw AI result into structured format."""
        enhancements_raw = raw.get("enhancements", {})

        # Parse each enhancement type
        bg = enhancements_raw.get("background_removal", {})
        background = BackgroundEnhancement(
            recommended=bg.get("recommended", False),
            reason=bg.get("reason", ""),
            suggested_background=bg.get("suggested_background"),
        )

        light = enhancements_raw.get("lighting_adjustment", {})
        light_adj = light.get("adjustments", {})
        lighting = LightingEnhancement(
            recommended=light.get("recommended", False),
            reason=light.get("reason", ""),
            adjustments=LightingAdjustment(
                brightness=self._clamp_adjustment(light_adj.get("brightness", 0)),
                contrast=self._clamp_adjustment(light_adj.get("contrast", 0)),
                highlights=self._clamp_adjustment(light_adj.get("highlights", 0)),
                shadows=self._clamp_adjustment(light_adj.get("shadows", 0)),
            ) if light.get("recommended") else None,
        )

        color = enhancements_raw.get("color_correction", {})
        color_adj = color.get("adjustments", {})
        color_correction = ColorEnhancement(
            recommended=color.get("recommended", False),
            reason=color.get("reason", ""),
            adjustments=ColorCorrection(
                saturation=self._clamp_adjustment(color_adj.get("saturation", 0)),
                vibrance=self._clamp_adjustment(color_adj.get("vibrance", 0)),
                temperature=self._clamp_adjustment(color_adj.get("temperature", 0)),
                tint=self._clamp_adjustment(color_adj.get("tint", 0)),
            ) if color.get("recommended") else None,
        )

        crop = enhancements_raw.get("cropping", {})
        cropping = CroppingEnhancement(
            recommended=crop.get("recommended", False),
            reason=crop.get("reason", ""),
            suggested_aspect_ratio=crop.get("suggested_aspect_ratio"),
            crop_focus=crop.get("crop_focus"),
        )

        sharp = enhancements_raw.get("sharpening", {})
        sharpening = SharpeningEnhancement(
            recommended=sharp.get("recommended", False),
            reason=sharp.get("reason", ""),
            intensity=sharp.get("intensity"),
        )

        shadow = enhancements_raw.get("shadow_addition", {})
        shadow_enhancement = ShadowEnhancement(
            recommended=shadow.get("recommended", False),
            reason=shadow.get("reason", ""),
            shadow_type=shadow.get("shadow_type"),
            intensity=shadow.get("intensity"),
        )

        plan = EnhancementPlan(
            background_removal=background,
            lighting_adjustment=lighting,
            color_correction=color_correction,
            cropping=cropping,
            sharpening=sharpening,
            shadow_addition=shadow_enhancement,
        )

        return EnhanceResult(
            current_assessment=raw.get("current_assessment", {}),
            enhancements=plan,
            enhancement_order=raw.get("enhancement_order", []),
            expected_score_after=raw.get("expected_score_after", 0),
            professional_notes=raw.get("professional_notes", ""),
        )

    def _is_recommended(self, result: EnhanceResult, enhancement_name: str) -> bool:
        """Check if an enhancement is recommended."""
        name_map = {
            "background_removal": result.enhancements.background_removal,
            "lighting_adjustment": result.enhancements.lighting_adjustment,
            "color_correction": result.enhancements.color_correction,
            "cropping": result.enhancements.cropping,
            "sharpening": result.enhancements.sharpening,
            "shadow_addition": result.enhancements.shadow_addition,
        }
        enhancement = name_map.get(enhancement_name.lower())
        return enhancement.recommended if enhancement else False

    def _apply_enhancement(
        self,
        image: Image.Image,
        enhancement_name: str,
        result: EnhanceResult,
    ) -> Image.Image:
        """Apply a single enhancement to an image."""
        logger.debug(f"Applying enhancement: {enhancement_name}")

        name = enhancement_name.lower()

        if name == "background_removal":
            bg = result.enhancements.background_removal
            color = (255, 255, 255)  # Default white
            if bg.suggested_background and bg.suggested_background.lower() != "white":
                # Could parse color names/hex here
                pass
            return self._image.remove_background(image, color)

        elif name == "lighting_adjustment":
            adj = result.enhancements.lighting_adjustment.adjustments
            if adj:
                return self._image.adjust_lighting(
                    image,
                    brightness=adj.brightness,
                    contrast=adj.contrast,
                    highlights=adj.highlights,
                    shadows=adj.shadows,
                )

        elif name == "color_correction":
            adj = result.enhancements.color_correction.adjustments
            if adj:
                return self._image.adjust_colors(
                    image,
                    saturation=adj.saturation,
                    vibrance=adj.vibrance,
                    temperature=adj.temperature,
                    tint=adj.tint,
                )

        elif name == "cropping":
            crop = result.enhancements.cropping
            ratio = crop.suggested_aspect_ratio if crop.suggested_aspect_ratio != "original" else None
            return self._image.auto_crop(image, aspect_ratio=ratio)

        elif name == "sharpening":
            sharp = result.enhancements.sharpening
            intensity = sharp.intensity or "medium"
            return self._image.sharpen(image, intensity=intensity)

        elif name == "shadow_addition":
            shadow = result.enhancements.shadow_addition
            return self._image.add_shadow(
                image,
                shadow_type=shadow.shadow_type or "drop",
                intensity=shadow.intensity or "medium",
            )

        return image

    @staticmethod
    def _clamp_adjustment(value: int | float | None) -> int:
        """Clamp adjustment value to valid range."""
        if value is None:
            return 0
        return max(-100, min(100, int(value)))
