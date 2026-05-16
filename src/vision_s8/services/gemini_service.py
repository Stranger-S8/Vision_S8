"""
Gemini AI Service for Vision_S8.

Handles all interactions with Google's Gemini API including prompt management,
retry logic, and response parsing.
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from PIL import Image

from ..config import settings
from ..utils.logger import get_logger

logger = get_logger("vision_s8.gemini")


class GeminiService:
    """Service for interacting with Google Gemini AI."""

    def __init__(self):
        """Initialize the Gemini client."""
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_model
        self._prompts_cache: dict[str, str] = {}

    def _load_prompt(self, prompt_name: str) -> str:
        """
        Load a prompt template from file.

        Args:
            prompt_name: Name of the prompt (without extension)

        Returns:
            Prompt template string
        """
        if prompt_name not in self._prompts_cache:
            prompt_path = settings.prompts_dir / f"{prompt_name}.txt"
            if not prompt_path.exists():
                raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
            self._prompts_cache[prompt_name] = prompt_path.read_text(encoding="utf-8")
        return self._prompts_cache[prompt_name]

    def _extract_json(self, text: str) -> dict[str, Any]:
        """
        Extract JSON from response text.

        Args:
            text: Response text potentially containing JSON

        Returns:
            Parsed JSON dictionary
        """
        # Try to find JSON in code blocks first
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r"\{[\s\S]*\}", text)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("No JSON found in response")

        return json.loads(json_str)

    async def _generate_with_retry(
        self,
        prompt: str,
        images: list[Image.Image],
        max_retries: int | None = None,
    ) -> str:
        """
        Generate content with retry logic.

        Args:
            prompt: Text prompt
            images: List of PIL images
            max_retries: Maximum retry attempts

        Returns:
            Generated text response
        """
        retries = max_retries or settings.max_retries
        last_error = None

        for attempt in range(retries):
            try:
                # Build content list
                contents: list[Any] = [prompt]
                contents.extend(images)

                # Generate response
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        temperature=0.3,  # Lower for more consistent JSON
                        top_p=0.95,
                        max_output_tokens=4096,
                    ),
                )

                if response.text:
                    return response.text

                raise ValueError("Empty response from Gemini")

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Gemini API attempt {attempt + 1}/{retries} failed: {e}"
                )
                if attempt < retries - 1:
                    await asyncio.sleep(settings.retry_delay * (attempt + 1))

        raise RuntimeError(f"Gemini API failed after {retries} attempts: {last_error}")

    async def analyze_image(
        self,
        image: Image.Image,
        prompt_name: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Analyze a single image with the specified prompt.

        Args:
            image: PIL Image to analyze
            prompt_name: Name of the prompt template
            **kwargs: Additional template variables

        Returns:
            Parsed JSON response
        """
        prompt = self._load_prompt(prompt_name)

        # Replace template variables
        for key, value in kwargs.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))

        logger.debug(f"Analyzing image with prompt: {prompt_name}")
        response_text = await self._generate_with_retry(prompt, [image])

        try:
            result = self._extract_json(response_text)
            logger.debug(f"Successfully parsed response for {prompt_name}")
            return result
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw response: {response_text}")
            raise ValueError(f"Invalid JSON response from Gemini: {e}")

    async def compare_images(
        self,
        main_image: Image.Image,
        competitor_images: list[Image.Image],
    ) -> dict[str, Any]:
        """
        Compare main image against competitor images.

        Args:
            main_image: Primary product image
            competitor_images: List of competitor images

        Returns:
            Comparison result
        """
        prompt = self._load_prompt("compare")

        # Add context about which image is which
        numbered_prompt = prompt + "\n\nImage order: First image is the MAIN image. Subsequent images are COMPETITORS numbered 1, 2, 3, etc."

        all_images = [main_image] + competitor_images
        logger.debug(f"Comparing main image against {len(competitor_images)} competitors")

        response_text = await self._generate_with_retry(numbered_prompt, all_images)
        return self._extract_json(response_text)

    async def predict_ab_test(
        self,
        variant_images: list[Image.Image],
    ) -> dict[str, Any]:
        """
        Predict A/B test performance for image variants.

        Args:
            variant_images: List of product image variants

        Returns:
            A/B test prediction result
        """
        prompt = self._load_prompt("ab_test")

        # Add context about variants
        numbered_prompt = prompt + f"\n\nAnalyzing {len(variant_images)} variants. Images are numbered 1 through {len(variant_images)} in order."

        logger.debug(f"Predicting A/B test for {len(variant_images)} variants")
        response_text = await self._generate_with_retry(numbered_prompt, variant_images)
        return self._extract_json(response_text)

    async def check_compliance(
        self,
        image: Image.Image,
        platform: str,
    ) -> dict[str, Any]:
        """
        Check image compliance for a specific platform.

        Args:
            image: Product image
            platform: Target platform name

        Returns:
            Compliance check result
        """
        return await self.analyze_image(image, "compliance", platform=platform)

    async def analyze_seo(self, image: Image.Image) -> dict[str, Any]:
        """
        Analyze image for SEO optimization.

        Args:
            image: Product image

        Returns:
            SEO analysis result
        """
        return await self.analyze_image(image, "seo")

    async def get_enhancement_plan(self, image: Image.Image) -> dict[str, Any]:
        """
        Get enhancement recommendations for an image.

        Args:
            image: Product image

        Returns:
            Enhancement plan
        """
        return await self.analyze_image(image, "enhance")

    async def audit_image(self, image: Image.Image) -> dict[str, Any]:
        """
        Perform a full audit on a product image.

        Args:
            image: Product image

        Returns:
            Audit result
        """
        return await self.analyze_image(image, "audit")

    async def verify_connection(self) -> bool:
        """
        Verify the Gemini API connection.

        Returns:
            True if connection is successful
        """
        try:
            # Simple test call
            response = self._client.models.generate_content(
                model=self._model,
                contents=["Say 'OK' if you can read this."],
                config=types.GenerateContentConfig(max_output_tokens=10),
            )
            return bool(response.text)
        except Exception as e:
            logger.error(f"Gemini connection verification failed: {e}")
            return False


# Singleton instance
_gemini_service: GeminiService | None = None


def get_gemini_service() -> GeminiService:
    """Get the Gemini service singleton."""
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service
