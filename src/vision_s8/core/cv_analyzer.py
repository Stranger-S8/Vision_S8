"""
Computer Vision Analyzer for Vision_S8.

Implements traditional computer vision algorithms using OpenCV for
technical image analysis that doesn't require AI.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Any

from ..utils.logger import get_logger

logger = get_logger("vision_s8.cv_analyzer")


class CVAnalyzer:
    """
    Computer Vision Analyzer using OpenCV.
    
    Performs technical analysis of product images using traditional
    computer vision algorithms - no AI required.
    """

    def __init__(self):
        """Initialize the CV analyzer."""
        pass

    def analyze(self, image: Image.Image) -> dict[str, Any]:
        """
        Perform comprehensive CV analysis on an image.

        Args:
            image: PIL Image to analyze

        Returns:
            Dictionary with all CV metrics
        """
        # Convert PIL to OpenCV format
        cv_image = self._pil_to_cv(image)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        return {
            "dimensions": self.get_dimensions(image),
            "sharpness": self.calculate_sharpness(gray),
            "brightness": self.calculate_brightness(cv_image),
            "contrast": self.calculate_contrast(gray),
            "color_analysis": self.analyze_colors(cv_image),
            "background_analysis": self.analyze_background(cv_image),
            "edge_density": self.calculate_edge_density(gray),
            "noise_level": self.estimate_noise(gray),
            "blur_detection": self.detect_blur(gray),
            "symmetry_score": self.calculate_symmetry(gray),
            "composition": self.analyze_composition(cv_image),
            "histogram_analysis": self.analyze_histogram(cv_image),
        }

    @staticmethod
    def _pil_to_cv(image: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV format (BGR)."""
        rgb = np.array(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _cv_to_pil(cv_image: np.ndarray) -> Image.Image:
        """Convert OpenCV image to PIL format."""
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def get_dimensions(self, image: Image.Image) -> dict[str, Any]:
        """Get image dimensions and aspect ratio."""
        width, height = image.size
        gcd = np.gcd(width, height)
        aspect_w, aspect_h = width // gcd, height // gcd

        return {
            "width": width,
            "height": height,
            "megapixels": round((width * height) / 1_000_000, 2),
            "aspect_ratio": f"{aspect_w}:{aspect_h}",
            "aspect_decimal": round(width / height, 3),
            "is_square": abs(width - height) < 10,
            "orientation": "landscape" if width > height else ("portrait" if height > width else "square"),
        }

    def calculate_sharpness(self, gray: np.ndarray) -> dict[str, Any]:
        """
        Calculate image sharpness using Laplacian variance.
        
        Higher values indicate sharper images.
        """
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()

        # Classify sharpness
        if variance > 500:
            quality = "excellent"
            score = 10
        elif variance > 200:
            quality = "good"
            score = 8
        elif variance > 100:
            quality = "acceptable"
            score = 6
        elif variance > 50:
            quality = "soft"
            score = 4
        else:
            quality = "blurry"
            score = 2

        return {
            "laplacian_variance": round(variance, 2),
            "quality": quality,
            "score": score,
            "is_sharp": variance > 100,
        }

    def calculate_brightness(self, cv_image: np.ndarray) -> dict[str, Any]:
        """
        Calculate image brightness metrics.
        
        Analyzes overall exposure and brightness distribution.
        """
        # Convert to LAB color space for perceptual brightness
        lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        mean_brightness = np.mean(l_channel)
        std_brightness = np.std(l_channel)

        # Classify brightness (L channel is 0-255)
        if mean_brightness < 50:
            level = "very_dark"
            score = 3
        elif mean_brightness < 80:
            level = "dark"
            score = 5
        elif mean_brightness < 120:
            level = "underexposed"
            score = 7
        elif mean_brightness < 180:
            level = "optimal"
            score = 10
        elif mean_brightness < 220:
            level = "bright"
            score = 7
        else:
            level = "overexposed"
            score = 4

        return {
            "mean": round(mean_brightness, 2),
            "std": round(std_brightness, 2),
            "level": level,
            "score": score,
            "is_optimal": 120 <= mean_brightness <= 180,
        }

    def calculate_contrast(self, gray: np.ndarray) -> dict[str, Any]:
        """
        Calculate image contrast using multiple methods.
        """
        # Method 1: Standard deviation (simple contrast)
        std_contrast = np.std(gray)

        # Method 2: Michelson contrast
        min_val, max_val = np.min(gray), np.max(gray)
        if max_val + min_val > 0:
            michelson = (max_val - min_val) / (max_val + min_val)
        else:
            michelson = 0

        # Method 3: RMS contrast
        rms_contrast = np.sqrt(np.mean((gray - np.mean(gray)) ** 2))

        # Score based on standard deviation
        if std_contrast > 60:
            quality = "high"
            score = 9
        elif std_contrast > 45:
            quality = "good"
            score = 8
        elif std_contrast > 30:
            quality = "moderate"
            score = 6
        else:
            quality = "low"
            score = 4

        return {
            "std_contrast": round(std_contrast, 2),
            "michelson_contrast": round(michelson, 3),
            "rms_contrast": round(rms_contrast, 2),
            "dynamic_range": int(max_val - min_val),
            "quality": quality,
            "score": score,
        }

    def analyze_colors(self, cv_image: np.ndarray) -> dict[str, Any]:
        """
        Analyze color distribution and saturation.
        """
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Saturation analysis
        mean_saturation = np.mean(s)
        saturation_std = np.std(s)

        # Color temperature estimation (simplified)
        b, g, r = cv2.split(cv_image)
        mean_r, mean_b = np.mean(r), np.mean(b)
        
        if mean_r > mean_b + 20:
            temperature = "warm"
        elif mean_b > mean_r + 20:
            temperature = "cool"
        else:
            temperature = "neutral"

        # Dominant color detection using k-means
        dominant_colors = self._get_dominant_colors(cv_image, k=5)

        # Color variety score
        color_variety = saturation_std / 128 * 10  # Normalize to 0-10

        return {
            "mean_saturation": round(mean_saturation, 2),
            "saturation_std": round(saturation_std, 2),
            "temperature": temperature,
            "dominant_colors": dominant_colors,
            "color_variety_score": round(min(color_variety, 10), 1),
            "is_vibrant": mean_saturation > 100,
            "is_muted": mean_saturation < 50,
        }

    def _get_dominant_colors(self, cv_image: np.ndarray, k: int = 5) -> list[dict]:
        """Extract dominant colors using k-means clustering."""
        # Reshape image for k-means
        pixels = cv_image.reshape(-1, 3).astype(np.float32)

        # Sample for performance (max 10000 pixels)
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]

        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Count pixels in each cluster
        unique, counts = np.unique(labels, return_counts=True)
        percentages = counts / len(labels) * 100

        # Sort by percentage
        sorted_indices = np.argsort(-percentages)

        colors = []
        for idx in sorted_indices:
            bgr = centers[idx].astype(int)
            rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
            hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)
            colors.append({
                "rgb": rgb,
                "hex": hex_color,
                "percentage": round(percentages[idx], 1),
            })

        return colors

    def analyze_background(self, cv_image: np.ndarray) -> dict[str, Any]:
        """
        Analyze background characteristics.
        
        Detects background type, color, and purity.
        """
        height, width = cv_image.shape[:2]

        # Sample corners and edges for background detection
        corner_size = min(width, height) // 10
        corners = [
            cv_image[0:corner_size, 0:corner_size],                    # Top-left
            cv_image[0:corner_size, -corner_size:],                    # Top-right
            cv_image[-corner_size:, 0:corner_size],                    # Bottom-left
            cv_image[-corner_size:, -corner_size:],                    # Bottom-right
        ]

        # Calculate mean color of corners
        corner_colors = [np.mean(corner, axis=(0, 1)) for corner in corners]
        avg_corner_color = np.mean(corner_colors, axis=0)

        # Check for white background
        is_white = np.all(avg_corner_color > 240)
        is_light = np.all(avg_corner_color > 200)

        # Check background uniformity
        corner_std = np.std(corner_colors, axis=0)
        is_uniform = np.all(corner_std < 20)

        # Background color classification
        b, g, r = avg_corner_color
        if is_white:
            bg_type = "pure_white"
        elif is_light:
            bg_type = "light"
        elif np.mean(avg_corner_color) < 50:
            bg_type = "dark"
        else:
            bg_type = "colored"

        # Calculate background purity score
        purity_score = 10 - min(np.mean(corner_std) / 5, 10)

        return {
            "type": bg_type,
            "is_white": bool(is_white),
            "is_uniform": bool(is_uniform),
            "average_color_rgb": tuple(avg_corner_color.astype(int).tolist()),
            "uniformity_score": round(purity_score, 1),
            "meets_amazon_requirements": bool(is_white and is_uniform),
        }

    def calculate_edge_density(self, gray: np.ndarray) -> dict[str, Any]:
        """
        Calculate edge density using Canny edge detection.
        
        Higher edge density often indicates more detail in the image.
        """
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Calculate edge density
        total_pixels = edges.size
        edge_pixels = np.sum(edges > 0)
        density = edge_pixels / total_pixels * 100

        # Classify
        if density > 15:
            level = "high"
            score = 8
        elif density > 8:
            level = "moderate"
            score = 10  # Optimal for product photos
        elif density > 3:
            level = "low"
            score = 6
        else:
            level = "very_low"
            score = 4

        return {
            "density_percent": round(density, 2),
            "edge_pixel_count": int(edge_pixels),
            "level": level,
            "score": score,
        }

    def estimate_noise(self, gray: np.ndarray) -> dict[str, Any]:
        """
        Estimate image noise level.
        
        Uses Laplacian-based noise estimation.
        """
        # Apply Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Estimate noise as median absolute deviation
        noise_estimate = np.median(np.abs(laplacian)) * 1.4826

        # Classify noise level
        if noise_estimate < 3:
            level = "very_low"
            score = 10
        elif noise_estimate < 6:
            level = "low"
            score = 9
        elif noise_estimate < 12:
            level = "moderate"
            score = 7
        elif noise_estimate < 20:
            level = "high"
            score = 4
        else:
            level = "very_high"
            score = 2

        return {
            "noise_estimate": round(noise_estimate, 2),
            "level": level,
            "score": score,
            "is_clean": noise_estimate < 6,
        }

    def detect_blur(self, gray: np.ndarray) -> dict[str, Any]:
        """
        Detect motion blur and focus blur.
        """
        # FFT-based blur detection
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log(np.abs(f_shift) + 1)

        # High frequency content indicates sharpness
        center = np.array(magnitude.shape) // 2
        high_freq_region = magnitude[
            center[0]-50:center[0]+50,
            center[1]-50:center[1]+50
        ]
        high_freq_energy = np.mean(high_freq_region)

        # Laplacian variance for blur detection
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Classify blur
        if laplacian_var > 300:
            blur_type = "none"
            score = 10
        elif laplacian_var > 100:
            blur_type = "slight"
            score = 7
        elif laplacian_var > 50:
            blur_type = "moderate"
            score = 5
        else:
            blur_type = "significant"
            score = 2

        return {
            "laplacian_variance": round(laplacian_var, 2),
            "high_freq_energy": round(high_freq_energy, 2),
            "blur_type": blur_type,
            "score": score,
            "is_blurry": laplacian_var < 100,
        }

    def calculate_symmetry(self, gray: np.ndarray) -> dict[str, Any]:
        """
        Calculate image symmetry score.
        
        Product images often benefit from symmetrical composition.
        """
        height, width = gray.shape

        # Horizontal symmetry
        left_half = gray[:, :width//2]
        right_half = cv2.flip(gray[:, width//2:], 1)
        # Resize to match if needed
        if left_half.shape[1] != right_half.shape[1]:
            right_half = cv2.resize(right_half, (left_half.shape[1], left_half.shape[0]))
        h_diff = np.mean(np.abs(left_half.astype(float) - right_half.astype(float)))
        h_symmetry = max(0, 100 - h_diff) / 10

        # Vertical symmetry
        top_half = gray[:height//2, :]
        bottom_half = cv2.flip(gray[height//2:, :], 0)
        if top_half.shape[0] != bottom_half.shape[0]:
            bottom_half = cv2.resize(bottom_half, (top_half.shape[1], top_half.shape[0]))
        v_diff = np.mean(np.abs(top_half.astype(float) - bottom_half.astype(float)))
        v_symmetry = max(0, 100 - v_diff) / 10

        return {
            "horizontal_score": round(h_symmetry, 1),
            "vertical_score": round(v_symmetry, 1),
            "overall_score": round((h_symmetry + v_symmetry) / 2, 1),
            "is_symmetrical": (h_symmetry + v_symmetry) / 2 > 6,
        }

    def analyze_composition(self, cv_image: np.ndarray) -> dict[str, Any]:
        """
        Analyze image composition using rule of thirds and center focus.
        """
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # Detect edges to find subject
        edges = cv2.Canny(gray, 50, 150)

        # Find center of mass of edges (approximate subject location)
        moments = cv2.moments(edges)
        if moments["m00"] > 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
        else:
            cx, cy = width // 2, height // 2

        # Calculate position relative to image
        rel_x = cx / width
        rel_y = cy / height

        # Check rule of thirds alignment
        thirds_x = [1/3, 2/3]
        thirds_y = [1/3, 2/3]

        x_alignment = min(abs(rel_x - t) for t in thirds_x + [0.5])
        y_alignment = min(abs(rel_y - t) for t in thirds_y + [0.5])

        # Centered score (good for product photos)
        center_distance = np.sqrt((rel_x - 0.5)**2 + (rel_y - 0.5)**2)
        center_score = max(0, 10 - center_distance * 20)

        # Rule of thirds score
        thirds_score = max(0, 10 - (x_alignment + y_alignment) * 20)

        # Product coverage estimation
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.size
        coverage = edge_pixels / total_pixels * 100 * 5  # Scale up

        return {
            "subject_center": {"x": round(rel_x, 2), "y": round(rel_y, 2)},
            "center_alignment_score": round(center_score, 1),
            "rule_of_thirds_score": round(thirds_score, 1),
            "estimated_product_coverage": round(min(coverage, 100), 1),
            "is_centered": center_distance < 0.15,
            "composition_score": round((center_score + thirds_score) / 2, 1),
        }

    def analyze_histogram(self, cv_image: np.ndarray) -> dict[str, Any]:
        """
        Analyze image histogram for exposure and dynamic range.
        """
        # Calculate histograms for each channel
        colors = ('b', 'g', 'r')
        histograms = {}

        for i, color in enumerate(colors):
            hist = cv2.calcHist([cv_image], [i], None, [256], [0, 256])
            histograms[color] = hist.flatten().tolist()

        # Gray histogram for overall analysis
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        gray_hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()

        # Analyze distribution
        total_pixels = np.sum(gray_hist)
        shadows = np.sum(gray_hist[:85]) / total_pixels * 100
        midtones = np.sum(gray_hist[85:170]) / total_pixels * 100
        highlights = np.sum(gray_hist[170:]) / total_pixels * 100

        # Check for clipping
        shadow_clipping = np.sum(gray_hist[:5]) / total_pixels * 100
        highlight_clipping = np.sum(gray_hist[250:]) / total_pixels * 100

        # Exposure assessment
        if shadow_clipping > 5:
            exposure = "underexposed"
            score = 5
        elif highlight_clipping > 5:
            exposure = "overexposed"
            score = 5
        elif 30 < midtones < 60:
            exposure = "balanced"
            score = 10
        else:
            exposure = "unbalanced"
            score = 7

        return {
            "shadows_percent": round(shadows, 1),
            "midtones_percent": round(midtones, 1),
            "highlights_percent": round(highlights, 1),
            "shadow_clipping_percent": round(shadow_clipping, 2),
            "highlight_clipping_percent": round(highlight_clipping, 2),
            "exposure_assessment": exposure,
            "exposure_score": score,
            "is_well_exposed": exposure == "balanced",
        }

    def get_quality_score(self, analysis: dict[str, Any]) -> dict[str, Any]:
        """
        Calculate overall quality score from all analyses.
        """
        scores = []
        weights = {
            "sharpness": 2.0,
            "brightness": 1.5,
            "contrast": 1.5,
            "blur_detection": 2.0,
            "noise_level": 1.5,
            "background_analysis": 1.0,
            "edge_density": 0.5,
            "composition": 1.0,
            "histogram_analysis": 1.0,
        }

        for key, weight in weights.items():
            if key in analysis and "score" in analysis[key]:
                scores.append((analysis[key]["score"], weight))

        if not scores:
            return {"overall_score": 5, "weighted_score": 5}

        weighted_sum = sum(score * weight for score, weight in scores)
        total_weight = sum(weight for _, weight in scores)
        weighted_score = weighted_sum / total_weight

        return {
            "overall_score": round(weighted_score, 1),
            "max_score": 10,
            "component_count": len(scores),
            "quality_tier": (
                "excellent" if weighted_score >= 8.5 else
                "good" if weighted_score >= 7 else
                "acceptable" if weighted_score >= 5 else
                "needs_improvement"
            ),
        }


# Singleton instance
_cv_analyzer: CVAnalyzer | None = None


def get_cv_analyzer() -> CVAnalyzer:
    """Get the CV analyzer singleton."""
    global _cv_analyzer
    if _cv_analyzer is None:
        _cv_analyzer = CVAnalyzer()
    return _cv_analyzer
