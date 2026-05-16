"""
Vision_S8 Web UI - Simple Interface for E-commerce Sellers

A user-friendly Gradio interface that non-technical users can actually use.
No coding required - just drag, drop, and get results!
"""

import io
import json
from pathlib import Path
import sys

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("Gradio not installed. Run: pip install gradio")

from PIL import Image

if __package__ in (None, ""):
    # Support direct script execution: python src/vision_s8/web_ui.py
    src_root = Path(__file__).resolve().parents[1]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from vision_s8.core.cv_analyzer import get_cv_analyzer
    from vision_s8.core.auditor import ProductAuditor
    from vision_s8.services.image_service import get_image_service
    from vision_s8.config import settings
else:
    from .core.cv_analyzer import get_cv_analyzer
    from .core.auditor import ProductAuditor
    from .services.image_service import get_image_service
    from .config import settings


def analyze_image(image, platform: str, use_ai: bool):
    """
    Analyze a product image and return user-friendly results.
    
    This is what e-commerce sellers actually need - simple, actionable feedback.
    """
    if image is None:
        return None, "❌ Please upload an image first!", "", ""
    
    # Convert to PIL if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Get CV analyzer
    cv_analyzer = get_cv_analyzer()
    image_service = get_image_service()
    
    # Perform analysis
    cv_results = cv_analyzer.analyze(image)
    quality_score = cv_analyzer.get_quality_score(cv_results)
    image_info = image_service.get_image_info(image)
    
    # Calculate overall score
    score = quality_score["overall_score"]
    tier = quality_score["quality_tier"]
    
    # Create user-friendly summary
    if score >= 8:
        emoji = "🌟"
        verdict = "Excellent! Your image is ready to sell!"
    elif score >= 6:
        emoji = "✅"
        verdict = "Good image, but could be improved."
    elif score >= 4:
        emoji = "⚠️"
        verdict = "Needs work before listing."
    else:
        emoji = "❌"
        verdict = "Poor quality - significant improvements needed."
    
    # Build summary
    summary = f"""
# {emoji} Overall Score: {score}/10 ({tier.upper()})

**{verdict}**

---

## 📐 Image Specs
- **Size:** {image_info['width']} x {image_info['height']} pixels
- **Megapixels:** {image_info['megapixels']} MP
- **Aspect Ratio:** {image_info['aspect_ratio']}

"""
    
    # Platform-specific check
    platform_requirements = {
        "amazon": {"min_size": 1000, "bg": "white", "name": "Amazon"},
        "etsy": {"min_size": 2000, "bg": "any", "name": "Etsy"},
        "shopify": {"min_size": 800, "bg": "any", "name": "Shopify"},
        "ebay": {"min_size": 500, "bg": "light", "name": "eBay"},
        "instagram": {"min_size": 1080, "bg": "any", "name": "Instagram"},
    }
    
    req = platform_requirements.get(platform.lower(), platform_requirements["amazon"])
    
    platform_pass = "✅" if min(image_info['width'], image_info['height']) >= req["min_size"] else "❌"
    summary += f"""
## 🎯 {req['name']} Compliance: {platform_pass}
- **Minimum Size Required:** {req['min_size']}x{req['min_size']} px
- **Your Image:** {image_info['width']}x{image_info['height']} px
- **Background:** {req['bg'].title()} recommended

"""
    
    # Technical details (simplified)
    sharpness = cv_results.get("sharpness", {})
    brightness = cv_results.get("brightness", {})
    contrast = cv_results.get("contrast", {})
    bg = cv_results.get("background_analysis", {})
    
    details = f"""
## 🔬 Technical Analysis

| Metric | Status | Details |
|--------|--------|---------|
| **Sharpness** | {"✅" if sharpness.get("score", 0) >= 6 else "⚠️"} | {sharpness.get("quality", "Unknown")} |
| **Brightness** | {"✅" if brightness.get("score", 0) >= 6 else "⚠️"} | {brightness.get("level", "Unknown")} |
| **Contrast** | {"✅" if contrast.get("score", 0) >= 6 else "⚠️"} | {contrast.get("quality", "Unknown")} |
| **Background** | {"✅" if bg.get("is_uniform", False) else "⚠️"} | {"Uniform" if bg.get("is_uniform", False) else "Mixed"} |

"""
    
    # Actionable improvements
    improvements = []
    
    if sharpness.get("score", 10) < 6:
        improvements.append("📸 **Improve Sharpness:** Use a tripod, ensure proper focus, or try sharpening in photo editing software.")
    
    if brightness.get("score", 10) < 6:
        level = brightness.get("level", "")
        if "dark" in level.lower():
            improvements.append("💡 **Increase Brightness:** Add more lighting or increase exposure in editing.")
        else:
            improvements.append("💡 **Reduce Brightness:** Your image may be overexposed. Reduce lighting or exposure.")
    
    if contrast.get("score", 10) < 6:
        improvements.append("🎨 **Improve Contrast:** Adjust contrast in photo editing to make the product pop.")
    
    if not bg.get("is_uniform", True):
        improvements.append("🖼️ **Clean Background:** Use a solid white/neutral background for professional look.")
    
    if min(image_info['width'], image_info['height']) < req["min_size"]:
        improvements.append(f"📐 **Increase Resolution:** Your image is too small for {req['name']}. Minimum {req['min_size']}x{req['min_size']} required.")
    
    if improvements:
        improvement_text = "## 🛠️ How to Improve\n\n" + "\n\n".join(improvements)
    else:
        improvement_text = "## ✨ No Major Issues Found!\n\nYour image looks great. Consider A/B testing different angles."
    
    # Dominant colors
    colors = cv_results.get("color_analysis", {}).get("dominant_colors", [])[:5]
    color_text = "## 🎨 Dominant Colors\n\n"
    for color in colors:
        hex_color = color.get("hex", "#000000")
        pct = color.get("percentage", 0)
        color_text += f"- `{hex_color}` ({pct:.1f}%)\n"
    
    return image, summary, details + "\n" + improvement_text, color_text


def compare_images(image_a, image_b, platform: str):
    """
    Compare two product images and determine which one is better.
    
    Uses Computer Vision analysis - NO AI required!
    """
    if image_a is None or image_b is None:
        return "❌ Please upload both images to compare!", "", ""
    
    # Convert to PIL if needed
    if not isinstance(image_a, Image.Image):
        image_a = Image.fromarray(image_a)
    if not isinstance(image_b, Image.Image):
        image_b = Image.fromarray(image_b)
    
    # Get analyzers
    cv_analyzer = get_cv_analyzer()
    image_service = get_image_service()
    
    # Analyze both images
    cv_results_a = cv_analyzer.analyze(image_a)
    cv_results_b = cv_analyzer.analyze(image_b)
    
    quality_a = cv_analyzer.get_quality_score(cv_results_a)
    quality_b = cv_analyzer.get_quality_score(cv_results_b)
    
    info_a = image_service.get_image_info(image_a)
    info_b = image_service.get_image_info(image_b)
    
    score_a = quality_a["overall_score"]
    score_b = quality_b["overall_score"]
    
    # Determine winner
    diff = abs(score_a - score_b)
    
    if diff < 0.5:
        winner_text = "🤝 **TIE! Both images are similar quality.**"
        recommendation = "Either image would work. Choose based on which shows your product better."
    elif score_a > score_b:
        winner_text = f"🏆 **PHOTO A WINS!** ({score_a:.1f} vs {score_b:.1f})"
        recommendation = "Use Photo A for your listing - it has better technical quality."
    else:
        winner_text = f"🏆 **PHOTO B WINS!** ({score_b:.1f} vs {score_a:.1f})"
        recommendation = "Use Photo B for your listing - it has better technical quality."
    
    # Build comparison summary
    summary = f"""
# 📊 A/B Comparison Results

{winner_text}

{recommendation}

---

## Score Comparison

| Metric | Photo A | Photo B | Winner |
|--------|---------|---------|--------|
| **Overall Score** | {score_a:.1f}/10 | {score_b:.1f}/10 | {"A" if score_a > score_b else "B" if score_b > score_a else "Tie"} |
| **Quality Tier** | {quality_a['quality_tier']} | {quality_b['quality_tier']} | - |

"""
    
    # Detailed metric comparison
    metrics = [
        ("Sharpness", "sharpness", "score"),
        ("Brightness", "brightness", "score"),
        ("Contrast", "contrast", "score"),
        ("Background", "background_analysis", "uniformity_score"),
        ("Noise Level", "noise_level", "score"),
        ("Blur Detection", "blur_detection", "score"),
    ]
    
    details = """
## 📋 Detailed Comparison

| Metric | Photo A | Photo B | Better |
|--------|---------|---------|--------|
"""
    
    for name, key, score_key in metrics:
        val_a = cv_results_a.get(key, {}).get(score_key, 5)
        val_b = cv_results_b.get(key, {}).get(score_key, 5)
        
        if val_a is None:
            val_a = 5
        if val_b is None:
            val_b = 5
            
        winner = "A ✓" if val_a > val_b else ("B ✓" if val_b > val_a else "Tie")
        details += f"| {name} | {val_a:.1f} | {val_b:.1f} | {winner} |\n"
    
    # Size comparison
    size_details = f"""

## 📐 Size Comparison

| Spec | Photo A | Photo B |
|------|---------|---------|
| **Dimensions** | {info_a['width']}x{info_a['height']} | {info_b['width']}x{info_b['height']} |
| **Megapixels** | {info_a['megapixels']} MP | {info_b['megapixels']} MP |
| **Aspect Ratio** | {info_a['aspect_ratio']} | {info_b['aspect_ratio']} |

"""
    
    # Platform compliance
    platform_requirements = {
        "amazon": {"min_size": 1000, "name": "Amazon"},
        "etsy": {"min_size": 2000, "name": "Etsy"},
        "shopify": {"min_size": 800, "name": "Shopify"},
        "ebay": {"min_size": 500, "name": "eBay"},
        "instagram": {"min_size": 1080, "name": "Instagram"},
    }
    
    req = platform_requirements.get(platform.lower(), platform_requirements["amazon"])
    
    compliant_a = min(info_a['width'], info_a['height']) >= req["min_size"]
    compliant_b = min(info_b['width'], info_b['height']) >= req["min_size"]
    
    compliance = f"""
## 🎯 {req['name']} Compliance

| Photo | Meets {req['min_size']}px Minimum? |
|-------|------------------------------|
| **Photo A** | {"✅ Yes" if compliant_a else "❌ No - too small!"} |
| **Photo B** | {"✅ Yes" if compliant_b else "❌ No - too small!"} |

"""
    
    # Specific recommendations
    recommendations = "\n## 🛠️ Recommendations\n\n"
    
    if score_a > score_b and diff >= 0.5:
        # Why A is better
        if cv_results_a.get("sharpness", {}).get("score", 0) > cv_results_b.get("sharpness", {}).get("score", 0):
            recommendations += "- **Photo A is sharper** - clearer product details\n"
        if cv_results_a.get("brightness", {}).get("score", 0) > cv_results_b.get("brightness", {}).get("score", 0):
            recommendations += "- **Photo A has better lighting** - more appealing exposure\n"
        if cv_results_a.get("background_analysis", {}).get("is_uniform", False) and not cv_results_b.get("background_analysis", {}).get("is_uniform", False):
            recommendations += "- **Photo A has cleaner background** - more professional\n"
    elif score_b > score_a and diff >= 0.5:
        # Why B is better
        if cv_results_b.get("sharpness", {}).get("score", 0) > cv_results_a.get("sharpness", {}).get("score", 0):
            recommendations += "- **Photo B is sharper** - clearer product details\n"
        if cv_results_b.get("brightness", {}).get("score", 0) > cv_results_a.get("brightness", {}).get("score", 0):
            recommendations += "- **Photo B has better lighting** - more appealing exposure\n"
        if cv_results_b.get("background_analysis", {}).get("is_uniform", False) and not cv_results_a.get("background_analysis", {}).get("is_uniform", False):
            recommendations += "- **Photo B has cleaner background** - more professional\n"
    else:
        recommendations += "Both images are similar in quality. Consider:\n"
        recommendations += "- Which angle shows your product better?\n"
        recommendations += "- Which lighting makes colors more accurate?\n"
        recommendations += "- Test both on your listing and track conversions!\n"
    
    return summary, details + size_details + compliance, recommendations


def create_demo_ui():
    """Create the Gradio interface."""
    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio is required for the web UI. Install with: pip install gradio")
    
    with gr.Blocks(
        title="Vision_S8 - Product Image Analyzer",
        theme=gr.themes.Soft(),
        css="""
        .main-title { text-align: center; margin-bottom: 20px; }
        .score-box { font-size: 24px; font-weight: bold; }
        .winner-box { background: #d4edda; padding: 10px; border-radius: 8px; }
        """
    ) as demo:
        gr.Markdown(
            """
            # 📸 Vision_S8 - Product Image Analyzer
            
            **Optimize your e-commerce product images for maximum sales!**
            
            ---
            """,
            elem_classes=["main-title"]
        )
        
        with gr.Tabs():
            # ============================================================
            # TAB 1: Single Image Analysis
            # ============================================================
            with gr.TabItem("📸 Analyze Single Image"):
                gr.Markdown("Upload a product photo to get instant quality feedback and improvement suggestions.")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(
                            label="📤 Upload Product Image",
                            type="pil",
                            height=400,
                        )
                        
                        platform = gr.Dropdown(
                            choices=["Amazon", "Etsy", "Shopify", "eBay", "Instagram"],
                            value="Amazon",
                            label="🎯 Target Platform",
                        )
                        
                        use_ai = gr.Checkbox(
                            label="🤖 Use AI Analysis (requires API key)",
                            value=False,
                            info="Enable for more detailed suggestions"
                        )
                        
                        analyze_btn = gr.Button(
                            "🔍 Analyze Image",
                            variant="primary",
                            size="lg",
                        )
                    
                    with gr.Column(scale=1):
                        output_image = gr.Image(
                            label="📷 Your Image",
                            height=300,
                        )
                        
                        summary_output = gr.Markdown(
                            label="📊 Summary",
                        )
                
                with gr.Row():
                    with gr.Column():
                        details_output = gr.Markdown(
                            label="📋 Details & Improvements",
                        )
                    
                    with gr.Column():
                        colors_output = gr.Markdown(
                            label="🎨 Color Analysis",
                        )
                
                # Connect the analyze button
                analyze_btn.click(
                    fn=analyze_image,
                    inputs=[input_image, platform, use_ai],
                    outputs=[output_image, summary_output, details_output, colors_output],
                )
            
            # ============================================================
            # TAB 2: A/B Comparison (NO AI REQUIRED!)
            # ============================================================
            with gr.TabItem("🆚 Compare Two Photos (A/B Test)"):
                gr.Markdown("""
                **Which photo should you use?** Upload two versions and find out instantly!
                
                ✅ **No AI required** - Uses computer vision to compare technical quality.
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        image_a = gr.Image(
                            label="📷 Photo A",
                            type="pil",
                            height=300,
                        )
                    
                    with gr.Column(scale=1):
                        image_b = gr.Image(
                            label="📷 Photo B",
                            type="pil",
                            height=300,
                        )
                
                with gr.Row():
                    compare_platform = gr.Dropdown(
                        choices=["Amazon", "Etsy", "Shopify", "eBay", "Instagram"],
                        value="Amazon",
                        label="🎯 Target Platform",
                    )
                    
                    compare_btn = gr.Button(
                        "🆚 Compare Photos",
                        variant="primary",
                        size="lg",
                    )
                
                with gr.Row():
                    comparison_summary = gr.Markdown(
                        label="🏆 Winner",
                    )
                
                with gr.Row():
                    with gr.Column():
                        comparison_details = gr.Markdown(
                            label="📊 Detailed Comparison",
                        )
                    
                    with gr.Column():
                        comparison_recommendations = gr.Markdown(
                            label="🛠️ Recommendations",
                        )
                
                # Connect the compare button
                compare_btn.click(
                    fn=compare_images,
                    inputs=[image_a, image_b, compare_platform],
                    outputs=[comparison_summary, comparison_details, comparison_recommendations],
                )
        
        # Tips section (outside tabs)
        gr.Markdown("---\n### 💡 Tips for Better Product Photos")
        gr.Markdown("""
        1. **Use Natural Light** - Soft, diffused daylight gives the best results
        2. **White Background** - Required for Amazon, recommended for most platforms
        3. **High Resolution** - At least 2000x2000 pixels for Etsy, 1000x1000 for Amazon
        4. **Sharp Focus** - Use a tripod and proper focus on the product
        5. **Multiple Angles** - Show the product from different perspectives
        """)
    
    return demo


def run_web_ui(share: bool = False):
    """Launch the web UI."""
    demo = create_demo_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=share,
        inbrowser=True,
    )


if __name__ == "__main__":
    run_web_ui()
