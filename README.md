# 📸 Vision_S8

**Product Image Analyzer for E-commerce Sellers**

Instantly analyze your product photos and get actionable feedback to increase sales! Works with Amazon, Etsy, Shopify, eBay, and more.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Easy to Use](https://img.shields.io/badge/Easy-One%20Click%20Start-green.svg)](#-quick-start-for-sellers)
[![No Coding](https://img.shields.io/badge/No%20Coding-Required-orange.svg)](#-quick-start-for-sellers)

---

## 🎯 What This Does For You

| Before Vision_S8 | After Vision_S8 |
|------------------|-----------------|
| "Is my photo good enough?" | **Know your exact score: 7.2/10** |
| "Why aren't my listings selling?" | **"Your image is too dark - here's how to fix it"** |
| "Does this meet Amazon requirements?" | **✅ Compliant or ❌ Fix these issues** |
| "Which photo should I use?" | **A/B test prediction shows Photo B wins** |

---

## 🚀 Quick Start (For Sellers)

### Step 1: Install Python
Download from [python.org](https://www.python.org/downloads/) (check "Add to PATH")

### Step 2: Double-Click to Start
- **Windows:** Double-click `START_VISION_S8.bat`
- **Mac/Linux:** Run `./start_vision_s8.sh`

### Step 3: Upload & Analyze!
Your browser opens automatically. Just drag your product photo and click "Analyze"!

![Demo Screenshot](docs/demo.png)

---

## 💡 What You Get

### Instant Quality Score
```
🌟 Overall Score: 8.2/10 (GOOD)
Your image is ready to sell!
```

### Platform Compliance Check
```
🎯 Amazon Compliance: ✅ PASS
- Size: 2000x2000 (✅ meets 1000x1000 minimum)
- Background: White (✅ required)
```

### Actionable Improvements
```
🛠️ How to Improve:
📸 Increase Sharpness: Use a tripod for sharper images
💡 Add more lighting: Your image is slightly underexposed
```

### Dominant Colors Analysis
```
🎨 Dominant Colors:
- #FFFFFF (62%) - White background ✓
- #8B4513 (18%) - Product brown
- #D2691E (12%) - Product highlight
```

---

## ✨ All Features

| Feature | Description | Requires AI? |
|---------|-------------|--------------|
| 📸 **Quick Analysis** | Score your image in seconds | ❌ No |
| 🎯 **Platform Check** | Amazon, Etsy, Shopify, eBay compliance | ❌ No |
| 🔬 **Technical Details** | Sharpness, brightness, contrast, background | ❌ No |
| 🎨 **Color Analysis** | Dominant colors, saturation, temperature | ❌ No |
| 🛠️ **Fix Suggestions** | Actionable improvement tips | ❌ No |
| 🤖 **AI Deep Analysis** | Professional assessment & competitor comparison | ✅ Yes |
| ✨ **Auto Enhancement** | One-click background removal & fixes | ✅ Yes |
| 📊 **A/B Testing** | Predict which photo will perform better | ✅ Yes |
| 📦 **Batch Processing** | Analyze entire catalog at once | ✅ Yes |

**Most features work WITHOUT an API key!** AI features need a [free Gemini API key](https://aistudio.google.com/apikey).

---

## 🔬 The Technology (For Curious Sellers)

Vision_S8 uses **real computer vision** - the same technology used by professional photo editing software:

| What We Check | How We Check It |
|---------------|-----------------|
| **Sharpness** | Laplacian edge detection algorithm |
| **Brightness** | LAB color space perceptual analysis |
| **Contrast** | Michelson contrast formula |
| **Background** | Corner sampling + uniformity check |
| **Blur** | FFT frequency domain analysis |
| **Colors** | K-means clustering extraction |

This means **accurate, consistent results** - not just "AI guessing."

---

## 🎯 Supported Platforms

| Platform | Min Size | Background | Our Check |
|----------|----------|------------|-----------|
| **Amazon** | 1000x1000 | Pure white | ✅ Full |
| **Etsy** | 2000x2000 | Any | ✅ Full |
| **Shopify** | 800x800 | Any | ✅ Full |
| **eBay** | 500x500 | Light | ✅ Full |
| **Instagram** | 1080x1080 | Any | ✅ Full |
| **Walmart** | 1000x1000 | Pure white | ✅ Full |

---
| 📊 **A/B Test Predictor** | Predict which image variant will perform best before running actual tests |
| 🎯 **Platform Optimization** | Optimize for Amazon, Shopify, Instagram, eBay, Etsy, Walmart |
| 🔎 **SEO Analyzer** | Generate alt text, filenames, keywords, and meta descriptions |
| ✅ **Compliance Checker** | Validate against marketplace requirements (dimensions, background, watermarks) |
| 📦 **Batch Processing** | Process entire catalogs with progress tracking and Excel/CSV/HTML reports |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Google Gemini API Key ([Get one here](https://aistudio.google.com/apikey)) - *Optional for CV-only analysis*

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vision_s8.git
cd vision_s8

# Install dependencies using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

### Configuration

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Edit `.env` and add your Gemini API key (optional for CV-only mode):
```env
GEMINI_API_KEY=your_api_key_here
```

### Run the API

```bash
# Using the CLI command
vision-s8

# Or directly with uvicorn
uvicorn vision_s8.main:app --reload

# Or with Python
python -m vision_s8.main
```

The API will be available at `http://127.0.0.1:8000`

- **API Docs**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

---

## 📚 API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/audit` | HYBRID audit (CV + AI) with `use_ai` parameter |
| `POST` | `/api/v1/audit/cv` | **CV-only** analysis (no API key required!) |
| `POST` | `/api/v1/enhance` | Get enhancement recommendations and apply fixes |
| `POST` | `/api/v1/compare` | Compare against competitor images |
| `POST` | `/api/v1/ab-test` | Predict A/B test performance |
| `POST` | `/api/v1/seo` | Generate SEO content for image |
| `POST` | `/api/v1/compliance/{platform}` | Check marketplace compliance |
| `POST` | `/api/v1/batch/upload` | Create batch processing job |
| `GET` | `/api/v1/batch/{id}` | Get batch job status |
| `GET` | `/api/v1/health` | Health check |

### Example: CV-Only Audit (No API Key Required)

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/audit/cv" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@product_image.jpg"
```

Response:
```json
{
  "success": true,
  "data": {
    "audit_id": "abc123...",
    "quality_score": {
      "overall_score": 7.8,
      "quality_tier": "good"
    },
    "analysis": {
      "sharpness": {
        "laplacian_variance": 245.67,
        "quality": "good",
        "score": 8
      },
      "brightness": {
        "mean": 156.32,
        "level": "optimal",
        "score": 10
      },
      "contrast": {
        "std_contrast": 52.4,
        "michelson_contrast": 0.85,
        "quality": "good"
      },
      "color_analysis": {
        "mean_saturation": 78.5,
        "temperature": "neutral",
        "dominant_colors": [...]
      },
      "background_analysis": {
        "is_uniform": true,
        "estimated_color": "#ffffff"
      }
    }
  },
  "meta": {
    "analysis_type": "computer_vision_only",
    "algorithms_used": [
      "laplacian_sharpness",
      "lab_brightness",
      "canny_edge_detection",
      "kmeans_color_clustering",
      "fft_blur_detection",
      "histogram_analysis"
    ]
  }
}
```

### Example: HYBRID Audit (CV + AI)

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/audit" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@product_image.jpg"
```

---

## 🎯 Supported Platforms

| Platform | Requirements |
|----------|--------------|
| **Amazon** | White background, 1000x1000px min, no watermarks |
| **Shopify** | Flexible, 800x800px min recommended |
| **Instagram** | Square (1:1) or vertical (4:5), 1080px |
| **eBay** | White/light background, 500x500px min |
| **Etsy** | Lifestyle welcome, 2000x2000px min |
| **Walmart** | White background, 1000x1000px min |

---

## 📁 Project Structure

```
vision_s8/
├── src/vision_s8/
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── api/                 # API routes
│   │   └── routes/
│   │       ├── audit.py     # HYBRID + CV-only endpoints
│   │       ├── enhance.py
│   │       ├── compare.py
│   │       ├── ab_test.py
│   │       ├── seo.py
│   │       ├── compliance.py
│   │       └── batch.py
│   ├── core/                # Business logic
│   │   ├── cv_analyzer.py   # 🔬 Computer Vision algorithms
│   │   ├── auditor.py       # HYBRID auditor (CV + AI)
│   │   ├── enhancer.py
│   │   ├── comparator.py
│   │   └── ...
│   ├── services/            # External services
│   │   ├── gemini_service.py
│   │   ├── image_service.py
│   │   └── scraper_service.py
│   ├── models/              # Data models
│   │   ├── schemas.py
│   │   └── database.py
│   └── utils/               # Utilities
│       ├── logger.py
│       ├── file_handler.py
│       └── report_generator.py
├── tests/
│   ├── test_cv_analyzer.py  # 35+ CV algorithm tests
│   ├── test_auditor_hybrid.py
│   ├── test_image_service.py
│   ├── test_api_endpoints.py
│   └── ...
├── .env.example
├── pyproject.toml
└── README.md
```

---

## 🛠️ Development

```bash
# Install dev dependencies
uv sync --all-extras

# Run all tests
pytest tests/ -v

# Run CV-specific tests
pytest tests/test_cv_analyzer.py -v

# Run with coverage
pytest tests/ --cov=vision_s8 --cov-report=html

# Run linter
ruff check src/

# Format code
ruff format src/
```

### Test Coverage

| Module | Tests | Status |
|--------|-------|--------|
| CV Analyzer | 35+ tests | ✅ |
| Hybrid Auditor | 20+ tests | ✅ |
| Image Service | 25+ tests | ✅ |
| API Endpoints | 30+ tests | ✅ |

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- Powered by [Google Gemini AI](https://ai.google.dev/)
- Computer Vision by [OpenCV](https://opencv.org/)
- Built with [FastAPI](https://fastapi.tiangolo.com/)
- Background removal by [rembg](https://github.com/danielgatis/rembg)
