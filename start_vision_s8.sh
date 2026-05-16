#!/bin/bash
# Vision_S8 Easy Launcher for Mac/Linux
# Run with: ./start_vision_s8.sh

echo ""
echo "========================================"
echo "  Vision_S8 - Product Image Analyzer"
echo "  For E-commerce Sellers"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python is not installed!"
    echo ""
    echo "Please install Python:"
    echo "  Mac: brew install python3"
    echo "  Ubuntu: sudo apt install python3"
    echo ""
    exit 1
fi

echo "[OK] Python found!"
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "[ERROR] Please run this from the Vision_S8 folder!"
    exit 1
fi

# Install dependencies if needed
if [ ! -f ".installed" ]; then
    echo "[*] First time setup - installing dependencies..."
    echo "    This may take a few minutes..."
    echo ""
    pip3 install -e ".[ui]" --quiet
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install dependencies."
        exit 1
    fi
    touch .installed
    echo "[OK] Dependencies installed!"
    echo ""
fi

# Start the web UI
echo "[*] Starting Vision_S8..."
echo ""
echo "    Your browser will open automatically."
echo "    If not, go to: http://127.0.0.1:7860"
echo ""
echo "    Press Ctrl+C to stop."
echo ""

python3 -m vision_s8.web_ui
