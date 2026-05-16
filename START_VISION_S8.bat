@echo off
:: Vision_S8 Easy Launcher for Windows
:: Double-click this file to start the image analyzer!

echo.
echo  ========================================
echo   Vision_S8 - Product Image Analyzer
echo   For E-commerce Sellers
echo  ========================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed!
    echo.
    echo Please download Python from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [OK] Python found!
echo.

:: Check if we're in the right directory
if not exist "pyproject.toml" (
    echo [ERROR] Please run this from the Vision_S8 folder!
    pause
    exit /b 1
)

:: Install dependencies if needed
if not exist ".installed" (
    echo [*] First time setup - installing dependencies...
    echo     This may take a few minutes...
    echo.
    pip install -e ".[ui]" --quiet
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
    echo. > .installed
    echo [OK] Dependencies installed!
    echo.
)

:: Start the web UI
echo [*] Starting Vision_S8...
echo.
echo     Your browser will open automatically.
echo     If not, go to: http://127.0.0.1:7860
echo.
echo     Press Ctrl+C to stop.
echo.

python -m vision_s8.web_ui

pause
