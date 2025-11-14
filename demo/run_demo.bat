@echo off
REM Windows batch script to launch demo
echo ================================================
echo  Fetal Head Segmentation Demo Platform
echo ================================================
echo.

echo [1/3] Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.7+
    pause
    exit /b 1
)

echo.
echo [2/3] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [3/3] Launching demo application...
echo.
echo ================================================
echo  Demo will open in your browser automatically
echo  Press Ctrl+C to stop the server
echo ================================================
echo.

python app.py

pause
