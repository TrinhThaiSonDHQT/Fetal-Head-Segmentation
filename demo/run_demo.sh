#!/bin/bash
# Linux/Mac shell script to launch demo

echo "================================================"
echo " Fetal Head Segmentation Demo Platform"
echo "================================================"
echo ""

echo "[1/3] Checking Python installation..."
python3 --version
if [ $? -ne 0 ]; then
    echo "ERROR: Python not found. Please install Python 3.7+"
    exit 1
fi

echo ""
echo "[2/3] Installing dependencies..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo ""
echo "[3/3] Launching demo application..."
echo ""
echo "================================================"
echo " Demo will open in your browser automatically"
echo " Press Ctrl+C to stop the server"
echo "================================================"
echo ""

python3 app.py
