#!/bin/bash

# Bitcoin Address Generator Launcher Script

echo "=========================================="
echo "  Bitcoin Address Generator"
echo "=========================================="
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed!"
    echo "Please install Python 3.7 or higher"
    exit 1
fi

# Check if dependencies are installed
if ! python3 -c "import mnemonic" 2>/dev/null; then
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
fi

# Check for CUDA/GPU support
echo "Checking for GPU/CUDA support..."
if python3 -c "import pycuda.driver; print('CUDA available')" 2>/dev/null; then
    echo "✓ CUDA detected - Using GPU accelerated version"
    python3 btc_gpu_generator.py
else
    echo "ℹ CUDA not available - Using CPU-only version"
    echo ""
    read -p "Install GPU support? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing CUDA dependencies..."
        pip3 install pycuda
        echo "Please restart the script to use GPU acceleration"
        exit 0
    else
        python3 btc_generator.py
    fi
fi
