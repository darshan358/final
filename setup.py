#!/usr/bin/env python3
"""
Setup script for Bitcoin Address Generator
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a Python package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to install {package}")
        return False

def check_cuda():
    """Check if CUDA is available"""
    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.DEVNULL)
        print("✓ NVIDIA GPU detected")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ No NVIDIA GPU detected or nvidia-smi not found")
        return False

def main():
    print("Bitcoin Address Generator Setup")
    print("=" * 40)
    
    # Required packages
    required_packages = [
        "mnemonic==0.21",
        "ecdsa==0.19.1",
        "base58==2.1.1",
        "numpy>=1.20.0"
    ]
    
    # Install required packages
    print("\nInstalling required packages...")
    all_success = True
    for package in required_packages:
        if not install_package(package):
            all_success = False
    
    # Check for CUDA and install PyCUDA if available
    print("\nChecking for CUDA support...")
    if check_cuda():
        print("Attempting to install PyCUDA...")
        if install_package("pycuda"):
            print("✓ GPU acceleration will be available")
        else:
            print("✗ PyCUDA installation failed, will use CPU-only mode")
    else:
        print("Will use CPU-only mode")
    
    # Create addresses.txt if it doesn't exist
    if not os.path.exists("addresses.txt"):
        with open("addresses.txt", "w") as f:
            f.write("# Bitcoin addresses to check against (one per line)\n")
            f.write("# Example addresses:\n")
            f.write("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa\n")
            f.write("12c6DSiU4Rq3P4ZxziKxzrL5LmMBrzjrJX\n")
        print("✓ Created addresses.txt file")
    
    print("\n" + "=" * 40)
    if all_success:
        print("✓ Setup completed successfully!")
        print("\nUsage:")
        print("  python3 main.py --help          # Show help")
        print("  python3 main.py --sample 5      # Generate 5 sample addresses")
        print("  python3 main.py --benchmark 30  # Run 30-second benchmark")
        print("  python3 main.py                 # Start mining")
    else:
        print("✗ Setup completed with errors")
        print("Some packages failed to install. The program may not work correctly.")

if __name__ == "__main__":
    main()