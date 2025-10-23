# Installation & Setup Guide

## 📦 Complete Installation Instructions

### Prerequisites

#### System Requirements
- **Operating System**: Linux, macOS, or Windows (WSL recommended for Windows)
- **Python**: 3.7 or higher
- **RAM**: 4GB minimum, 8GB+ recommended
- **CPU**: Multi-core processor recommended
- **GPU**: NVIDIA GPU with CUDA support (optional, for GPU acceleration)

### Step-by-Step Installation

#### 1. Verify Python Installation

```bash
python3 --version
```

Should show Python 3.7 or higher. If not installed:

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3 python3-pip
```

**macOS:**
```bash
brew install python3
```

**Windows (WSL):**
```bash
sudo apt update && sudo apt install python3 python3-pip
```

#### 2. Install Python Dependencies

**Option A: Full Installation (with GPU support)**
```bash
cd /workspace
pip3 install -r requirements.txt
```

**Option B: CPU-Only (faster, no GPU)**
```bash
pip3 install mnemonic ecdsa base58 numpy coincurve tqdm colorama
```

**Option C: Virtual Environment (recommended)**
```bash
# Create virtual environment
python3 -m venv btc_env

# Activate it
source btc_env/bin/activate  # Linux/Mac
# or
btc_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

#### 3. Install CUDA (Optional, for GPU acceleration)

**Only needed if you want GPU acceleration!**

**Ubuntu/Debian:**
```bash
# Check if NVIDIA GPU exists
lspci | grep -i nvidia

# Install NVIDIA drivers
sudo apt update
sudo apt install nvidia-driver-530

# Reboot
sudo reboot

# Verify driver installation
nvidia-smi

# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda

# Add to PATH (add to ~/.bashrc for persistence)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install PyCUDA
pip3 install pycuda

# Test CUDA
python3 -c "import pycuda.driver; print('CUDA OK')"
```

**macOS:**
CUDA is not supported on macOS (Apple Silicon or Intel). Use CPU-only mode.

**Windows:**
1. Download CUDA Toolkit from NVIDIA website
2. Install NVIDIA drivers
3. Install CUDA Toolkit
4. Install PyCUDA: `pip install pycuda`

### 4. Verify Installation

Run the test suite:

```bash
cd /workspace
python3 test_generator.py
```

You should see:
- ✅ 3 test addresses generated
- ✅ Address checking test passed
- ✅ Performance metrics shown
- ✅ "All tests passed!"

## 🎯 Configuration

### Edit Target Addresses

```bash
nano addresses.txt
# or
vim addresses.txt
# or use any text editor
```

Add Bitcoin addresses (one per line):
```
1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
1FeexV6bAHb8ybZjqQMjJrcCrHGW9sb6uF
bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq
```

### Customize Settings

Edit `config.py` to change:
```python
DEFAULT_ITERATIONS = 10000      # Number of addresses to generate
BATCH_SIZE_CPU = 100            # CPU batch size
BATCH_SIZE_GPU = 256            # GPU batch size
MNEMONIC_STRENGTH = 256         # 256=24 words, 128=12 words
USE_GPU = True                  # Enable/disable GPU
```

## 🚀 Running the Application

### Method 1: Auto-Detection Script (Easiest)

```bash
./run.sh
```

This will:
- Check for dependencies
- Auto-detect GPU/CUDA
- Run the appropriate version
- Offer to install missing components

### Method 2: CPU Version (No GPU needed)

```bash
python3 btc_generator.py
```

When prompted:
```
Enter number of addresses to generate (default 10000): 50000
```

### Method 3: GPU Version (Requires CUDA)

```bash
python3 btc_gpu_generator.py
```

### Method 4: With Custom Config

```bash
# Edit config first
nano config.py

# Then run
python3 btc_gpu_generator.py
```

## 📊 Usage Examples

### Example 1: Quick Test (1,000 addresses)
```bash
python3 btc_generator.py
# Enter: 1000
# Takes ~1 second on 8-core CPU
```

### Example 2: Medium Run (100,000 addresses)
```bash
python3 btc_generator.py
# Enter: 100000
# Takes ~90 seconds on 8-core CPU
```

### Example 3: Large Run with GPU (1,000,000 addresses)
```bash
python3 btc_gpu_generator.py
# Enter: 1000000
# Takes ~250 seconds with RTX 3060
```

### Example 4: Background Run
```bash
# Run in background
nohup python3 btc_gpu_generator.py > output.log 2>&1 &

# Check progress
tail -f output.log

# Check if running
ps aux | grep btc_gpu
```

## 🔧 Troubleshooting

### Issue: "No module named 'mnemonic'"

**Solution:**
```bash
pip3 install -r requirements.txt
```

### Issue: "CUDA not available"

**Solution:**
```bash
# Check if GPU is detected
nvidia-smi

# If error, install NVIDIA drivers
sudo apt install nvidia-driver-530

# Reboot and try again
sudo reboot
```

### Issue: "Permission denied"

**Solution:**
```bash
chmod +x run.sh btc_generator.py btc_gpu_generator.py test_generator.py
```

### Issue: Slow performance

**Possible causes:**
1. Other programs using CPU/GPU
2. Not using all cores
3. Insufficient RAM

**Solutions:**
```bash
# Check CPU usage
htop

# Check GPU usage (if applicable)
nvidia-smi

# Close other applications
# Try smaller batch size in config.py
```

### Issue: Import error on coincurve

**Solution:**
```bash
# Install build dependencies
sudo apt install build-essential libssl-dev libffi-dev python3-dev

# Reinstall coincurve
pip3 uninstall coincurve
pip3 install coincurve
```

### Issue: PyCUDA compilation error

**Solution:**
```bash
# Install CUDA development tools
sudo apt install nvidia-cuda-toolkit

# Reinstall PyCUDA
pip3 uninstall pycuda
pip3 install pycuda

# If still fails, use CPU-only mode
python3 btc_generator.py
```

## 📁 File Structure After Installation

```
/workspace/
├── btc_generator.py          # ✅ Main CPU generator
├── btc_gpu_generator.py      # ✅ GPU accelerated generator
├── test_generator.py         # ✅ Test suite
├── config.py                 # ✅ Configuration file
├── run.sh                    # ✅ Launcher script
├── requirements.txt          # ✅ Dependencies
├── addresses.txt             # ✅ Target addresses
├── .gitignore               # ✅ Git ignore rules
├── README.md                 # 📖 Full documentation
├── QUICKSTART.md            # 📖 Quick start guide
├── INSTALLATION.md          # 📖 This file
├── PROJECT_OVERVIEW.md      # 📖 Technical overview
└── found_addresses.txt      # 📝 Generated when matches found
```

## ✅ Post-Installation Checklist

- [ ] Python 3.7+ installed and working
- [ ] All dependencies installed (`pip3 install -r requirements.txt`)
- [ ] Test suite passes (`python3 test_generator.py`)
- [ ] Scripts are executable (`chmod +x *.sh *.py`)
- [ ] Target addresses added to `addresses.txt`
- [ ] CUDA working (if using GPU) - `nvidia-smi` shows GPU
- [ ] Configuration reviewed in `config.py`

## 🎓 Next Steps

1. ✅ Run test suite: `python3 test_generator.py`
2. ✅ Start with small run: `python3 btc_generator.py` (enter 1000)
3. ✅ Review results
4. ✅ Scale up as needed
5. ✅ Read documentation in README.md

## 💡 Tips

1. **Start small**: Test with 1,000-10,000 addresses first
2. **Monitor resources**: Use `htop` or `nvidia-smi` to watch usage
3. **Use screen/tmux**: For long-running sessions
4. **Virtual environment**: Keep dependencies isolated
5. **Backup results**: Save `found_addresses.txt` if you get matches

## 🆘 Getting Help

If you encounter issues:

1. **Check test output**: `python3 test_generator.py`
2. **Review logs**: Look for error messages
3. **Verify dependencies**: `pip3 list`
4. **Check GPU**: `nvidia-smi` (if using GPU)
5. **Read documentation**: See README.md and PROJECT_OVERVIEW.md

## 🎉 You're Ready!

Everything should now be installed and working. Run:

```bash
./run.sh
```

And start generating Bitcoin addresses!

---

**Happy generating! 🚀**
