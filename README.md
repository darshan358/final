# Bitcoin Address Generator with GPU/CUDA Acceleration

A high-performance Bitcoin address generator that uses random seed phrases to generate BTC addresses and checks them against a target list. Features GPU acceleration with CUDA, multi-threading, and parallel processing for maximum performance.

## 🚀 Features

- **GPU Acceleration**: CUDA support for parallel address generation
- **Multi-threading**: Uses all available CPU cores for parallel processing
- **Multiple Address Formats**: Generates Legacy (P2PKH), SegWit (P2SH), and uncompressed addresses
- **BIP39 Mnemonic**: Uses standard 24-word seed phrases
- **Real-time Progress**: Live progress bar with speed metrics
- **Automatic Fallback**: Falls back to CPU if GPU/CUDA is not available
- **Colorful Output**: Beautiful terminal output with color coding

## 📋 Requirements

### Software Requirements
- Python 3.7 or higher
- CUDA Toolkit (optional, for GPU acceleration)
  - NVIDIA GPU with CUDA support
  - CUDA 11.0 or higher recommended

### Python Libraries
All required libraries are listed in `requirements.txt`:
- mnemonic
- ecdsa
- base58
- pycuda (optional for GPU)
- numpy
- coincurve
- tqdm
- colorama

## 🔧 Installation

### 1. Clone or Download the Project

```bash
cd /path/to/bitcoin-generator
```

### 2. Install Python Dependencies

**For CPU-only mode:**
```bash
pip install mnemonic ecdsa base58 numpy coincurve tqdm colorama
```

**For GPU/CUDA support:**
```bash
pip install -r requirements.txt
```

### 3. Install CUDA (Optional, for GPU acceleration)

**Ubuntu/Debian:**
```bash
# Install NVIDIA drivers
sudo apt update
sudo apt install nvidia-driver-530

# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
sudo apt update
sudo apt install cuda

# Install PyCUDA
pip install pycuda
```

**Verify CUDA installation:**
```bash
nvidia-smi
nvcc --version
```

## 📝 Setup

### 1. Prepare Target Addresses

Edit `addresses.txt` and add Bitcoin addresses you want to check against (one per line):

```
1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
1FeexV6bAHb8ybZjqQMjJrcCrHGW9sb6uF
3J98t1WpEZ73CNmYviecrnyiWrnqRhWNLy
bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq
```

The file is automatically created with sample addresses if it doesn't exist.

## 🎮 Usage

### Basic Usage (CPU-only)

```bash
python3 btc_generator.py
```

### GPU-Accelerated Version

```bash
python3 btc_gpu_generator.py
```

### Interactive Mode

When you run the program, it will ask:
```
Enter number of addresses to generate (default 10000):
```

Enter the number of addresses you want to generate and press Enter.

## 📊 Output

The program will display:
- Number of target addresses loaded
- CPU cores being used
- GPU status (if applicable)
- Real-time progress bar
- Generation speed (addresses/second)
- Match notifications (if found)

### Example Output

```
================================================================================
  Bitcoin Address Generator - GPU/CUDA + Multi-threading Edition
================================================================================

Loaded 10 target addresses from addresses.txt
CPU cores available: 8
CUDA available: True
Using 8 parallel workers

Enter number of addresses to generate (default 10000): 50000

Configuration:
  - Total addresses to generate: 50,000
  - Workers: 8
  - Addresses per worker: 6,250
  - GPU acceleration: Yes

Starting generation...

Generating addresses: 100%|████████████| 50000/50000 [00:45<00:00, 1100.5 addr/s]

================================================================================
Summary
================================================================================
Total addresses checked: 50,000
Time elapsed: 45.32 seconds
Speed: 1103.45 addresses/second
Matches found: 0

Done!
```

### If a Match is Found

```
================================================================================
✓ MATCH FOUND by Worker 3!
================================================================================
Address: 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
Mnemonic: abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about
Private Key: 1234567890abcdef...
WIF: 5HpHagT65TZzG1PH3CSu63k8DbpvD8s5ip4nEB3kEsreAnchuDf
================================================================================

Saving results to found_addresses.txt
```

Results are automatically saved to `found_addresses.txt`.

## 🏗️ Project Structure

```
bitcoin-generator/
├── btc_generator.py          # CPU-only version with multi-threading
├── btc_gpu_generator.py      # GPU/CUDA accelerated version
├── addresses.txt              # Target addresses to check
├── requirements.txt           # Python dependencies
├── found_addresses.txt        # Results (created when matches found)
└── README.md                  # This file
```

## ⚡ Performance Tips

1. **GPU vs CPU**: GPU version can be 5-10x faster if you have a CUDA-capable GPU
2. **Batch Size**: GPU version uses larger batches (256) vs CPU (100)
3. **Workers**: The program automatically uses all CPU cores
4. **Target List**: Smaller target lists check faster

### Expected Performance

- **CPU-only** (8-core): ~500-1,500 addresses/second
- **GPU (RTX 3060)**: ~2,000-5,000 addresses/second
- **GPU (RTX 4090)**: ~10,000+ addresses/second

## ⚠️ Important Notes

### Security Warnings

1. **Educational Purpose**: This tool is for educational purposes only
2. **Random Search**: Finding a match with an existing funded address is astronomically unlikely (2^256 possibilities)
3. **Never Use Found Keys**: Any address found is likely unfunded and for testing only
4. **Private Keys**: Keep generated private keys secure if you plan to use them

### Legal Disclaimer

- This software is provided for educational purposes
- Do not use for unauthorized access to Bitcoin addresses
- The authors are not responsible for misuse of this software
- Always comply with local laws and regulations

### Technical Notes

1. **Address Types Generated**:
   - Legacy (P2PKH) - starts with 1
   - SegWit (P2SH) - starts with 3
   - Uncompressed Legacy - starts with 1

2. **Not Generated** (would require additional implementation):
   - Native SegWit (Bech32) - starts with bc1
   - Taproot - starts with bc1p

3. **Probability**:
   - Total possible addresses: ~2^160 for each format
   - Finding a specific address: virtually impossible
   - This is a demonstration of parallel processing techniques

## 🔍 Troubleshooting

### CUDA Not Available

If you see "CUDA not available", the program will automatically use CPU-only mode. To fix:

1. Verify NVIDIA GPU: `nvidia-smi`
2. Check CUDA installation: `nvcc --version`
3. Reinstall PyCUDA: `pip install --upgrade pycuda`

### Import Errors

```bash
# Reinstall all dependencies
pip install --upgrade -r requirements.txt
```

### Permission Errors

```bash
# Make scripts executable
chmod +x btc_generator.py btc_gpu_generator.py
```

### Slow Performance

1. Reduce number of workers if RAM is limited
2. Ensure no other heavy processes are running
3. Check GPU utilization: `nvidia-smi` (for GPU mode)

## 🤝 Contributing

This is an educational project. Feel free to:
- Report bugs
- Suggest improvements
- Add new features
- Optimize performance

## 📄 License

This project is provided as-is for educational purposes.

## 🙏 Acknowledgments

- BIP39 Standard for mnemonic generation
- Bitcoin Core for address format specifications
- PyCUDA for GPU acceleration framework
- All open-source libraries used in this project

---

**Remember**: The probability of finding a funded Bitcoin address through random generation is effectively zero. This tool demonstrates parallel processing and GPU acceleration techniques.
