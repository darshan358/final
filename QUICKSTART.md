# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Minimum (CPU-only):**
```bash
pip install mnemonic coincurve base58 tqdm colorama
```

### Step 2: Add Target Addresses

Edit `addresses.txt` and add Bitcoin addresses to check (one per line):

```
1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
1FeexV6bAHb8ybZjqQMjJrcCrHGW9sb6uF
bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq
```

### Step 3: Run the Generator

**Easy way (auto-detects GPU):**
```bash
./run.sh
```

**CPU-only version:**
```bash
python3 btc_generator.py
```

**GPU-accelerated version:**
```bash
python3 btc_gpu_generator.py
```

## 🧪 Test First

Run the test script to verify everything works:

```bash
python3 test_generator.py
```

This will:
- Generate test addresses
- Verify the generation process
- Show performance metrics
- Confirm all libraries are working

## ⚡ Performance Tips

1. **More cores = faster**: The program uses all CPU cores automatically
2. **GPU boost**: If you have NVIDIA GPU with CUDA, use `btc_gpu_generator.py`
3. **Batch processing**: GPU version processes 256 addresses at once
4. **Target list**: Keep your addresses.txt file focused on specific addresses

## 📊 Expected Performance

| Configuration | Speed (addresses/sec) |
|--------------|----------------------|
| 4-core CPU   | ~500-800            |
| 8-core CPU   | ~1,000-1,500        |
| 16-core CPU  | ~2,000-3,000        |
| RTX 3060     | ~2,000-5,000        |
| RTX 4090     | ~10,000+            |

## ❓ Common Issues

### "Module not found"
```bash
pip install -r requirements.txt
```

### "CUDA not available"
```bash
# Install CUDA toolkit for your system
# Ubuntu/Debian:
sudo apt install nvidia-cuda-toolkit
pip install pycuda
```

### Script won't run
```bash
chmod +x run.sh btc_generator.py btc_gpu_generator.py
```

## 🎯 What This Does

1. **Generates** random BIP39 seed phrases (24 words)
2. **Derives** Bitcoin private keys from seeds
3. **Creates** multiple address formats:
   - Legacy (starts with 1)
   - SegWit (starts with 3)
   - Uncompressed (starts with 1)
4. **Checks** each address against your target list
5. **Saves** any matches to `found_addresses.txt`

## ⚠️ Reality Check

The probability of finding a funded Bitcoin address through random generation is:

**1 in 2^160** (approximately **1 in 1,461,501,637,330,902,918,203,684,832,716,283,019,655,932,542,976**)

This is:
- More than the number of atoms in the observable universe
- Effectively **impossible**
- A demonstration of cryptographic security

**This tool demonstrates:**
- Parallel processing techniques
- GPU acceleration
- Bitcoin address generation
- Multi-threading in Python

## 📚 Learn More

See [README.md](README.md) for full documentation.

---

**Ready to start?**

```bash
./run.sh
```

Good luck! 🍀
