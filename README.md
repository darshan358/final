# Bitcoin Address Generator & Checker

A high-performance Bitcoin address generator that uses random seed phrases to generate private keys and addresses, then checks them against a target list. Features GPU acceleration (CUDA), multithreading, and parallel processing for maximum performance.

## Features

- **Random Seed Generation**: Uses cryptographically secure random seed phrases (BIP39)
- **Multiple Address Types**: Generates P2PKH (Legacy), P2SH (Script Hash), and Bech32 (Segwit) addresses
- **GPU Acceleration**: CUDA support for parallel hash computations (when available)
- **Multithreading**: Parallel processing using multiple CPU threads
- **Address Checking**: Efficiently checks generated addresses against target list
- **Real-time Statistics**: Live performance monitoring and statistics
- **Automatic Saving**: Found matches are automatically saved to file

## Installation

1. Clone or download this repository
2. Run the setup script:
   ```bash
   python3 setup.py
   ```

### Manual Installation

If the setup script fails, install dependencies manually:

```bash
pip3 install mnemonic ecdsa base58 numpy

# For GPU support (optional):
pip3 install pycuda
```

## Usage

### Basic Usage

```bash
# Start mining (runs indefinitely)
python3 main.py

# Run for specific duration (60 seconds)
python3 main.py --duration 60

# Use specific number of threads
python3 main.py --threads 16

# Disable GPU acceleration
python3 main.py --no-gpu
```

### Testing & Benchmarking

```bash
# Generate sample addresses
python3 main.py --sample 10

# Run benchmark test
python3 main.py --benchmark 60

# Add target address to check list
python3 main.py --add-address 1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
```

### Configuration

Edit `addresses.txt` to add Bitcoin addresses you want to check against:

```
# Bitcoin addresses to check against (one per line)
1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa
12c6DSiU4Rq3P4ZxziKxzrL5LmMBrzjrJX
1HLoD9E4SDFFPDiYfNYnkBLQ85Y51J3Zb1
bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh
```

## Command Line Options

```
--threads N         Number of worker threads (default: 8)
--batch-size N      Addresses per batch (default: 1000)
--duration N        Run for N seconds (default: indefinite)
--no-gpu           Disable GPU acceleration
--sample N         Generate N sample addresses and exit
--benchmark N      Run benchmark for N seconds
--add-address ADDR Add address to target list
```

## Performance

Performance varies based on hardware:

- **CPU Only**: ~1,000-10,000 addresses/second
- **With GPU**: ~10,000-100,000+ addresses/second (depending on GPU)

## Output Files

- `found_addresses.txt`: Contains any matching addresses found
- `addresses.txt`: Target addresses to check against

## How It Works

1. **Seed Generation**: Creates random BIP39 mnemonic phrases
2. **Key Derivation**: Uses BIP32/BIP44 to derive private keys from seeds
3. **Address Generation**: Creates multiple address types from each private key
4. **Parallel Checking**: Efficiently checks generated addresses against target list
5. **Match Recording**: Saves any matches with full details

## Security Notes

- This tool generates truly random private keys using cryptographically secure methods
- All generated keys and addresses are mathematically valid
- Found matches include the full mnemonic phrase and private key for recovery
- Keep `found_addresses.txt` secure as it contains private keys

## Technical Details

- Uses BIP39 for mnemonic generation
- Implements BIP32/BIP44 key derivation
- Supports P2PKH, P2SH, and Bech32 address formats
- CUDA kernels for parallel SHA256 computation
- Thread-safe address checking with efficient hash table lookups

## Requirements

- Python 3.7+
- NumPy
- mnemonic
- ecdsa
- base58
- PyCUDA (optional, for GPU acceleration)
- NVIDIA GPU with CUDA support (optional)

## License

This project is for educational purposes. Use responsibly and in accordance with local laws.

## Disclaimer

This tool is for educational and research purposes only. The probability of finding a collision with existing Bitcoin addresses is astronomically low. Do not expect to find matches with addresses containing actual Bitcoin.