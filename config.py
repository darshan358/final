"""
Configuration file for Bitcoin Address Generator
"""

# Generation settings
DEFAULT_ITERATIONS = 10000
BATCH_SIZE_CPU = 100
BATCH_SIZE_GPU = 256

# Mnemonic settings
MNEMONIC_STRENGTH = 256  # 256 bits = 24 words, 128 bits = 12 words

# GPU settings
USE_GPU = True  # Set to False to force CPU-only mode
GPU_DEVICE_ID = 0  # CUDA device ID to use

# Performance settings
PROGRESS_UPDATE_INTERVAL = 0.1  # seconds
AUTO_SAVE_RESULTS = True

# File paths
TARGET_ADDRESSES_FILE = "addresses.txt"
RESULTS_FILE = "found_addresses.txt"

# Address formats to check
CHECK_COMPRESSED = True
CHECK_UNCOMPRESSED = True
CHECK_SEGWIT = True

# Display settings
SHOW_COLORFUL_OUTPUT = True
SHOW_PROGRESS_BAR = True

# Advanced settings
MAX_WORKERS = None  # None = use all CPU cores, or specify a number
WORKER_TIMEOUT = None  # Timeout for worker processes in seconds (None = no timeout)
