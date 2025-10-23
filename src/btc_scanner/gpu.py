from __future__ import annotations

import os
from typing import Optional

import numpy as np

try:
    from numba import cuda
except Exception:  # pragma: no cover - numba not installed or no cuda
    cuda = None  # type: ignore


def is_cuda_available() -> bool:
    try:
        return cuda is not None and cuda.is_available()
    except Exception:
        return False


if cuda is not None:

    @cuda.jit
    def _xorshift_fill_random(output: np.ndarray, seeds: np.ndarray):
        idx = cuda.grid(1)
        if idx >= output.shape[0]:
            return
        x = seeds[idx]
        # Generate 4 bytes at a time using xorshift32
        # Fill output[idx, :]
        nbytes = output.shape[1]
        for i in range(0, nbytes, 4):
            x ^= (x << 13) & 0xFFFFFFFF
            x ^= (x >> 17) & 0xFFFFFFFF
            x ^= (x << 5) & 0xFFFFFFFF
            b0 = x & 0xFF
            b1 = (x >> 8) & 0xFF
            b2 = (x >> 16) & 0xFF
            b3 = (x >> 24) & 0xFF
            if i < nbytes:
                output[idx, i] = b0
            if i + 1 < nbytes:
                output[idx, i + 1] = b1
            if i + 2 < nbytes:
                output[idx, i + 2] = b2
            if i + 3 < nbytes:
                output[idx, i + 3] = b3


def gpu_generate_entropy(batch_size: int, bytes_per_item: int) -> Optional[np.ndarray]:
    """
    Generate pseudorandom bytes on GPU using a simple XORSHIFT32 PRNG, seeded from
    OS entropy. Not cryptographically secure; intended for high-throughput candidate generation.
    Returns an array of shape (batch_size, bytes_per_item) dtype=uint8, or None if CUDA unavailable.
    """
    if not is_cuda_available():
        return None

    import numpy as np  # local import to ensure available

    seeds = np.frombuffer(os.urandom(batch_size * 4), dtype=np.uint32)
    output = np.zeros((batch_size, bytes_per_item), dtype=np.uint8)

    threads_per_block = 128
    blocks_per_grid = (batch_size + threads_per_block - 1) // threads_per_block

    _xorshift_fill_random[blocks_per_grid, threads_per_block](output, seeds)
    cuda.synchronize()
    return output
