import math
import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from numba import cuda  # type: ignore
    _NUMBA_CUDA_AVAILABLE = cuda.is_available()
except Exception:  # pragma: no cover - environment without numba/cuda
    _NUMBA_CUDA_AVAILABLE = False
    cuda = None  # type: ignore


def _sha256_two_hashes(data: bytes) -> Tuple[int, int]:
    d1 = hashlib.sha256(data).digest()
    # Derive two 64-bit integers from the digest using different segments
    h1 = int.from_bytes(d1[0:8], byteorder="big", signed=False)
    h2 = int.from_bytes(d1[8:16], byteorder="big", signed=False)
    if h2 == 0:
        # Avoid zero increment in double hashing
        h2 = int.from_bytes(d1[16:24], byteorder="big", signed=False) or 0x9e3779b97f4a7c15
    return h1, h2


def _optimal_bloom_params(expected_items: int, false_positive_rate: float) -> Tuple[int, int]:
    if expected_items <= 0:
        raise ValueError("expected_items must be positive")
    if not (0 < false_positive_rate < 1):
        raise ValueError("false_positive_rate must be in (0,1)")
    ln2 = math.log(2)
    m = int(math.ceil(-(expected_items * math.log(false_positive_rate)) / (ln2 ** 2)))  # bits
    k = max(1, int(round((m / expected_items) * ln2)))
    return m, k


@dataclass
class BloomSpec:
    bit_size: int
    num_hashes: int


class NumpyBloomFilter:
    def __init__(self, expected_items: int, false_positive_rate: float = 1e-6) -> None:
        bit_size, num_hashes = _optimal_bloom_params(expected_items, false_positive_rate)
        byte_size = (bit_size + 7) // 8
        self._bits = np.zeros(byte_size, dtype=np.uint8)
        self._spec = BloomSpec(bit_size=bit_size, num_hashes=num_hashes)

    @property
    def spec(self) -> BloomSpec:
        return self._spec

    @property
    def bits(self) -> np.ndarray:
        return self._bits

    def _bit_indices(self, h1: int, h2: int) -> Iterable[int]:
        m = self._spec.bit_size
        for j in range(self._spec.num_hashes):
            yield (h1 + j * h2) % m

    def add(self, item: str) -> None:
        h1, h2 = _sha256_two_hashes(item.encode("utf-8"))
        for idx in self._bit_indices(h1, h2):
            byte_index = idx >> 3
            bit_mask = 1 << (idx & 7)
            self._bits[byte_index] |= bit_mask

    def add_all(self, items: Iterable[str]) -> None:
        for it in items:
            s = it.strip()
            if not s:
                continue
            self.add(s)

    def contains(self, item: str) -> bool:
        h1, h2 = _sha256_two_hashes(item.encode("utf-8"))
        return self.contains_h(h1, h2)

    def contains_h(self, h1: int, h2: int) -> bool:
        for idx in self._bit_indices(h1, h2):
            byte_index = idx >> 3
            bit_mask = 1 << (idx & 7)
            if (self._bits[byte_index] & bit_mask) == 0:
                return False
        return True

    def contains_batch(self, items: Sequence[str]) -> List[bool]:
        out: List[bool] = []
        for it in items:
            if not it:
                out.append(False)
                continue
            out.append(self.contains(it))
        return out


class SharedBloomReader:
    """Read-only Bloom filter backed by a shared-memory numpy array.

    This avoids duplicating the bit array across processes.
    """

    def __init__(self, shm_buf: memoryview, spec: BloomSpec) -> None:
        # Wrap the shared memory buffer as a numpy array without copying
        self._bits = np.frombuffer(shm_buf, dtype=np.uint8)
        self._spec = spec

    @property
    def spec(self) -> BloomSpec:
        return self._spec

    def _bit_indices(self, h1: int, h2: int) -> Iterable[int]:
        m = self._spec.bit_size
        for j in range(self._spec.num_hashes):
            yield (h1 + j * h2) % m

    def contains(self, item: str) -> bool:
        h1, h2 = _sha256_two_hashes(item.encode("utf-8"))
        return self.contains_h(h1, h2)

    def contains_h(self, h1: int, h2: int) -> bool:
        for idx in self._bit_indices(h1, h2):
            byte_index = idx >> 3
            bit_mask = 1 << (idx & 7)
            if (self._bits[byte_index] & bit_mask) == 0:
                return False
        return True

    def contains_batch(self, items: Sequence[str]) -> List[bool]:
        out: List[bool] = []
        for it in items:
            if not it:
                out.append(False)
                continue
            out.append(self.contains(it))
        return out


# Optional GPU acceleration for membership checks
if _NUMBA_CUDA_AVAILABLE:

    @cuda.jit
    def _bloom_contains_kernel(bit_array: np.ndarray, bit_size: int, num_hashes: int,
                               h1_arr: np.ndarray, h2_arr: np.ndarray, out_arr: np.ndarray) -> None:
        i = cuda.grid(1)
        if i >= h1_arr.size:
            return
        h1 = h1_arr[i]
        h2 = h2_arr[i]
        is_member = True
        for j in range(num_hashes):
            idx = (h1 + j * h2) % bit_size
            byte_index = idx >> 3
            bit_mask = 1 << (idx & 7)
            if (bit_array[byte_index] & bit_mask) == 0:
                is_member = False
                break
        out_arr[i] = 1 if is_member else 0


class CudaBloomChecker:
    def __init__(self, bit_bytes: np.ndarray, spec: BloomSpec) -> None:
        if not _NUMBA_CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")
        self._spec = spec
        # Copy bit array to device
        self._d_bits = cuda.to_device(bit_bytes)

    @property
    def spec(self) -> BloomSpec:
        return self._spec

    def contains_batch_h(self, h_pairs: Sequence[Tuple[int, int]]) -> List[bool]:
        if not h_pairs:
            return []
        h1_arr = np.array([h[0] for h in h_pairs], dtype=np.int64)
        h2_arr = np.array([h[1] for h in h_pairs], dtype=np.int64)
        out_arr = np.zeros(len(h_pairs), dtype=np.uint8)
        d_h1 = cuda.to_device(h1_arr)
        d_h2 = cuda.to_device(h2_arr)
        d_out = cuda.to_device(out_arr)
        threads_per_block = 128
        blocks = (len(h_pairs) + threads_per_block - 1) // threads_per_block
        _bloom_contains_kernel[blocks, threads_per_block](self._d_bits, self._spec.bit_size, self._spec.num_hashes, d_h1, d_h2, d_out)
        d_out.copy_to_host(out_arr)
        return [bool(x) for x in out_arr]

    def contains_batch(self, items: Sequence[str]) -> List[bool]:
        h_pairs: List[Tuple[int, int]] = []
        for it in items:
            if not it:
                h_pairs.append((0, 0))
            else:
                h_pairs.append(_sha256_two_hashes(it.encode("utf-8")))
        return self.contains_batch_h(h_pairs)
