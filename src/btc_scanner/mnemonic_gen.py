from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
from mnemonic import Mnemonic

from .gpu import gpu_generate_entropy


_WORDS_TO_BYTES = {12: 16, 15: 20, 18: 24, 21: 28, 24: 32}


def _entropy_cpu(batch_size: int, num_bytes: int) -> np.ndarray:
    data = os.urandom(batch_size * num_bytes)
    return np.frombuffer(data, dtype=np.uint8).reshape(batch_size, num_bytes)


def generate_mnemonics_batch(
    batch_size: int,
    words: int = 12,
    use_gpu: bool = False,
) -> List[str]:
    if words not in _WORDS_TO_BYTES:
        raise ValueError("words must be one of 12,15,18,21,24")
    num_bytes = _WORDS_TO_BYTES[words]

    ent: Optional[np.ndarray] = None
    if use_gpu:
        ent = gpu_generate_entropy(batch_size, num_bytes)
    if ent is None:
        ent = _entropy_cpu(batch_size, num_bytes)

    mnemo = Mnemonic("english")
    mnemonics: List[str] = []
    for i in range(batch_size):
        mn = mnemo.to_mnemonic(bytes(ent[i].tolist()))
        mnemonics.append(mn)
    return mnemonics
