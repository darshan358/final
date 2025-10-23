from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Set, Tuple

from .derive import AddressType, derive_addresses_from_mnemonic
from .mnemonic_gen import generate_mnemonics_batch
from .utils import normalize_address


@dataclass
class WorkerConfig:
    addresses: Set[str]
    addr_type: AddressType
    lookahead: int
    batch_size: int
    use_gpu: bool


@dataclass
class WorkerResult:
    processed: int
    hits: List[Tuple[str, str, str]]  # (address, mnemonic, wif)


def scan_batch(config: WorkerConfig) -> WorkerResult:
    mnemonics = generate_mnemonics_batch(config.batch_size, words=12, use_gpu=config.use_gpu)
    hits: List[Tuple[str, str, str]] = []
    processed = 0

    for mn in mnemonics:
        derived = derive_addresses_from_mnemonic(mn, addr_type=config.addr_type, lookahead=config.lookahead)
        for d in derived:
            processed += 1
            addr = normalize_address(d["address"])
            if addr in config.addresses:
                hits.append((d["address"], mn, d["wif"]))
    return WorkerResult(processed=processed, hits=hits)
