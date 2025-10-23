from __future__ import annotations

import hashlib
from typing import Iterable, Set

import base58


def normalize_address(addr: str) -> str:
    a = addr.strip()
    if not a:
        return a
    # Bech32 addresses are case-insensitive but should be normalized to lowercase
    if a.lower().startswith(("bc1", "tb1", "bcrt1")):
        return a.lower()
    return a


def load_addresses_set(lines: Iterable[str]) -> Set[str]:
    result: Set[str] = set()
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        result.add(normalize_address(line))
    return result


def to_wif(privkey_bytes: bytes, compressed: bool = True, testnet: bool = False) -> str:
    if len(privkey_bytes) != 32:
        raise ValueError("Private key must be 32 bytes")
    prefix = b"\x80" if not testnet else b"\xEF"
    payload = prefix + privkey_bytes + (b"\x01" if compressed else b"")
    checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    return base58.b58encode(payload + checksum).decode("ascii")
