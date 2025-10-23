from __future__ import annotations

from pathlib import Path
from typing import Set

from .utils import load_addresses_set


def load_addresses_file(path: str | Path) -> Set[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"addresses file not found: {p}")
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        return load_addresses_set(f)
