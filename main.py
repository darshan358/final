from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List

import click
from tqdm import tqdm

from src.btc_scanner.gpu import is_cuda_available
from src.btc_scanner.loader import load_addresses_file
from src.btc_scanner.worker import WorkerConfig, scan_batch


@dataclass
class ScanOptions:
    addresses_path: Path
    addr_type: str
    workers: int
    use_gpu: bool
    lookahead: int
    total_candidates: int
    batch_size: int
    hits_output: Path


@click.command()
@click.option("--addresses", "addresses_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), default=Path("addresses.txt"), show_default=True, help="Path to addresses.txt")
@click.option("--addr-type", type=click.Choice(["p2pkh", "p2wpkh", "p2sh-p2wpkh"], case_sensitive=False), default="p2wpkh", show_default=True, help="Address type to derive")
@click.option("--workers", type=int, default=max(1, os.cpu_count() or 1), show_default=True, help="Number of CPU worker processes")
@click.option("--gpu/--no-gpu", "use_gpu", default=False, show_default=True, help="Use CUDA GPU to generate entropy for mnemonics")
@click.option("--lookahead", type=int, default=10, show_default=True, help="Addresses per mnemonic to derive (index 0..n-1)")
@click.option("--total", "total_candidates", type=int, default=10000, show_default=True, help="Total addresses to generate & check")
@click.option("--batch-size", type=int, default=256, show_default=True, help="Mnemonics per batch per worker")
@click.option("--hits-output", type=click.Path(dir_okay=False, path_type=Path), default=Path("hits.txt"), show_default=True, help="Where to append found hits")
def cli(addresses_path: Path, addr_type: str, workers: int, use_gpu: bool, lookahead: int, total_candidates: int, batch_size: int, hits_output: Path):
    """Scan random BIP39 mnemonics, derive BTC addresses, and check against a target list."""
    if use_gpu and not is_cuda_available():
        click.echo("[warn] CUDA not available; continuing without GPU")
        use_gpu = False
    if use_gpu and workers > 1:
        click.echo("[info] GPU mode enabled; using a single process to avoid CUDA contention")
        workers = 1

    click.echo(f"Loading addresses from {addresses_path} ...")
    addresses = load_addresses_file(addresses_path)
    click.echo(f"Loaded {len(addresses)} addresses")

    # Each mnemonic yields `lookahead` addresses
    total_mnemonics_needed = max(1, math.ceil(total_candidates / max(1, lookahead)))

    batches_needed = math.ceil(total_mnemonics_needed / batch_size)
    click.echo(f"Planned: {batches_needed} batches x {batch_size} mnemonics")

    hits_output.parent.mkdir(parents=True, exist_ok=True)

    progress = tqdm(total=total_candidates, unit="addr", smoothing=0.05)
    total_processed = 0
    total_hits = 0

    def make_config() -> WorkerConfig:
        return WorkerConfig(
            addresses=addresses,
            addr_type=addr_type,
            lookahead=lookahead,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )

    if workers <= 1:
        for _ in range(batches_needed):
            res = scan_batch(make_config())
            total_processed += res.processed
            progress.update(res.processed)
            if res.hits:
                total_hits += len(res.hits)
                _append_hits(hits_output, res.hits)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(scan_batch, make_config()) for _ in range(batches_needed)]
            for fut in as_completed(futures):
                res = fut.result()
                total_processed += res.processed
                progress.update(res.processed)
                if res.hits:
                    total_hits += len(res.hits)
                    _append_hits(hits_output, res.hits)

    progress.close()
    click.echo(f"Processed ~{total_processed} derived addresses; hits: {total_hits}")
    if total_hits:
        click.echo(f"Saved hits to {hits_output}")


def _append_hits(path: Path, hits: List[tuple[str, str, str]]) -> None:
    # (address, mnemonic, wif)
    with path.open("a", encoding="utf-8") as f:
        for addr, mnemonic, wif in hits:
            f.write(f"{addr}\t{wif}\t{mnemonic}\n")


if __name__ == "__main__":
    cli()
