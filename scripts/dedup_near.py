#!/usr/bin/env python3
"""Near-duplicate removal for the processed corpus (MinHash + LSH).

``merge_datasets`` only drops exact duplicates. Mirrored writeups, reposted
advisories and templated pages differ by a few words and slip through, and
on a small token budget every repeated document is wasted compute
(FineWeb, Penedo et al. 2024). Documents are shingled into word 5-grams,
MinHashed with 128 permutations and bucketed by 16 bands of 8 rows, which
flags pairs above roughly 0.7 Jaccard similarity.

Validation records are registered first and always kept; a training record
is dropped if it collides with any validation record or any earlier kept
training record, so near-copies of val text cannot leak into training.

Usage:
    python scripts/dedup_near.py --train data/processed/train.jsonl \
        --val data/processed/val.jsonl --out data/processed/train.dedup.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zlib
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

NUM_PERM = 128
BANDS = 16
ROWS = NUM_PERM // BANDS
SHINGLE = 5
MIN_WORDS = 20
_PRIME = np.uint64((1 << 61) - 1)
_rng = np.random.default_rng(1234)
_A = _rng.integers(1, (1 << 61) - 1, NUM_PERM, dtype=np.uint64)
_B = _rng.integers(0, (1 << 61) - 1, NUM_PERM, dtype=np.uint64)
_WORD = re.compile(r"\w+")


def minhash(text: str) -> Optional[np.ndarray]:
    """128-value MinHash signature of the text's word 5-grams, or None if too short."""
    words = _WORD.findall(text.lower())
    if len(words) < MIN_WORDS:
        return None
    shingles = {" ".join(words[i:i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}
    h = np.fromiter((zlib.crc32(s.encode()) for s in shingles), dtype=np.uint64, count=len(shingles))
    # Universal hashing (a*h + b) mod p; h < 2^32 and a < 2^61 can overflow
    # uint64, which only reshuffles values and keeps the scheme consistent.
    return ((np.outer(_A, h) + _B[:, None]) % _PRIME).min(axis=1)


def band_keys(sig: np.ndarray) -> List[bytes]:
    # Built-in hash() is salted per process, and Pool workers are separate
    # processes, so keys must come from a deterministic digest.
    return [hashlib.blake2b(sig[b * ROWS:(b + 1) * ROWS].tobytes(), digest_size=8).digest()
            for b in range(BANDS)]


def _signature(line: str):
    rec = json.loads(line)
    sig = minhash(rec.get("text", ""))
    return None if sig is None else band_keys(sig)


def iter_lines(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield line


def dedup(train: Path, val: Optional[Path], out: Path, workers: int = 8) -> dict:
    seen = [set() for _ in range(BANDS)]
    stats = Counter()
    dropped_by_source = Counter()

    def register(keys):
        for b, k in enumerate(keys):
            seen[b].add(k)

    with Pool(workers) as pool:
        if val and val.exists():
            for keys in pool.imap(_signature, iter_lines(val), chunksize=256):
                if keys is not None:
                    register(keys)
                stats["val"] += 1

        with out.open("w", encoding="utf-8") as f_out:
            lines = iter_lines(train)
            for line, keys in zip(iter_lines(train), pool.imap(_signature, lines, chunksize=256)):
                stats["train_in"] += 1
                if keys is not None and any(k in seen[b] for b, k in enumerate(keys)):
                    stats["dropped"] += 1
                    dropped_by_source[json.loads(line).get("source", "unknown")] += 1
                    continue
                if keys is not None:
                    register(keys)
                f_out.write(line)
                stats["train_out"] += 1

    return {"stats": dict(stats), "dropped_by_source": dict(dropped_by_source.most_common())}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--train", default="data/processed/train.jsonl")
    p.add_argument("--val", default="data/processed/val.jsonl")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    report = dedup(Path(args.train), Path(args.val), Path(args.out), args.workers)
    s = report["stats"]
    pct = 100 * s.get("dropped", 0) / max(1, s.get("train_in", 0))
    print(f"near-dedup: {s.get('train_in', 0):,} train in, {s.get('dropped', 0):,} dropped "
          f"({pct:.1f}%), {s.get('train_out', 0):,} kept; {s.get('val', 0):,} val registered")
    for source, n in list(report["dropped_by_source"].items())[:15]:
        print(f"  {source:28s} {n:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
