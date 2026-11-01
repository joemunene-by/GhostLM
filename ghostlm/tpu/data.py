"""Memory-mapped token streams for the JAX trainer.

Validation reads fixed non-overlapping blocks like ``GhostBinDataset``; training draws
domain-weighted random blocks seeded per sample exactly like ``MultiDomainBinDataset``,
so a run started here and one started on MLX see the same sequence of batches. On a
TPU VM, copy the ``.bin`` shards from the bucket to local disk first; memmap needs a
real file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np

from ghostlm.curriculum import DomainCurriculum

Batch = Tuple[np.ndarray, np.ndarray]


def open_tokens(path: str | Path) -> np.memmap:
    path = Path(path)
    meta = path.with_suffix(".meta.json")
    dtype = "uint16"
    if meta.exists():
        dtype = json.loads(meta.read_text()).get("dtype", "uint16")
    return np.memmap(path, dtype=np.dtype(dtype), mode="r")


def load_manifest(path: str | Path) -> Dict[str, str]:
    """``{domain: bin_path}`` with relative paths resolved against the manifest's folder."""
    path = Path(path)
    raw = json.loads(path.read_text())
    return {d: str(p if Path(p).is_absolute() else (path.parent / p)) for d, p in raw.items()}


def val_batches(path: str | Path, context: int, batch_size: int, num_batches: int) -> Iterator[Batch]:
    """Sequential (x, y) blocks from the start of the file; the final target is padded with -1."""
    tokens = open_tokens(path)
    n_blocks = min(len(tokens) // context, batch_size * num_batches)
    for b in range(0, n_blocks - batch_size + 1, batch_size):
        xs, ys = [], []
        for i in range(b, b + batch_size):
            start = i * context
            x = tokens[start:start + context].astype(np.int32)
            y = tokens[start + 1:start + context + 1].astype(np.int32)
            if len(y) < len(x):
                y = np.concatenate([y, np.full(len(x) - len(y), -1, dtype=np.int32)])
            xs.append(x)
            ys.append(y)
        yield np.stack(xs), np.stack(ys)


class CurriculumSampler:
    """Domain-weighted random blocks, one RNG per global sample index (seed, rank, 0, n)."""

    def __init__(self, domain_bins: Dict[str, str], context: int, curriculum: DomainCurriculum,
                 seed: int = 42, rank: int = 0):
        self.context = context
        self.curriculum = curriculum
        self.seed = seed
        self.rank = rank
        self.domains = []
        self.streams: Dict[str, np.memmap] = {}
        for domain, path in domain_bins.items():
            arr = open_tokens(path)
            if len(arr) < context + 1:
                print(f"  skip domain {domain}: only {len(arr)} tokens", flush=True)
                continue
            self.domains.append(domain)
            self.streams[domain] = arr
            print(f"  domain {domain}: {len(arr):,} tokens ({path})", flush=True)
        if not self.domains:
            raise ValueError("no usable domain bins")

    def weights(self, progress: float) -> np.ndarray:
        w = self.curriculum.weights_at(progress)
        v = np.array([max(0.0, w.get(d, 0.0)) for d in self.domains], dtype=np.float64)
        s = v.sum()
        return v / s if s > 0 else np.full(len(self.domains), 1.0 / len(self.domains))

    def sample(self, n: int, progress: float) -> Batch:
        rng = np.random.default_rng([self.seed, self.rank, 0, n])
        domain = self.domains[rng.choice(len(self.domains), p=self.weights(progress))]
        stream = self.streams[domain]
        start = int(rng.integers(0, len(stream) - self.context))
        x = stream[start:start + self.context].astype(np.int32)
        y = stream[start + 1:start + self.context + 1].astype(np.int32)
        return x, y

    def batch(self, first_sample: int, size: int, progress: float) -> Batch:
        pairs = [self.sample(first_sample + i, progress) for i in range(size)]
        return np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])


class SingleStreamSampler:
    """Uniform random blocks from one ``.bin`` when no curriculum manifest is given."""

    def __init__(self, path: str | Path, context: int, seed: int = 42):
        self.tokens = open_tokens(path)
        self.context = context
        self.seed = seed
        print(f"  Mapped {len(self.tokens):,} tokens from {path}", flush=True)

    def batch(self, first_sample: int, size: int, progress: float) -> Batch:
        xs, ys = [], []
        for i in range(size):
            rng = np.random.default_rng([self.seed, first_sample + i])
            start = int(rng.integers(0, len(self.tokens) - self.context))
            xs.append(self.tokens[start:start + self.context].astype(np.int32))
            ys.append(self.tokens[start + 1:start + self.context + 1].astype(np.int32))
        return np.stack(xs), np.stack(ys)
