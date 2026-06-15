"""GhostLM dataset — converts processed JSONL data into PyTorch DataLoader-ready tensors."""

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset

from ghostlm.config import GhostLMConfig
from ghostlm.curriculum import DomainCurriculum
from ghostlm.tokenizer import GhostTokenizer


class GhostDataset(Dataset):
    """PyTorch Dataset for GhostLM language model training.

    Loads tokenized text from a JSONL file, concatenates all tokens
    into a single flat sequence, and yields fixed-length chunks for
    autoregressive language modeling (x, y shifted by one token).

    Every record is terminated with the tokenizer's EOS token so the
    model sees explicit document boundaries instead of unrelated texts
    flowing into each other.

    Note: this tokenizes the whole file into memory at startup, which
    is fine for small corpora but slow and memory-hungry at hundreds of
    millions of tokens. For large runs, pretokenize once with
    ``scripts/pretokenize.py`` and use ``GhostBinDataset`` (selected
    automatically by ``build_dataloaders`` for ``.bin`` paths).
    """

    def __init__(self, jsonl_path: str, tokenizer: GhostTokenizer, config: GhostLMConfig):
        """Initialize the dataset from a JSONL file.

        Reads all records, tokenizes the "text" field of each (with a
        trailing EOS document separator), and concatenates them into
        one continuous token stream.

        Args:
            jsonl_path: Path to the processed JSONL file.
            tokenizer: GhostTokenizer instance for encoding text.
            config: GhostLMConfig containing context_length.
        """
        self.context_length = config.context_length
        self.tokens: List[int] = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                text = record.get("text", "")
                if text:
                    self.tokens.extend(tokenizer.encode(text, add_eos=True))

        print(f"  Loaded {len(self.tokens):,} tokens from {jsonl_path}")

    def __len__(self) -> int:
        """Return the number of non-overlapping context-length chunks.

        Returns:
            Integer count of available training samples.
        """
        return len(self.tokens) // self.context_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve a single (input, target) token chunk.

        The target sequence is the input sequence shifted left by one
        token, enabling next-token prediction training.

        Args:
            idx: Index of the chunk to retrieve.

        Returns:
            Tuple of (x, y) tensors, each of shape (context_length,).
        """
        start = idx * self.context_length
        end = start + self.context_length

        x = self.tokens[start:end]
        y = self.tokens[start + 1 : end + 1]

        # Pad target with -1 if we hit the end of data (cross-entropy ignores -1)
        if len(y) < len(x):
            y = y + [-1] * (len(x) - len(y))

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )


class GhostBinDataset(Dataset):
    """Memory-mapped dataset over a pretokenized ``.bin`` token file.

    The ``.bin`` file is a flat array of token ids written by
    ``scripts/pretokenize.py`` (uint16 when the vocab fits, uint32
    otherwise; recorded in the sidecar ``meta.json``). ``np.memmap``
    keeps resident memory near zero and startup instant regardless of
    corpus size — a Python-list token stream costs ~28 bytes/token,
    which at v1.0-corpus scale (~422M tokens) is >10 GB of RAM plus a
    full re-tokenization on every launch.
    """

    def __init__(self, bin_path: str, config: GhostLMConfig):
        """Open the memmap and read its dtype from the sidecar meta.json.

        Args:
            bin_path: Path to the ``.bin`` token file.
            config: GhostLMConfig containing context_length.
        """
        self.context_length = config.context_length
        bin_path = Path(bin_path)

        meta_path = bin_path.with_suffix(".meta.json")
        dtype = "uint16"
        if meta_path.exists():
            with open(meta_path) as f:
                dtype = json.load(f).get("dtype", "uint16")

        self.tokens = np.memmap(bin_path, dtype=np.dtype(dtype), mode="r")
        print(f"  Mapped {len(self.tokens):,} tokens from {bin_path}")

    def __len__(self) -> int:
        """Return the number of non-overlapping context-length chunks."""
        return len(self.tokens) // self.context_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve a single (input, target) token chunk (y shifted by one)."""
        start = idx * self.context_length
        end = start + self.context_length

        x = torch.from_numpy(self.tokens[start:end].astype(np.int64))
        y_np = self.tokens[start + 1 : end + 1].astype(np.int64)
        if len(y_np) < len(x):
            y_np = np.concatenate([y_np, np.full(len(x) - len(y_np), -1, dtype=np.int64)])
        return x, torch.from_numpy(y_np)


class MultiDomainBinDataset(IterableDataset):
    """Domain-weighted, curriculum-aware sampler over per-domain ``.bin`` files.

    Each domain is a separate memory-mapped token stream (written by
    ``scripts/pretokenize.py --by-domain``). On every step the sampler asks
    the ``DomainCurriculum`` for the domain weights at the current training
    progress, picks a domain by those weights, and returns a random
    context-length block from it. This is the multi-stage data schedule
    used by SmolLM2 / H2O-Danube3: a fixed corpus, but a *mixture that
    shifts over training*.

    Infinite by design (an ``IterableDataset``): training is step-bounded,
    not epoch-bounded, so there is no natural length. ``progress_fn``
    returns the current step / max_steps in [0, 1]; the trainer updates the
    value it closes over each step. With ``num_workers=0`` (GhostLM's
    default) the dataset shares the trainer's process, so the closure sees
    live progress.
    """

    def __init__(
        self,
        domain_bins: Dict[str, str],
        config: GhostLMConfig,
        curriculum: DomainCurriculum,
        progress_fn: Callable[[], float],
        seed: int = 42,
    ):
        self.context_length = config.context_length
        self.curriculum = curriculum
        self.progress_fn = progress_fn
        self.seed = seed
        self.domains: List[str] = []
        self.streams: Dict[str, np.memmap] = {}
        for domain, path in domain_bins.items():
            p = Path(path)
            meta = p.with_suffix(".meta.json")
            dtype = "uint16"
            if meta.exists():
                with open(meta) as f:
                    dtype = json.load(f).get("dtype", "uint16")
            arr = np.memmap(p, dtype=np.dtype(dtype), mode="r")
            if len(arr) < self.context_length + 1:
                print(f"  skip domain {domain}: only {len(arr)} tokens")
                continue
            self.domains.append(domain)
            self.streams[domain] = arr
            print(f"  domain {domain}: {len(arr):,} tokens ({path})")
        if not self.domains:
            raise ValueError("no usable domain bins (all empty or too short)")

    def _weight_vector(self, progress: float) -> np.ndarray:
        w = self.curriculum.weights_at(progress)
        v = np.array([max(0.0, w.get(d, 0.0)) for d in self.domains], dtype=np.float64)
        s = v.sum()
        return v / s if s > 0 else np.full(len(self.domains), 1.0 / len(self.domains))

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        rank = int(os.environ.get("RANK", "0"))
        rng = np.random.default_rng(self.seed + 1009 * rank + worker_id)
        ctx = self.context_length
        while True:
            v = self._weight_vector(self.progress_fn())
            domain = self.domains[rng.choice(len(self.domains), p=v)]
            stream = self.streams[domain]
            start = int(rng.integers(0, len(stream) - ctx))
            x = torch.from_numpy(stream[start:start + ctx].astype(np.int64))
            y = torch.from_numpy(stream[start + 1:start + ctx + 1].astype(np.int64))
            yield x, y


def build_curriculum_train_loader(
    domain_bins: Dict[str, str],
    config: GhostLMConfig,
    curriculum: DomainCurriculum,
    progress_fn: Callable[[], float],
    seed: Optional[int] = None,
) -> DataLoader:
    """Build an infinite, curriculum-weighted training DataLoader.

    Pair with a step-bounded training loop. The val loader stays the plain
    ``GhostBinDataset`` so val loss remains comparable across runs.
    """
    ds = MultiDomainBinDataset(
        domain_bins, config, curriculum, progress_fn,
        seed=config.seed if seed is None else seed,
    )
    pin = torch.cuda.is_available()
    return DataLoader(
        ds, batch_size=config.batch_size, num_workers=0,
        pin_memory=pin, drop_last=True,
    )


def _make_dataset(path: str, tokenizer: GhostTokenizer, config: GhostLMConfig) -> Dataset:
    """Pick the dataset backend from the file extension."""
    if str(path).endswith(".bin"):
        return GhostBinDataset(path, config)
    return GhostDataset(path, tokenizer, config)


def build_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer: GhostTokenizer,
    config: GhostLMConfig,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders from JSONL or .bin files.

    Paths ending in ``.bin`` use the memory-mapped ``GhostBinDataset``
    (pretokenize with ``scripts/pretokenize.py``); anything else is
    tokenized in-process via ``GhostDataset``.

    Under torchrun (WORLD_SIZE > 1) the train loader is sharded with a
    ``DistributedSampler`` so each rank sees a disjoint slice of the
    data — without it, every rank trains on identical batches and DDP
    buys nothing. The val loader stays unsharded so the reported val
    loss means the same thing at any world size.

    Args:
        train_path: Path to the training JSONL or .bin file.
        val_path: Path to the validation JSONL or .bin file.
        tokenizer: GhostTokenizer instance for encoding.
        config: GhostLMConfig with batch_size and context_length.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_dataset = _make_dataset(train_path, tokenizer, config)
    val_dataset = _make_dataset(val_path, tokenizer, config)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    train_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=int(os.environ.get("RANK", "0")),
            shuffle=True,
            drop_last=True,
            seed=config.seed,
        )

    # pin_memory only helps (and only works) for CUDA host->device copies.
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        num_workers=0,
        pin_memory=pin,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=pin,
    )

    return train_loader, val_loader
