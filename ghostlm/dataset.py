"""GhostLM dataset — converts processed JSONL data into PyTorch DataLoader-ready tensors."""

import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from ghostlm.config import GhostLMConfig
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
