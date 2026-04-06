"""GhostLM dataset — converts processed JSONL data into PyTorch DataLoader-ready tensors."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from ghostlm.config import GhostLMConfig
from ghostlm.tokenizer import GhostTokenizer


class GhostDataset(Dataset):
    """PyTorch Dataset for GhostLM language model training.

    Loads tokenized text from a JSONL file, concatenates all tokens
    into a single flat sequence, and yields fixed-length chunks for
    autoregressive language modeling (x, y shifted by one token).
    """

    def __init__(self, jsonl_path: str, tokenizer: GhostTokenizer, config: GhostLMConfig):
        """Initialize the dataset from a JSONL file.

        Reads all records, tokenizes the "text" field of each, and
        concatenates them into one continuous token stream.

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
                    self.tokens.extend(tokenizer.encode(text))

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


def build_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer: GhostTokenizer,
    config: GhostLMConfig,
) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders from JSONL files.

    Creates GhostDataset instances for both splits and wraps them
    in PyTorch DataLoaders with appropriate batching and shuffling.

    Args:
        train_path: Path to the training JSONL file.
        val_path: Path to the validation JSONL file.
        tokenizer: GhostTokenizer instance for encoding.
        config: GhostLMConfig with batch_size and context_length.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_dataset = GhostDataset(train_path, tokenizer, config)
    val_dataset = GhostDataset(val_path, tokenizer, config)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader
