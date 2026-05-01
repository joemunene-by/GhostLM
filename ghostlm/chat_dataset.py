"""GhostLM chat dataset — JSONL of role-tagged turns into masked SFT tensors.

Each input record is a list of turns:

    {"turns": [{"role": "user", "content": "..."},
               {"role": "assistant", "content": "..."}], "source": "..."}

The dataset tokenizes with chat role markers via GhostTokenizer.encode_chat
and emits (input_ids, target_ids) where target_ids is set to ``-1`` everywhere
the loss should be ignored — i.e. on every position except the assistant's
content and its trailing <|ghost_end|>. ``F.cross_entropy(..., ignore_index=-1)``
in the model takes care of the masking automatically, so the existing trainer
loop works without modification.
"""

import json
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from ghostlm.config import GhostLMConfig
from ghostlm.tokenizer import GhostTokenizer

IGNORE_INDEX = -1


class ChatDataset(Dataset):
    """Variable-length packed chat dataset for SFT.

    Conversations shorter than ``context_length`` are right-padded with the
    PAD token (with target -1, so padding contributes no loss). Conversations
    longer than ``context_length`` are dropped — the synthetic Q&A generator
    is responsible for staying within budget.
    """

    def __init__(self, jsonl_path: str, tokenizer: GhostTokenizer, config: GhostLMConfig):
        """Load a chat JSONL and pre-tokenize every conversation.

        Args:
            jsonl_path: Path to the JSONL file. Each line: {"turns": [...]}.
            tokenizer: GhostTokenizer with chat role markers defined.
            config: GhostLMConfig — only ``context_length`` is read.
        """
        self.context_length = config.context_length
        self.pad_id = tokenizer._special_tokens[tokenizer.PAD]

        self.samples: List[Tuple[List[int], List[int]]] = []
        dropped_long = 0
        path = Path(jsonl_path)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                turns = rec["turns"]
                ids, mask = tokenizer.encode_chat(turns)
                # We need len(ids) <= context_length + 1 because targets shift
                # by one. Drop anything longer.
                if len(ids) > self.context_length + 1:
                    dropped_long += 1
                    continue
                self.samples.append((ids, mask))

        print(
            f"  Loaded {len(self.samples):,} chat samples from {jsonl_path}"
            + (f" (dropped {dropped_long} > context_length)" if dropped_long else "")
        )

    def __len__(self) -> int:
        """Return the number of conversations in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a padded (input_ids, target_ids) pair for sample ``idx``.

        Targets are masked to IGNORE_INDEX (-1) on every position the loss
        should not see — user prompts, role markers, and right-padding. The
        assistant's content tokens and its trailing <|ghost_end|> are the
        only positions with real targets.
        """
        ids, mask = self.samples[idx]
        L = self.context_length

        # input is ids[:-1], target is ids[1:] shifted — predict next from current.
        # mask describes whether *position i* is an assistant token. For LM loss
        # on token t we want target_t to be supervised when ids[t+1] is an
        # assistant token, i.e. mask[t+1] == 1.
        x_full = ids[:-1]
        y_full = [tok if mask[i + 1] == 1 else IGNORE_INDEX for i, tok in enumerate(ids[1:])]

        # Pad to context_length
        pad_x = L - len(x_full)
        if pad_x > 0:
            x_full = x_full + [self.pad_id] * pad_x
            y_full = y_full + [IGNORE_INDEX] * pad_x
        else:
            x_full = x_full[:L]
            y_full = y_full[:L]

        return (
            torch.tensor(x_full, dtype=torch.long),
            torch.tensor(y_full, dtype=torch.long),
        )


def build_chat_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer: GhostTokenizer,
    config: GhostLMConfig,
) -> Tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders for chat SFT.

    Args:
        train_path: Path to train JSONL of chat conversations.
        val_path: Path to val JSONL of chat conversations.
        tokenizer: GhostTokenizer (must include chat role markers).
        config: GhostLMConfig with ``batch_size`` and ``context_length``.

    Returns:
        Tuple (train_loader, val_loader).
    """
    train_ds = ChatDataset(train_path, tokenizer, config)
    val_ds = ChatDataset(val_path, tokenizer, config)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, val_loader
