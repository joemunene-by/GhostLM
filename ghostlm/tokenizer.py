"""GhostLM tokenizer — wraps tiktoken's GPT-2 BPE tokenizer with cybersecurity-aware utilities."""

import json
import os
from pathlib import Path
from typing import List, Optional, Union

import tiktoken
import torch


class GhostTokenizer:
    """Wrapper around tiktoken GPT-2 BPE tokenizer with GhostLM utilities.

    Provides encoding, decoding, batching, padding, and text chunking
    utilities tailored for cybersecurity document processing.
    """

    # Special token strings
    BOS = "<|ghost_bos|>"
    EOS = "<|ghost_eos|>"
    PAD = "<|ghost_pad|>"
    UNK = "<|ghost_unk|>"

    def __init__(self):
        """Initialize the GhostTokenizer with the GPT-2 BPE encoding.

        Loads the tiktoken gpt2 encoding and assigns special token IDs
        beyond the standard vocabulary for begin-of-sequence, end-of-sequence,
        padding, and unknown tokens.
        """
        self._encoder = tiktoken.get_encoding("gpt2")
        self._vocab_size = self._encoder.n_vocab

        # Assign special token IDs beyond the base vocabulary
        self._special_tokens = {
            self.BOS: self._vocab_size,
            self.EOS: self._vocab_size + 1,
            self.PAD: self._vocab_size + 2,
            self.UNK: self._vocab_size + 3,
        }

        # Reverse mapping for quick lookup
        self._id_to_special = {v: k for k, v in self._special_tokens.items()}

    @property
    def vocab_size(self) -> int:
        """Return the effective vocabulary size including special tokens.

        Returns:
            Total vocabulary size (base vocab + 4 special tokens).
        """
        return self._vocab_size + len(self._special_tokens)

    def _special_token_ids(self) -> set:
        """Return a set of all special token IDs.

        Returns:
            Set of integer token IDs reserved for special tokens.
        """
        return set(self._special_tokens.values())

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode a text string into a list of token IDs.

        Args:
            text: Input text to encode.
            add_bos: If True, prepend the BOS token ID.
            add_eos: If True, append the EOS token ID.

        Returns:
            List of integer token IDs.
        """
        ids = self._encoder.encode(text, allowed_special="all")

        if add_bos:
            ids = [self._special_tokens[self.BOS]] + ids
        if add_eos:
            ids = ids + [self._special_tokens[self.EOS]]

        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode a list of token IDs back into a text string.

        Args:
            ids: List of integer token IDs to decode.
            skip_special: If True, filter out special token IDs before decoding.

        Returns:
            Decoded text string.
        """
        if skip_special:
            special_ids = self._special_token_ids()
            ids = [i for i in ids if i not in special_ids]

        return self._encoder.decode(ids)

    def encode_batch(self, texts: List[str], add_bos: bool = False, add_eos: bool = False) -> List[List[int]]:
        """Encode a list of text strings into lists of token IDs.

        Args:
            texts: List of input text strings to encode.
            add_bos: If True, prepend BOS token ID to each sequence.
            add_eos: If True, append EOS token ID to each sequence.

        Returns:
            List of lists of integer token IDs, one per input text.
        """
        return [self.encode(text, add_bos=add_bos, add_eos=add_eos) for text in texts]

    def to_tensor(self, ids: List[int], device: str = "cpu") -> torch.Tensor:
        """Convert a list of token IDs to a PyTorch tensor.

        Args:
            ids: List of integer token IDs.
            device: Target device for the tensor (default: "cpu").

        Returns:
            torch.LongTensor of shape (1, len(ids)).
        """
        return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    def pad_batch(self, batch: List[List[int]], pad_left: bool = False) -> tuple:
        """Pad a batch of token ID lists to the same length.

        Pads all sequences in the batch to the length of the longest sequence
        using the PAD token ID. Returns both the padded tensor and an attention
        mask indicating real tokens (1) vs padding (0).

        Args:
            batch: List of token ID lists, each potentially different length.
            pad_left: If True, pad on the left side (useful for generation).
                      If False, pad on the right side (default).

        Returns:
            Tuple of (padded_tensor, attention_mask) where:
                - padded_tensor: torch.LongTensor of shape (batch_size, max_len)
                - attention_mask: torch.LongTensor of shape (batch_size, max_len)
        """
        max_len = max(len(seq) for seq in batch)
        pad_id = self._special_tokens[self.PAD]

        padded = []
        masks = []

        for seq in batch:
            pad_count = max_len - len(seq)
            if pad_left:
                padded_seq = [pad_id] * pad_count + seq
                mask = [0] * pad_count + [1] * len(seq)
            else:
                padded_seq = seq + [pad_id] * pad_count
                mask = [1] * len(seq) + [0] * pad_count

            padded.append(padded_seq)
            masks.append(mask)

        padded_tensor = torch.tensor(padded, dtype=torch.long)
        mask_tensor = torch.tensor(masks, dtype=torch.long)

        return padded_tensor, mask_tensor

    def chunk_text(self, text: str, chunk_size: int = 1024, overlap: int = 64) -> List[List[int]]:
        """Encode text and split into overlapping token chunks.

        Useful for processing long cybersecurity documents that exceed
        the model's context length. Overlapping chunks preserve context
        continuity across boundaries.

        Args:
            text: Input text string to chunk.
            chunk_size: Maximum number of tokens per chunk.
            overlap: Number of overlapping tokens between consecutive chunks.

        Returns:
            List of token ID lists, each of length at most chunk_size.
        """
        ids = self.encode(text)

        if len(ids) <= chunk_size:
            return [ids]

        chunks = []
        stride = chunk_size - overlap

        for i in range(0, len(ids), stride):
            chunk = ids[i : i + chunk_size]
            chunks.append(chunk)
            if i + chunk_size >= len(ids):
                break

        return chunks

    def save(self, path: str) -> None:
        """Save tokenizer metadata to a JSON file.

        Stores vocabulary size, special token strings, and their assigned
        IDs so the tokenizer can be reconstructed later.

        Args:
            path: File path to save the JSON metadata.
        """
        metadata = {
            "vocab_size": self._vocab_size,
            "special_tokens": self._special_tokens,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "GhostTokenizer":
        """Load a GhostTokenizer from saved metadata JSON.

        Reconstructs the tokenizer by reading special token assignments
        from the saved metadata file.

        Args:
            path: File path to the saved JSON metadata.

        Returns:
            GhostTokenizer instance loaded with the saved configuration.
        """
        with open(path, "r") as f:
            metadata = json.load(f)

        tokenizer = cls()

        # Restore special token mappings
        tokenizer._special_tokens = {k: int(v) for k, v in metadata["special_tokens"].items()}
        tokenizer._id_to_special = {v: k for k, v in tokenizer._special_tokens.items()}

        return tokenizer

    def __len__(self) -> int:
        """Return the effective vocabulary size.

        Returns:
            Integer count of tokens including special tokens.
        """
        return self.vocab_size

    def __repr__(self) -> str:
        """Return a concise string representation of the tokenizer.

        Returns:
            String like: GhostTokenizer(vocab_size=50261, special_tokens=4)
        """
        return f"GhostTokenizer(vocab_size={self.vocab_size}, special_tokens={len(self._special_tokens)})"
