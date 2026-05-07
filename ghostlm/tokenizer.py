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
    # Chat role markers (added in v0.5 chat-tuning) — IDs appended after the
    # original four so pre-chat checkpoints can be expanded by 3 rows rather
    # than reshuffled.
    USER = "<|ghost_user|>"
    ASSISTANT = "<|ghost_assistant|>"
    END = "<|ghost_end|>"

    def __init__(self):
        """Initialize the GhostTokenizer with the GPT-2 BPE encoding.

        Loads the tiktoken gpt2 encoding and assigns special token IDs
        beyond the standard vocabulary for begin-of-sequence, end-of-sequence,
        padding, unknown, and chat role markers.
        """
        self._encoder = tiktoken.get_encoding("gpt2")
        self._vocab_size = self._encoder.n_vocab

        # Assign special token IDs beyond the base vocabulary
        self._special_tokens = {
            self.BOS: self._vocab_size,
            self.EOS: self._vocab_size + 1,
            self.PAD: self._vocab_size + 2,
            self.UNK: self._vocab_size + 3,
            self.USER: self._vocab_size + 4,
            self.ASSISTANT: self._vocab_size + 5,
            self.END: self._vocab_size + 6,
        }

        # Reverse mapping for quick lookup
        self._id_to_special = {v: k for k, v in self._special_tokens.items()}

    @property
    def vocab_size(self) -> int:
        """Return the effective vocabulary size including special tokens.

        Returns:
            Total vocabulary size (base vocab + 7 special tokens).
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

    def encode_chat(self, turns: List[dict]) -> tuple:
        """Encode a multi-turn chat conversation with role markers and a loss mask.

        Format: <|ghost_user|>{content}<|ghost_end|><|ghost_assistant|>{content}<|ghost_end|>...
        The loss mask is 1 on assistant content tokens and the assistant's trailing
        <|ghost_end|> (so the model learns to stop), and 0 everywhere else (user
        prompts and role markers themselves).

        Args:
            turns: List of {"role": "user"|"assistant", "content": str} dicts,
                strictly alternating starting with "user".

        Returns:
            Tuple (token_ids, loss_mask) — same length, both lists of int.
        """
        user_id = self._special_tokens[self.USER]
        assistant_id = self._special_tokens[self.ASSISTANT]
        end_id = self._special_tokens[self.END]

        ids: List[int] = []
        mask: List[int] = []

        for turn in turns:
            role = turn["role"]
            content_ids = self._encoder.encode(turn["content"], allowed_special="all")
            if role == "user":
                ids.append(user_id)
                mask.append(0)
                ids.extend(content_ids)
                mask.extend([0] * len(content_ids))
                ids.append(end_id)
                mask.append(0)
            elif role == "assistant":
                ids.append(assistant_id)
                mask.append(0)
                ids.extend(content_ids)
                mask.extend([1] * len(content_ids))
                ids.append(end_id)
                mask.append(1)
            else:
                raise ValueError(f"Unknown role: {role!r}")

        return ids, mask

    def format_chat_prompt(self, turns: List[dict]) -> List[int]:
        """Encode a chat history and append <|ghost_assistant|> ready for generation.

        Used at inference: feed the resulting token ids to the model; it should
        generate the assistant's reply followed by <|ghost_end|>.

        Args:
            turns: List of {"role": "user"|"assistant", "content": str}, ending
                with a "user" turn (the prompt awaiting a reply).

        Returns:
            List of token IDs ending in the assistant role marker.
        """
        ids, _ = self.encode_chat(turns)
        ids.append(self._special_tokens[self.ASSISTANT])
        return ids

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


class GhostTokenizerV05:
    """v0.5 tokenizer — domain-trained 32K BPE via HuggingFace tokenizers.

    Drop-in replacement for ``GhostTokenizer`` with the same API surface
    (`encode`, `decode`, `encode_chat`, `format_chat_prompt`, `vocab_size`,
    `_special_tokens`) so the existing dataset / trainer / chat code paths
    work unchanged. The seven GhostLM special tokens land at the start of
    the vocab (IDs 0-6) — different from v0.4's tail placement, but the
    chat-format machinery only cares about the name -> ID mapping, not
    the absolute IDs.

    Trained by ``scripts/train_tokenizer.py``. Load via ``from_file``.
    """

    # Same special-token names as the legacy tokenizer.
    BOS = GhostTokenizer.BOS
    EOS = GhostTokenizer.EOS
    PAD = GhostTokenizer.PAD
    UNK = GhostTokenizer.UNK
    USER = GhostTokenizer.USER
    ASSISTANT = GhostTokenizer.ASSISTANT
    END = GhostTokenizer.END

    def __init__(self, path: str):
        """Load the trained tokenizer.json file."""
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(path)
        self._vocab_size = self._tok.get_vocab_size()
        self._special_tokens = {
            name: self._tok.token_to_id(name)
            for name in (
                self.BOS, self.EOS, self.PAD, self.UNK,
                self.USER, self.ASSISTANT, self.END,
            )
        }
        self._id_to_special = {v: k for k, v in self._special_tokens.items()}

    @classmethod
    def from_file(cls, path: str) -> "GhostTokenizerV05":
        """Alias for the constructor — matches the GhostTokenizer.load shape."""
        return cls(path)

    @property
    def vocab_size(self) -> int:
        """Return the total vocabulary size (~32,000 by default)."""
        return self._vocab_size

    def _special_token_ids(self) -> set:
        """Return the set of special-token integer IDs."""
        return set(self._special_tokens.values())

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode text to token IDs (matches GhostTokenizer.encode)."""
        ids = self._tok.encode(text).ids
        if add_bos:
            ids = [self._special_tokens[self.BOS]] + ids
        if add_eos:
            ids = ids + [self._special_tokens[self.EOS]]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode token IDs to text (matches GhostTokenizer.decode)."""
        if skip_special:
            specials = self._special_token_ids()
            ids = [i for i in ids if i not in specials]
        return self._tok.decode(ids)

    def encode_batch(self, texts: List[str], add_bos: bool = False, add_eos: bool = False) -> List[List[int]]:
        """Encode a batch of texts."""
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

    def encode_chat(self, turns: List[dict]) -> tuple:
        """Build (token_ids, loss_mask) for a chat conversation.

        Mirrors ``GhostTokenizer.encode_chat`` exactly — only the underlying
        BPE differs.
        """
        user_id = self._special_tokens[self.USER]
        assistant_id = self._special_tokens[self.ASSISTANT]
        end_id = self._special_tokens[self.END]
        ids: List[int] = []
        mask: List[int] = []
        for turn in turns:
            role = turn["role"]
            content_ids = self._tok.encode(turn["content"]).ids
            if role == "user":
                ids.append(user_id); mask.append(0)
                ids.extend(content_ids); mask.extend([0] * len(content_ids))
                ids.append(end_id); mask.append(0)
            elif role == "assistant":
                ids.append(assistant_id); mask.append(0)
                ids.extend(content_ids); mask.extend([1] * len(content_ids))
                ids.append(end_id); mask.append(1)
            else:
                raise ValueError(f"Unknown role: {role!r}")
        return ids, mask

    def format_chat_prompt(self, turns: List[dict]) -> List[int]:
        """Build an inference prompt ending in <|ghost_assistant|>."""
        ids, _ = self.encode_chat(turns)
        ids.append(self._special_tokens[self.ASSISTANT])
        return ids

    def to_tensor(self, ids: List[int], device: str = "cpu") -> torch.Tensor:
        """Convert ids to a (1, T) torch.LongTensor."""
        return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    def __len__(self) -> int:
        """Return the vocab size."""
        return self._vocab_size

    def __repr__(self) -> str:
        """Concise summary."""
        return f"GhostTokenizerV05(vocab_size={self._vocab_size}, special_tokens={len(self._special_tokens)})"


class GhostTokenizerV1:
    """v1 tokenizer: 32K BPE retrained on the v1.0 corpus by
    ``scripts/train_v1_bpe.py``. Differs from V05 in the special-token
    naming scheme plus four extra tool-use tags:

      Specials in v1 (per ``special_tokens_map.json``):
        <|endoftext|>, <|pad|>, <|bos|>, <|eos|>,
        <|ghost_user|>, <|ghost_assistant|>, <|ghost_end|>,
        <|tool_call|>, <|/tool_call|>,
        <|tool_response|>, <|/tool_response|>

    The four tool tags exist so SFT data produced by
    ``scripts/distill_tool_use.py`` (bet 1 from
    ``docs/differentiation.md``) tokenizes as exactly four atomic IDs,
    not 8-12 BPE fragments, which keeps the loss-mask boundaries
    stable across tokenizer choices.

    The bet 3 compression report on a 99-record sample showed v1 BPE
    is +1.6% denser than GPT-2's 50K BPE on the v1.0 corpus, well
    below the +25-35% hypothesis. As of 2026-05-08 this class exists
    primarily so the artifact is reachable from training code if a
    downstream benchmark proves it earns its keep; ghost-base default
    remains GPT-2 BPE."""

    BOS = "<|bos|>"
    EOS = "<|eos|>"
    PAD = "<|pad|>"
    UNK = "<|endoftext|>"
    USER = "<|ghost_user|>"
    ASSISTANT = "<|ghost_assistant|>"
    END = "<|ghost_end|>"
    # New for v1 (bet 1 tool-use SFT format).
    TOOL_CALL = "<|tool_call|>"
    TOOL_CALL_END = "<|/tool_call|>"
    TOOL_RESPONSE = "<|tool_response|>"
    TOOL_RESPONSE_END = "<|/tool_response|>"

    def __init__(self, path: str):
        """Load tokenizer.json (and sibling special_tokens_map.json if present)."""
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(path)
        self._vocab_size = self._tok.get_vocab_size()
        # Bind every named special. token_to_id returns None for missing
        # tokens; we want a hard error so misconfigured tokenizer files
        # don't silently mask bugs at training time.
        names = (self.BOS, self.EOS, self.PAD, self.UNK,
                 self.USER, self.ASSISTANT, self.END,
                 self.TOOL_CALL, self.TOOL_CALL_END,
                 self.TOOL_RESPONSE, self.TOOL_RESPONSE_END)
        self._special_tokens = {}
        for name in names:
            tid = self._tok.token_to_id(name)
            if tid is None:
                raise ValueError(
                    f"v1 tokenizer at {path} missing special token {name!r}; "
                    f"retrain via scripts/train_v1_bpe.py"
                )
            self._special_tokens[name] = tid
        self._id_to_special = {v: k for k, v in self._special_tokens.items()}

    @classmethod
    def from_file(cls, path: str) -> "GhostTokenizerV1":
        """Constructor alias matching V05.from_file."""
        return cls(path)

    @property
    def vocab_size(self) -> int:
        """Total vocab size (32000 by default)."""
        return self._vocab_size

    def _special_token_ids(self) -> set:
        """Set of all special-token integer IDs."""
        return set(self._special_tokens.values())

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode text -> ids (matches V05.encode shape)."""
        ids = self._tok.encode(text).ids
        if add_bos:
            ids = [self._special_tokens[self.BOS]] + ids
        if add_eos:
            ids = ids + [self._special_tokens[self.EOS]]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """Decode ids -> text (matches V05.decode shape)."""
        if skip_special:
            specials = self._special_token_ids()
            ids = [i for i in ids if i not in specials]
        return self._tok.decode(ids)

    def encode_batch(self, texts: List[str], add_bos: bool = False, add_eos: bool = False) -> List[List[int]]:
        """Batch encode."""
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

    def encode_chat(self, turns: List[dict]) -> tuple:
        """Build (token_ids, loss_mask) for chat (matches V05.encode_chat)."""
        user_id = self._special_tokens[self.USER]
        assistant_id = self._special_tokens[self.ASSISTANT]
        end_id = self._special_tokens[self.END]
        ids: List[int] = []
        mask: List[int] = []
        for turn in turns:
            role = turn["role"]
            content_ids = self._tok.encode(turn["content"]).ids
            if role == "user":
                ids.append(user_id); mask.append(0)
                ids.extend(content_ids); mask.extend([0] * len(content_ids))
                ids.append(end_id); mask.append(0)
            elif role == "assistant":
                ids.append(assistant_id); mask.append(0)
                ids.extend(content_ids); mask.extend([1] * len(content_ids))
                ids.append(end_id); mask.append(1)
            else:
                raise ValueError(f"Unknown role: {role!r}")
        return ids, mask

    def format_chat_prompt(self, turns: List[dict]) -> List[int]:
        """Inference-side prompt ending in <|ghost_assistant|>."""
        ids, _ = self.encode_chat(turns)
        ids.append(self._special_tokens[self.ASSISTANT])
        return ids

    def to_tensor(self, ids: List[int], device: str = "cpu") -> torch.Tensor:
        """Convert ids -> (1, T) torch.LongTensor."""
        return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    def __len__(self) -> int:
        """Return vocab size."""
        return self._vocab_size

    def __repr__(self) -> str:
        """Concise summary."""
        return f"GhostTokenizerV1(vocab_size={self._vocab_size}, special_tokens={len(self._special_tokens)})"


def _looks_like_v1(path: Path) -> bool:
    """Heuristic: v1 tokenizers have a sibling ``special_tokens_map.json``
    that lists the four tool-use tags. v0.5 tokenizers don't."""
    sibling = path.parent / "special_tokens_map.json"
    if not sibling.exists():
        return False
    try:
        meta = json.loads(sibling.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    extras = meta.get("additional_special_tokens", [])
    return "<|tool_call|>" in extras


def load_tokenizer(path: Optional[str] = None):
    """Factory: pick the right tokenizer backend.

    Resolution order:
      - No path (or path missing): legacy GhostTokenizer (GPT-2 BPE).
      - Path resolves AND looks like a v1 artifact (has the four
        tool-use tags in sibling special_tokens_map.json): V1 backend.
      - Otherwise: V05 backend.

    Train code paths can call this with ``config.tokenizer_path`` and
    not care about the backend.
    """
    if path and Path(path).exists():
        if _looks_like_v1(Path(path)):
            return GhostTokenizerV1(path)
        return GhostTokenizerV05(path)
    return GhostTokenizer()
