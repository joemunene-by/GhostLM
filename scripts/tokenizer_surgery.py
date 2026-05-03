#!/usr/bin/env python3
"""Tokenizer surgery — add small-talk anchor tokens to the v0.5 BPE.

The v0.5 BPE was trained on the cybersec corpus only. Common chat words
that the chat dataset relies on ("hi", "hello", "thanks", "you", etc.)
get split character-level — no gradient signal connects them to
assistant-mode behavior. This is the dominant cause of the v0.5 chat
plateau per the research-agent diagnosis.

Rather than retraining the whole BPE (which would force a 24h
re-pretrain), we add ~30 anchor tokens to the existing tokenizer
vocab as added (non-special, non-segmentation-blocking) tokens. The
model's input embedding is then expanded by N rows, initialized with
the *average* of the constituent old-token embeddings — so "hi" starts
out close to wherever "h" + "i" would have lived, then SFT can move
it. This is the warm-start trick from the SmolLM2 retrospective and
the DepthUpscaling paper (Komatsuzaki et al., May 2025).

Output:
- ``data/tokenizer_v05_surgery/tokenizer.json`` — new BPE with anchors
- Optional: a script-side helper to expand a model checkpoint's
  embedding to match the new vocab size, called from finetune_chat.py
  via the same expand_token_embedding mechanism we already use for
  the chat-role tokens.

After surgery: re-SFT v0.5 with ``--tokenizer
data/tokenizer_v05_surgery/tokenizer.json``. The chat records pass
through the new BPE → "hi" becomes a single id → model gets clean
gradient on chat-shape tokens.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List


# Curated 32-token chat-anchor list. High-frequency English words from the
# small_talk.jsonl seed and the chat assistant turns. Kept tight — every
# token here costs a row in the embedding matrix and an opportunity for
# BPE to over-eagerly match a substring (e.g. "hi" inside "hide").
ANCHOR_TOKENS: List[str] = [
    # Greetings / acknowledgments — most common 1-3 char words
    " hi", " hello", " hey", " thanks", " thank", " ok", " yes", " no",
    " sure", " bye",
    # Identity-question pronouns / verbs that reach the assistant turn
    " I", " I'm", " you", " you're", " your", " what", " who",
    " where", " when", " why", " how",
    # Top assistant verbs from the small_talk dataset
    " am", " are", " is", " was", " can", " can't", " do", " don't",
    " have", " want",
]


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="Add chat anchors to v0.5 BPE")
    p.add_argument("--in-tokenizer", default="data/tokenizer_v05/tokenizer.json")
    p.add_argument("--out-dir", default="data/tokenizer_v05_surgery")
    p.add_argument("--anchors", nargs="*", default=None,
                   help="Override the default anchor list")
    return p.parse_args()


def main() -> None:
    """Run the surgery and dump the new tokenizer."""
    from tokenizers import Tokenizer

    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Start from the existing v0.5 tokenizer.
    tokenizer = Tokenizer.from_file(args.in_tokenizer)
    base_vocab_size = tokenizer.get_vocab_size()
    print(f"Starting vocab size: {base_vocab_size:,}")

    anchors = list(args.anchors) if args.anchors else list(ANCHOR_TOKENS)

    # Filter out any anchors that already happen to be a single token (e.g.
    # because the corpus contained them often enough). No need to add those.
    new_anchors: List[str] = []
    for tok in anchors:
        ids = tokenizer.encode(tok).ids
        if len(ids) == 1:
            print(f"  skip (already single-token): {tok!r}")
            continue
        new_anchors.append(tok)

    print(f"Adding {len(new_anchors)} anchor tokens:")
    for tok in new_anchors:
        ids = tokenizer.encode(tok).ids
        print(f"  {tok!r}  was {len(ids)} ids: {ids}")

    # add_tokens returns the number actually added (some may already exist).
    added = tokenizer.add_tokens(new_anchors)
    print(f"Added {added} tokens (resulting vocab: {tokenizer.get_vocab_size():,})")

    # Save the modified tokenizer.
    out_path = out_dir / "tokenizer.json"
    tokenizer.save(str(out_path))
    print(f"Saved: {out_path}")

    # Also save a manifest of what changed — useful for the embedding
    # expansion logic on the model side.
    new_ids = [tokenizer.token_to_id(tok) for tok in new_anchors]
    manifest = {
        "base_vocab_size": base_vocab_size,
        "new_vocab_size": tokenizer.get_vocab_size(),
        "added_tokens": [
            {"text": tok, "new_id": tid, "was_ids": Tokenizer.from_file(args.in_tokenizer).encode(tok).ids}
            for tok, tid in zip(new_anchors, new_ids)
        ],
    }
    (out_dir / "surgery_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Manifest: {out_dir / 'surgery_manifest.json'}")

    # Quick after-surgery encoding test.
    print()
    print("Sanity check — anchors should now encode to 1 token:")
    for tok in new_anchors[:8]:
        ids = tokenizer.encode(tok).ids
        marker = "✓" if len(ids) == 1 else f"✗ ({len(ids)} ids)"
        print(f"  {tok!r}: {marker}")


if __name__ == "__main__":
    main()
