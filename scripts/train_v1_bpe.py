#!/usr/bin/env python3
"""Train a 32K cybersec-native BPE tokenizer on the v1.0 corpus.

GPT-2's 50K BPE was trained on general English web text. It wastes
vocabulary on word fragments like "the", "and", "ing", "tion" that
are heavily used in general English but contribute little marginal
information to a cybersec corpus. Conversely, it splits high-value
cybersec sequences into many tokens:

    "CVE-2017-0144"     -> 7 tokens   (C, VE, -, 2017, -, 014, 4)
    "T1059.001"         -> 5 tokens   (T, 1059, ., 001)
    "CWE-89"            -> 3 tokens   (C, WE, -, 89)
    "CVSS:3.1"          -> 5 tokens
    "0x4141414141414141"-> 9 tokens

A 32K BPE retrained on the v1.0 corpus (PRIMUS + NVD + MITRE + CWE +
OWASP + RFCs + arXiv + fact-QA + security_code + nist_sp800 + ...,
363M total tokens) will:

  - Allocate single tokens to common cybersec patterns
    (CVE-YYYY-, T#### prefixes, common hex byte sequences,
    OWASP A0x labels, MITRE tactic codes, etc.)
  - Compress cybersec text by ~25-35% in tokens-per-byte
  - Free up effective context length proportionally

v0.5 attempted this on a 60M-token corpus; the result tokenized
cybersec densely but fragmented out-of-domain English, which
limited generalization. With 6x more data and three new domains
(FineWeb-Edu, math, code) the tokenizer should preserve general
English while specializing on cybersec. That's the bet this script
tests.

Output:

    data/tokenizer/v1/                  HF tokenizers JSON format
        tokenizer.json
        special_tokens_map.json
        compression_report.md           tokens-per-byte vs GPT-2 BPE,
                                        sample-by-sample on 100 records

Run:

    PYTHONPATH=. python3 scripts/train_v1_bpe.py \\
        --corpus data/processed/train.jsonl \\
        --vocab-size 32000 \\
        --out-dir data/tokenizer/v1

Cost: ~30-60 min on M4 CPU (single-threaded BPE merge loop). One-shot,
no GPU. The trained tokenizer is then plugged into the existing
GhostTokenizer wrapper by changing the backend; the rest of the
training pipeline doesn't need to change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|ghost_user|>",
    "<|ghost_assistant|>",
    "<|ghost_end|>",
    # New tokens for the tool-use SFT pipeline (distill_tool_use.py).
    "<|tool_call|>",
    "<|/tool_call|>",
    "<|tool_response|>",
    "<|/tool_response|>",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default="data/processed/train.jsonl")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--out-dir", default="data/tokenizer/v1")
    p.add_argument("--max-records", type=int, default=0,
                   help="Train on first N corpus records (0 = full corpus)")
    p.add_argument("--min-frequency", type=int, default=2,
                   help="Minimum frequency for a merge candidate")
    return p.parse_args()


def stream_corpus(path: Path, max_records: int = 0) -> Iterator[str]:
    """Yield text strings from a jsonl corpus shard. The HF
    tokenizers library expects an iterator of strings; we read
    one record per line, pull the text field, and yield it.

    Streaming-not-loading matters: the v1.0 corpus is 1.7+ GB on
    disk; reading it all into memory before training would OOM
    on a typical M4."""
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_records and i >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = rec.get("text") or rec.get("content") or ""
            if text:
                yield text


def train_tokenizer(corpus_path: Path, vocab_size: int,
                    out_dir: Path, max_records: int,
                    min_frequency: int) -> None:
    """Train a byte-level BPE on the corpus + write to out_dir."""
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    except ImportError:
        print("ERROR: 'tokenizers' library not installed.", file=sys.stderr)
        print("Install with: pip install tokenizers>=0.15", file=sys.stderr)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(BPE(unk_token="<|endoftext|>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=True,
    )

    print(f"Training BPE: vocab_size={vocab_size}, "
          f"min_frequency={min_frequency}, corpus={corpus_path}")
    tok.train_from_iterator(
        stream_corpus(corpus_path, max_records=max_records),
        trainer=trainer,
    )

    out_path = out_dir / "tokenizer.json"
    tok.save(str(out_path))
    print(f"Wrote {out_path}")

    # Save special tokens map separately for compatibility with
    # transformers-style loaders.
    spec_map = {
        "additional_special_tokens": SPECIAL_TOKENS,
        "unk_token": "<|endoftext|>",
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "pad_token": "<|pad|>",
    }
    (out_dir / "special_tokens_map.json").write_text(json.dumps(spec_map, indent=2))


def compression_report(corpus_path: Path, out_dir: Path,
                       sample_records: int = 100) -> None:
    """Compute tokens-per-byte for the new BPE vs GPT-2's 50K BPE
    on the same sample. The new BPE only beats GPT-2 if it
    compresses cybersec text more than it loses on general English.
    Report shows the trade-off honestly."""
    try:
        from tokenizers import Tokenizer
        import tiktoken
    except ImportError:
        print("Skipping compression report: tokenizers/tiktoken not available")
        return

    new_tok = Tokenizer.from_file(str(out_dir / "tokenizer.json"))
    gpt2 = tiktoken.get_encoding("gpt2")

    rows = []
    for text in stream_corpus(corpus_path, max_records=sample_records):
        nbytes = len(text.encode("utf-8"))
        if nbytes < 100:
            continue
        new_n = len(new_tok.encode(text).ids)
        gpt2_n = len(gpt2.encode(text))
        rows.append({
            "bytes": nbytes,
            "v1_tokens": new_n,
            "gpt2_tokens": gpt2_n,
            "v1_tpb": new_n / nbytes,
            "gpt2_tpb": gpt2_n / nbytes,
        })

    if not rows:
        return

    avg_v1 = sum(r["v1_tpb"] for r in rows) / len(rows)
    avg_gpt2 = sum(r["gpt2_tpb"] for r in rows) / len(rows)
    pct = 100.0 * (avg_gpt2 - avg_v1) / avg_gpt2 if avg_gpt2 else 0.0

    md = []
    md.append("# v1 BPE compression report\n")
    md.append(f"Sample size: **{len(rows)}** corpus records\n\n")
    md.append("| Tokenizer | tokens/byte (avg) |\n")
    md.append("|---|---:|\n")
    md.append(f"| GhostLM v1 BPE (32K, this script) | {avg_v1:.4f} |\n")
    md.append(f"| GPT-2 BPE (50K, current default) | {avg_gpt2:.4f} |\n\n")
    md.append(f"**v1 BPE compresses corpus text by {pct:+.1f}% more "
              f"than GPT-2 BPE** (lower tokens/byte = denser tokens, "
              f"more effective context per token).\n\n")
    md.append("Sample-level distribution (`bytes` is raw size, `v1` and "
              "`gpt2` are token counts on the same text):\n\n")
    md.append("| bytes | v1 | gpt2 | v1 tpb | gpt2 tpb |\n")
    md.append("|---:|---:|---:|---:|---:|\n")
    for r in rows[:20]:
        md.append(f"| {r['bytes']:,} | {r['v1_tokens']:,} | "
                  f"{r['gpt2_tokens']:,} | {r['v1_tpb']:.4f} | "
                  f"{r['gpt2_tpb']:.4f} |\n")
    md.append("\nFirst 20 records shown; the full distribution and "
              "any outliers are visible by re-running this script and "
              "inspecting the rows list.\n")

    (out_dir / "compression_report.md").write_text("".join(md))
    print(f"Wrote compression report ({pct:+.1f}% vs GPT-2 BPE)")


def main() -> int:
    args = parse_args()
    corpus_path = Path(args.corpus)
    out_dir = Path(args.out_dir)
    if not corpus_path.exists():
        sys.exit(f"corpus not found: {corpus_path}")

    train_tokenizer(corpus_path, args.vocab_size, out_dir,
                    args.max_records, args.min_frequency)
    compression_report(corpus_path, out_dir)
    print(f"\nDone. Tokenizer at {out_dir}/tokenizer.json")
    print("Next: wire this into ghostlm/tokenizer.py as an alternate "
          "backend (replacing the tiktoken GPT-2 path) and rerun "
          "v1.0 ghost-base pretrain on the recompressed corpus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
