#!/usr/bin/env python3
"""Score any tokenizer.json against any corpus subset.

The bet 3 hypothesis was 'cybersec-native BPE compresses cybersec
text 25-35% denser than GPT-2 BPE.' The first measurement (on the
mixed v1.0 corpus) landed at +1.6%, far below that target. The
question this script answers: if we train v1 BPE on cybersec-only
text, does it crush cybersec text more decisively, or is the
overlap with GPT-2's English vocab just structurally tight?

To answer that we need to score the same tokenizer on:
  - cybersec-only text  (drop fineweb_edu / math_reasoning)
  - general text only   (only fineweb_edu / math_reasoning)
  - mixed (full corpus)

Plus the same comparisons against GPT-2's tiktoken BPE as a fixed
baseline. ``train_v1_bpe.py``'s built-in compression_report only
covers the corpus it trains on, so we need a standalone scorer.

Run:

    PYTHONPATH=. python3 scripts/score_tokenizer.py \\
        --tokenizer data/tokenizer/v1_cyber/tokenizer.json \\
        --corpus data/processed/train.jsonl \\
        --max-records 500 \\
        --filter-source primus_seed,nvd,cwe,owasp_top10

Outputs the avg tokens-per-byte for the supplied tokenizer and
GPT-2 BPE on the same records, plus the per-record distribution
sliced by ``source`` so the by-domain story is visible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set


def stream_records(path: Path, max_records: int = 0,
                   keep_sources: Optional[Set[str]] = None,
                   drop_sources: Optional[Set[str]] = None,
                   ) -> Iterator[Dict[str, str]]:
    """Yield records from a jsonl file with optional source filtering.

    ``keep_sources`` (allowlist) and ``drop_sources`` (denylist) can
    both be set; allowlist wins. Records lacking a non-empty ``text``
    field are skipped silently."""
    n_yielded = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = rec.get("source", "unknown")
            if keep_sources is not None and src not in keep_sources:
                continue
            if drop_sources and src in drop_sources:
                continue
            text = rec.get("text") or rec.get("content") or ""
            if not text:
                continue
            yield {"source": src, "text": text}
            n_yielded += 1
            if max_records and n_yielded >= max_records:
                break


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer", required=True,
                   help="Path to tokenizer.json (HuggingFace tokenizers format)")
    p.add_argument("--corpus", required=True,
                   help="JSONL with text field per record")
    p.add_argument("--max-records", type=int, default=500,
                   help="Cap records scored. Default 500 is enough "
                        "for stable averages without burning cycles.")
    p.add_argument("--filter-source",
                   help="Comma-separated source allowlist; only records "
                        "with these source values are scored")
    p.add_argument("--drop-source",
                   help="Comma-separated source denylist; records with "
                        "these source values are skipped")
    p.add_argument("--no-gpt2", action="store_true",
                   help="Skip the GPT-2 BPE baseline (avoid the "
                        "tiktoken dependency cost)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    tok_path = Path(args.tokenizer)
    if not tok_path.exists():
        sys.exit(f"tokenizer not found: {tok_path}")
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        sys.exit(f"corpus not found: {corpus_path}")

    keep = set(s.strip() for s in args.filter_source.split(",")
               if s.strip()) if args.filter_source else None
    drop = set(s.strip() for s in args.drop_source.split(",")
               if s.strip()) if args.drop_source else None

    try:
        from tokenizers import Tokenizer
    except ImportError:
        sys.exit("install with: pip install tokenizers>=0.15")
    tok = Tokenizer.from_file(str(tok_path))

    gpt2 = None
    if not args.no_gpt2:
        try:
            import tiktoken
            gpt2 = tiktoken.get_encoding("gpt2")
        except ImportError:
            print("(tiktoken not installed; skipping GPT-2 baseline)")

    # Per-source aggregation.
    by_source: Dict[str, Dict[str, int]] = {}
    n = 0
    for rec in stream_records(corpus_path, args.max_records, keep, drop):
        nbytes = len(rec["text"].encode("utf-8"))
        if nbytes < 100:
            continue
        new_n = len(tok.encode(rec["text"]).ids)
        gpt2_n = len(gpt2.encode(rec["text"])) if gpt2 else 0
        agg = by_source.setdefault(rec["source"], {
            "n": 0, "bytes": 0, "tok": 0, "gpt2": 0,
        })
        agg["n"] += 1
        agg["bytes"] += nbytes
        agg["tok"] += new_n
        agg["gpt2"] += gpt2_n
        n += 1

    if not by_source:
        sys.exit("no records scored; check --filter-source / --corpus")

    print(f"Tokenizer:  {tok_path}")
    print(f"Corpus:     {corpus_path}")
    print(f"Records:    {n}")
    print(f"Filter:     keep={keep or '*'} drop={drop or '-'}")
    print()
    header = "| source | n | bytes | tok t/b | gpt2 t/b | win % |"
    sep    = "|---|---:|---:|---:|---:|---:|"
    print(header)
    print(sep)
    total_b = total_t = total_g = 0
    for src in sorted(by_source.keys()):
        a = by_source[src]
        tpb = a["tok"] / a["bytes"]
        gpb = a["gpt2"] / a["bytes"] if a["gpt2"] else 0.0
        win = 100 * (gpb - tpb) / gpb if gpb else 0.0
        total_b += a["bytes"]
        total_t += a["tok"]
        total_g += a["gpt2"]
        print(f"| {src} | {a['n']} | {a['bytes']:,} | "
              f"{tpb:.4f} | {gpb:.4f} | {win:+.1f}% |")
    overall_tpb = total_t / total_b
    overall_gpb = total_g / total_b if total_g else 0.0
    overall_win = 100 * (overall_gpb - overall_tpb) / overall_gpb \
                  if overall_gpb else 0.0
    print(f"| **OVERALL** | **{n}** | **{total_b:,}** | "
          f"**{overall_tpb:.4f}** | **{overall_gpb:.4f}** | "
          f"**{overall_win:+.1f}%** |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
