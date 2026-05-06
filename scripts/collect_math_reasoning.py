#!/usr/bin/env python3
"""Stream a math / reasoning sample for ghost-base CoT capability.

The v0.9 corpus has zero math content. Even a small share of
mathematical reasoning text noticeably lifts chain-of-thought
emergence in the literature (SmolLM2 ablation showed ~3-5 pp lift on
GSM8K from a 5%-of-tokens math share). For ghost-base v1.0 we want
the model to be able to follow numeric / logical reasoning chains in
cybersec contexts (CVSS scoring, port-number arithmetic,
exploitability calculations) so a math sample is in scope.

Default source: ``open-web-math/open-web-math``. ODC-BY licensed,
~13B-token web subset filtered for math content. Streamed via HF
datasets so we don't pay full-download cost.

Output: ``data/raw/math_reasoning.jsonl`` with the standard
``{"id", "source", "text"}`` schema. Source field is
``math_reasoning``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="Stream math/reasoning sample")
    p.add_argument("--out", default="data/raw/math_reasoning.jsonl")
    p.add_argument("--repo", default="open-web-math/open-web-math",
                   help="HF dataset repo (open-web-math, EleutherAI/proof-pile-2, etc.)")
    p.add_argument("--config", default=None,
                   help="Optional dataset config (e.g. 'algebraic-stack' for proof-pile-2)")
    p.add_argument("--split", default="train")
    p.add_argument("--max-records", type=int, default=20_000,
                   help="Number of records to pull (~20M tokens at default chars)")
    p.add_argument("--min-chars", type=int, default=500)
    p.add_argument("--max-chars", type=int, default=10_000)
    return p.parse_args()


def main() -> None:
    """Stream and write."""
    from datasets import load_dataset
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    seen: set = set()
    if out.exists():
        with out.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("id"):
                        seen.add(rec["id"])
        print(f"  resume: {len(seen)} records on disk")

    print(f"streaming {args.repo}{(' [' + args.config + ']') if args.config else ''}:{args.split} "
          f"(target {args.max_records})...")
    if args.config:
        ds = load_dataset(args.repo, name=args.config, split=args.split, streaming=True)
    else:
        ds = load_dataset(args.repo, split=args.split, streaming=True)

    out_fh = out.open("a", encoding="utf-8", buffering=1)
    written = 0
    skipped_short = 0
    truncated = 0
    skipped_dup = 0

    for i, rec in enumerate(ds):
        rec_id = f"math_reasoning_{i}"
        if rec_id in seen:
            skipped_dup += 1
            if written + skipped_dup >= args.max_records:
                break
            continue

        text = rec.get("text") or rec.get("content") or rec.get("body") or ""
        if not text:
            text = "\n\n".join(str(v) for v in rec.values()
                               if isinstance(v, str) and len(v) > 50)
        if len(text) < args.min_chars:
            skipped_short += 1
            continue
        if len(text) > args.max_chars:
            text = text[: args.max_chars].rsplit(" ", 1)[0]
            truncated += 1

        out_fh.write(json.dumps({
            "id": rec_id,
            "source": "math_reasoning",
            "text": text,
        }, ensure_ascii=False) + "\n")
        written += 1

        if written >= args.max_records:
            break
        if written % 2000 == 0:
            print(f"  [{i + 1}] written={written} short={skipped_short} truncated={truncated}")

    out_fh.close()
    print(f"\nDone. Wrote {written} math-reasoning records to {out}")
    if skipped_short:
        print(f"  skipped {skipped_short} too-short")
    if skipped_dup:
        print(f"  skipped {skipped_dup} already on disk")
    if truncated:
        print(f"  truncated {truncated} long records to {args.max_chars} chars")


if __name__ == "__main__":
    main()
