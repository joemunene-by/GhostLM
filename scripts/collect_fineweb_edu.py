#!/usr/bin/env python3
"""Stream FineWeb-Edu samples from HuggingFace into the corpus.

The v0.9 corpus is cybersec-only. To make GhostLM able to do general
language tasks (write coherent paragraphs about non-security topics,
follow instructions on novel inputs, reason with everyday vocabulary)
we need a serving of high-quality general-language text. FineWeb-Edu
(HuggingFaceFW/fineweb-edu) is a 1.3T-token educational subset of
CommonCrawl, classifier-filtered for textbook-style content. It's
the closest thing to a permissively-licensed Cosmopedia at scale.

This puller streams a configurable shard count and writes
``data/raw/fineweb_edu.jsonl``. Defaults are conservative for M4
disk: 50K records (~50M tokens) is a good ghost-base seed.

License: ODC-BY (same family as Primus-FineWeb), safe to redistribute
as derived training data when shipping checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="Stream FineWeb-Edu into JSONL")
    p.add_argument("--out", default="data/raw/fineweb_edu.jsonl")
    p.add_argument("--repo", default="HuggingFaceFW/fineweb-edu",
                   help="HF dataset repo")
    p.add_argument("--name", default="sample-10BT",
                   help="Dataset config name (sample-10BT, sample-100BT, default)")
    p.add_argument("--max-records", type=int, default=50_000,
                   help="Number of records to pull")
    p.add_argument("--min-chars", type=int, default=400)
    p.add_argument("--max-chars", type=int, default=10_000)
    p.add_argument("--min-edu-score", type=float, default=3.0,
                   help="Skip records below this educational-quality score")
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
        print(f"  resume: {len(seen)} records already on disk")

    print(f"streaming {args.repo}:{args.name} (target {args.max_records})...")
    ds = load_dataset(args.repo, name=args.name, split="train", streaming=True)

    out_fh = out.open("a", encoding="utf-8", buffering=1)
    written = 0
    skipped_short = 0
    skipped_score = 0
    skipped_dup = 0
    truncated = 0
    seen_idx = -1
    for i, rec in enumerate(ds):
        seen_idx = i
        rec_id = f"fineweb_edu_{i}"
        if rec_id in seen:
            skipped_dup += 1
            if written + skipped_dup >= args.max_records:
                break
            continue

        text = rec.get("text") or ""
        score = rec.get("score") or rec.get("edu_score") or 0.0
        if score < args.min_edu_score:
            skipped_score += 1
            continue
        if len(text) < args.min_chars:
            skipped_short += 1
            continue
        if len(text) > args.max_chars:
            text = text[: args.max_chars].rsplit(" ", 1)[0]
            truncated += 1

        out_fh.write(json.dumps({
            "id": rec_id,
            "source": "fineweb_edu",
            "text": text,
            "score": float(score),
        }, ensure_ascii=False) + "\n")
        written += 1

        if written >= args.max_records:
            break
        if (written) % 5000 == 0 and written > 0:
            print(f"  [{seen_idx + 1}] written={written} short={skipped_short} "
                  f"low_score={skipped_score} truncated={truncated}")

    out_fh.close()
    print(f"\nDone. Wrote {written} FineWeb-Edu records to {out}")
    if skipped_short:
        print(f"  skipped {skipped_short} too-short")
    if skipped_score:
        print(f"  skipped {skipped_score} below score {args.min_edu_score}")
    if skipped_dup:
        print(f"  skipped {skipped_dup} already-on-disk")
    if truncated:
        print(f"  truncated {truncated} long records to {args.max_chars} chars")


if __name__ == "__main__":
    main()
