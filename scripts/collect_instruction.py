#!/usr/bin/env python3
"""Collect a general instruction-following sample for the instruction domain.

A generalist base model needs to have seen instruction -> response structure
in pretraining, not just free web text. GhostLM's corpus has none of this
outside the small hand-written chat SFT seeds, so the `instruction` domain
(budgeted in the generalist corpus profile) starts empty. This collector
fills it with a broad, permissively-licensed instruction dataset.

Default source: ``databricks/databricks-dolly-15k`` (CC BY-SA 3.0), a
15K-record human-written set spanning eight task categories (open/closed
QA, brainstorming, classification, creative writing, summarization,
information extraction, general QA). Streamed via HF ``datasets``.

Each record is rendered as a natural instruction/response passage so the
base model learns the mapping by adjacency:

    <instruction>

    <context, if present>

    <response>

Output: ``data/raw/instruction.jsonl`` with ``{id, source, category,
license, text}``; ``source == "instruction"`` so it maps to the
``instruction`` training domain (see ``data.collect.SOURCE_DOMAINS``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="Collect general instruction data")
    p.add_argument("--out", default="data/raw/instruction.jsonl")
    p.add_argument("--repo", default="databricks/databricks-dolly-15k",
                   help="HF dataset repo (instruction/response schema)")
    p.add_argument("--split", default="train")
    p.add_argument("--max-records", type=int, default=15_000)
    p.add_argument("--min-chars", type=int, default=40)
    p.add_argument("--max-chars", type=int, default=8_000)
    p.add_argument("--license", default="CC-BY-SA-3.0",
                   help="License tag recorded on each output record")
    return p.parse_args()


def format_record(instruction: str, context: str, response: str) -> str:
    """Render an instruction/context/response triple as a pretrain passage.

    Context is included only when present. Returns an empty string if either
    the instruction or the response is missing, so the caller can skip it.
    """
    instruction = (instruction or "").strip()
    context = (context or "").strip()
    response = (response or "").strip()
    if not instruction or not response:
        return ""
    parts = [instruction]
    if context:
        parts.append(context)
    parts.append(response)
    return "\n\n".join(parts)


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

    print(f"streaming {args.repo}:{args.split} (target {args.max_records})...")
    ds = load_dataset(args.repo, split=args.split, streaming=True)

    out_fh = out.open("a", encoding="utf-8", buffering=1)
    written = 0
    skipped_short = 0
    truncated = 0
    skipped_dup = 0

    for i, rec in enumerate(ds):
        rec_id = f"instruction_{i}"
        if rec_id in seen:
            skipped_dup += 1
            continue
        text = format_record(
            rec.get("instruction", ""),
            rec.get("context", ""),
            rec.get("response", ""),
        )
        if len(text) < args.min_chars:
            skipped_short += 1
            continue
        if len(text) > args.max_chars:
            text = text[: args.max_chars].rsplit(" ", 1)[0]
            truncated += 1

        out_fh.write(json.dumps({
            "id": rec_id,
            "source": "instruction",
            "category": rec.get("category", ""),
            "license": args.license,
            "text": text,
        }, ensure_ascii=False) + "\n")
        written += 1

        if written >= args.max_records:
            break
        if written % 3000 == 0:
            print(f"  written={written} short={skipped_short} truncated={truncated}")

    out_fh.close()
    print(f"\nDone. Wrote {written} instruction records to {out}")
    if skipped_short:
        print(f"  skipped {skipped_short} too-short")
    if skipped_dup:
        print(f"  skipped {skipped_dup} already on disk")
    if truncated:
        print(f"  truncated {truncated} long records to {args.max_chars} chars")


if __name__ == "__main__":
    main()
