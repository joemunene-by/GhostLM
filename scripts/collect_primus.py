#!/usr/bin/env python3
"""Pull a subset of the PRIMUS cybersec corpus (Trend Micro AI Lab).

Two sources fit GhostLM's "from-scratch" constraint as additional pretrain
text:

- **Primus-Seed**: hand-curated cybersec text (security company sites,
  wikis, MITRE official dumps). Highest density. Pulled in full (16
  files, all small).
- **Primus-FineWeb**: TinyBERT-filtered cybersec subset of CommonCrawl
  FineWeb. 1601 files, 2.57B tokens total. Pulled as a configurable
  sample to fit M4 disk + training budget.

Output: `data/raw/primus_seed.jsonl` and `data/raw/primus_fineweb.jsonl`,
each one JSON record per line with our standard `{"id", "source", "text"}`
schema. Source field is `primus_seed` or `primus_fineweb` so the
existing build_chat_dataset.py mechanism can reference / exclude them.

License: ODC-BY (FineWeb subset) and MIT-style (Primus-Seed). Safe for
distribution as derived training data when we ship checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="Download PRIMUS subset for GhostLM pretrain")
    p.add_argument("--seed-out", default="data/raw/primus_seed.jsonl",
                   help="Output JSONL for full Primus-Seed (small, hand-curated)")
    p.add_argument("--fineweb-out", default="data/raw/primus_fineweb.jsonl",
                   help="Output JSONL for Primus-FineWeb sample")
    p.add_argument("--fineweb-files", type=int, default=10,
                   help="Number of Primus-FineWeb shards to pull "
                        "(out of 1601). 10 ~ 16M tokens, 60 ~ 100M tokens.")
    p.add_argument("--max-text-chars", type=int, default=8000,
                   help="Truncate over-long records to keep training-time "
                        "compute predictable.")
    p.add_argument("--min-text-chars", type=int, default=200,
                   help="Drop records shorter than this (mostly noise).")
    p.add_argument("--skip-seed", action="store_true",
                   help="Skip Primus-Seed download (already done).")
    p.add_argument("--skip-fineweb", action="store_true",
                   help="Skip Primus-FineWeb download (already done).")
    return p.parse_args()


def write_record(out_fh, idx: int, source: str, text: str, max_chars: int, min_chars: int) -> bool:
    """Append one record to the output file. Returns True if written."""
    text = (text or "").strip()
    if len(text) < min_chars:
        return False
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0]
    rec = {
        "id": f"{source}_{idx}",
        "source": source,
        "text": text,
    }
    out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return True


def stream_dataset(repo_id: str, split: str = "train") -> Iterable[dict]:
    """Stream records from HF dataset; uses streaming so we don't pay
    full-download disk cost up front."""
    from datasets import load_dataset
    ds = load_dataset(repo_id, split=split, streaming=True)
    yield from ds


def main() -> None:
    """Download Primus-Seed in full and Primus-FineWeb sample."""
    args = parse_args()
    seed_out = Path(args.seed_out)
    fineweb_out = Path(args.fineweb_out)
    seed_out.parent.mkdir(parents=True, exist_ok=True)
    fineweb_out.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_seed:
        print(f"=== Primus-Seed ===")
        print(f"  -> {seed_out}")
        seen = set()
        if seed_out.exists():
            with seed_out.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        if rec.get("id"):
                            seen.add(rec["id"])
            print(f"  resume: {len(seen)} records already done")

        out_fh = seed_out.open("a", encoding="utf-8", buffering=1)
        written = 0
        skipped = 0
        for i, rec in enumerate(stream_dataset("trendmicro-ailab/Primus-Seed")):
            rec_id = f"primus_seed_{i}"
            if rec_id in seen:
                continue
            text = rec.get("content") or rec.get("text") or ""
            if not text:
                text = "\n\n".join([str(v) for v in rec.values() if isinstance(v, str)])
            ok = write_record(out_fh, i, "primus_seed", text,
                              args.max_text_chars, args.min_text_chars)
            if ok:
                written += 1
            else:
                skipped += 1
            if (i + 1) % 1000 == 0:
                print(f"  [{i + 1}] written={written} skipped={skipped}")
        out_fh.close()
        print(f"Done. Wrote {written} Primus-Seed records (skipped {skipped})")

    if not args.skip_fineweb:
        print(f"\n=== Primus-FineWeb (sample {args.fineweb_files} shards) ===")
        print(f"  -> {fineweb_out}")
        seen = set()
        if fineweb_out.exists():
            with fineweb_out.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        if rec.get("id"):
                            seen.add(rec["id"])
            print(f"  resume: {len(seen)} records already done")

        # FineWeb has 1601 shards; we use streaming and break after
        # processing approximately --fineweb-files shards' worth of
        # records. Average shard is ~10K records, so stop at
        # fineweb_files * 10000 records.
        target = args.fineweb_files * 10_000
        out_fh = fineweb_out.open("a", encoding="utf-8", buffering=1)
        written = 0
        skipped = 0
        for i, rec in enumerate(stream_dataset("trendmicro-ailab/Primus-FineWeb")):
            if i >= target:
                break
            rec_id = f"primus_fineweb_{i}"
            if rec_id in seen:
                continue
            text = rec.get("text") or rec.get("content") or ""
            if not text:
                text = "\n\n".join([str(v) for v in rec.values() if isinstance(v, str)])
            ok = write_record(out_fh, i, "primus_fineweb", text,
                              args.max_text_chars, args.min_text_chars)
            if ok:
                written += 1
            else:
                skipped += 1
            if (i + 1) % 5000 == 0:
                print(f"  [{i + 1}/{target}] written={written} skipped={skipped}")
        out_fh.close()
        print(f"Done. Wrote {written} Primus-FineWeb records (skipped {skipped})")


if __name__ == "__main__":
    main()
