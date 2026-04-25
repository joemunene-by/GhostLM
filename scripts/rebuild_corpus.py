"""Rebuild the train/val splits after a corpus pull.

Re-runs the merge step against the current ``data/raw/*.jsonl`` files
without touching the collectors. Designed for the Phase 3 post-NVD-pull
workflow: once ``data/raw/cve_full.jsonl`` is on disk (from
``scripts/collect_nvd_full.py``), this swaps it in as the CVE source for
the merge and writes fresh ``data/processed/{train,val}.jsonl``.

The deterministic-hash split is preserved — identical texts always land
in the same bucket, so re-running this is idempotent.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.collect import merge_datasets


def parse_args():
    p = argparse.ArgumentParser(description="Rebuild train/val splits from data/raw/.")
    p.add_argument("--raw-dir", default="data/raw", help="Directory containing raw JSONL files.")
    p.add_argument("--output", default="data/processed/train.jsonl",
                   help="Output train path (val is sibling).")
    p.add_argument("--val-split", type=float, default=0.05, help="Validation fraction.")
    p.add_argument(
        "--prefer-full-nvd",
        action="store_true",
        default=True,
        help="If data/raw/cve_full.jsonl exists, use it instead of cve.jsonl. Default true.",
    )
    p.add_argument(
        "--no-prefer-full-nvd",
        dest="prefer_full_nvd",
        action="store_false",
        help="Force using the legacy cve.jsonl even if cve_full.jsonl exists.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    raw = Path(args.raw_dir)
    if not raw.is_dir():
        sys.exit(f"raw dir not found: {raw}")

    # Discover all raw sources and pick which CVE file to feed in.
    candidates = sorted(raw.glob("*.jsonl"))
    cve_full = raw / "cve_full.jsonl"
    cve_legacy = raw / "cve.jsonl"

    use_full = args.prefer_full_nvd and cve_full.exists()
    sources = []
    for p in candidates:
        if p.name == "cve.jsonl" and use_full:
            continue  # superseded by cve_full
        if p.name == "cve_full.jsonl" and not use_full:
            continue
        sources.append(str(p))

    cve_choice = cve_full if use_full else (cve_legacy if cve_legacy.exists() else None)
    print("Rebuild corpus")
    print(f"  raw dir:    {raw}")
    print(f"  CVE source: {cve_choice}")
    print(f"  sources ({len(sources)}):")
    for s in sources:
        print(f"    - {s}")

    if not sources:
        sys.exit("no raw sources found — did the collectors run?")

    merge_datasets(
        input_paths=sources,
        output_path=args.output,
        val_split=args.val_split,
    )


if __name__ == "__main__":
    main()
