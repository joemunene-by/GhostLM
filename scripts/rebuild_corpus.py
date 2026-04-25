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


def select_corpus_sources(raw_dir, prefer_full_nvd=True):
    """Pick the JSONL files to feed into the merge from a raw/ directory.

    When both ``cve.jsonl`` (the v0.3.0 baseline corpus) and
    ``cve_full.jsonl`` (the post-Phase-3 NVD pull) are present, only one
    should go into the merge. By default ``cve_full.jsonl`` wins;
    ``prefer_full_nvd=False`` forces the legacy file for reproducibility.

    Args:
        raw_dir: Path to the directory containing ``*.jsonl`` raw sources.
        prefer_full_nvd: If True (default) and ``cve_full.jsonl`` exists,
            it is selected and ``cve.jsonl`` is excluded. If False (or
            ``cve_full.jsonl`` is absent), ``cve.jsonl`` is selected.

    Returns:
        ``(sources, cve_choice)`` where ``sources`` is a list of selected
        JSONL paths as strings and ``cve_choice`` is the Path to the CVE
        file that won the selection (or ``None`` if neither exists).
    """
    raw_dir = Path(raw_dir)
    candidates = sorted(raw_dir.glob("*.jsonl"))
    cve_full = raw_dir / "cve_full.jsonl"
    cve_legacy = raw_dir / "cve.jsonl"

    use_full = prefer_full_nvd and cve_full.exists()
    sources = []
    for p in candidates:
        if p.name == "cve.jsonl" and use_full:
            continue  # superseded by cve_full
        if p.name == "cve_full.jsonl" and not use_full:
            continue
        sources.append(str(p))

    cve_choice = cve_full if use_full else (cve_legacy if cve_legacy.exists() else None)
    return sources, cve_choice


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

    sources, cve_choice = select_corpus_sources(raw, prefer_full_nvd=args.prefer_full_nvd)
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
