"""CLI entry point for the full NVD CVE pull.

Wraps ``data.collect.collect_cve_full`` so it can be invoked from the
command line with explicit year ranges and a resume-friendly output path.

Examples:
    # Full pull, 1999-present (uses NVD_API_KEY if set)
    python scripts/collect_nvd_full.py

    # Pull a specific year range to a separate file
    python scripts/collect_nvd_full.py --start-year 2020 --end-year 2025 \\
        --output data/raw/cve_2020_2025.jsonl

    # Resume an interrupted pull (output_path is loaded as the resume
    # state, so just re-run the same command)
    python scripts/collect_nvd_full.py
"""

import argparse
import datetime
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.collect import collect_cve_full


def parse_args():
    p = argparse.ArgumentParser(description="Pull the full NVD CVE corpus.")
    p.add_argument("--output", default="data/raw/cve_full.jsonl",
                   help="Output JSONL path (resume-aware).")
    p.add_argument("--start-year", type=int, default=1999,
                   help="First year to query (inclusive).")
    p.add_argument("--end-year", type=int, default=None,
                   help="Last year (defaults to current UTC year).")
    p.add_argument("--page-size", type=int, default=2000,
                   help="resultsPerPage (NVD max 2000).")
    p.add_argument("--flush-every", type=int, default=5000,
                   help="Flush to disk after this many new records.")
    return p.parse_args()


def main():
    args = parse_args()
    end_year = args.end_year or datetime.datetime.utcnow().year
    print(f"NVD full pull: {args.start_year}..{end_year} -> {args.output}")
    collect_cve_full(
        output_path=args.output,
        start_year=args.start_year,
        end_year=end_year,
        page_size=args.page_size,
        flush_every=args.flush_every,
    )


if __name__ == "__main__":
    main()
