"""CLI entry point for the arXiv full-text PDF collector.

Wraps ``data.collect.collect_arxiv_full_text``. Reads the abstracts
already pulled by ``collect_security_papers`` (default
``data/raw/papers.jsonl``) and re-fetches each as a PDF, extracting
body text via pymupdf.

A 2000-paper full-text pull at the default 1 req/sec is ~35 minutes
of wall clock plus a few minutes of extraction. The output is saved
to ``data/raw/arxiv_full.jsonl`` and the merge step picks it up as
a new ``arxiv_full`` source distinct from the abstract source.

Usage:
    python scripts/collect_arxiv_full.py
    python scripts/collect_arxiv_full.py --max-records 100 --request-delay 0.5
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.collect import collect_arxiv_full_text


def parse_args():
    p = argparse.ArgumentParser(
        description="Fetch full-text PDFs for arXiv abstracts and extract text."
    )
    p.add_argument(
        "--output", default="data/raw/arxiv_full.jsonl",
        help="Output JSONL path. Resumes if it already exists.",
    )
    p.add_argument(
        "--abstracts", default="data/raw/papers.jsonl",
        help=(
            "Path to the abstracts JSONL whose ids are re-fetched as "
            "PDFs. Run scripts/collect_security_papers / make data first "
            "to seed this list."
        ),
    )
    p.add_argument(
        "--max-records", type=int, default=500,
        help="Total record cap for this run.",
    )
    p.add_argument(
        "--min-chars", type=int, default=1500,
        help=(
            "Drop papers whose extracted text is shorter than this. "
            "Scanned-image PDFs and pymupdf-unfriendly papers fall here."
        ),
    )
    p.add_argument(
        "--max-chars", type=int, default=60000,
        help=(
            "Truncate extracted text to this many chars. Title prepended "
            "from the abstract record survives at the start."
        ),
    )
    p.add_argument(
        "--request-delay", type=float, default=1.0,
        help="Seconds between PDF fetches (be polite to arXiv's CDN).",
    )
    p.add_argument(
        "--request-timeout", type=int, default=60,
        help="Per-request HTTP timeout.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    collect_arxiv_full_text(
        output_path=args.output,
        max_records=args.max_records,
        abstract_input=args.abstracts,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        request_delay=args.request_delay,
        request_timeout=args.request_timeout,
    )


if __name__ == "__main__":
    main()
