#!/usr/bin/env python3
"""Ingest OWASP CheatSheetSeries markdown into a GhostLM JSONL corpus.

The OWASP CheatSheetSeries (CC BY-SA 4.0) is 138 markdown documents
covering specific security topics with concrete defensive guidance. The
fact density is high (much higher than CTF writeups) and the writing is
in a "developer reference" register that complements the existing
CVE / MITRE / CTFtime / arxiv mix.

Each cheat sheet becomes one record. Title is preserved. The first H1
is stripped from the body since the title is already in the record key.
Frontmatter (if present) is dropped.

Output: ``data/raw/owasp_cheatsheets.jsonl`` with the standard
``{"id", "source", "text"}`` schema. Source is ``owasp_cheatsheets``.

Usage:
    python scripts/collect_owasp_cheatsheets.py \\
        --src ~/Desktop/joemunene-repos/CheatSheetSeries
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="Ingest OWASP CheatSheetSeries")
    p.add_argument("--src", required=True,
                   help="Path to the CheatSheetSeries repo root")
    p.add_argument("--out", default="data/raw/owasp_cheatsheets.jsonl")
    p.add_argument("--min-chars", type=int, default=300,
                   help="Drop docs shorter than this (mostly stubs)")
    p.add_argument("--max-chars", type=int, default=12000,
                   help="Truncate over-long docs to keep training-time "
                        "compute predictable")
    return p.parse_args()


FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def extract_title_and_body(md: str, fallback_name: str) -> tuple[str, str]:
    """Return (title, body) from a markdown document."""
    md = FRONTMATTER_RE.sub("", md, count=1)
    m = H1_RE.search(md)
    if m:
        title = m.group(1).strip()
        body = md[m.end():].strip()
    else:
        title = fallback_name
        body = md.strip()
    return title, body


def main() -> None:
    """Walk the CheatSheets directory and emit one record per file."""
    args = parse_args()
    src = Path(args.src).expanduser()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cheatsheets_dir = src / "cheatsheets"
    if not cheatsheets_dir.exists():
        raise SystemExit(f"No cheatsheets dir at {cheatsheets_dir}")

    out_fh = out_path.open("w", encoding="utf-8")
    written = 0
    skipped_short = 0
    truncated = 0

    for md_path in sorted(cheatsheets_dir.glob("*.md")):
        raw = md_path.read_text(encoding="utf-8", errors="ignore")
        fallback_name = md_path.stem.replace("_", " ").replace("-", " ")
        title, body = extract_title_and_body(raw, fallback_name)

        text = f"{title}\n\n{body}".strip()
        if len(text) < args.min_chars:
            skipped_short += 1
            continue
        if len(text) > args.max_chars:
            text = text[: args.max_chars].rsplit("\n\n", 1)[0]
            truncated += 1

        rec = {
            "id": md_path.stem,
            "source": "owasp_cheatsheets",
            "text": text,
            "title": title,
        }
        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        written += 1

    out_fh.close()
    print(f"Wrote {written} OWASP cheatsheets to {out_path}")
    if skipped_short:
        print(f"  Skipped {skipped_short} short stubs")
    if truncated:
        print(f"  Truncated {truncated} over-long docs to {args.max_chars} chars")


if __name__ == "__main__":
    main()
