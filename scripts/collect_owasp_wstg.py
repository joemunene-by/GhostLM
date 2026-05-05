#!/usr/bin/env python3
"""Ingest the OWASP Web Security Testing Guide (WSTG) markdown corpus.

WSTG (CC BY-SA 4.0) is the canonical web pentesting methodology, ~194
markdown documents covering specific test cases (e.g. "Testing for
SQL Injection", "Testing for XSS", "Testing for Broken Authentication").
Fact density is high and the writing is procedural / how-to, which
complements the descriptive register of NVD / MITRE / CWE.

Source: https://github.com/OWASP/wstg (`document/` subtree)
Output: ``data/raw/owasp_wstg.jsonl`` with the standard
``{"id", "source", "text"}`` schema. Source is ``owasp_wstg``.

Each markdown becomes one record. Title from H1 if present, else
filename-derived.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="Ingest OWASP WSTG markdown corpus")
    p.add_argument("--src", required=True,
                   help="Path to the wstg repo root")
    p.add_argument("--out", default="data/raw/owasp_wstg.jsonl")
    p.add_argument("--min-chars", type=int, default=300)
    p.add_argument("--max-chars", type=int, default=15000)
    return p.parse_args()


FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def extract_title_and_body(md: str, fallback: str) -> tuple[str, str]:
    """Strip YAML frontmatter, pull H1 as title."""
    md = FRONTMATTER_RE.sub("", md, count=1)
    m = H1_RE.search(md)
    if m:
        title = m.group(1).strip()
        body = md[m.end():].strip()
    else:
        title = fallback
        body = md.strip()
    return title, body


def main() -> None:
    """Walk all .md files under src/ and emit one record each."""
    args = parse_args()
    src = Path(args.src).expanduser()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md_files = list(src.rglob("*.md"))
    if not md_files:
        raise SystemExit(f"No markdown files found under {src}")

    out_fh = out_path.open("w", encoding="utf-8")
    written = 0
    skipped_short = 0
    truncated = 0

    for md_path in sorted(md_files):
        # Skip non-content markdowns (READMEs, contributing docs)
        rel_lower = str(md_path.relative_to(src)).lower()
        if any(skip in rel_lower for skip in ("readme.md", "contributing", "license", "code-of-conduct")):
            continue

        raw = md_path.read_text(encoding="utf-8", errors="ignore")
        fallback = md_path.stem.replace("_", " ").replace("-", " ")
        title, body = extract_title_and_body(raw, fallback)
        text = f"{title}\n\n{body}".strip()

        if len(text) < args.min_chars:
            skipped_short += 1
            continue
        if len(text) > args.max_chars:
            text = text[: args.max_chars].rsplit("\n\n", 1)[0]
            truncated += 1

        rec = {
            "id": str(md_path.relative_to(src)).replace("/", "_").replace(".md", ""),
            "source": "owasp_wstg",
            "text": text,
            "title": title,
        }
        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        written += 1

    out_fh.close()
    print(f"Wrote {written} OWASP WSTG records to {out_path}")
    if skipped_short:
        print(f"  Skipped {skipped_short} short/non-content files")
    if truncated:
        print(f"  Truncated {truncated} long docs to {args.max_chars} chars")


if __name__ == "__main__":
    main()
