#!/usr/bin/env python3
"""Ingest OWASP Top 10 (2021) — the canonical web-app risk list.

The Top 10 is OWASP's most cited document: ten categories of the most
critical web-application security risks, each with description,
attack scenarios, prevention, references. Cribbed by every appsec
training program. CC BY-SA 4.0.

Source: https://github.com/OWASP/Top10 (2021/docs/en/*.md)
Output: ``data/raw/owasp_top10.jsonl`` with the standard
``{"id", "source", "text"}`` schema. Source is ``owasp_top10``.

We fetch each markdown file individually via raw.githubusercontent.com
because git-clone of the full repo has been unreliable on flaky
networks. Exposes ``--year`` to allow ingesting other Top-10 cycles
(2017, 2017, etc.) by changing the path.

Reference usage of ``collect_common`` (fetch + JsonlWriter); new
collectors should follow this shape.
"""

from __future__ import annotations

import argparse
import re

from collect_common import JsonlWriter, http_get_json, http_get_text


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="Ingest OWASP Top 10 markdown corpus")
    p.add_argument("--year", default="2021",
                   help="Top-10 cycle year (2021, 2017, ...)")
    p.add_argument("--lang", default="en")
    p.add_argument("--out", default="data/raw/owasp_top10.jsonl")
    p.add_argument("--request-delay", type=float, default=0.5)
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--max-chars", type=int, default=12000)
    return p.parse_args()


H1_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


def list_markdown_urls(year: str, lang: str) -> list[tuple[str, str]]:
    """Return [(filename, raw_url), ...] for the year/lang docs folder."""
    api = (f"https://api.github.com/repos/OWASP/Top10/"
           f"contents/{year}/docs/{lang}")
    items = http_get_json(api, headers={"Accept": "application/vnd.github+json"})
    out = []
    for item in items:
        if item.get("name", "").endswith(".md") and item.get("download_url"):
            out.append((item["name"], item["download_url"]))
    return out


def extract_title_and_body(md: str, fallback: str) -> tuple[str, str]:
    """Pull H1 as title, return (title, body)."""
    m = H1_RE.search(md)
    if m:
        title = m.group(1).strip()
        body = md[m.end():].strip()
    else:
        title = fallback
        body = md.strip()
    return title, body


def main() -> None:
    """Fetch all year/lang markdowns, emit JSONL."""
    args = parse_args()

    print(f"  listing OWASP Top 10 {args.year}/{args.lang} markdowns...")
    files = list_markdown_urls(args.year, args.lang)
    print(f"  found {len(files)} markdown files")

    with JsonlWriter(args.out, source="owasp_top10",
                     min_chars=args.min_chars, max_chars=args.max_chars,
                     request_delay=args.request_delay) as out:
        for fname, url in files:
            try:
                raw = http_get_text(url)
            except Exception as e:
                out.count_failure(f"{fname}: fetch error {e}")
                continue

            fallback = fname.replace(".md", "").replace("_", " ").replace("-", " ")
            title, body = extract_title_and_body(raw, fallback)
            text = f"{title}\n\n{body}".strip()

            rec_id = f"OWASP-Top10-{args.year}-{fname.replace('.md', '')}"
            out.write(rec_id=rec_id, text=text, title=title)


if __name__ == "__main__":
    main()
