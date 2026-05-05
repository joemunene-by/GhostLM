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
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path


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
    req = urllib.request.Request(
        api, headers={"User-Agent": "GhostLM-Top10/0.6",
                      "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        items = json.loads(resp.read().decode("utf-8"))
    out = []
    for item in items:
        if item.get("name", "").endswith(".md") and item.get("download_url"):
            out.append((item["name"], item["download_url"]))
    return out


def fetch_markdown(url: str) -> str:
    """Fetch one markdown file as text."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "GhostLM-Top10/0.6"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="ignore")


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
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"  listing OWASP Top 10 {args.year}/{args.lang} markdowns...")
    files = list_markdown_urls(args.year, args.lang)
    print(f"  found {len(files)} markdown files")

    out_fh = out_path.open("w", encoding="utf-8")
    written = 0
    skipped_short = 0
    truncated = 0
    failed = 0

    for fname, url in files:
        try:
            raw = fetch_markdown(url)
        except Exception as e:
            print(f"  {fname}: fetch error {e}")
            failed += 1
            time.sleep(args.request_delay)
            continue

        fallback = fname.replace(".md", "").replace("_", " ").replace("-", " ")
        title, body = extract_title_and_body(raw, fallback)
        text = f"{title}\n\n{body}".strip()

        if len(text) < args.min_chars:
            skipped_short += 1
            time.sleep(args.request_delay)
            continue
        if len(text) > args.max_chars:
            text = text[:args.max_chars].rsplit("\n\n", 1)[0]
            truncated += 1

        rec_id = f"OWASP-Top10-{args.year}-{fname.replace('.md', '')}"
        rec = {
            "id": rec_id,
            "source": "owasp_top10",
            "text": text,
            "title": title,
        }
        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        written += 1
        time.sleep(args.request_delay)

    out_fh.close()
    print(f"Wrote {written} OWASP Top 10 records to {out_path}")
    if skipped_short:
        print(f"  Skipped {skipped_short} too-short")
    if truncated:
        print(f"  Truncated {truncated} long docs to {args.max_chars} chars")
    if failed:
        print(f"  Failed {failed}")


if __name__ == "__main__":
    main()
