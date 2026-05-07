#!/usr/bin/env python3
"""Collect CISA Cybersecurity Advisories from cisa.gov.

Distinct from CISA KEV (which is the Known-Exploited Vulnerabilities
catalog already covered by ``collect_cisa_kev.py``). CSAs are detailed
technical bulletins that CISA publishes about specific threat actor
TTPs, joint advisories with FBI/NSA/MI5, ICS-CERT advisories about
industrial control system vulnerabilities, and emergency directives.
The textual content (5-50 paragraphs per advisory) is much richer
than the KEV catalog rows.

Source: the CISA news-events feed at cisa.gov plus the all-advisories
RSS. RSS gives recent activity; the news-events index covers the
back catalog. The collector pulls both and dedupes by URL.

License posture: cisa.gov content is US Government work, public
domain. No attribution required, but per-record source_url is
preserved anyway for traceability.

Output: ``data/raw/cisa_advisories.jsonl`` with the standard
``{"id", "source", "text"}`` schema. Source field is
``cisa_advisories``.

Run:

    PYTHONPATH=. python3 scripts/collect_cisa_advisories.py \\
        --max-records 500
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional, Set, Tuple


# CISA publishes via Drupal; the all-advisories XML feed is the
# canonical source. Multiple aliases work; we keep two as a fallback
# in case Drupal renames one.
FEEDS: List[Tuple[str, str]] = [
    ("cisa-advisories",   "https://www.cisa.gov/cybersecurity-advisories/all.xml"),
    ("cisa-news-events",  "https://www.cisa.gov/news-events/cybersecurity-advisories/all.xml"),
    ("cisa-ics-cert",     "https://www.cisa.gov/uscert/ics/advisories/advisories.xml"),
]

USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "GhostLM-corpus-collector/0.9.3 Safari/537.36")

CONTENT_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "pre", "code", "blockquote", "td", "th"}
SKIP_TAGS = {"nav", "script", "style", "aside", "header", "footer", "form", "noscript", "iframe", "svg"}


class CISAArticleExtractor(HTMLParser):
    """Body extractor tuned for cisa.gov's Drupal templates. Captures
    paragraph-level text inside the article body, drops nav chrome
    and the ubiquitous 'Share This' / 'Subscribe' boilerplate that
    bookends every advisory."""

    def __init__(self):
        super().__init__()
        self._buf: List[str] = []
        self._skip = 0
        self._capture: List[bool] = []

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self._skip += 1
        if tag in CONTENT_TAGS:
            self._capture.append(True)
            if tag.startswith("h"):
                self._buf.append("\n\n")

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS and self._skip > 0:
            self._skip -= 1
        if tag in CONTENT_TAGS and self._capture:
            self._capture.pop()
            if tag in {"p", "pre", "blockquote", "li", "td"} or tag.startswith("h"):
                self._buf.append("\n\n")

    def handle_data(self, data):
        if self._skip > 0:
            return
        if self._capture and self._capture[-1]:
            self._buf.append(data)

    def get_text(self) -> str:
        text = "".join(self._buf)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Drop the boilerplate footer CISA appends to most pages.
        for boil in (
            "Please share your thoughts.",
            "We recently updated our anonymous product survey",
            "Subscribe to receive email alerts",
        ):
            i = text.find(boil)
            if i > 0:
                text = text[:i].rstrip()
                break
        return text.strip()


def http_get(url: str, timeout: int = 30) -> Optional[str]:
    """GET with a real User-Agent. cisa.gov 403s bare urllib otherwise."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = resp.headers.get("Content-Type", "")
            charset = "utf-8"
            m = re.search(r"charset=([\w-]+)", ctype, re.I)
            if m:
                charset = m.group(1)
            return resp.read().decode(charset, errors="replace")
    except Exception as e:
        print(f"    fetch failed: {url[:90]}: {type(e).__name__}: {e}")
        return None


def parse_rss(xml: str) -> List[Tuple[str, str, Optional[str]]]:
    """Pull (title, link, published) from RSS or Atom XML. Same shape
    as the parser in collect_vendor_research; copied here so the
    collectors are independent."""
    out: List[Tuple[str, str, Optional[str]]] = []
    for m in re.finditer(r"<item\b.*?</item>", xml, re.S | re.I):
        block = m.group(0)
        title = _tag(block, "title")
        link = _tag(block, "link")
        pub = _tag(block, "pubDate")
        if link:
            out.append((title or "", link, pub))
    for m in re.finditer(r"<entry\b.*?</entry>", xml, re.S | re.I):
        block = m.group(0)
        title = _tag(block, "title")
        link_m = re.search(r'<link[^>]*\bhref=["\']([^"\']+)["\']', block, re.I)
        link = link_m.group(1) if link_m else None
        pub = _tag(block, "updated") or _tag(block, "published")
        if link:
            out.append((title or "", link, pub))
    return out


def _tag(block: str, tag: str) -> Optional[str]:
    m = re.search(fr"<{tag}\b[^>]*>(.*?)</{tag}>", block, re.S | re.I)
    if not m:
        return None
    raw = m.group(1).strip()
    raw = re.sub(r"^<!\[CDATA\[(.*?)\]\]>$", r"\1", raw, flags=re.S)
    return html.unescape(raw).strip()


def load_seen_urls(path: Path) -> Set[str]:
    seen: Set[str] = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                u = rec.get("source_url") or rec.get("url")
                if u:
                    seen.add(u)
            except json.JSONDecodeError:
                continue
    return seen


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/raw/cisa_advisories.jsonl")
    p.add_argument("--max-per-feed", type=int, default=200,
                   help="Cap entries pulled per feed (avoids hammering CISA)")
    p.add_argument("--max-records", type=int, default=0,
                   help="Total cap across all feeds; 0 = no cap")
    p.add_argument("--min-words", type=int, default=120,
                   help="Drop advisories shorter than this after extraction")
    p.add_argument("--request-delay", type=float, default=1.5,
                   help="Polite delay between HTTP requests, seconds")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = load_seen_urls(out_path)
    print(f"Output:    {out_path}")
    print(f"Resuming:  {len(seen)} URLs already in output")
    print(f"Feeds:     {len(FEEDS)}")
    print()

    total_written = 0
    fh = out_path.open("a", encoding="utf-8")
    try:
        for feed_name, feed_url in FEEDS:
            if args.max_records and total_written >= args.max_records:
                break
            print(f"=== {feed_name} ({feed_url}) ===")
            xml = http_get(feed_url)
            if not xml:
                print("    feed fetch failed; skipping")
                continue
            entries = parse_rss(xml)
            if not entries:
                print("    no entries parsed")
                continue
            per_feed = 0
            for title, url, published in entries[: args.max_per_feed]:
                if args.max_records and total_written >= args.max_records:
                    break
                if url in seen:
                    continue
                time.sleep(args.request_delay)
                page = http_get(url)
                if not page:
                    continue
                ext = CISAArticleExtractor()
                try:
                    ext.feed(page)
                except Exception as e:
                    print(f"    parse failed: {url[:80]}: {type(e).__name__}")
                    continue
                body = ext.get_text()
                if len(body.split()) < args.min_words:
                    continue
                rec = {
                    "id": f"cisa_advisories#{feed_name}#{re.sub(r'[^a-z0-9]', '-', url.lower())[:80]}",
                    "source": "cisa_advisories",
                    "feed": feed_name,
                    "source_url": url,
                    "title": title,
                    "published": published,
                    "text": (f"{title}\n\n{body}" if title else body),
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                seen.add(url)
                per_feed += 1
                total_written += 1
            print(f"    wrote {per_feed} new advisories")
    finally:
        fh.close()
    print(f"\nDone. {total_written} new advisories written to {out_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
