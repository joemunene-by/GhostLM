#!/usr/bin/env python3
"""Collect technical threat-intel research from major vendor blogs.

Companion collector to ``collect_security_blogs.py``. The existing
script covers research-shop blogs (Project Zero, Trail of Bits,
PortSwigger, NCC Group) and security commentary (Krebs, Microsoft
SRT). This one covers the **vendor threat-intelligence research**
register: Cisco Talos campaign analysis, Mandiant attribution
write-ups, Crowdstrike OverWatch reports, Unit 42 malware family
deep-dives. The two registers are different in flavor (vendor TI is
more campaign-detail / IOC-heavy / attribution-discussion) so we
write to a separate output stream and source-tag accordingly.

Each post's HTML is fetched and the article body extracted via a
lightweight stdlib parser tuned for blog markup (mirrors the parser
in ``collect_security_blogs.py``; not imported from there to keep the
two collectors fully independent of each other's failure modes).

Output: ``data/raw/vendor_research.jsonl`` with the standard
``{"id", "source", "text"}`` schema. Source field is
``vendor_research``. Per-record ``feed`` and ``source_url`` fields
preserve attribution.

License posture: each feed allows research / non-commercial use with
attribution; the corpus stays attributable per record. No off-domain
links followed; only the article body of each post is captured.

Run:

    PYTHONPATH=. python3 scripts/collect_vendor_research.py \\
        --out data/raw/vendor_research.jsonl \\
        --max-per-feed 50

Throughput on a normal residential connection: ~5-15 minutes for the
default ~500-record limit. Resume-safe: re-running picks up where it
left off (skips entries whose URL is already in the output).
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


# Vendor TI research feeds, ordered roughly by output volume.
FEEDS: List[Tuple[str, str]] = [
    ("talos",            "https://blog.talosintelligence.com/rss/"),
    ("unit42",           "https://unit42.paloaltonetworks.com/feed/"),
    ("crowdstrike",      "https://www.crowdstrike.com/blog/feed/"),
    ("mandiant",         "https://cloud.google.com/blog/topics/threat-intelligence/rss/"),
    ("rapid7",           "https://www.rapid7.com/blog/rss/"),
    ("tenable",          "https://www.tenable.com/blog/feed"),
    ("sophos",           "https://news.sophos.com/en-us/feed/"),
    ("eset",             "https://www.welivesecurity.com/en/rss/"),
    ("trend-micro",      "https://www.trendmicro.com/en_us/research.rss"),
    ("sans-isc",         "https://isc.sans.edu/rssfeed.xml"),
    ("recorded-future",  "https://www.recordedfuture.com/feed"),
]

USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
              "AppleWebKit/537.36 (KHTML, like Gecko) "
              "GhostLM-corpus-collector/0.9.2 Safari/537.36")

CONTENT_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "pre", "code", "blockquote"}
SKIP_TAGS = {"nav", "script", "style", "aside", "header", "footer", "form", "noscript",
             "iframe", "svg", "figure"}


class ArticleBodyExtractor(HTMLParser):
    """Lightweight body extractor: collects text inside paragraph-like
    tags, drops navigation chrome and scripts. Same shape as the
    parser in collect_security_blogs.py; deliberately copied to keep
    this collector independent."""

    def __init__(self) -> None:
        super().__init__()
        self._buf: List[str] = []
        self._skip_depth = 0
        self._capture: List[bool] = []

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self._skip_depth += 1
        if tag in CONTENT_TAGS:
            self._capture.append(True)
            if tag.startswith("h"):
                self._buf.append("\n\n")

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag in CONTENT_TAGS and self._capture:
            self._capture.pop()
            if tag in {"p", "pre", "blockquote", "li"} or tag.startswith("h"):
                self._buf.append("\n\n")

    def handle_data(self, data):
        if self._skip_depth > 0:
            return
        if self._capture and self._capture[-1]:
            self._buf.append(data)

    def get_text(self) -> str:
        text = "".join(self._buf)
        # Collapse runs of whitespace, preserve paragraph breaks.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def http_get(url: str, timeout: int = 30) -> Optional[str]:
    """GET with a real-browser User-Agent; many vendor sites 403 a
    bare urllib request."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = resp.headers.get("Content-Type", "")
            charset = "utf-8"
            m = re.search(r"charset=([\w-]+)", ctype, re.I)
            if m:
                charset = m.group(1)
            return resp.read().decode(charset, errors="replace")
    except Exception as e:  # noqa: BLE001 - one bad fetch is normal
        print(f"    fetch failed: {url[:80]}: {type(e).__name__}: {e}")
        return None


def parse_feed_entries(xml: str, feed_name: str) -> List[Tuple[str, str, Optional[str]]]:
    """Pull (title, link, published) from RSS or Atom XML.

    Both formats use distinct tag names; we cover both with a shared
    regex pass. Production-grade parsing would use feedparser; we use
    stdlib re to avoid a dependency for one pass-through transform."""
    entries: List[Tuple[str, str, Optional[str]]] = []
    # RSS 2.0: <item><title>...</title><link>...</link><pubDate>...</pubDate></item>
    for m in re.finditer(r"<item\b.*?</item>", xml, re.S | re.I):
        block = m.group(0)
        title = _extract_tag(block, "title")
        link = _extract_tag(block, "link")
        published = _extract_tag(block, "pubDate")
        if link:
            entries.append((title or "", link, published))
    # Atom 1.0: <entry><title>...</title><link href="..."/><updated>...</updated></entry>
    for m in re.finditer(r"<entry\b.*?</entry>", xml, re.S | re.I):
        block = m.group(0)
        title = _extract_tag(block, "title")
        # Atom links are <link href="..."/>; pull the href attr.
        link_m = re.search(r'<link[^>]*\bhref=["\']([^"\']+)["\']', block, re.I)
        link = link_m.group(1) if link_m else None
        published = _extract_tag(block, "updated") or _extract_tag(block, "published")
        if link:
            entries.append((title or "", link, published))
    return entries


def _extract_tag(block: str, tag: str) -> Optional[str]:
    m = re.search(fr"<{tag}\b[^>]*>(.*?)</{tag}>", block, re.S | re.I)
    if not m:
        return None
    raw = m.group(1).strip()
    # Strip CDATA wrapper.
    raw = re.sub(r"^<!\[CDATA\[(.*?)\]\]>$", r"\1", raw, flags=re.S)
    return html.unescape(raw).strip()


def load_existing_urls(out_path: Path) -> Set[str]:
    """For resume-safety: read the output JSONL and collect URLs that
    have already been ingested."""
    seen: Set[str] = set()
    if not out_path.exists():
        return seen
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            url = rec.get("source_url") or rec.get("url")
            if url:
                seen.add(url)
    return seen


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/raw/vendor_research.jsonl")
    p.add_argument("--max-per-feed", type=int, default=50,
                   help="Cap posts pulled per feed (avoids hammering small vendors)")
    p.add_argument("--min-words", type=int, default=80,
                   help="Drop posts shorter than this after extraction")
    p.add_argument("--request-delay", type=float, default=1.0,
                   help="Polite delay between HTTP requests, seconds")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen_urls = load_existing_urls(out_path)
    print(f"Output:    {out_path}")
    print(f"Resuming:  {len(seen_urls)} URLs already in output")
    print(f"Feeds:     {len(FEEDS)}")
    print()

    total_written = 0
    fh = out_path.open("a", encoding="utf-8")

    for feed_name, feed_url in FEEDS:
        print(f"=== {feed_name} ({feed_url}) ===")
        xml = http_get(feed_url)
        if not xml:
            print("    feed fetch failed; skipping")
            continue
        entries = parse_feed_entries(xml, feed_name)
        if not entries:
            print("    no entries parsed (feed format unrecognized?)")
            continue
        per_feed_written = 0
        for title, url, published in entries[: args.max_per_feed]:
            if url in seen_urls:
                continue
            time.sleep(args.request_delay)
            html_text = http_get(url)
            if not html_text:
                continue
            extractor = ArticleBodyExtractor()
            try:
                extractor.feed(html_text)
            except Exception as e:  # noqa: BLE001 - some pages have junk markup
                print(f"    parse failed: {url[:80]}: {type(e).__name__}")
                continue
            body = extractor.get_text()
            if len(body.split()) < args.min_words:
                continue
            rec = {
                "id": f"vendor_research#{feed_name}#{re.sub(r'[^a-z0-9]', '-', url.lower())[:80]}",
                "source": "vendor_research",
                "feed": feed_name,
                "source_url": url,
                "title": title,
                "published": published,
                "text": (f"{title}\n\n{body}" if title else body),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            seen_urls.add(url)
            per_feed_written += 1
            total_written += 1
        print(f"    wrote {per_feed_written} new posts")

    fh.close()
    print(f"\nDone. {total_written} new posts written to {out_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
