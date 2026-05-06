#!/usr/bin/env python3
"""Pull primary-source security research blogs into the corpus.

Curated set of high-signal cybersecurity research blogs. Each is a
hand-picked source where the writing is technical, primary-source,
and well-known to the field: Project Zero exploit chains, PortSwigger
web research, Trail of Bits binary analysis, Google security
announcements, NCC Group consulting writeups, etc. The goal is
training data that exposes the model to the *register* of practicing
security researchers, not just the catalog-style text in NVD/MITRE.

Each feed is RSS or Atom (parsed via feedparser). For each post we
fetch the HTML and extract article-body text using a stdlib HTML
parser tuned for blog-style markup (collect text inside <p>, <h*>,
<li>, <pre>; ignore <nav>, <script>, <style>, <aside>).

License posture: each source allows non-commercial / research use
with attribution; per-record `source_url` and `feed_name` make
attribution auditable. No off-domain links followed.

Output: ``data/raw/security_blogs.jsonl`` with the standard
``{"id", "source", "text"}`` schema. Source field is
``security_blogs``.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path


# Curated feeds. Each tuple: (name, feed_url).
FEEDS = [
    ("project-zero",    "https://googleprojectzero.blogspot.com/feeds/posts/default?max-results=200"),
    ("portswigger",     "https://portswigger.net/research/rss"),
    ("trail-of-bits",   "https://blog.trailofbits.com/feed/"),
    ("google-security", "https://security.googleblog.com/feeds/posts/default?max-results=200"),
    ("github-securitylab", "https://securitylab.github.com/feed.xml"),
    ("ncc-group",       "https://research.nccgroup.com/feed/"),
    ("doyensec",        "https://blog.doyensec.com/atom.xml"),
    ("krebs",           "https://krebsonsecurity.com/feed/"),
    ("dfir-report",     "https://thedfirreport.com/feed/"),
    ("ret2-systems",    "https://blog.ret2.io/feed.xml"),
    ("microsoft-srt",   "https://msrc.microsoft.com/blog/feed/"),
]


# Tags where the actual article text lives
CONTENT_TAGS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "pre", "code", "blockquote"}
# Tags whose content we ignore entirely (navigation chrome, JS, etc.)
SKIP_TAGS = {"nav", "script", "style", "aside", "header", "footer", "form", "noscript"}


class ArticleBodyExtractor(HTMLParser):
    """Pull text from blog-post bodies, ignoring chrome / boilerplate."""

    def __init__(self) -> None:
        super().__init__()
        self._buffer: list[str] = []
        self._in_skip_depth = 0
        self._capture_stack: list[bool] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in SKIP_TAGS:
            self._in_skip_depth += 1
        if tag in CONTENT_TAGS:
            self._capture_stack.append(True)
            # Insert spacing for headings
            if tag.startswith("h"):
                self._buffer.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS and self._in_skip_depth > 0:
            self._in_skip_depth -= 1
        if tag in CONTENT_TAGS and self._capture_stack:
            self._capture_stack.pop()
            if tag in {"p", "pre", "blockquote", "li"} or tag.startswith("h"):
                self._buffer.append("\n\n")

    def handle_data(self, data: str) -> None:
        if self._in_skip_depth > 0:
            return
        if self._capture_stack and self._capture_stack[-1]:
            self._buffer.append(data)

    def text(self) -> str:
        out = "".join(self._buffer)
        out = html.unescape(out)
        out = re.sub(r"[ \t]+", " ", out)
        out = re.sub(r"\n[ \t]+", "\n", out)
        out = re.sub(r"\n{3,}", "\n\n", out)
        return out.strip()


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="Pull primary-source security research blogs")
    p.add_argument("--out", default="data/raw/security_blogs.jsonl")
    p.add_argument("--max-posts-per-feed", type=int, default=80)
    p.add_argument("--min-chars", type=int, default=600)
    p.add_argument("--max-chars", type=int, default=20000)
    p.add_argument("--request-delay", type=float, default=1.0)
    return p.parse_args()


def fetch_html(url: str, timeout: int = 30) -> str:
    """Fetch raw HTML bytes -> string. Returns '' on error."""
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (compatible; GhostLM-BlogCollector/0.9; "
                       "+https://github.com/joemunene-by/GhostLM)",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    enc = "utf-8"
    ct = resp.headers.get("content-type", "")
    m = re.search(r"charset=([\w-]+)", ct, re.I)
    if m:
        enc = m.group(1)
    return raw.decode(enc, errors="ignore")


def extract_body(html_text: str) -> str:
    """Return the article body text (no chrome)."""
    parser = ArticleBodyExtractor()
    parser.feed(html_text)
    return parser.text()


def main() -> None:
    """Iterate every feed, fetch posts, write JSONL."""
    import feedparser  # type: ignore
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("id"):
                        seen.add(rec["id"])
        print(f"  resume: {len(seen)} posts already on disk")

    out_fh = out_path.open("a", encoding="utf-8", buffering=1)
    total_written = 0
    total_failed = 0
    for name, feed_url in FEEDS:
        print(f"\n[{name}] {feed_url}")
        try:
            feed = feedparser.parse(feed_url, request_headers={
                "User-Agent": "Mozilla/5.0 (GhostLM-BlogCollector/0.9)",
            })
        except Exception as e:
            print(f"  feed parse failed: {e}")
            continue
        if not feed.entries:
            print("  no entries")
            continue
        per_feed = 0
        for i, entry in enumerate(feed.entries[: args.max_posts_per_feed]):
            url = entry.get("link") or ""
            title = entry.get("title") or ""
            if not url:
                continue
            rec_id = f"{name}_{abs(hash(url)) % 10**12}"
            if rec_id in seen:
                continue
            try:
                html_text = fetch_html(url)
                body = extract_body(html_text)
            except Exception as e:
                total_failed += 1
                print(f"  {url}: fetch/parse error {e}")
                time.sleep(args.request_delay)
                continue
            if len(body) < args.min_chars:
                continue
            if len(body) > args.max_chars:
                body = body[: args.max_chars].rsplit("\n", 1)[0]
            text = f"{title}\n\n{body}" if title else body
            rec = {
                "id": rec_id,
                "source": "security_blogs",
                "text": text,
                "feed": name,
                "url": url,
                "title": title,
            }
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            per_feed += 1
            total_written += 1
            time.sleep(args.request_delay)
        print(f"  wrote {per_feed} posts")

    out_fh.close()
    print(f"\nDone. Wrote {total_written} blog posts to {out_path}")
    if total_failed:
        print(f"  failed {total_failed}")


if __name__ == "__main__":
    main()
