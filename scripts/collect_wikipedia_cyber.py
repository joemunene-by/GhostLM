#!/usr/bin/env python3
"""Pull Wikipedia cybersecurity articles via the public API.

Wikipedia has a "Computer security" category tree with thousands of
articles spanning attacks, defenses, vulnerabilities, protocols,
specific malware families, threat actors, and security concepts.
Coverage is broad and the writing is encyclopedic (vs the CTF-writeup
style that dominates our existing corpus).

Source: en.wikipedia.org public API.
License: CC BY-SA 4.0 with attribution at the dataset level.

Output: ``data/raw/wikipedia_cyber.jsonl`` with the standard
``{"id", "source", "text"}`` schema. The ``id`` is the article title,
``source`` is ``wikipedia_cyber``.

The script does a breadth-first walk of subcategories starting from
"Category:Computer security", caps total articles, dedupes by title,
fetches the full plaintext (extract format) for each. Polite by
default (1 req/sec).
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Optional, Set


WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "GhostLM-CyberCorpus/0.6 (research; github.com/joemunene-by/GhostLM)"


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="Collect Wikipedia cybersecurity articles")
    p.add_argument("--out", default="data/raw/wikipedia_cyber.jsonl")
    p.add_argument("--root-categories", nargs="*", default=[
        "Category:Computer_security",
        "Category:Cyberattacks",
        "Category:Cryptography",
        "Category:Computer_security_exploits",
        "Category:Hacking_(computer_security)",
        "Category:Cybercrime",
    ])
    p.add_argument("--max-articles", type=int, default=2000,
                   help="Cap total articles to avoid runaway crawls")
    p.add_argument("--max-depth", type=int, default=2,
                   help="Subcategory recursion depth")
    p.add_argument("--request-delay", type=float, default=0.6,
                   help="Seconds between API calls (Wikipedia asks for >0.5)")
    p.add_argument("--min-chars", type=int, default=400,
                   help="Drop articles shorter than this (mostly stubs)")
    p.add_argument("--max-chars", type=int, default=15000)
    return p.parse_args()


def api_call(params: dict) -> dict:
    """Make one API call to Wikipedia."""
    params = {**params, "format": "json"}
    qs = urllib.parse.urlencode(params)
    req = urllib.request.Request(
        f"{WIKI_API}?{qs}",
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def get_category_members(category: str, request_delay: float) -> tuple[List[str], List[str]]:
    """Return (article_titles, subcategory_titles) inside a category."""
    articles: List[str] = []
    subcats: List[str] = []
    cmcontinue: Optional[str] = None
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": 500,
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = api_call(params)
        for m in data.get("query", {}).get("categorymembers", []):
            if m["ns"] == 14:  # 14 = Category
                subcats.append(m["title"])
            elif m["ns"] == 0:  # 0 = main article
                articles.append(m["title"])
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        time.sleep(request_delay)
        if not cmcontinue:
            break
    return articles, subcats


def get_extract(title: str, request_delay: float) -> str:
    """Fetch the plaintext extract for one article."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "exsectionformat": "plain",
        "titles": title,
    }
    data = api_call(params)
    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        text = page.get("extract", "")
        time.sleep(request_delay)
        return text
    time.sleep(request_delay)
    return ""


def main() -> None:
    """BFS the cybersec category tree, fetch each article, write JSONL."""
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip articles already in the output file.
    seen: Set[str] = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("id"):
                        seen.add(rec["id"])
        print(f"  resume: {len(seen)} articles already done")

    # BFS over categories
    visited_cats: Set[str] = set()
    queue: List[tuple[str, int]] = [(c, 0) for c in args.root_categories]
    article_titles: List[str] = []
    article_set: Set[str] = set(seen)

    while queue and len(article_titles) + len(seen) < args.max_articles:
        cat, depth = queue.pop(0)
        if cat in visited_cats:
            continue
        visited_cats.add(cat)
        try:
            arts, subs = get_category_members(cat, args.request_delay)
        except Exception as e:
            print(f"  error on {cat}: {e}")
            continue
        new_arts = [a for a in arts if a not in article_set]
        for a in new_arts:
            article_set.add(a)
            article_titles.append(a)
        if depth + 1 <= args.max_depth:
            for sc in subs:
                if sc not in visited_cats:
                    queue.append((sc, depth + 1))
        print(f"  [{cat}] +{len(new_arts)} articles (total queued: {len(article_titles)})")

    article_titles = article_titles[: max(0, args.max_articles - len(seen))]
    print(f"\nFetching {len(article_titles)} articles...")

    out_fh = out_path.open("a", encoding="utf-8", buffering=1)
    written = 0
    skipped = 0
    for i, title in enumerate(article_titles):
        if title in seen:
            continue
        try:
            text = get_extract(title, args.request_delay)
        except Exception as e:
            print(f"  fetch error on '{title}': {e}")
            continue
        if len(text) < args.min_chars:
            skipped += 1
            continue
        if len(text) > args.max_chars:
            text = text[: args.max_chars].rsplit("\n\n", 1)[0]
        rec = {
            "id": title,
            "source": "wikipedia_cyber",
            "text": f"{title}\n\n{text}",
        }
        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_fh.flush()
        written += 1
        if (i + 1) % 25 == 0:
            print(f"  [{i + 1}/{len(article_titles)}] written={written} skipped={skipped}")
    out_fh.close()
    print(f"\nDone. Wrote {written} Wikipedia cyber articles to {out_path}")


if __name__ == "__main__":
    main()
