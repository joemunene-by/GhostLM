#!/usr/bin/env python3
"""Stream a broad (non-cybersec) Wikipedia sample for general knowledge.

GhostLM's pretrain corpus is cybersec-heavy, and the ghost-small line's
fact-recall floor (0-1% on the v2 bench) is the clearest symptom: the
model matches register and topic but binds almost no retrievable facts.
Broad encyclopedic text is the substrate generalist factual recall is
built on, and the corpus has had only a narrow ``wikipedia_cyber`` BFS
slice of it. This collector pulls a wide, category-agnostic Wikipedia
sample so the ``knowledge`` domain carries real share under the
generalist corpus profile (see ``data.collect.SOURCE_DOMAINS`` /
``scripts/rebuild_corpus.py``).

Default source: ``wikimedia/wikipedia`` config ``20231101.en``. Each
record exposes ``{id, url, title, text}``. Streamed via HF ``datasets``
so we don't pay the full-download cost. Licensed CC BY-SA 4.0 + GFDL;
the per-record ``url`` is preserved for attribution.

Output: ``data/raw/wikipedia_general.jsonl`` with the standard
``{"id", "source", "text", ...}`` schema, ``source == "wikipedia"``.

Disambiguation pages, list/index stubs, and very short articles are
skipped so the sample skews to substantive prose.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Title prefixes/markers that signal non-prose pages we don't want.
_SKIP_TITLE_MARKERS = (
    "List of ", "Index of ", "Outline of ", "Timeline of ",
    "Glossary of ", "Table of ",
)


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="Stream a broad Wikipedia sample")
    p.add_argument("--out", default="data/raw/wikipedia_general.jsonl")
    p.add_argument("--repo", default="wikimedia/wikipedia",
                   help="HF dataset repo for Wikipedia dumps")
    p.add_argument("--name", default="20231101.en",
                   help="Dataset config (a dated language snapshot)")
    p.add_argument("--split", default="train")
    p.add_argument("--max-records", type=int, default=80_000,
                   help="Number of articles to keep (~80M tokens at default caps)")
    p.add_argument("--min-chars", type=int, default=600,
                   help="Skip stubs shorter than this many characters")
    p.add_argument("--max-chars", type=int, default=8_000,
                   help="Truncate long articles to this many characters")
    p.add_argument("--sample-every", type=int, default=7,
                   help="Keep 1 of every N eligible articles, to spread the "
                        "sample across the alphabet instead of taking a dense "
                        "prefix of the dump. Set 1 to keep every eligible article.")
    return p.parse_args()


def _is_low_value(title: str, text: str) -> bool:
    """True for disambiguation / list / index pages we want to skip."""
    if any(title.startswith(m) for m in _SKIP_TITLE_MARKERS):
        return True
    if title.endswith("(disambiguation)"):
        return True
    # Disambiguation bodies are short and end with the marker line.
    if "may refer to:" in text[:200].lower():
        return True
    return False


def main() -> None:
    """Stream and write."""
    from datasets import load_dataset
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    seen: set = set()
    if out.exists():
        with out.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("id"):
                        seen.add(rec["id"])
        print(f"  resume: {len(seen)} records on disk")

    print(f"streaming {args.repo}:{args.name}:{args.split} (target {args.max_records})...")
    ds = load_dataset(args.repo, name=args.name, split=args.split, streaming=True)

    out_fh = out.open("a", encoding="utf-8", buffering=1)
    written = 0
    skipped_short = 0
    skipped_lowvalue = 0
    truncated = 0
    skipped_dup = 0
    eligible = 0

    for rec in ds:
        title = (rec.get("title") or "").strip()
        text = (rec.get("text") or "").strip()
        url = rec.get("url") or ""
        rec_id = f"wiki_{rec.get('id') or title}"

        if rec_id in seen:
            skipped_dup += 1
            continue
        if len(text) < args.min_chars:
            skipped_short += 1
            continue
        if _is_low_value(title, text):
            skipped_lowvalue += 1
            continue

        # Spread the sample across the dump instead of a dense prefix.
        eligible += 1
        if args.sample_every > 1 and (eligible % args.sample_every) != 0:
            continue

        if len(text) > args.max_chars:
            text = text[: args.max_chars].rsplit(" ", 1)[0]
            truncated += 1

        out_fh.write(json.dumps({
            "id": rec_id,
            "source": "wikipedia",
            "title": title,
            "url": url,
            "text": text,
            "license": "CC-BY-SA-4.0",
        }, ensure_ascii=False) + "\n")
        written += 1

        if written >= args.max_records:
            break
        if written % 5000 == 0:
            print(f"  written={written} short={skipped_short} "
                  f"lowvalue={skipped_lowvalue} truncated={truncated}")

    out_fh.close()
    print(f"\nDone. Wrote {written} Wikipedia records to {out}")
    if skipped_short:
        print(f"  skipped {skipped_short} too-short / stub")
    if skipped_lowvalue:
        print(f"  skipped {skipped_lowvalue} list/disambiguation pages")
    if skipped_dup:
        print(f"  skipped {skipped_dup} already on disk")
    if truncated:
        print(f"  truncated {truncated} long articles to {args.max_chars} chars")


if __name__ == "__main__":
    main()
