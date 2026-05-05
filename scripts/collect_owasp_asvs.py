#!/usr/bin/env python3
"""Ingest OWASP Application Security Verification Standard (ASVS).

ASVS is OWASP's formal security verification checklist (v5.0). Source
of truth is the project's signed release: a flat JSON with every
requirement, grouped by chapter and section. CC BY-SA 4.0.

Source: https://github.com/OWASP/ASVS/releases/latest (flat.json asset)
Output: ``data/raw/owasp_asvs.jsonl`` with the standard
``{"id", "source", "text"}`` schema. Source is ``owasp_asvs``.

Each (chapter, section) pair becomes one record so requirements stay
in context with their section header. ~80 records typically.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


ASVS_URL = (
    "https://github.com/OWASP/ASVS/releases/download/latest/"
    "OWASP_Application_Security_Verification_Standard_5.0.0_en.flat.json"
)


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="Ingest OWASP ASVS requirements")
    p.add_argument("--src", default=None,
                   help="Path to a pre-downloaded flat.json (skips network)")
    p.add_argument("--out", default="data/raw/owasp_asvs.jsonl")
    p.add_argument("--min-chars", type=int, default=200)
    return p.parse_args()


def fetch_asvs(cache: Path) -> dict:
    """Download (or read cached) ASVS flat JSON."""
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"  downloading {ASVS_URL}")
        urllib.request.urlretrieve(ASVS_URL, cache)
        print(f"  cached to {cache}")
    return json.loads(cache.read_text(encoding="utf-8"))


def render_section(chapter_id: str, chapter_name: str,
                   section_id: str, section_name: str,
                   reqs: list[dict]) -> str:
    """Compose a section + its requirements into one training paragraph."""
    lines = [
        f"OWASP ASVS {section_id}: {section_name}",
        f"(Chapter {chapter_id}: {chapter_name})",
        "",
    ]
    for r in reqs:
        rid = r.get("req_id", "?")
        level = r.get("L") or ""
        desc = (r.get("req_description") or "").strip()
        if not desc:
            continue
        prefix = f"L{level} " if level else ""
        lines.append(f"  - [{rid}] {prefix}{desc}")
    return "\n".join(lines)


def main() -> None:
    """Pull ASVS, group requirements by section, emit JSONL."""
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.src:
        data = json.loads(Path(args.src).read_text(encoding="utf-8"))
    else:
        cache = out_path.parent / ".asvs_flat.json"
        data = fetch_asvs(cache)

    reqs = data.get("requirements", []) if isinstance(data, dict) else data
    if not reqs:
        raise SystemExit("No ASVS requirements parsed from JSON")

    grouped: dict = {}
    for r in reqs:
        key = (r.get("chapter_id"), r.get("chapter_name"),
               r.get("section_id"), r.get("section_name"))
        grouped.setdefault(key, []).append(r)

    out_fh = out_path.open("w", encoding="utf-8")
    written = 0
    skipped = 0
    for (cid, cname, sid, sname), section_reqs in sorted(grouped.items()):
        text = render_section(cid or "?", cname or "?", sid or "?",
                              sname or "?", section_reqs)
        if len(text) < args.min_chars:
            skipped += 1
            continue
        rec = {
            "id": f"ASVS-{sid}",
            "source": "owasp_asvs",
            "text": text,
        }
        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        written += 1
    out_fh.close()
    print(f"Wrote {written} ASVS records to {out_path} (skipped {skipped} too-short)")


if __name__ == "__main__":
    main()
