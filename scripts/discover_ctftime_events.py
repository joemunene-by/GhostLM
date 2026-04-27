"""Discover CTFtime events for the writeup-collector config.

Hits CTFtime's public events API (``/api/v1/events/``) for a date range,
filters by minimum weight and participant count, and writes a JSON
config compatible with ``scripts/collect_ctftime.py``. Replaces the
manual curation step that was capping ``data/ctftime_events.json`` at
28 hand-picked events.

Filtering rationale:

- ``weight`` is CTFtime's quality score for an event (0-100). Top-tier
  CTFs usually weigh 50+. Setting the floor at 30 keeps mid-tier
  competitions whose writeups still teach real techniques while
  excluding the long tail of toy CTFs.
- ``participants`` filters out competitions nobody played — those
  have few writeups and the ones that exist are often bad.
- ``format`` is restricted to Jeopardy because the CTFtime writeup
  page format used by ``collect_ctftime_writeups`` is built for
  per-task writeups; Attack-Defense events don't fit that pattern.

Usage:
    python scripts/discover_ctftime_events.py --years 2020 2024
    python scripts/discover_ctftime_events.py --years 2020 2026 \\
        --min-weight 40 --min-participants 100 --merge data/ctftime_events.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Dict, List

import requests


CTFTIME_EVENTS_URL = "https://ctftime.org/api/v1/events/"
USER_AGENT = "GhostLM-corpus-tool/0.3 (research; https://github.com/joemunene-by/GhostLM)"


def year_to_unix(year: int, end: bool = False) -> int:
    """Convert a calendar year to a Unix timestamp at Jan 1 (or Dec 31 23:59:59 UTC)."""
    if end:
        dt = datetime.datetime(year, 12, 31, 23, 59, 59, tzinfo=datetime.timezone.utc)
    else:
        dt = datetime.datetime(year, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
    return int(dt.timestamp())


def fetch_events(start_year: int, end_year: int, limit: int = 1000,
                 timeout: int = 30) -> List[Dict]:
    """Pull all events from the CTFtime API for the given year range.

    The API caps results per request, so the date range is split into
    yearly chunks for safety. Failed years are logged but don't abort
    the discovery — partial results are better than none.

    Args:
        start_year: First calendar year to include (inclusive).
        end_year: Last calendar year to include (inclusive).
        limit: Per-request result cap (CTFtime accepts up to 1000).
        timeout: HTTP timeout per request.

    Returns:
        Flat list of event dicts as returned by the API.
    """
    all_events: List[Dict] = []
    headers = {"User-Agent": USER_AGENT}
    for year in range(start_year, end_year + 1):
        params = {
            "limit": limit,
            "start": year_to_unix(year, end=False),
            "finish": year_to_unix(year, end=True),
        }
        try:
            resp = requests.get(CTFTIME_EVENTS_URL, params=params,
                                headers=headers, timeout=timeout)
            resp.raise_for_status()
            year_events = resp.json()
        except Exception as e:
            print(f"  WARN: year {year} fetch failed: {e}", file=sys.stderr)
            continue
        all_events.extend(year_events)
        print(f"  {year}: pulled {len(year_events)} events")
    return all_events


def select_events(events: List[Dict], min_weight: float, min_participants: int,
                  jeopardy_only: bool = True) -> List[Dict]:
    """Apply the quality filter and shape the output to the writeup-config schema.

    Returns a list of ``{"id": int, "name": str}`` dicts so the result
    drops directly into ``scripts/collect_ctftime.py`` via
    ``--config``.
    """
    selected = []
    for ev in events:
        if jeopardy_only and ev.get("format") != "Jeopardy":
            continue
        if (ev.get("weight") or 0) < min_weight:
            continue
        if (ev.get("participants") or 0) < min_participants:
            continue
        eid = ev.get("id")
        title = (ev.get("title") or "").strip()
        if not eid or not title:
            continue
        selected.append({"id": eid, "name": title})
    # Stable ordering by event id for deterministic configs.
    selected.sort(key=lambda e: e["id"])
    return selected


def merge_existing(new_events: List[Dict], existing_path: Path) -> List[Dict]:
    """Union ``new_events`` with the existing config; existing entries win on id collisions.

    Manual curation in ``data/ctftime_events.json`` may include events
    the API filter would drop (e.g. low-participant events with great
    writeups). Preserve those rather than overwriting.
    """
    if not existing_path.exists():
        return new_events
    with existing_path.open("r", encoding="utf-8") as f:
        existing = json.load(f)
    by_id: Dict[int, Dict] = {e["id"]: e for e in new_events}
    for e in existing:
        # Existing entries take precedence — replace any matching new id.
        by_id[e["id"]] = e
    merged = list(by_id.values())
    merged.sort(key=lambda e: e["id"])
    return merged


def parse_args():
    p = argparse.ArgumentParser(description="Discover CTFtime events for writeup ingestion.")
    p.add_argument(
        "--years", nargs=2, type=int, metavar=("START", "END"),
        default=[2020, datetime.datetime.now().year],
        help="Inclusive year range to query (default: 2020 to current year).",
    )
    p.add_argument(
        "--min-weight", type=float, default=30.0,
        help=(
            "CTFtime weight floor (0-100). Defaults to 30 — keeps mid-tier "
            "competitions whose writeups still teach real technique."
        ),
    )
    p.add_argument(
        "--min-participants", type=int, default=50,
        help="Exclude competitions with fewer than this many participants.",
    )
    p.add_argument(
        "--include-non-jeopardy", action="store_true",
        help=(
            "Allow Attack-Defense and other formats. Off by default because "
            "the CTFtime writeup-page schema collect_ctftime_writeups walks "
            "is built around per-task Jeopardy writeups."
        ),
    )
    p.add_argument(
        "--output", default="data/ctftime_events.json",
        help="Output config path (overwrites unless --merge is specified).",
    )
    p.add_argument(
        "--merge", default=None, type=Path,
        help=(
            "Existing config to union into the output. Existing entries win "
            "on id collisions (preserves manual curation overrides)."
        ),
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be written without touching disk.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    start_year, end_year = args.years
    print(f"Discovering CTFtime events for {start_year}-{end_year} "
          f"(weight ≥ {args.min_weight}, participants ≥ {args.min_participants})...")

    events = fetch_events(start_year, end_year)
    print(f"Total events fetched: {len(events)}")

    selected = select_events(
        events,
        min_weight=args.min_weight,
        min_participants=args.min_participants,
        jeopardy_only=not args.include_non_jeopardy,
    )
    print(f"Selected after filters: {len(selected)}")

    if args.merge:
        merged = merge_existing(selected, args.merge)
        print(f"After merge with {args.merge}: {len(merged)} events "
              f"(was {len(selected)} discovered, "
              f"{len(merged) - len(selected)} preserved from existing)")
        selected = merged

    if args.dry_run:
        print(f"[dry-run] Would write {len(selected)} events to {args.output}")
        for e in selected[:10]:
            print(f"  {e['id']:>6}  {e['name']}")
        if len(selected) > 10:
            print(f"  ... and {len(selected) - 10} more")
        return

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(selected)} events to {out}")


if __name__ == "__main__":
    main()
