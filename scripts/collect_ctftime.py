"""CLI entry point for the CTFtime writeup collector.

Reads a JSON config listing the CTFtime event IDs to ingest and dispatches
to ``data.collect.collect_ctftime_writeups``. Like the GitHub-CTF-repos
collector, the event list is config-driven rather than hardcoded so the
licensing posture (which CTFs we treat as research-archivable) is
transparent and auditable.

Example config (``data/ctftime_events.example.json``):
    [
      {"id": 1405, "name": "FwordCTF 2021"},
      {"id": 2230, "name": "JerseyCTF IV"}
    ]

Usage:
    python scripts/collect_ctftime.py --config data/ctftime_events.json
    python scripts/collect_ctftime.py --config data/ctftime_events.json \\
        --max-writeups 200 --request-delay 1.5
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.collect import collect_ctftime_writeups


def parse_args():
    p = argparse.ArgumentParser(description="Collect CTFtime inline writeups for a list of events.")
    p.add_argument("--config", required=True,
                   help="Path to JSON config (list of {id, name} objects).")
    p.add_argument("--output", default="data/raw/ctftime.jsonl",
                   help="Output JSONL path (resume-aware).")
    p.add_argument("--request-delay", type=float, default=1.0,
                   help="Seconds to sleep between HTTP requests (be polite).")
    p.add_argument("--request-timeout", type=int, default=30,
                   help="Per-request HTTP timeout in seconds.")
    p.add_argument("--min-chars", type=int, default=200,
                   help="Drop writeups shorter than this.")
    p.add_argument("--max-chars", type=int, default=30000,
                   help="Truncate writeups longer than this. Default 30000 — real CTF "
                        "writeups with full exploit transcripts often run 15-25K chars; "
                        "the previous 12K cap was truncating most records mid-exploit.")
    p.add_argument("--max-writeups", type=int, default=None,
                   help="Stop after collecting this many new writeups (smoke testing).")
    return p.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    if not isinstance(events, list):
        sys.exit("config must be a JSON array of {id, name} objects")

    event_ids = []
    for entry in events:
        if not isinstance(entry, dict) or "id" not in entry:
            sys.exit(f"config entry missing required 'id' key: {entry!r}")
        event_ids.append(int(entry["id"]))

    print(f"Loaded {len(event_ids)} events from {config_path}")
    collect_ctftime_writeups(
        event_ids=event_ids,
        output_path=args.output,
        request_delay=args.request_delay,
        request_timeout=args.request_timeout,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        max_writeups=args.max_writeups,
    )


if __name__ == "__main__":
    main()
