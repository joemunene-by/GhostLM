"""CLI entry point for the CTF-writeup-repo collector.

Reads a JSON config listing the repos to clone (with their SPDX licenses)
and dispatches to ``data.collect.collect_ctf_repos``. The JSON config keeps
the "which repos to ingest" decision transparent and auditable — license
choices live in the config, not in code.

Example config (``data/ctf_repos.example.json``):
    [
      {"url": "https://github.com/some-team/ctf-writeups", "license": "MIT"},
      {"url": "https://github.com/another-team/writeups",  "license": "CC-BY-4.0",
       "subdir": "2024"}
    ]

Usage:
    python scripts/collect_ctf_repos.py --config data/ctf_repos.example.json

The output JSONL records each carry the source repo URL, file path, and
license SPDX so downstream auditors can spot-check attribution.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.collect import collect_ctf_repos


def parse_args():
    p = argparse.ArgumentParser(description="Collect CTF writeups from permissively-licensed repos.")
    p.add_argument("--config", required=True, help="Path to JSON config (list of {url, license, [subdir]}).")
    p.add_argument("--output", default="data/raw/ctf_repos.jsonl", help="Output JSONL path.")
    p.add_argument("--min-chars", type=int, default=200, help="Drop files shorter than this.")
    p.add_argument("--max-chars", type=int, default=12000, help="Truncate files longer than this.")
    return p.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        repos = json.load(f)

    if not isinstance(repos, list):
        sys.exit("config must be a JSON array of {url, license, [subdir]} objects")

    for entry in repos:
        if not isinstance(entry, dict) or "url" not in entry or "license" not in entry:
            sys.exit(f"config entry missing required keys (url, license): {entry!r}")

    print(f"Loaded {len(repos)} repo entries from {config_path}")
    collect_ctf_repos(
        repos=repos,
        output_path=args.output,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )


if __name__ == "__main__":
    main()
