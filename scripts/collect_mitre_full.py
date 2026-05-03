#!/usr/bin/env python3
"""Pull MITRE ATT&CK Enterprise STIX bundle and emit non-technique objects.

The existing ``data/raw/mitre_attack.jsonl`` covers techniques + sub-techniques
(691 records). The STIX bundle also contains mitigations (~140), intrusion
groups (~150), malware/tools (~700), data sources (~40), and tactics (14) —
all written in the same MITRE house style and useful as cybersec pretrain
material. This script pulls the bundle once and writes those non-technique
objects to ``data/raw/mitre_full.jsonl``.

Output schema matches existing ``mitre_attack.jsonl``::

    {"id": "M1234", "text": "MITRE ATT&CK Mitigation M1234: ...\\n\\n<body>"}

Resume-safe — re-running picks up where it left off (skips ids already in
the output file). Single download is ~30 MB; the parse + emit is < 1 min.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Set


STIX_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/"
    "enterprise-attack.json"
)


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="Collect MITRE ATT&CK non-technique STIX objects")
    p.add_argument("--output", default="data/raw/mitre_full.jsonl")
    p.add_argument("--bundle-cache", default="data/raw/.mitre_stix_bundle.json",
                   help="Cache the STIX bundle locally so re-runs don't re-download")
    p.add_argument("--types", nargs="*", default=None,
                   help="STIX types to include (default: course-of-action, "
                        "intrusion-set, malware, tool, x-mitre-data-source, "
                        "x-mitre-tactic). Pass --types attack-pattern to also "
                        "re-pull techniques.")
    return p.parse_args()


def fetch_bundle(cache_path: Path) -> Dict:
    """Download the STIX bundle (cached)."""
    if cache_path.exists():
        print(f"  using cached bundle: {cache_path}")
        return json.loads(cache_path.read_text(encoding="utf-8"))
    print(f"  downloading {STIX_URL} ...")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(STIX_URL, timeout=120) as resp:
        data = resp.read()
    cache_path.write_bytes(data)
    print(f"  cached to {cache_path} ({len(data) / 1e6:.1f} MB)")
    return json.loads(data)


def get_external_id(obj: Dict) -> Optional[str]:
    """Extract the MITRE ATT&CK id (M1234, G0001, S0001, TA0001, ...) from
    the ``external_references`` block."""
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack":
            return ref.get("external_id")
    return None


def render_record(obj: Dict, ext_id: str) -> Optional[Dict]:
    """Format a STIX object as a pretrain text record. Returns None if the
    object is missing the bits needed to make a useful record."""
    name = (obj.get("name") or "").strip()
    description = (obj.get("description") or "").strip()
    if not name or not description:
        return None

    type_label = {
        "course-of-action": "Mitigation",
        "intrusion-set": "Group",
        "malware": "Software (Malware)",
        "tool": "Software (Tool)",
        "x-mitre-data-source": "Data Source",
        "x-mitre-tactic": "Tactic",
        "campaign": "Campaign",
        "attack-pattern": "Technique",
    }.get(obj.get("type", ""), obj.get("type", "Object"))

    aliases = obj.get("aliases") or obj.get("x_mitre_aliases") or []
    alias_line = ""
    if aliases and len(aliases) > 1:
        # First alias is usually the same as `name`.
        extras = [a for a in aliases if a != name]
        if extras:
            alias_line = f"Aliases: {', '.join(extras[:6])}\n\n"

    platforms = obj.get("x_mitre_platforms") or []
    platform_line = f"Platforms: {', '.join(platforms)}\n" if platforms else ""

    text = (
        f"MITRE ATT&CK {type_label} {ext_id}: {name}\n"
        f"{platform_line}\n"
        f"{alias_line}"
        f"{description}"
    ).strip()

    return {"id": ext_id, "source": "mitre_full", "text": text}


def main() -> None:
    """Pull bundle, emit one record per non-technique STIX object."""
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: Set[str] = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("id"):
                        seen.add(rec["id"])
        print(f"  resume: {len(seen)} records already done")

    bundle = fetch_bundle(Path(args.bundle_cache))
    objects = bundle.get("objects", [])
    print(f"  bundle contains {len(objects):,} STIX objects")

    target_types = set(args.types) if args.types else {
        "course-of-action", "intrusion-set", "malware", "tool",
        "x-mitre-data-source", "x-mitre-tactic", "campaign",
    }

    type_counts: Dict[str, int] = {}
    revoked_or_deprecated = 0
    written = 0
    skipped_missing = 0

    out_fh = out_path.open("a", encoding="utf-8", buffering=1)
    for obj in objects:
        if obj.get("type") not in target_types:
            continue
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            revoked_or_deprecated += 1
            continue

        ext_id = get_external_id(obj)
        if not ext_id:
            continue
        if ext_id in seen:
            continue

        rec = render_record(obj, ext_id)
        if rec is None:
            skipped_missing += 1
            continue

        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_fh.flush()
        written += 1
        seen.add(ext_id)
        type_counts[obj["type"]] = type_counts.get(obj["type"], 0) + 1

    out_fh.close()
    print()
    print(f"Done. Wrote {written} new records to {out_path}")
    for t, c in sorted(type_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {t}: {c}")
    if revoked_or_deprecated:
        print(f"  Skipped {revoked_or_deprecated} revoked/deprecated objects")
    if skipped_missing:
        print(f"  Skipped {skipped_missing} objects missing name/description")


if __name__ == "__main__":
    main()
