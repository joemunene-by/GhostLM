#!/usr/bin/env python3
"""Pull CISA's Known Exploited Vulnerabilities catalog.

CISA KEV is a curated list of CVEs being *actively* exploited in the wild —
high-signal threat intelligence that anchors which vulnerabilities matter
in practice. The full catalog is one JSON file (~1300 entries as of 2026)
that updates roughly weekly.

Each output record describes one actively-exploited CVE in a few sentences,
formatted to match the existing pretrain corpus style::

    {"id": "CVE-2024-X", "source": "cisa_kev",
     "text": "CISA KEV — CVE-2024-X (Vendor Product): name\\n\\nDescription...\\n\\nRequired remediation: ..."}

Source: https://www.cisa.gov/known-exploited-vulnerabilities-catalog
Bulk feed: https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Set


KEV_URL = (
    "https://www.cisa.gov/sites/default/files/feeds/"
    "known_exploited_vulnerabilities.json"
)


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="Collect CISA Known Exploited Vulnerabilities")
    p.add_argument("--output", default="data/raw/cisa_kev.jsonl")
    return p.parse_args()


def render_record(entry: dict) -> dict:
    """Format one KEV entry as a pretrain text record."""
    cve = entry.get("cveID", "").strip()
    vendor = entry.get("vendorProject", "").strip()
    product = entry.get("product", "").strip()
    name = entry.get("vulnerabilityName", "").strip()
    desc = entry.get("shortDescription", "").strip()
    required = entry.get("requiredAction", "").strip()
    due_date = entry.get("dueDate", "").strip()
    date_added = entry.get("dateAdded", "").strip()
    ransomware = entry.get("knownRansomwareCampaignUse", "").strip()
    cwes = entry.get("cwes", []) or []

    head = f"CISA KEV — {cve}"
    if vendor or product:
        head += f" ({vendor} {product})".rstrip(" )") + ")"
        head = head.replace(" )", ")")  # cleanup if vendor empty
    head += f": {name}"

    parts = [head, "", desc]
    if cwes:
        parts.append(f"CWE: {', '.join(cwes[:5])}")
    if ransomware and ransomware.lower() not in {"unknown", "n/a", ""}:
        parts.append(f"Known ransomware use: {ransomware}")
    if required:
        parts.append(f"Required remediation: {required}")
    if due_date:
        parts.append(f"Federal civilian agencies due date: {due_date}")
    if date_added:
        parts.append(f"Added to KEV catalog: {date_added}")

    return {
        "id": cve,
        "source": "cisa_kev",
        "text": "\n\n".join(p for p in parts if p),
    }


def main() -> None:
    """Download the KEV catalog and emit one record per CVE (resume-safe)."""
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
        print(f"  resume: {len(seen)} CVEs already done")

    print(f"  downloading {KEV_URL} ...")
    with urllib.request.urlopen(KEV_URL, timeout=60) as resp:
        data = json.loads(resp.read())

    catalog = data.get("vulnerabilities", [])
    print(f"  catalog has {len(catalog):,} CVEs (catalog version {data.get('catalogVersion', '?')})")

    written = 0
    out_fh = out_path.open("a", encoding="utf-8", buffering=1)
    for entry in catalog:
        cve = entry.get("cveID", "").strip()
        if not cve or cve in seen:
            continue
        rec = render_record(entry)
        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_fh.flush()
        written += 1
        seen.add(cve)
    out_fh.close()

    print()
    print(f"Done. Wrote {written} new CISA KEV records to {out_path}")


if __name__ == "__main__":
    main()
