#!/usr/bin/env python3
"""Pull MITRE CWE (Common Weakness Enumeration) into a GhostLM corpus.

CWE is a structured catalog of software/hardware weaknesses,
maintained by MITRE under a free-redistribution license. Each entry
has a numeric ID (e.g. CWE-79 = "Improper Neutralization of Input
During Web Page Generation (XSS)"), a name, a description, technical
details on how the weakness manifests, common consequences, mitigation
guidance, and references. Fact density is very high.

Source: https://cwe.mitre.org/data/xml/cwec_latest.xml.zip
License: free for research and commercial use.

Output: ``data/raw/cwe.jsonl`` with the standard
``{"id", "source", "text"}`` schema. Each CWE becomes one record. The
text is assembled as title + description + extended description +
common consequences + mitigations.
"""

from __future__ import annotations

import argparse
import io
import json
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path


CWE_URL = "https://cwe.mitre.org/data/xml/cwec_latest.xml.zip"
NS = {"cwe": "http://cwe.mitre.org/cwe-7"}


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="Collect MITRE CWE records")
    p.add_argument("--out", default="data/raw/cwe.jsonl")
    p.add_argument("--cache", default="data/raw/.cwe_xml.zip",
                   help="Cache the downloaded zip so re-runs do not re-download")
    return p.parse_args()


def fetch_xml(cache_path: Path) -> bytes:
    """Download (or read cached) CWE zip and return the inner XML bytes."""
    if not cache_path.exists():
        print(f"  downloading {CWE_URL}...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CWE_URL, cache_path)
        print(f"  cached to {cache_path}")
    with zipfile.ZipFile(cache_path) as zf:
        # The zip contains exactly one XML; pick the first non-directory entry.
        for name in zf.namelist():
            if name.endswith(".xml"):
                return zf.read(name)
    raise SystemExit("No XML found inside CWE zip")


def text_from(elem) -> str:
    """Recursively extract text content from a CWE XHTML-bearing element."""
    if elem is None:
        return ""
    parts = []
    if elem.text:
        parts.append(elem.text.strip())
    for child in list(elem):
        parts.append(text_from(child))
        if child.tail:
            parts.append(child.tail.strip())
    return " ".join(p for p in parts if p)


def render_cwe(weakness) -> dict | None:
    """Convert one Weakness XML element to a training record dict."""
    cwe_id = weakness.get("ID")
    name = weakness.get("Name") or ""
    abstraction = weakness.get("Abstraction") or ""
    if not cwe_id or not name:
        return None

    description = text_from(weakness.find("cwe:Description", NS))
    extended = text_from(weakness.find("cwe:Extended_Description", NS))

    consequences = []
    for cons in weakness.findall("cwe:Common_Consequences/cwe:Consequence", NS):
        scope = ", ".join(text_from(s) for s in cons.findall("cwe:Scope", NS) if text_from(s))
        impact = ", ".join(text_from(i) for i in cons.findall("cwe:Impact", NS) if text_from(i))
        note = text_from(cons.find("cwe:Note", NS))
        bits = []
        if scope:
            bits.append(f"Scope: {scope}")
        if impact:
            bits.append(f"Impact: {impact}")
        if note:
            bits.append(note)
        if bits:
            consequences.append(" - " + ". ".join(bits))

    mitigations = []
    for mit in weakness.findall("cwe:Potential_Mitigations/cwe:Mitigation", NS):
        phase = ", ".join(text_from(p) for p in mit.findall("cwe:Phase", NS) if text_from(p))
        desc = text_from(mit.find("cwe:Description", NS))
        if desc:
            prefix = f"[{phase}] " if phase else ""
            mitigations.append(f" - {prefix}{desc}")

    parts = [f"CWE-{cwe_id}: {name}"]
    if abstraction:
        parts.append(f"Abstraction: {abstraction}")
    if description:
        parts.append(f"\n{description}")
    if extended:
        parts.append(f"\n{extended}")
    if consequences:
        parts.append("\nCommon Consequences:\n" + "\n".join(consequences))
    if mitigations:
        parts.append("\nPotential Mitigations:\n" + "\n".join(mitigations))

    return {
        "id": f"CWE-{cwe_id}",
        "source": "cwe",
        "text": "\n".join(parts).strip(),
    }


def main() -> None:
    """Pull CWE XML, parse, emit JSONL."""
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    xml_bytes = fetch_xml(Path(args.cache))
    root = ET.fromstring(xml_bytes)
    weaknesses = root.findall(".//cwe:Weaknesses/cwe:Weakness", NS)
    print(f"  parsed {len(weaknesses)} weaknesses")

    out_fh = out_path.open("w", encoding="utf-8")
    written = 0
    skipped = 0
    for w in weaknesses:
        rec = render_cwe(w)
        if rec is None or len(rec["text"]) < 100:
            skipped += 1
            continue
        out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        written += 1
    out_fh.close()
    print(f"Wrote {written} CWE records to {out_path} (skipped {skipped})")


if __name__ == "__main__":
    main()
