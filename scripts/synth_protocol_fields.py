#!/usr/bin/env python3
"""Templated synthesis of network-protocol-field training records (bet 12).

Bet 12 ([docs/differentiation.md](differentiation.md) §"Bet 12: network
protocol field reading") trains ghost-base on protocol-aware reading
of wire-format hex bytes. One layer up from bet 8 (file-magic and
binary literacy), this is about wire formats: TLS handshakes, DNS
queries, HTTP/2 frames, BGP UPDATEs, ICMP, IP, TCP, Ethernet, ARP,
SMB, Kerberos, QUIC, MQTT, RDP, plus JA3 TLS fingerprinting.

Real network forensics workflows are byte-by-byte: an analyst stares
at a hex dump or a Wireshark frame and asks 'what protocol is this,
which fields are at which offsets, what's anomalous'. Big LMs do
this poorly because their pretrain saw essentially zero raw protocol
bytes. A small from-scratch LM trained natively on this distribution
is the genuinely novel artifact.

Seed: ``data/raw/protocol_field_patterns.jsonl``, 20 patterns
across 6 protocol layers (datalink, network, transport, application,
plus QUIC mixing transport+application, plus JA3 as a derived
fingerprint).

Output formats per pattern:

  1. ``pretrain_prose``   flat markdown article: protocol, hex
                           pattern, fields-at-offsets table, prose
                           explanation. Right shape for pretrain
                           mixing.
  2. ``identify_protocol`` chat Q&A. USER pastes hex bytes, asks
                            'what protocol/message is this?'.
                            ASSISTANT names it + walks through the
                            byte-level reasoning.
  3. ``read_field``        chat Q&A. USER pastes hex + asks 'what is
                            the value of field X?'. ASSISTANT cites
                            the offset and decodes the field.

20 patterns x 3 variants = 60 records. Most patterns produce all 3
variants; the JA3-style derived fingerprint pattern only produces
pretrain_prose since it isn't a literal hex dump.

Run:

    PYTHONPATH=. python3 scripts/synth_protocol_fields.py \\
        --bank data/raw/protocol_field_patterns.jsonl \\
        --out data/processed/synth_protocol_fields.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_record(seed_id: str, variant: str, text: str) -> Dict[str, str]:
    h = hashlib.sha1(
        f"{seed_id}\n{variant}\n{text}".encode("utf-8")
    ).hexdigest()[:10]
    return {
        "id": f"synth_protocol_fields#{seed_id}#{variant}#{h}",
        "source": "synth_protocol_fields",
        "teacher": "templated",
        "seed_source": variant,
        "seed_id": seed_id,
        "text": text,
    }


def fields_table(fields: List[Dict[str, str]]) -> str:
    if not fields:
        return ""
    lines = ["| Offset | Field | Value |", "|---|---|---|"]
    for f in fields:
        lines.append(f"| `{f.get('offset', '?')}` | "
                     f"{f.get('name', '?')} | "
                     f"{f.get('value', '?')} |")
    return "\n".join(lines)


def fence(blob: str) -> str:
    return f"```\n{blob}\n```"


def pretrain_prose(p: Dict) -> str:
    return (
        f"# Protocol pattern: {p['protocol']} ({p['layer']} layer)\n\n"
        f"## Wire bytes\n\n{fence(p['hex_pattern'])}\n\n"
        f"ASCII decode: `{p.get('ascii_decode', '(non-printable)')}`\n\n"
        f"## Fields at offsets\n\n"
        f"{fields_table(p.get('fields_at_offsets', []))}\n\n"
        f"## Analyst interpretation\n\n{p['explanation']}\n"
    )


def identify_protocol(p: Dict) -> str:
    return (
        f"USER: I see this byte sequence on the wire. What protocol "
        f"and message type is this, and how can you tell?\n\n"
        f"{fence(p['hex_pattern'])}\n\n"
        f"ASSISTANT: This is {p['protocol']} ({p['layer']} layer). "
        f"{p['explanation']}\n"
    )


def read_field(p: Dict) -> str:
    fields = p.get("fields_at_offsets", [])
    if not fields:
        return ""  # Caller will skip empty results.
    bullets = "\n".join(
        f"- Offset `{f.get('offset', '?')}`: **{f.get('name', '?')}** = "
        f"{f.get('value', '?')}"
        for f in fields
    )
    return (
        f"USER: Walk me through the fields in this {p['protocol']} "
        f"capture, byte by byte:\n\n"
        f"{fence(p['hex_pattern'])}\n\n"
        f"ASSISTANT: Field-by-field breakdown:\n\n"
        f"{bullets}\n\n"
        f"{p['explanation']}\n"
    )


VARIANTS = [
    ("pretrain_prose", pretrain_prose),
    ("identify_protocol", identify_protocol),
    ("read_field", read_field),
]


def quality_ok(text: str, min_words: int = 60, max_words: int = 1800) -> bool:
    if not text:
        return False
    words = text.split()
    if not (min_words <= len(words) <= max_words):
        return False
    if "```" not in text:
        return False
    return True


def stream_bank(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", default="data/raw/protocol_field_patterns.jsonl")
    p.add_argument("--out", default="data/processed/synth_protocol_fields.jsonl")
    p.add_argument("--variants", default=",".join(v[0] for v in VARIANTS))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bank_path = REPO_ROOT / args.bank if not Path(args.bank).is_absolute() \
                else Path(args.bank)
    out_path = REPO_ROOT / args.out if not Path(args.out).is_absolute() \
               else Path(args.out)
    if not bank_path.exists():
        sys.exit(f"pattern bank not found: {bank_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wanted = {v.strip() for v in args.variants.split(",") if v.strip()}
    variants = [(name, fn) for name, fn in VARIANTS if name in wanted]
    if not variants:
        sys.exit(f"no valid variants selected; available: "
                 f"{[v[0] for v in VARIANTS]}")

    counts: Dict[str, int] = {}
    rejects: Dict[str, int] = {}
    n_total = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for pattern in stream_bank(bank_path):
            for variant_name, fn in variants:
                text = fn(pattern)
                if not quality_ok(text):
                    rejects[variant_name] = rejects.get(variant_name, 0) + 1
                    continue
                rec = build_record(pattern["id"], variant_name, text)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[variant_name] = counts.get(variant_name, 0) + 1
                n_total += 1

    print(f"Wrote {n_total} records to {out_path}")
    print(f"  by variant: {counts}")
    print(f"  rejects:    {rejects}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
