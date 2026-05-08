#!/usr/bin/env python3
"""Templated synthesis of binary / hex literacy training records (bet 8).

Bet 8 ([docs/differentiation.md](differentiation.md) §"Bet 8: binary-
and-hex literacy") aims to give ghost-base reading comprehension on
the byte-level artifacts that real reverse engineers and forensics
analysts handle: PE / ELF / Mach-O headers, packer signatures,
shellcode patterns, raw hex dumps, disassembly snippets. Big LMs do
this poorly because their pretrain saw vanishingly little of it.

This is the most novel bet in the strategic frame: no other small
cybersec LM trains on this distribution natively. The capability
maps directly to malware analysis and incident-response workflows
where the analyst is staring at hexdump / objdump / strings output.

Seed: ``data/raw/binary_literacy_patterns.jsonl``, a hand-curated
bank of (category, name, hex pattern, ASCII decode, explanation,
examples) tuples covering five categories: file_magic, packer,
shellcode, pe_field, disassembly.

Output formats per pattern:

  1. **`pretrain_prose`**: flat markdown article presenting the
     hex pattern, ASCII decode, longer signature, and explanation.
     Right shape for pretrain corpus mixing.
  2. **`identify_hex`**: USER shows hex bytes, asks 'what is this';
     ASSISTANT names the format and explains the byte-level reasoning.
  3. **`show_magic`**: USER asks 'show me the magic bytes of <X>';
     ASSISTANT gives hex + ASCII decode + structural context. (Skipped
     for disassembly patterns where 'magic' is not a meaningful
     concept.)

15 patterns x 2-3 variants = ~40-45 records, all parser-clean.

Run:

    PYTHONPATH=. python3 scripts/synth_binary_literacy.py \\
        --bank data/raw/binary_literacy_patterns.jsonl \\
        --out data/processed/synth_binary_literacy.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_record(seed_id: str, variant: str, text: str) -> Dict[str, str]:
    h = hashlib.sha1(
        f"{seed_id}\n{variant}\n{text}".encode("utf-8")
    ).hexdigest()[:10]
    return {
        "id": f"synth_binary_literacy#{seed_id}#{variant}#{h}",
        "source": "synth_binary_literacy",
        "teacher": "templated",
        "seed_source": variant,
        "seed_id": seed_id,
        "text": text,
    }


def examples_phrase(examples: List[str]) -> str:
    if not examples:
        return ""
    if len(examples) == 1:
        return f"\n\nReal-world example: {examples[0]}."
    return f"\n\nReal-world examples: {'; '.join(examples)}."


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


def pretrain_prose(p: Dict) -> str:
    """Variant 1: flat markdown article."""
    cat_label = p["category"].replace("_", " ").title()
    return (
        f"# Binary literacy: {p['name']} ({cat_label})\n\n"
        f"## Bytes\n\n"
        f"At offset 0 (or anchor point): `{p['hex_at_offset_0']}`\n"
        f"ASCII decode: `{p['ascii_decode']}`\n"
        f"Longer pattern: `{p['longer_pattern']}`\n\n"
        f"## What this means\n\n"
        f"{p['explanation']}"
        f"{examples_phrase(p.get('examples', []))}\n"
    )


def identify_hex(p: Dict) -> str:
    """Variant 2: hex -> name + reasoning."""
    return (
        f"USER: I am looking at this byte sequence in a hex editor at "
        f"the start of a file: `{p['longer_pattern']}`. What is this, "
        f"and how do you know?\n\n"
        f"ASSISTANT: This is {p['name']}. {p['explanation']}"
        f"{examples_phrase(p.get('examples', []))}\n"
    )


def show_magic(p: Dict) -> str:
    """Variant 3: name -> magic bytes + context."""
    return (
        f"USER: Show me the magic bytes / signature of {p['name']}, "
        f"and explain what comes immediately after.\n\n"
        f"ASSISTANT: The signature is `{p['hex_at_offset_0']}` "
        f"(ASCII: `{p['ascii_decode']}`). A longer canonical pattern "
        f"is `{p['longer_pattern']}`. {p['explanation']}\n"
    )


# Categories where show_magic is meaningful. Disassembly snippets
# do not have a 'magic' concept.
SHOW_MAGIC_CATEGORIES = {"file_magic", "packer", "shellcode", "pe_field"}


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------


def quality_ok(text: str, min_words: int = 50, max_words: int = 1500) -> bool:
    words = text.split()
    if not (min_words <= len(words) <= max_words):
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
    p.add_argument("--bank", default="data/raw/binary_literacy_patterns.jsonl")
    p.add_argument("--out", default="data/processed/synth_binary_literacy.jsonl")
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

    counts: Dict[str, int] = {}
    rejects: Dict[str, int] = {}
    n_total = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for pattern in stream_bank(bank_path):
            variants_for_this = [("pretrain_prose", pretrain_prose),
                                 ("identify_hex", identify_hex)]
            if pattern.get("category") in SHOW_MAGIC_CATEGORIES:
                variants_for_this.append(("show_magic", show_magic))

            for vname, fn in variants_for_this:
                text = fn(pattern)
                if not quality_ok(text):
                    rejects[vname] = rejects.get(vname, 0) + 1
                    continue
                rec = build_record(pattern["id"], vname, text)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[vname] = counts.get(vname, 0) + 1
                n_total += 1

    print(f"Wrote {n_total} records to {out_path}")
    print(f"  by variant: {counts}")
    print(f"  rejects:    {rejects}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
