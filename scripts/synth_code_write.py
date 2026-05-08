#!/usr/bin/env python3
"""Templated synthesis of code-writing training records (v0.9.24).

Companion to scripts/synth_code_explain.py. Where code-explain
shows a snippet and asks the model to interpret it, code-write
shows a description and asks the model to produce the code.

Variants per pattern:
  1. pretrain_prose   markdown article describing how to do X in Y,
                       with the canonical implementation.
  2. write_function   USER asks "write a function that does X in Y";
                       ASSISTANT gives the code plus a short note.
  3. write_idiomatic  USER asks "what is the idiomatic Y way to X";
                       ASSISTANT shows the implementation and
                       explains why it's idiomatic.
  4. compare          When the pattern has alternative_implementations,
                       USER asks "show two ways to X in Y";
                       ASSISTANT shows both with tradeoffs.

40 patterns × 3-4 variants = 120-160 records.

Run:

    PYTHONPATH=. python3 scripts/synth_code_write.py \\
        --bank data/raw/code_write_patterns.jsonl \\
        --out data/processed/synth_code_write.jsonl
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
        "id": f"synth_code_write#{seed_id}#{variant}#{h}",
        "source": "synth_code_write",
        "teacher": "templated",
        "seed_source": variant,
        "seed_id": seed_id,
        "text": text,
    }


def fence(code: str, lang: str) -> str:
    return f"```{lang}\n{code}\n```"


def _qa(user: str, asst: str) -> str:
    return f"USER: {user}\n\nASSISTANT: {asst}\n"


def pretrain_prose(p: Dict) -> str:
    parts = [
        f"# {p['language'].title()}: {p['description']}",
        "",
        p["explanation"],
        "",
        fence(p["implementation"], p["language"]),
    ]
    return "\n".join(parts)


def write_function_qa(p: Dict) -> str:
    user = (
        f"Write a {p['language']} function that does the following: "
        f"{p['description'].lower()}"
    )
    asst = (
        f"{fence(p['implementation'], p['language'])}\n\n"
        + p["explanation"]
    )
    return _qa(user, asst)


def write_idiomatic_qa(p: Dict) -> str:
    user = (
        f"What is the idiomatic {p['language'].title()} way to "
        f"{p['description'].lower()}?"
    )
    asst = (
        p["explanation"]
        + "\n\n"
        + fence(p["implementation"], p["language"])
    )
    return _qa(user, asst)


def compare_qa(p: Dict) -> str:
    alts = p.get("alternative_implementations") or []
    if not alts:
        return ""
    user = (
        f"Show two different ways to {p['description'].lower()} in "
        f"{p['language'].title()}. Briefly note the tradeoffs."
    )
    primary_block = fence(p["implementation"], p["language"])
    alt_blocks = []
    for alt in alts[:2]:  # cap at 2 alternatives for record size
        alt_blocks.append(fence(alt["code"], p["language"]))
        alt_blocks.append(alt.get("note", ""))
    asst = (
        "Approach 1:\n\n"
        + primary_block
        + f"\n\n{p['explanation']}\n\n"
        + "Approach 2:\n\n"
        + "\n\n".join(alt_blocks)
    )
    return _qa(user, asst)


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


def quality_ok(text: str, min_words: int = 25,
                max_words: int = 1500) -> bool:
    n = len(text.split())
    return min_words <= n <= max_words


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="scripts/synth_code_write.py")
    p.add_argument("--bank",
                    default="data/raw/code_write_patterns.jsonl")
    p.add_argument("--out",
                    default="data/processed/synth_code_write.jsonl")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bank = Path(args.bank)
    if not bank.exists():
        print(f"[error] bank not found: {bank}", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    counts: Dict[str, int] = {}
    rejects: Dict[str, int] = {}
    n_total = 0
    with out.open("w", encoding="utf-8") as fout:
        for pattern in stream_bank(bank):
            pid = pattern["id"]
            variants = [
                ("pretrain_prose", pretrain_prose),
                ("write_function", write_function_qa),
                ("write_idiomatic", write_idiomatic_qa),
                ("compare", compare_qa),
            ]
            for vname, fn in variants:
                text = fn(pattern)
                if not text or not quality_ok(text):
                    rejects[vname] = rejects.get(vname, 0) + 1
                    continue
                rec = build_record(pid, vname, text)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[vname] = counts.get(vname, 0) + 1
                n_total += 1

    print(f"Wrote {n_total} records to {out}")
    print(f"  by variant: {counts}")
    if rejects:
        print(f"  rejects:    {rejects}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
