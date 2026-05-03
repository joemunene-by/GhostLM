#!/usr/bin/env python3
"""Convert letter-only MCQ chat data into text-loss format.

Reads ``data/raw/chat/mcq.jsonl`` (assistant turn = "B") and writes
``data/raw/chat/mcq_text.jsonl`` where the assistant turn is the FULL
option text (e.g. "Phishing attacks rely on social engineering..."),
optionally prefixed with the letter ("B. Phishing attacks rely on...").

Why: the eval_debiased.py / eval_text_scoring.py results show that
letter-only SFT teaches the model to emit a single high-frequency letter
without learning content. Switching the SFT loss to score the answer
TEXT removes the letter-shortcut path and aligns training with the
text-scoring evaluation.

Two output formats:

- ``letter+text`` (default): assistant = "B. Phishing attacks ..." so the
  model still learns to emit a letter for backward-compat with the
  letter-scoring bench, but the loss is dominated by the text.
- ``text-only``: assistant = "Phishing attacks ..." (no letter prefix).
  Pure text-loss; cleanest test of whether content learning works.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Optional


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="Convert letter-only MCQ to text-loss format")
    p.add_argument("--in-mcq", default="data/raw/chat/mcq.jsonl",
                   help="Input letter-only MCQ JSONL from build_mcq_data.py")
    p.add_argument("--out", default="data/raw/chat/mcq_text.jsonl")
    p.add_argument("--format", choices=["letter+text", "text-only"],
                   default="letter+text",
                   help="letter+text keeps the letter prefix (B. <text>); "
                        "text-only is just the option content (no letter)")
    return p.parse_args()


def parse_existing_mcq(record: Dict) -> Optional[Dict]:
    """Recover (question, options A/B/C/D, correct letter) from a letter-only
    MCQ record. Same logic as build_mcq_cot_data.parse_existing_mcq."""
    user_text = record["turns"][0]["content"]
    assistant_text = record["turns"][1]["content"]

    options: Dict[str, str] = {}
    for letter in "ABCD":
        opt_match = re.search(rf"^{letter}\)\s*(.+?)$", user_text, re.MULTILINE)
        if opt_match:
            options[letter] = opt_match.group(1).strip()
    if len(options) != 4:
        return None

    correct_letter = assistant_text.strip()[:1].upper()
    if correct_letter not in options:
        return None

    return {
        "options": options,
        "letter": correct_letter,
        "correct_text": options[correct_letter],
    }


def main() -> None:
    """Rewrite each MCQ record with text-loss assistant turn."""
    args = parse_args()
    in_path = Path(args.in_mcq)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    out_fh = out_path.open("w", encoding="utf-8")
    written = 0
    skipped = 0
    for rec in records:
        parsed = parse_existing_mcq(rec)
        if parsed is None:
            skipped += 1
            continue

        if args.format == "letter+text":
            assistant_content = f"{parsed['letter']}. {parsed['correct_text']}"
        else:  # text-only
            assistant_content = parsed["correct_text"]

        new_rec = {
            "turns": [
                {"role": "user", "content": rec["turns"][0]["content"]},
                {"role": "assistant", "content": assistant_content},
            ],
            "source": "mcq_text",
        }
        out_fh.write(json.dumps(new_rec, ensure_ascii=False) + "\n")
        written += 1
    out_fh.close()

    print(f"Wrote {written} text-loss MCQ records to {out_path} ({args.format} format)")
    if skipped:
        print(f"  Skipped {skipped} unparseable inputs")


if __name__ == "__main__":
    main()
