#!/usr/bin/env python3
"""Fetch SecQA v1 + v2 from HuggingFace and convert to GhostLM MCQ format.

External cybersec MCQ bench for cross-validating the v0.9 in-repo CTF
result. SecQA was published with the LLM-cybersec literature
(Liu et al.) and is structurally similar to CTIBench: 4-way MCQ with
a single correct letter, but the topic mix is broader (general
cybersec knowledge rather than CTI-specific). Output schema matches
``data/raw/ctf_eval_bench.jsonl`` so ``eval_text_scoring.py
--bench-jsonl`` can run it without modification.

Source: https://huggingface.co/datasets/zefang-liu/secqa
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="Fetch SecQA and convert to MCQ JSONL")
    p.add_argument("--out", default="data/raw/secqa.jsonl")
    p.add_argument("--configs", nargs="+", default=["secqa_v1", "secqa_v2"])
    p.add_argument("--split", default="test")
    return p.parse_args()


def main() -> None:
    """Pull and write."""
    from datasets import load_dataset
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out.open("w", encoding="utf-8") as f:
        for conf in args.configs:
            ds = load_dataset("zefang-liu/secqa", conf, split=args.split)
            for i, rec in enumerate(ds):
                row = {
                    "id": f"{conf}-{args.split}-{i:03d}",
                    "source": "secqa",
                    "question": rec["Question"],
                    "choices": {
                        "A": rec["A"],
                        "B": rec["B"],
                        "C": rec["C"],
                        "D": rec["D"],
                    },
                    "answer": rec["Answer"],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
    print(f"Wrote {written} SecQA records to {out}")


if __name__ == "__main__":
    main()
