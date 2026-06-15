#!/usr/bin/env python3
"""Fetch general-domain MCQ benchmarks and convert to GhostLM MCQ format.

GhostLM's eval suite is entirely cybersecurity (CTIBench, SecQA, the
in-repo CTF set, the cybersec fact-recall bench). That measures the
narrow specialty but says nothing about whether the generalist corpus
pivot is working. This fetcher pulls three standard general-knowledge /
reasoning MCQ rulers so general capability is measurable with the same
debiased text-scoring methodology:

  - ARC-Easy        grade-school science, easier split   (CC BY-SA 4.0)
  - ARC-Challenge   grade-school science, harder split   (CC BY-SA 4.0)
  - OpenBookQA      science + commonsense application     (Apache-2.0)

All three are 4-way MCQ with a single correct letter, structurally
identical to CTIBench, so ``scripts/eval_text_scoring.py --bench-jsonl
<out> --prompt-style general`` runs them without modification. Records
carry ``"domain": "general"`` so the eval drops the cybersec framing
automatically.

Output schema matches ``data/raw/ctf_eval_bench.jsonl``:
``{id, source, bench, domain, question, choices: {A,B,C,D}, answer}``.
Questions whose option count is not exactly 4 are skipped (a small tail
of ARC items have 3 or 5 options) so every record maps cleanly to A-D.

Source datasets:
  https://huggingface.co/datasets/allenai/ai2_arc
  https://huggingface.co/datasets/allenai/openbookqa
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Numeric answer keys appear in a few ARC items ("1".."4"); normalize them.
_NUM_TO_LETTER = {"1": "A", "2": "B", "3": "C", "4": "D"}
_LETTERS = ["A", "B", "C", "D"]


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="Fetch general MCQ benches to JSONL")
    p.add_argument("--out", default="data/raw/general_mcq_bench.jsonl")
    p.add_argument("--split", default="test",
                   help="Dataset split (test is the standard eval split)")
    p.add_argument("--benches", nargs="+",
                   default=["arc_easy", "arc_challenge", "openbookqa"],
                   help="Which benches to include")
    p.add_argument("--limit-per-bench", type=int, default=None,
                   help="Cap records per bench (for a quick smoke set)")
    return p.parse_args()


def _norm_answer(key: str) -> str:
    """Map a raw answerKey to an A-D letter, or '' if unmappable."""
    key = (key or "").strip()
    if key in _NUM_TO_LETTER:
        return _NUM_TO_LETTER[key]
    if key in _LETTERS:
        return key
    return ""


def _arc_rows(split: str, config: str):
    """Yield normalized rows from an ARC config."""
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", config, split=split)
    for i, rec in enumerate(ds):
        texts = rec["choices"]["text"]
        labels = rec["choices"]["label"]
        if len(texts) != 4:
            continue
        # ARC labels are usually A-D but occasionally 1-4; remap by position
        # to keep a clean A-D mapping and translate the answer key the same way.
        pos_by_label = {lab: idx for idx, lab in enumerate(labels)}
        ans_raw = rec["answerKey"].strip()
        if ans_raw in pos_by_label:
            ans_pos = pos_by_label[ans_raw]
        elif ans_raw in _NUM_TO_LETTER and _NUM_TO_LETTER[ans_raw] in pos_by_label:
            ans_pos = pos_by_label[_NUM_TO_LETTER[ans_raw]]
        else:
            continue
        yield {
            "question": rec["question"],
            "choices": {_LETTERS[j]: texts[j] for j in range(4)},
            "answer": _LETTERS[ans_pos],
            "idx": i,
        }


def _openbookqa_rows(split: str):
    """Yield normalized rows from OpenBookQA (config 'main')."""
    from datasets import load_dataset
    ds = load_dataset("allenai/openbookqa", "main", split=split)
    for i, rec in enumerate(ds):
        texts = rec["choices"]["text"]
        if len(texts) != 4:
            continue
        ans = _norm_answer(rec["answerKey"])
        if not ans:
            continue
        yield {
            "question": rec["question_stem"],
            "choices": {_LETTERS[j]: texts[j] for j in range(4)},
            "answer": ans,
            "idx": i,
        }


_BENCH_SOURCES = {
    "arc_easy": lambda split: _arc_rows(split, "ARC-Easy"),
    "arc_challenge": lambda split: _arc_rows(split, "ARC-Challenge"),
    "openbookqa": lambda split: _openbookqa_rows(split),
}


def main() -> None:
    """Pull and write."""
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    per_bench: dict = {}
    with out.open("w", encoding="utf-8") as f:
        for bench in args.benches:
            if bench not in _BENCH_SOURCES:
                print(f"  skip unknown bench: {bench}")
                continue
            n = 0
            for row in _BENCH_SOURCES[bench](args.split):
                if args.limit_per_bench and n >= args.limit_per_bench:
                    break
                f.write(json.dumps({
                    "id": f"{bench}-{args.split}-{row['idx']:04d}",
                    "source": "general_mcq",
                    "bench": bench,
                    "domain": "general",
                    "question": row["question"],
                    "choices": row["choices"],
                    "answer": row["answer"],
                }, ensure_ascii=False) + "\n")
                n += 1
                written += 1
            per_bench[bench] = n
            print(f"  {bench}: {n} records")

    print(f"\nWrote {written} general-MCQ records to {out}")
    for bench, n in per_bench.items():
        print(f"  {bench:16s} {n}")


if __name__ == "__main__":
    main()
