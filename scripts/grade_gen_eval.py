#!/usr/bin/env python3
"""Grade a model's free-text answers against the GhostLM generation eval set.

    python scripts/grade_gen_eval.py answers.jsonl [--eval data/raw/ghostlm_gen_eval.jsonl ...]

The eval items live in data/raw/ghostlm_gen_eval*.jsonl, each with a prompt (a string or
a list of chat turns), a human reference answer, and mechanical ``checks``. ``answers.jsonl``
is ``{"id": ..., "answer": ...}`` per line, produced by running the prompts through the model
under test. The mechanical checks give a fast, reproducible first-pass score; the reference
text is there so a human can spot-grade the borderline ones, which is the intended use for the
reasoning and refusal categories. Prints a per-category table and writes answers.graded.jsonl.

Check keys (all case-insensitive substring tests on the answer):
  must_contain_all    every phrase present
  must_contain_any*   at least one phrase present (any number of independent groups: must_contain_any, _any_2, ...)
  must_not_contain    none of these phrases present
  runnable_python     the answer's first fenced python block compiles
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVALS = sorted((ROOT / "data/raw").glob("ghostlm_gen_eval*.jsonl"))
CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def load_eval(paths: list[Path]) -> dict:
    items = {}
    for p in paths:
        for line in p.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                items[rec["id"]] = rec
    return items


def python_compiles(answer: str) -> bool:
    m = CODE_BLOCK.search(answer)
    code = m.group(1) if m else answer
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def grade_one(answer: str, checks: dict) -> tuple[bool, list[str]]:
    low = answer.lower()
    fails = []
    for key, value in checks.items():
        if key == "must_contain_all":
            missing = [p for p in value if p.lower() not in low]
            if missing:
                fails.append(f"missing required: {missing}")
        elif key.startswith("must_contain_any"):
            if not any(p.lower() in low for p in value):
                fails.append(f"none of: {value}")
        elif key == "must_not_contain":
            present = [p for p in value if p.lower() in low]
            if present:
                fails.append(f"forbidden present: {present}")
        elif key == "runnable_python":
            if value and not python_compiles(answer):
                fails.append("python does not parse")
    return (not fails), fails


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("answers", type=Path)
    p.add_argument("--eval", type=Path, nargs="*", default=DEFAULT_EVALS)
    a = p.parse_args()
    items = load_eval(a.eval)
    answers = {}
    for line in a.answers.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            answers[r["id"]] = r.get("answer", "")

    by_cat: dict[str, list[bool]] = defaultdict(list)
    graded = []
    missing_answers = 0
    for item_id, item in items.items():
        if item_id not in answers:
            missing_answers += 1
            continue
        passed, fails = grade_one(answers[item_id], item.get("checks", {}))
        by_cat[item["category"]].append(passed)
        graded.append({"id": item_id, "category": item["category"], "trap": item.get("trap", False),
                       "passed": passed, "fails": fails})

    out = a.answers.with_suffix(".graded.jsonl")
    out.write_text("".join(json.dumps(g, ensure_ascii=False) + "\n" for g in graded))

    print(f"Graded {len(graded)} of {len(items)} items ({missing_answers} without an answer)\n")
    print(f"{'Category':28s} {'pass':>6s} {'n':>4s}  rate")
    print("-" * 52)
    all_pass = all_n = 0
    for cat in sorted(by_cat):
        res = by_cat[cat]
        pc = sum(res)
        all_pass += pc
        all_n += len(res)
        print(f"{cat:28s} {pc:>6d} {len(res):>4d}  {pc / len(res) * 100:4.0f}%")
    print("-" * 52)
    if all_n:
        print(f"{'TOTAL':28s} {all_pass:>6d} {all_n:>4d}  {all_pass / all_n * 100:4.0f}%")
    traps = [g for g in graded if g["trap"]]
    if traps:
        tp = sum(g["passed"] for g in traps)
        print(f"\nTraps (hallucination/refusal): {tp}/{len(traps)} handled correctly")
    print(f"\nPer-item detail in {out}. Reference answers in the eval files for human spot-grading.")


if __name__ == "__main__":
    main()
