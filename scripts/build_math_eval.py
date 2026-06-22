#!/usr/bin/env python3
"""Build a deterministic math/reasoning MCQ benchmark for GhostLM.

GhostLM had no math eval. This generates `data/raw/math_mcq_bench.jsonl`
in the project's standard MCQ schema ({question, choices:{A..D}, answer,
bench}) so it scores through the same debiased text-scoring path as the
general rulers. Problems are templated and deterministic (fixed seed),
with a computed gold answer and three plausible numeric distractors, so
the set is reproducible and not scraped (zero contamination risk).

Coverage: arithmetic, percentages, ratios/rates, simple linear algebra,
sequences, and short word problems — the everyday numeracy band a small
model can plausibly reach, and a clean ruler to watch climb with scale.

Usage:
    python scripts/build_math_eval.py --n 120 --out data/raw/math_mcq_bench.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from math import gcd as _gcd
from pathlib import Path


def _distractors(ans: int, rng: random.Random, k: int = 3) -> list[int]:
    out: set[int] = set()
    deltas = [1, -1, 2, -2, 5, -5, 10, -10, ans and ans // 2, ans + ans // 3]
    while len(out) < k:
        d = rng.choice(deltas) or rng.randint(2, 9)
        cand = ans + d
        if cand != ans and cand not in out and cand >= 0:
            out.add(cand)
    return list(out)


def _mc(question: str, ans: int, rng: random.Random, bench: str) -> dict:
    opts = _distractors(ans, rng) + [ans]
    rng.shuffle(opts)
    letters = ["A", "B", "C", "D"]
    choices = {letters[i]: str(opts[i]) for i in range(4)}
    gold = letters[opts.index(ans)]
    return {"question": question, "choices": choices, "answer": gold, "bench": bench}


def gen(n: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    out: list[dict] = []
    makers = ["arith", "pct", "rate", "algebra", "seq", "word"]
    while len(out) < n:
        kind = makers[len(out) % len(makers)]
        if kind == "arith":
            a, b, c = rng.randint(12, 99), rng.randint(3, 19), rng.randint(2, 12)
            q = f"Compute: {a} + {b} * {c}."
            out.append(_mc(q, a + b * c, rng, "math_arithmetic"))
        elif kind == "pct":
            p = rng.choice([5, 10, 15, 20, 25, 40, 50])
            # pick a base that divides evenly so the gold answer is exact
            base = rng.randint(2, 20) * (100 // _gcd(p, 100))
            q = f"What is {p}% of {base}?"
            out.append(_mc(q, base * p // 100, rng, "math_percent"))
        elif kind == "rate":
            speed, hrs = rng.randint(30, 90), rng.randint(2, 9)
            q = f"A vehicle travels at {speed} km/h for {hrs} hours. How many km does it cover?"
            out.append(_mc(q, speed * hrs, rng, "math_rate"))
        elif kind == "algebra":
            x, m, b = rng.randint(2, 20), rng.randint(2, 9), rng.randint(1, 30)
            y = m * x + b
            q = f"If {m}x + {b} = {y}, what is x?"
            out.append(_mc(q, x, rng, "math_algebra"))
        elif kind == "seq":
            start, step = rng.randint(1, 12), rng.randint(2, 9)
            terms = [start + step * i for i in range(4)]
            q = f"What is the next number in the sequence {terms[0]}, {terms[1]}, {terms[2]}, {terms[3]}, ...?"
            out.append(_mc(q, terms[3] + step, rng, "math_sequence"))
        else:  # word
            had, gave, got = rng.randint(20, 80), rng.randint(5, 19), rng.randint(5, 25)
            q = f"Sam had {had} tokens, gave away {gave}, then earned {got} more. How many tokens does Sam have now?"
            out.append(_mc(q, had - gave + got, rng, "math_word"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/raw/math_mcq_bench.jsonl")
    a = ap.parse_args()
    recs = gen(a.n, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    by: dict[str, int] = {}
    for r in recs:
        by[r["bench"]] = by.get(r["bench"], 0) + 1
    print(f"wrote {len(recs)} math MCQ records -> {out}")
    for k, v in sorted(by.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
