#!/usr/bin/env python3
"""Templated synthesis of code-REASONING / debugging training records.

Fills a gap in the code SFT banks: explain / write / security existed, but
nothing taught the model to *trace a bug, reason about why it fails, and
fix it*. Each pattern carries buggy code, the bug class, a step-by-step
diagnosis, and the corrected code. Output is shaped identically to
synth_code_explain.jsonl ({id, source, teacher, seed_source, seed_id,
text}) so it folds straight into the combined synth corpus.

Usage:
    python scripts/synth_code_reasoning.py --out data/processed/synth_code_reasoning.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

# Each pattern: a real, common bug with an accurate diagnosis + fix.
PATTERNS: List[Dict] = [
    {
        "id": "offbyone_range", "lang": "python", "bug": "off-by-one",
        "buggy": "def last_n(xs, n):\n    return [xs[i] for i in range(len(xs) - n - 1, len(xs))]",
        "symptom": "It returns n+1 items, or raises IndexError when n == len(xs).",
        "diag": "The slice start is `len(xs) - n - 1`, one index too low. To take the last n items the start must be `len(xs) - n`. The extra `- 1` includes one earlier element (n+1 total) and, when n == len(xs), makes the start negative-by-one and the loop reach an out-of-range index.",
        "fix": "def last_n(xs, n):\n    return xs[len(xs) - n:]",
    },
    {
        "id": "mutable_default", "lang": "python", "bug": "mutable default argument",
        "buggy": "def add_tag(tag, tags=[]):\n    tags.append(tag)\n    return tags",
        "symptom": "Tags from previous calls leak into later calls.",
        "diag": "The default list is created once, at function-definition time, and shared across every call that omits `tags`. Each call appends to the same list, so state accumulates between unrelated calls.",
        "fix": "def add_tag(tag, tags=None):\n    if tags is None:\n        tags = []\n    tags.append(tag)\n    return tags",
    },
    {
        "id": "recursion_base", "lang": "python", "bug": "missing base case",
        "buggy": "def fact(n):\n    return n * fact(n - 1)",
        "symptom": "RecursionError: maximum recursion depth exceeded.",
        "diag": "There is no base case, so the recursion never terminates — `fact(0)` calls `fact(-1)`, and so on past the recursion limit. A correct factorial must stop at n <= 1 and return 1.",
        "fix": "def fact(n):\n    if n <= 1:\n        return 1\n    return n * fact(n - 1)",
    },
    {
        "id": "int_div", "lang": "python", "bug": "integer vs float division",
        "buggy": "def average(xs):\n    total = 0\n    for x in xs:\n        total += x\n    return total // len(xs)",
        "symptom": "The average is truncated (e.g. average([1, 2]) returns 1, not 1.5).",
        "diag": "`//` is floor division, which discards the fractional part. An average needs true division `/`. Also guard against an empty list to avoid ZeroDivisionError.",
        "fix": "def average(xs):\n    if not xs:\n        return 0.0\n    return sum(xs) / len(xs)",
    },
    {
        "id": "mutate_while_iter", "lang": "python", "bug": "mutating a list while iterating",
        "buggy": "def drop_evens(xs):\n    for x in xs:\n        if x % 2 == 0:\n            xs.remove(x)\n    return xs",
        "symptom": "Some even numbers survive (e.g. [2, 4, 6] -> [4]).",
        "diag": "Removing items shifts every later element left, so the iterator skips the element that moves into the freed slot. Build a new list (or iterate a copy) instead of mutating the list you are walking.",
        "fix": "def drop_evens(xs):\n    return [x for x in xs if x % 2 != 0]",
    },
    {
        "id": "nil_deref_go", "lang": "go", "bug": "nil map write",
        "buggy": "func count(words []string) map[string]int {\n    var m map[string]int\n    for _, w := range words {\n        m[w]++\n    }\n    return m\n}",
        "symptom": "panic: assignment to entry in nil map.",
        "diag": "`var m map[string]int` declares a nil map. Reading a nil map is fine, but writing to one panics. The map must be allocated with make before any write.",
        "fix": "func count(words []string) map[string]int {\n    m := make(map[string]int)\n    for _, w := range words {\n        m[w]++\n    }\n    return m\n}",
    },
    {
        "id": "await_missing", "lang": "javascript", "bug": "missing await",
        "buggy": "async function load(id) {\n    const res = fetch(`/api/${id}`);\n    return res.json();\n}",
        "symptom": "TypeError: res.json is not a function.",
        "diag": "`fetch` returns a Promise; without `await`, `res` is the Promise itself, not the resolved Response, so `res.json` is undefined. Await the fetch (and the json parse).",
        "fix": "async function load(id) {\n    const res = await fetch(`/api/${id}`);\n    return await res.json();\n}",
    },
    {
        "id": "closure_loop", "lang": "javascript", "bug": "var capture in loop",
        "buggy": "const fns = [];\nfor (var i = 0; i < 3; i++) {\n    fns.push(() => i);\n}\n// fns.map(f => f()) -> [3, 3, 3]",
        "symptom": "Every closure returns 3 instead of 0, 1, 2.",
        "diag": "`var` is function-scoped, so all three closures capture the same `i`, which is 3 after the loop ends. Use `let`, which is block-scoped and creates a fresh binding per iteration.",
        "fix": "const fns = [];\nfor (let i = 0; i < 3; i++) {\n    fns.push(() => i);\n}",
    },
    {
        "id": "resource_leak", "lang": "python", "bug": "unclosed file handle",
        "buggy": "def read_all(path):\n    f = open(path)\n    return f.read()",
        "symptom": "File descriptors leak under load; data may not flush on write.",
        "diag": "`open` is never closed, so the OS handle stays open until garbage collection (non-deterministic). Use a `with` block so the file is closed deterministically even if `read` raises.",
        "fix": "def read_all(path):\n    with open(path) as f:\n        return f.read()",
    },
    {
        "id": "equality_identity", "lang": "python", "bug": "is vs == for values",
        "buggy": "def is_target(x):\n    return x is 1000",
        "symptom": "Returns False for a value that equals 1000.",
        "diag": "`is` tests object identity, not equality. Small integers are cached and may share identity, but 1000 is outside the cache, so `x is 1000` can be False even when `x == 1000`. Compare values with `==`.",
        "fix": "def is_target(x):\n    return x == 1000",
    },
    {
        "id": "shadowed_builtin", "lang": "python", "bug": "shadowed builtin",
        "buggy": "def total(list):\n    return sum(list) / len(list)\n\nlist = []  # later in the module",
        "symptom": "TypeError: 'list' object is not callable, elsewhere in the file.",
        "diag": "Naming a parameter and a module variable `list` rebinds the builtin `list`. Any later `list(...)` call in scope then fails. Rename to a non-builtin like `values`.",
        "fix": "def total(values):\n    return sum(values) / len(values)",
    },
    {
        "id": "index_modulo", "lang": "rust", "bug": "panic on empty slice index",
        "buggy": "fn first(xs: &[i32]) -> i32 {\n    xs[0]\n}",
        "symptom": "thread panicked: index out of bounds on an empty slice.",
        "diag": "Indexing `xs[0]` panics when the slice is empty. Rust's safe idiom is `.first()`, which returns `Option<&i32>`, forcing the empty case to be handled.",
        "fix": "fn first(xs: &[i32]) -> Option<i32> {\n    xs.first().copied()\n}",
    },
]

VARIANTS = ["find", "trace", "fix", "prose"]


def fence(code: str, lang: str) -> str:
    return f"```{lang}\n{code}\n```"


def _qa(user: str, asst: str) -> str:
    return f"USER: {user}\n\nASSISTANT: {asst}\n"


def render(p: Dict, variant: str) -> str:
    buggy, lang = fence(p["buggy"], p["lang"]), p["lang"]
    fixed = fence(p["fix"], p["lang"])
    if variant == "find":
        return _qa(
            f"There is a bug in this {lang} code. What is wrong with it?\n\n{buggy}",
            f"This is a {p['bug']} bug. {p['symptom']}\n\n{p['diag']}",
        )
    if variant == "trace":
        return _qa(
            f"Walk through why this {lang} code misbehaves, step by step.\n\n{buggy}",
            f"Symptom: {p['symptom']}\n\nDiagnosis: {p['diag']}\n\nIt is a classic {p['bug']} bug.",
        )
    if variant == "fix":
        return _qa(
            f"Fix the bug in this {lang} code and explain the fix.\n\n{buggy}",
            f"The bug is a {p['bug']}. {p['diag']}\n\nFixed:\n\n{fixed}",
        )
    # prose (pretrain-style)
    return (
        f"A common {lang} bug is the {p['bug']}. Consider:\n\n{p['buggy']}\n\n"
        f"{p['symptom']} {p['diag']} The corrected version is:\n\n{p['fix']}\n"
    )


def build_record(seed_id: str, variant: str, text: str) -> Dict[str, str]:
    h = hashlib.sha1(f"{seed_id}\n{variant}\n{text}".encode()).hexdigest()[:10]
    return {
        "id": f"synth_code_reasoning#{seed_id}#{variant}#{h}",
        "source": "synth_code_reasoning",
        "teacher": "templated",
        "seed_source": variant,
        "seed_id": seed_id,
        "text": text,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/synth_code_reasoning.jsonl")
    a = ap.parse_args()
    recs = [build_record(p["id"], v, render(p, v)) for p in PATTERNS for v in VARIANTS]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(recs)} records ({len(PATTERNS)} patterns x {len(VARIANTS)} variants) -> {out}")


if __name__ == "__main__":
    main()
