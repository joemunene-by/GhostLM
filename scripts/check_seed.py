"""Validate hand-written chat seed files in data/chat_gen/seed_*.jsonl.

Same bar as gen_chat_sft.py, plus no em/en dashes and no duplicate opening questions across files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_chat_sft import EMOJI, FILLER, LEAK  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAX_CHARS = 4000


def problems(rec: dict) -> list[str]:
    turns = rec.get("turns")
    if not isinstance(turns, list) or len(turns) < 2 or len(turns) % 2:
        return ["turns must be an even-length list"]
    out = []
    for i, t in enumerate(turns):
        want = "user" if i % 2 == 0 else "assistant"
        if t.get("role") != want:
            out.append(f"turn {i} should be {want}")
        c = t.get("content", "")
        if not isinstance(c, str) or not c.strip():
            out.append(f"turn {i} empty")
            continue
        if t.get("role") == "assistant":
            if LEAK.search(c):
                out.append(f"turn {i} names another model or company")
            if FILLER.match(c.strip()):
                out.append(f"turn {i} opens with filler")
            if EMOJI.search(c):
                out.append(f"turn {i} has emoji")
            if "—" in c or "–" in c:
                out.append(f"turn {i} has an em or en dash")
    if sum(len(t.get("content", "")) for t in turns) > MAX_CHARS:
        out.append(f"over {MAX_CHARS} chars")
    if not str(rec.get("source", "")).startswith("seed_"):
        out.append("source must start with seed_")
    return out


def main(paths: list[str]) -> int:
    files = [Path(p) for p in paths] or sorted((ROOT / "data/chat_gen").glob("seed_*.jsonl"))
    seen: dict[str, str] = {}
    bad = total = 0
    for f in files:
        for n, line in enumerate(f.read_text().splitlines(), 1):
            if not line.strip():
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"{f.name}:{n}: bad JSON ({e})")
                bad += 1
                continue
            errs = problems(rec)
            first = rec["turns"][0]["content"].strip().lower() if not errs else ""
            if first and first in seen:
                errs.append(f"duplicate opening question (also in {seen[first]})")
            seen.setdefault(first, f"{f.name}:{n}")
            if errs:
                bad += 1
                print(f"{f.name}:{n}: " + "; ".join(errs))
    print(f"{total} conversations, {bad} with problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
