#!/usr/bin/env python3
"""Combine every chat SFT source into data/processed/chat_train.jsonl and chat_val.jsonl.

Sources: the original builder output (kept as chat_base_*.jsonl), templated synth_*.jsonl
records with USER/ASSISTANT turns, and every model-written conversation (grounded teacher
answers, seeds, generated chats) that passed review_chat.py (data/chat_reviewed/).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data/processed"
VAL_FRAC = 0.03
TURN = re.compile(r"^(USER|ASSISTANT|TOOL): ?", re.M)


def read(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def synth_turns(text: str):
    parts = TURN.split(text)
    if len(parts) < 3 or "TOOL" in parts[1::2]:
        return None
    turns = [{"role": r.lower(), "content": c.strip()} for r, c in zip(parts[1::2], parts[2::2])]
    if turns[0]["role"] != "user" or turns[-1]["role"] != "assistant" or not all(t["content"] for t in turns):
        return None
    return turns


def is_val(rec: dict) -> bool:
    h = hashlib.sha1(rec["turns"][0]["content"].encode()).digest()
    return h[0] / 256 < VAL_FRAC


def main() -> int:
    subprocess.run([sys.executable, str(ROOT / "scripts/teacher_sft.py"), "merge"], check=True)
    train, val = read(PROC / "chat_base_train.jsonl"), read(PROC / "chat_base_val.jsonl")
    seen = {r["turns"][0]["content"] for r in train + val}
    new = []
    for f in sorted(PROC.glob("synth_*.jsonl")):
        for r in read(f):
            turns = synth_turns(r.get("text", ""))
            if turns:
                new.append({"turns": turns, "source": r.get("source", f.stem)})
    # Model-written data (teacher answers, seeds, generated chats) only counts once review_chat.py has passed it.
    reviewed = [f for f in sorted((ROOT / "data/chat_reviewed").glob("*.jsonl")) if f.name != "dropped.jsonl"]
    for f in reviewed:
        new += [{"turns": r["turns"], "source": r["source"]} for r in read(f)]
    added = Counter()
    for r in new:
        first = r["turns"][0]["content"]
        if first in seen:
            continue
        seen.add(first)
        (val if is_val(r) else train).append(r)
        added[r["source"]] += 1
    for name, rows in (("chat_train.jsonl", train), ("chat_val.jsonl", val)):
        # Rebuilt on a schedule, so never leave chat_sft a half-written file.
        tmp = PROC / f"{name}.tmp"
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        tmp.replace(PROC / name)
    print(f"train {len(train)}, val {len(val)}; added {sum(added.values())}: {dict(added.most_common())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
