#!/usr/bin/env python3
"""Convert every SFT-shape synth record into ChatDataset format and merge.

The existing ``prep_tool_use_sft.py`` handles bet 1 / bet 9 four-message
tool-use traces. After v0.9.5 added bet 7 (code-security) and bet 8
(binary-literacy), and v0.9.8 added bets 10/11/12 (log-analysis /
IaC-security / protocol-fields), and v0.9.23/.24 added the general
code-explain + code-write banks, we have ~2,500 SFT-shape records
across two trace formats:

  4-message:  USER: q
              ASSISTANT: <|tool_call|>{...}<|/tool_call|>
              TOOL: <|tool_response|>{...}<|/tool_response|>
              ASSISTANT: <answer with <|cite|> tags>
              (bets 1, 9 — the tool-using bets)

  2-message:  USER: q
              ASSISTANT: a
              (bets 7, 8, 10, 11, 12, 23, 24 — Q&A-style)

This script accepts the full ``synth_v15_combined.jsonl`` (every SFT-shape
variant), parses both shapes, converts each into the ``{"turns": [...]}``
record format the chat tokenizer expects, and merges with an existing
chat train/val dataset. Loss masking happens automatically inside
``ChatDataset`` — only assistant tokens contribute.

CLI:

    PYTHONPATH=. python3 scripts/prep_full_sft.py \\
        --in-combined data/processed/synth_v15_combined.jsonl \\
        --base-train data/processed/chat_train.jsonl \\
        --base-val data/processed/chat_val.jsonl \\
        --out-train data/processed/chat_train_full_sft.jsonl \\
        --out-val data/processed/chat_val_full_sft.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

_USER_PFX = "USER:"
_ASST_PFX = "ASSISTANT:"
_TOOL_PFX = "TOOL:"


def _strip_prefix(line: str, pfx: str) -> str:
    return line[len(pfx):].lstrip()


def parse_trace(text: str) -> Optional[List[Dict[str, str]]]:
    """Parse a USER/ASSISTANT(/TOOL/ASSISTANT) trace into role-content turns.

    Returns a list of ``{"role": "user|assistant", "content": str}`` dicts
    on success, or ``None`` if the trace doesn't match a known shape.
    Supports both 4-message tool traces and 2-message Q&A.
    """
    if not text or not isinstance(text, str):
        return None
    lines = text.strip().split("\n")
    if len(lines) < 2:
        return None

    # Find prefix-bearing line indices to avoid breaking on multi-line content.
    boundaries: List[tuple[int, str]] = []
    for i, line in enumerate(lines):
        for pfx, role in ((_USER_PFX, "user"), (_ASST_PFX, "assistant"),
                           (_TOOL_PFX, "tool")):
            if line.startswith(pfx):
                boundaries.append((i, role))
                break
    if not boundaries:
        return None
    if boundaries[0][1] != "user":
        return None

    # Slice the text into per-turn chunks.
    turns: List[Dict[str, str]] = []
    for j, (start, role) in enumerate(boundaries):
        end = boundaries[j + 1][0] if j + 1 < len(boundaries) else len(lines)
        chunk = "\n".join(lines[start:end])
        for pfx in (_USER_PFX, _ASST_PFX, _TOOL_PFX):
            if chunk.startswith(pfx):
                chunk = _strip_prefix(chunk, pfx)
                break
        if role == "tool":
            # ChatDataset has no `tool` role; the bet-1 convention folds tool
            # responses into the next user turn so the model still learns
            # "given this output, here's the answer."
            role = "user"
        turns.append({"role": role, "content": chunk.strip()})

    if len(turns) < 2:
        return None
    if turns[-1]["role"] != "assistant":
        return None
    return turns


def trace_to_chat_record(turns: List[Dict[str, str]], rec: Dict) -> Dict:
    return {
        "turns": turns,
        "source": rec.get("source", "synth"),
        "seed_source": rec.get("seed_source", ""),
        "seed_id": rec.get("seed_id", ""),
    }


def stream_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def hash_for_split(record: Dict) -> int:
    rid = (record.get("source", "") + "|"
           + str(record.get("seed_id", "")) + "|"
           + record.get("seed_source", ""))
    return int(hashlib.sha256(rid.encode()).hexdigest(), 16)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-combined",
                   default="data/processed/synth_v15_combined.jsonl",
                   help="Combined synth file with format_type tags.")
    p.add_argument("--base-train",
                   default="data/processed/chat_train.jsonl")
    p.add_argument("--base-val",
                   default="data/processed/chat_val.jsonl")
    p.add_argument("--out-train",
                   default="data/processed/chat_train_full_sft.jsonl")
    p.add_argument("--out-val",
                   default="data/processed/chat_val_full_sft.jsonl")
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_path = Path(args.in_combined)
    if not in_path.exists():
        print(f"ERROR: {in_path} not found", file=sys.stderr)
        return 1

    converted: List[Dict] = []
    skipped_pretrain = 0
    parse_failures = 0
    by_source: Dict[str, int] = {}

    for rec in stream_jsonl(in_path):
        if rec.get("format_type") != "sft":
            skipped_pretrain += 1
            continue
        turns = parse_trace(rec.get("text", ""))
        if not turns:
            parse_failures += 1
            continue
        out = trace_to_chat_record(turns, rec)
        converted.append(out)
        by_source[rec.get("source", "?")] = by_source.get(rec.get("source", "?"), 0) + 1

    print(f"Converted {len(converted)} SFT records "
          f"(skipped {skipped_pretrain} pretrain-shape, "
          f"{parse_failures} parse failures)")
    for s, n in sorted(by_source.items(), key=lambda kv: -kv[1]):
        print(f"  {s:<32} {n}")

    # 95/5 deterministic split.
    train_synth: List[Dict] = []
    val_synth: List[Dict] = []
    val_threshold = int(args.val_fraction * 2 ** 256)
    for r in converted:
        if hash_for_split(r) % (2 ** 256) < val_threshold:
            val_synth.append(r)
        else:
            train_synth.append(r)
    print(f"Split: train={len(train_synth)} val={len(val_synth)}")

    base_train = list(stream_jsonl(Path(args.base_train))) \
        if Path(args.base_train).exists() else []
    base_val = list(stream_jsonl(Path(args.base_val))) \
        if Path(args.base_val).exists() else []
    print(f"Base train: {len(base_train)} val: {len(base_val)}")

    final_train = base_train + train_synth
    final_val = base_val + val_synth

    out_train = Path(args.out_train)
    out_val = Path(args.out_val)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    with out_train.open("w", encoding="utf-8") as f:
        for r in final_train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with out_val.open("w", encoding="utf-8") as f:
        for r in final_val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(final_train)} -> {out_train}")
    print(f"Wrote {len(final_val)} -> {out_val}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
