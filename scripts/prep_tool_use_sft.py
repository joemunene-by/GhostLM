#!/usr/bin/env python3
"""Convert bet 1 / bet 9 synth tool-use traces into chat-SFT records.

The bet-1 quality filter wrote synth records as one continuous trace
string in the `text` field:

    USER: <question>
    ASSISTANT: <|tool_call|>{...}<|/tool_call|>
    TOOL: <|tool_response|>{...}<|/tool_response|>
    ASSISTANT: <final answer with <|cite|> tags>

The chat SFT pipeline (`finetune_chat.py` -> `ChatDataset`) expects
records of the form:

    {"turns": [{"role": "user|assistant", "content": "..."}], "source": "..."}

This script bridges the two. For each synth record:

  1. Parses the 4-message trace.
  2. Maps the bet-1 four roles into the two-role chat tokenizer:
       USER         -> user
       ASSISTANT 1  -> assistant (the tool call)
       TOOL         -> user (the tool response, wrapping tags preserved)
       ASSISTANT 2  -> assistant (the cite-tagged final answer)
     Loss is automatically masked to the two assistant turns by
     ChatDataset, so the model learns BOTH "when to emit a tool call"
     and "how to synthesize cite-tagged answers from tool responses".
  3. Optionally mixes the converted records with an existing chat
     train file (so v0.9 doesn't lose its small-talk + identity SFT
     when we add tool-use on top).
  4. Writes a 95/5 train/val split, deterministic by record id.

CLI:

    PYTHONPATH=. python3 scripts/prep_tool_use_sft.py \\
        --in-tool-use data/processed/synth_tool_use.jsonl \\
        --in-provenance data/processed/synth_tool_use_provenance.jsonl \\
        --base-train data/processed/chat_train.jsonl \\
        --base-val data/processed/chat_val.jsonl \\
        --out-train data/processed/chat_train_with_tools.jsonl \\
        --out-val data/processed/chat_val_with_tools.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional


_USER_PFX = "USER: "
_ASST_PFX = "ASSISTANT: "
_TOOL_PFX = "TOOL: "


def parse_trace(text: str) -> Optional[Dict[str, str]]:
    """Parse a USER / ASSISTANT / TOOL / ASSISTANT trace string.

    Returns a dict with keys ``user``, ``tool_call``, ``tool_response``,
    ``answer`` on success, or ``None`` if the trace does not match the
    expected 4-message shape.
    """
    if not text or not isinstance(text, str):
        return None
    lines = text.strip().split("\n")
    if len(lines) < 4:
        return None
    if not lines[0].startswith(_USER_PFX):
        return None
    if not lines[1].startswith(_ASST_PFX):
        return None
    if not lines[2].startswith(_TOOL_PFX):
        return None
    # The final assistant answer may wrap onto subsequent lines if a
    # template embedded a newline; concatenate the tail.
    if not lines[3].startswith(_ASST_PFX):
        return None
    answer_block = "\n".join(lines[3:])
    return {
        "user": lines[0][len(_USER_PFX):].strip(),
        "tool_call": lines[1][len(_ASST_PFX):].strip(),
        "tool_response": lines[2][len(_TOOL_PFX):].strip(),
        "answer": answer_block[len(_ASST_PFX):].strip(),
    }


def trace_to_chat_record(parsed: Dict[str, str], rec: Dict) -> Dict:
    """Convert a parsed trace into a ChatDataset-shaped record."""
    return {
        "turns": [
            {"role": "user", "content": parsed["user"]},
            {"role": "assistant", "content": parsed["tool_call"]},
            {"role": "user", "content": parsed["tool_response"]},
            {"role": "assistant", "content": parsed["answer"]},
        ],
        "source": rec.get("source", "synth_tool_use"),
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
    """Stable per-record hash for deterministic train/val splitting."""
    rid = record.get("source", "") + "|" + record.get("seed_id", "") \
          + "|" + record.get("turns", [{}])[0].get("content", "")[:100]
    return int(hashlib.sha1(rid.encode("utf-8")).hexdigest()[:8], 16)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="scripts/prep_tool_use_sft.py",
        description="Convert bet 1 / bet 9 synth traces into chat SFT.",
    )
    p.add_argument("--in-tool-use",
                    default="data/processed/synth_tool_use.jsonl",
                    help="Path to bet 1 synth jsonl.")
    p.add_argument("--in-provenance",
                    default="data/processed/synth_tool_use_provenance.jsonl",
                    help="Path to bet 9 synth jsonl.")
    p.add_argument("--base-train", default=None,
                    help="Optional existing chat_train.jsonl to mix in. "
                         "Skip to create a tool-use-only file.")
    p.add_argument("--base-val", default=None,
                    help="Optional existing chat_val.jsonl to mix in.")
    p.add_argument("--out-train",
                    default="data/processed/chat_train_with_tools.jsonl")
    p.add_argument("--out-val",
                    default="data/processed/chat_val_with_tools.jsonl")
    p.add_argument("--val-fraction", type=float, default=0.05,
                    help="Held-out fraction of converted tool-use records.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    # Convert each synth source into chat records.
    converted: List[Dict] = []
    parse_failures = 0
    for label, path_str in [("bet1", args.in_tool_use),
                              ("bet9", args.in_provenance)]:
        path = Path(path_str)
        if not path.exists():
            print(f"[warn] {label}: {path} not found, skipping",
                  file=sys.stderr)
            continue
        n_label = 0
        for rec in stream_jsonl(path):
            text = rec.get("text", "")
            parsed = parse_trace(text)
            if parsed is None:
                parse_failures += 1
                continue
            converted.append(trace_to_chat_record(parsed, rec))
            n_label += 1
        print(f"  {label}: {n_label} records from {path}")

    if parse_failures:
        print(f"  parse_failures: {parse_failures} (skipped)")

    if not converted:
        print("[error] No records parsed; nothing to write.", file=sys.stderr)
        return 1

    # Deterministic split by hash so the same record always lands in
    # the same split across runs (matters when comparing checkpoints).
    train_recs: List[Dict] = []
    val_recs: List[Dict] = []
    val_threshold = int(args.val_fraction * 256)  # use top byte
    for rec in converted:
        bucket = hash_for_split(rec) % 256
        (val_recs if bucket < val_threshold else train_recs).append(rec)

    print(f"  tool-use train: {len(train_recs)}, val: {len(val_recs)}")

    # Mix with the existing chat data if a base path is provided.
    if args.base_train:
        base_path = Path(args.base_train)
        if base_path.exists():
            n_base = 0
            for rec in stream_jsonl(base_path):
                if "turns" not in rec:
                    continue
                train_recs.append(rec)
                n_base += 1
            print(f"  base-train mixed in: {n_base}")
        else:
            print(f"[warn] base-train {base_path} not found, skipping",
                  file=sys.stderr)

    if args.base_val:
        base_path = Path(args.base_val)
        if base_path.exists():
            n_base = 0
            for rec in stream_jsonl(base_path):
                if "turns" not in rec:
                    continue
                val_recs.append(rec)
                n_base += 1
            print(f"  base-val mixed in: {n_base}")

    # Shuffle so tool-use and base-chat records interleave in batches.
    random.shuffle(train_recs)
    random.shuffle(val_recs)

    # Write outputs.
    out_train = Path(args.out_train)
    out_val = Path(args.out_val)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_val.parent.mkdir(parents=True, exist_ok=True)
    with out_train.open("w", encoding="utf-8") as f:
        for rec in train_recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with out_val.open("w", encoding="utf-8") as f:
        for rec in val_recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print()
    print(f"Wrote {len(train_recs)} train records -> {out_train}")
    print(f"Wrote {len(val_recs)} val records   -> {out_val}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
