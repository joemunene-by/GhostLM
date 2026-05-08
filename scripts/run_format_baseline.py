#!/usr/bin/env python3
"""Run a checkpoint over the bet 6 format-aware seed prompts and write
predictions JSONL ready to feed into ``scripts/eval_format_compliance.py``.

Establishes the structural-compliance baseline for any GhostLM
checkpoint. Use it once today against v0.9 chat to capture the
"before bet 6 training" floor; use it again after bet 6 lands to
measure the lift.

Usage:

    PYTHONPATH=. python3 scripts/run_format_baseline.py \\
        --checkpoint checkpoints/phase19_chat_v09/best_model.pt \\
        --out logs/format_baseline_v09_chat.jsonl
        # --seeds defaults to data/raw/format_aware_eval.jsonl, the
        # held-out eval set. Use --seeds data/raw/format_aware_seeds.jsonl
        # to score against the few-shot bank instead (debug only).

    PYTHONPATH=. python3 scripts/eval_format_compliance.py \\
        --predictions logs/format_baseline_v09_chat.jsonl

The baseline script preserves the original ``required_fields`` and
``required_substrings`` tags on each record so the eval harness can
score the predictions against the same expectations as the gold
eval set. Note: the eval records have only ``prompt`` and the
required-content tags; they deliberately do NOT carry a gold
``artifact`` field, since that lives in the few-shot bank and we
want to keep the two cleanly separated.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ghostlm.tokenizer import GhostTokenizer  # noqa: E402

# Reuse chat.py's well-tested inference primitives.
from scripts.chat import (  # noqa: E402
    generate_until_end, load_model, resolve_device,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True,
                   help="Path to the .pt checkpoint to evaluate")
    p.add_argument("--seeds", default="data/raw/format_aware_eval.jsonl",
                   help="JSONL with format / prompt / required_fields / "
                        "required_substrings entries. Default points at "
                        "the held-out eval set (no overlap with the "
                        "few-shot bank that distillation reads from).")
    p.add_argument("--out", required=True,
                   help="Output predictions JSONL path")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=600,
                   help="Cap per-prompt generation length. Format "
                        "artifacts are typically 200-500 tokens; 600 "
                        "is generous without burning M4 inference time.")
    p.add_argument("--repetition-penalty", type=float, default=1.2)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Loading {args.checkpoint} on {device}")
    model, _config = load_model(args.checkpoint, device)
    tokenizer = GhostTokenizer()
    end_id = tokenizer._special_tokens[tokenizer.END]

    seeds_path = REPO_ROOT / args.seeds if not Path(args.seeds).is_absolute() \
                 else Path(args.seeds)
    if not seeds_path.exists():
        sys.exit(f"seeds not found: {seeds_path}")

    out_path = REPO_ROOT / args.out if not Path(args.out).is_absolute() \
               else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    t0 = time.time()
    with seeds_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            prompt = rec.get("prompt", "")
            if not prompt:
                continue

            chat = [{"role": "user", "content": prompt}]
            prompt_ids = tokenizer.format_chat_prompt(chat)
            new_ids = generate_until_end(
                model, prompt_ids,
                end_id=end_id,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=device,
                repetition_penalty=args.repetition_penalty,
            )
            predicted = tokenizer.decode(new_ids).strip()

            out_rec = {
                "format": rec.get("format"),
                "prompt": prompt,
                "predicted_artifact": predicted,
            }
            for k in ("required_fields", "required_substrings"):
                if k in rec:
                    out_rec[k] = rec[k]
            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()
            n += 1
            print(f"  [{n}] {rec.get('format')}: "
                  f"{len(new_ids)} new tokens in "
                  f"{time.time()-t0:.1f}s")

    print(f"\nDone. Wrote {n} predictions to {out_path}")
    print(f"Score with: PYTHONPATH=. python3 "
          f"scripts/eval_format_compliance.py --predictions {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
