#!/usr/bin/env python3
"""Evaluate GhostAgent against a held-out provenance / tool-use eval set.

Each eval record has:

    {"prompt": "...",
     "required_substrings": ["<|tool_call|>", "<|cite|>", "CVE-XXXX", ...]}

The script runs the agent loop on every prompt, concatenates the full
trace (every message's content), and scores how many required
substrings appear. Two metrics:

  strict pass-rate    fraction of prompts where ALL substrings appear
  soft pass-rate      mean fraction of substrings that appear per prompt

Wilson 95% CI is reported for both.

A ``--baseline`` mode runs the same agent runtime but with
``max_iters=1`` so the model never gets a chance to dispatch tools or
see tool responses. This is the paired-comparison "model-only vs
model-with-tools" reference we use to claim the SFT actually improved
tool-use behaviour.

CLI:

    PYTHONPATH=. python3 scripts/eval_agent.py \\
        --checkpoint checkpoints/phase20_chat_v09_tools/best_model.pt \\
        --eval data/raw/provenance_eval.jsonl

    PYTHONPATH=. python3 scripts/eval_agent.py \\
        --checkpoint checkpoints/phase19_chat_v09/best_model.pt \\
        --eval data/raw/provenance_eval.jsonl --baseline
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ghostlm.agent import AgentTrace, GhostAgent, RuntimeConfig  # noqa: E402
from ghostlm.agent.runner import make_generator  # noqa: E402


def trace_to_full_text(trace: AgentTrace) -> str:
    """Concatenate model-output + tool-response message contents.

    Scoring deliberately excludes SYSTEM and USER messages: many eval
    prompts mention the entity (e.g. "What is CVE-2017-0144?"), so
    including the user turn would credit substrings that were already
    in the question rather than substrings the model produced or
    retrieved through tool dispatch. We keep TOOL responses because
    they represent successful grounding through a tool the model
    chose to invoke.
    """
    from ghostlm.agent import MessageRole
    keep = {MessageRole.ASSISTANT, MessageRole.TOOL}
    return "\n".join(m.content for m in trace.history if m.role in keep)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson 95% CI for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def score_record(trace: AgentTrace, required: List[str]) -> Dict:
    """Return per-record scoring details."""
    full_text = trace_to_full_text(trace)
    matches = {sub: (sub in full_text) for sub in required}
    n_hit = sum(1 for v in matches.values() if v)
    n_req = len(required)
    return {
        "matches": matches,
        "n_hit": n_hit,
        "n_required": n_req,
        "fraction": n_hit / n_req if n_req else 0.0,
        "all_present": n_hit == n_req,
        "terminated_reason": trace.terminated_reason,
        "iterations": trace.iterations,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="scripts/eval_agent.py")
    p.add_argument("--checkpoint", default=None,
                    help="Checkpoint path. Omit for random ghost-tiny "
                         "smoke run.")
    p.add_argument("--eval",
                    default="data/raw/provenance_eval.jsonl",
                    help="JSONL eval file with prompt + required_substrings.")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-iters", type=int, default=6,
                    help="Cap on agent iterations per prompt.")
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--repetition-penalty", type=float, default=1.15)
    p.add_argument("--baseline", action="store_true",
                    help="Run with max_iters=1 (model-only, no tool "
                         "dispatch). Use as the no-tools control.")
    p.add_argument("--offline", action="store_true",
                    help="Force tool backends to use offline caches.")
    p.add_argument("--output", default=None,
                    help="Optional path to dump per-record JSONL.")
    p.add_argument("--quiet", action="store_true",
                    help="Suppress per-record output, print summary only.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.offline:
        os.environ["GHOST_AGENT_OFFLINE"] = "1"

    # Load eval set.
    eval_path = Path(args.eval)
    if not eval_path.exists():
        print(f"[error] eval file not found: {eval_path}", file=sys.stderr)
        return 1
    records: List[Dict] = []
    with eval_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        print("[error] eval file is empty", file=sys.stderr)
        return 1
    print(f"  Loaded {len(records)} eval prompts from {eval_path}")

    # Build generator + agent once; reuse across all prompts.
    generator, is_random = make_generator(
        args.checkpoint,
        args.device,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        args.top_k,
        args.repetition_penalty,
    )
    iters = 1 if args.baseline else args.max_iters
    cfg = RuntimeConfig(
        max_iters=iters,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    agent = GhostAgent(generator, cfg)

    label = "baseline" if args.baseline else "agent"
    if is_random:
        print(f"[note] random ghost-tiny weights, output will be noise")
    print(f"  Mode: {label} (max_iters={iters})")
    print()

    # Score every prompt.
    per_record: List[Dict] = []
    out_fp = None
    if args.output:
        out_fp = Path(args.output).open("w", encoding="utf-8")

    n_strict = 0
    soft_total = 0.0
    iter_total = 0
    for i, rec in enumerate(records, 1):
        prompt = rec["prompt"]
        required = rec["required_substrings"]
        trace = agent.run(prompt)
        score = score_record(trace, required)
        per_record.append(score)
        n_strict += int(score["all_present"])
        soft_total += score["fraction"]
        iter_total += score["iterations"]
        if not args.quiet:
            mark = "PASS" if score["all_present"] else \
                   ("partial" if score["n_hit"] > 0 else "FAIL")
            print(f"  [{i:>2}/{len(records)}] {mark} "
                  f"({score['n_hit']}/{score['n_required']}) "
                  f"iter={score['iterations']} "
                  f"reason={score['terminated_reason']}: "
                  f"{prompt[:60]}")
        if out_fp:
            out_fp.write(json.dumps({
                "prompt": prompt,
                "required": required,
                "score": score,
                "trace": trace.to_dict(),
            }, ensure_ascii=False) + "\n")
    if out_fp:
        out_fp.close()

    n = len(records)
    strict_rate = n_strict / n
    soft_rate = soft_total / n
    s_lo, s_hi = wilson_ci(n_strict, n)

    print()
    print("=" * 60)
    print(f"Mode:           {label}")
    if args.checkpoint:
        print(f"Checkpoint:     {args.checkpoint}")
    print(f"Eval set:       {eval_path}")
    print(f"N prompts:      {n}")
    print(f"Strict pass:    {n_strict}/{n} = {strict_rate:.1%} "
          f"(95% CI: {s_lo:.1%} - {s_hi:.1%})")
    print(f"Soft pass mean: {soft_rate:.1%} "
          f"(avg fraction of required substrings present)")
    print(f"Mean iters:     {iter_total/n:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
