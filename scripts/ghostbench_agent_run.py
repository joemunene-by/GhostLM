#!/usr/bin/env python3
"""Run GhostAgent against every Bench in a GhostBench Suite.

GhostBench's Bench/Suite/RunReport machinery is intentionally
prediction-driven: you run a model offline, save predictions to a
JSONL per Bench, and feed them through ``python -m ghostbench score``
or ``python -m ghostbench compare`` to get scored RunReports.

This script is the prediction-generator side for the agent loop:

  1. Load any GhostLM checkpoint (or ghost-tiny random for smoke).
  2. Wrap it in GhostAgent with the canonical tools registry.
  3. For each Bench discovered in --eval-dir, run the agent loop
     on every prompt, serialise the trace into Prediction shape,
     and write to ``--predictions-dir/<bench_name>.jsonl``.
  4. Optionally write a sidecar trace file per bench
     (``<bench_name>.traces.jsonl``) for audit / replay.

The Prediction's ``predicted_artifact`` is the trace's scored text:
ASSISTANT messages plus TOOL responses, USER and SYSTEM excluded.
This is the same convention scripts/eval_agent.py uses, factored
onto ``AgentTrace.to_scored_text`` in v0.9.11.

A ``--baseline`` flag forces ``max_iters=1`` so the model emits one
message and the loop terminates without dispatching any tool. This
is the no-tools control for paired comparison: same prompt, same
generation params, same agent runtime, same system prompt, but the
model never sees a tool response. Compare via:

    python -m ghostbench compare --eval data/raw/<eval>.jsonl \\
        --a-predictions logs/<run>/<bench>.jsonl --a-name agent \\
        --b-predictions logs/<run>_baseline/<bench>.jsonl --b-name baseline

CLI:

    PYTHONPATH=. python3 scripts/ghostbench_agent_run.py \\
        --checkpoint checkpoints/phase19_chat_v09/best_model.pt \\
        --eval-dir data/raw \\
        --predictions-dir logs/v09_agent \\
        --run-name v09_agent

    # Paired baseline (same checkpoint, max_iters=1):
    PYTHONPATH=. python3 scripts/ghostbench_agent_run.py \\
        --checkpoint checkpoints/phase19_chat_v09/best_model.pt \\
        --eval-dir data/raw \\
        --predictions-dir logs/v09_baseline \\
        --run-name v09_baseline --baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ghostbench.bench import Suite  # noqa: E402
from ghostbench.parsers import DEFAULT_PARSERS  # noqa: E402
from ghostlm.agent import (  # noqa: E402
    AgentTrace,
    GhostAgent,
    RuntimeConfig,
)
from ghostlm.agent.runner import make_generator  # noqa: E402


def trace_to_prediction(trace: AgentTrace, eval_record: Dict) -> Dict:
    """Convert an AgentTrace into a GhostBench Prediction record.

    The Prediction shape (see ghostbench.bench.Prediction) propagates
    eval-record tags into the prediction file so the scorer doesn't
    have to cross-reference eval and prediction files.
    """
    return {
        "format": eval_record.get("format", ""),
        "prompt": eval_record.get("prompt", ""),
        "predicted_artifact": trace.to_scored_text(),
        "required_fields": eval_record.get("required_fields", []) or [],
        "required_substrings": eval_record.get("required_substrings",
                                                []) or [],
        "seed_id": eval_record.get("seed_id"),
    }


def stream_eval_jsonl(path: Path):
    """Yield raw eval-record dicts from a JSONL."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="scripts/ghostbench_agent_run.py")
    p.add_argument("--checkpoint", default=None,
                    help="Path to a GhostLM .pt checkpoint. Omit for "
                         "random ghost-tiny smoke.")
    p.add_argument("--eval-dir", default="data/raw",
                    help="Directory containing the held-out bet eval "
                         "JSONL files (Suite.from_dir scans this).")
    p.add_argument("--predictions-dir", required=True,
                    help="Output directory; written as one JSONL per "
                         "discovered Bench.")
    p.add_argument("--run-name", default="agent_run",
                    help="Identifier carried into RunReports.")
    p.add_argument("--baseline", action="store_true",
                    help="Force max_iters=1 (model-only, no tool "
                         "dispatch). Pair with the same checkpoint's "
                         "non-baseline run for the with-vs-without-"
                         "tools comparison.")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-iters", type=int, default=6)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--repetition-penalty", type=float, default=1.15)
    p.add_argument("--offline", action="store_true",
                    help="Force tool backends to use offline caches.")
    p.add_argument("--write-traces", action="store_true",
                    help="Also write per-bench *.traces.jsonl with the "
                         "full trace structure for audit / replay.")
    p.add_argument("--only", default=None,
                    help="Optional comma-separated list of bench names "
                         "to run; default is every bench in the Suite.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.offline:
        os.environ["GHOST_AGENT_OFFLINE"] = "1"

    # Discover the Suite.
    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        print(f"[error] eval-dir not found: {eval_dir}", file=sys.stderr)
        return 1
    suite = Suite.from_dir(eval_dir, parsers=DEFAULT_PARSERS)
    if not len(suite):
        print(f"[error] no benches discovered in {eval_dir}",
              file=sys.stderr)
        return 1

    only: Optional[set] = None
    if args.only:
        only = {n.strip() for n in args.only.split(",") if n.strip()}

    benches = [b for b in suite
               if (only is None or b.name in only)]
    if not benches:
        print(f"[error] --only filter matched no benches; "
              f"available: {[b.name for b in suite]}", file=sys.stderr)
        return 1

    # Build agent once; reuse across all benches and prompts.
    generator, is_random = make_generator(
        args.checkpoint, args.device,
        args.max_new_tokens, args.temperature,
        args.top_p, args.top_k, args.repetition_penalty,
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
    pred_dir = Path(args.predictions_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Mode:           {label} (max_iters={iters})")
    if args.checkpoint:
        print(f"  Checkpoint:     {args.checkpoint}")
    if is_random:
        print(f"  [note] random ghost-tiny weights (output is noise)")
    print(f"  Predictions:    {pred_dir}")
    print(f"  Run name:       {args.run_name}")
    print(f"  Benches:        {[b.name for b in benches]}")
    print()

    overall_start = time.time()

    # Run agent against each bench.
    for bench in benches:
        bench_path = eval_dir / _bench_to_filename(bench.name)
        out_path = pred_dir / f"{bench.name}.jsonl"
        traces_path = pred_dir / f"{bench.name}.traces.jsonl" \
            if args.write_traces else None
        n_done = 0
        bench_start = time.time()
        with out_path.open("w", encoding="utf-8") as fout, \
                (traces_path.open("w", encoding="utf-8")
                 if traces_path else _NullCM()) as ftrace:
            for raw_rec in stream_eval_jsonl(bench_path):
                prompt = raw_rec.get("prompt", "")
                if not prompt:
                    continue
                trace = agent.run(prompt)
                pred = trace_to_prediction(trace, raw_rec)
                fout.write(json.dumps(pred, ensure_ascii=False) + "\n")
                if ftrace:
                    ftrace.write(json.dumps({
                        "prompt": prompt,
                        "trace": trace.to_dict(),
                    }, ensure_ascii=False) + "\n")
                n_done += 1
        elapsed = time.time() - bench_start
        print(f"  {bench.name}: {n_done} predictions in "
              f"{elapsed:.1f}s -> {out_path}")

    overall_elapsed = time.time() - overall_start
    print()
    print(f"Done in {overall_elapsed:.1f}s. Score with:")
    print(f"  python -m ghostbench summary \\")
    print(f"    --eval-dir {eval_dir} \\")
    print(f"    --predictions-dir {pred_dir} \\")
    print(f"    --run-name {args.run_name}")
    return 0


def _bench_to_filename(bench_name: str) -> str:
    """Inverse of the Suite.from_dir mapping."""
    inverse = {
        "bet6_format_aware": "format_aware_eval.jsonl",
        "bet7_code_security": "code_security_eval.jsonl",
        "bet8_binary_literacy": "binary_literacy_eval.jsonl",
        "bet9_provenance": "provenance_eval.jsonl",
        "bet10_log_analysis": "log_analysis_eval.jsonl",
        "bet11_iac_security": "iac_security_eval.jsonl",
        "bet12_protocol_fields": "protocol_fields_eval.jsonl",
    }
    return inverse.get(bench_name, f"{bench_name}.jsonl")


class _NullCM:
    """A no-op context manager that returns None on enter."""
    def __enter__(self): return None
    def __exit__(self, *a): return False


if __name__ == "__main__":
    sys.exit(main())
