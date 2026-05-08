#!/usr/bin/env python3
"""One-command reproducer for the v0.9.5 differentiation-bet baselines.

Walks the four held-out eval sets (bets 6, 7, 8, 9) against any
GhostLM checkpoint, runs inference, scores each, and prints a
single combined results table. Anyone who clones the repo can
reproduce every baseline measurement with one command.

Eval sets:

  bet 6  format-aware       data/raw/format_aware_eval.jsonl  (n=32)
  bet 7  code-for-security  data/raw/code_security_eval.jsonl (n=20)
  bet 8  binary literacy    data/raw/binary_literacy_eval.jsonl (n=20)
  bet 9  provenance         data/raw/provenance_eval.jsonl    (n=15)

Run (against v0.9 chat as currently deployed):

    PYTHONPATH=. python3 scripts/run_all_baselines.py \\
        --checkpoint checkpoints/phase19_chat_v09/best_model.pt \\
        --run-name v09_chat

Run (against any future ghost-base checkpoint):

    PYTHONPATH=. python3 scripts/run_all_baselines.py \\
        --checkpoint checkpoints/phase21_ghost_base/best_model.pt \\
        --run-name ghost_base_v1

Outputs:

  logs/baselines_<run_name>/<bet>.jsonl    raw predictions per bet
  logs/baselines_<run_name>/<bet>_score.md scoring report per bet
  logs/baselines_<run_name>/summary.md     combined headline table
  logs/baselines_<run_name>/summary.json   machine-readable summary

The summary.md is the table that goes in the comparison-rows
section of docs/baselines_v09_bets789.md and
docs/format_baseline_v09.md.

Cost: ~5-10 min on M4 MPS for v0.9 (87 prompts total, mostly short
generations); proportionally faster on a real GPU.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent


EVAL_BETS = [
    ("bet6_format_aware",   "data/raw/format_aware_eval.jsonl"),
    ("bet7_code_security",  "data/raw/code_security_eval.jsonl"),
    ("bet8_binary_literacy", "data/raw/binary_literacy_eval.jsonl"),
    ("bet9_provenance",     "data/raw/provenance_eval.jsonl"),
]


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson 95% CI as percentages. Same formula as
    eval_format_compliance.wilson_ci so the numbers match."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5 / denom
    lo = max(0.0, center - spread)
    hi = min(1.0, center + spread)
    return (100 * lo, 100 * hi)


def run_one_bet(bet_name: str, eval_path: Path, ckpt_path: Path,
                out_dir: Path, max_tokens: int) -> Dict:
    """Run inference + scoring for one bet. Returns a dict with
    n / parse_pass / fields_pass for the summary table."""
    pred_path = out_dir / f"{bet_name}.jsonl"
    score_path = out_dir / f"{bet_name}_score.md"
    print(f"\n=== {bet_name} ===")
    inference_cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "run_format_baseline.py"),
        "--checkpoint", str(ckpt_path),
        "--seeds", str(eval_path),
        "--out", str(pred_path),
        "--max-tokens", str(max_tokens),
    ]
    print("  " + " ".join(inference_cmd))
    subprocess.run(inference_cmd, check=True, cwd=str(REPO_ROOT))

    score_cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "eval_format_compliance.py"),
        "--predictions", str(pred_path),
        "--out", str(score_path),
    ]
    score_proc = subprocess.run(score_cmd, capture_output=True, text=True,
                                 cwd=str(REPO_ROOT))
    print("  " + score_proc.stdout.strip())

    # Re-evaluate so we can extract the structured numbers; the
    # subprocess produced the markdown report but not a machine-readable
    # summary. Cheap to redo.
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.eval_format_compliance import evaluate_record  # noqa: E402

    n = parse_pass = fields_pass = 0
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev = evaluate_record(rec, "predicted_artifact")
            n += 1
            if ev["parse_ok"]:
                parse_pass += 1
            if ev["fields_ok"]:
                fields_pass += 1

    plo, phi = wilson_ci(parse_pass, n)
    flo, fhi = wilson_ci(fields_pass, n)
    return {
        "bet": bet_name,
        "n": n,
        "parse_pass": parse_pass,
        "fields_pass": fields_pass,
        "parse_pct": 100 * parse_pass / n if n else 0.0,
        "fields_pct": 100 * fields_pass / n if n else 0.0,
        "parse_ci": (plo, phi),
        "fields_ci": (flo, fhi),
    }


def render_summary(rows: List[Dict], run_name: str) -> str:
    lines = [f"# Baseline summary: {run_name}", ""]
    lines.append("| Bet | n | parse % (95% CI) | fields % (95% CI) |")
    lines.append("|---|---:|---|---|")
    for r in rows:
        plo, phi = r["parse_ci"]
        flo, fhi = r["fields_ci"]
        lines.append(
            f"| {r['bet']} | {r['n']} | "
            f"{r['parse_pct']:.1f}% [{plo:.1f}-{phi:.1f}] | "
            f"{r['fields_pct']:.1f}% [{flo:.1f}-{fhi:.1f}] |"
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True,
                   help="Path to the .pt checkpoint to score")
    p.add_argument("--run-name", required=True,
                   help="Short identifier used to namespace logs/")
    p.add_argument("--max-tokens", type=int, default=400,
                   help="Per-prompt generation cap")
    p.add_argument("--bets", default=",".join(b[0] for b in EVAL_BETS),
                   help="Comma-separated subset of bets to run")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        sys.exit(f"checkpoint not found: {ckpt}")

    wanted = {b.strip() for b in args.bets.split(",") if b.strip()}
    selected = [(name, path) for name, path in EVAL_BETS if name in wanted]
    if not selected:
        sys.exit(f"no valid bets selected; available: {[b[0] for b in EVAL_BETS]}")

    out_dir = REPO_ROOT / "logs" / f"baselines_{args.run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint:  {ckpt}")
    print(f"Run name:    {args.run_name}")
    print(f"Output dir:  {out_dir}")
    print(f"Bets to run: {[b[0] for b in selected]}")

    rows = []
    for bet_name, eval_rel in selected:
        eval_path = REPO_ROOT / eval_rel
        if not eval_path.exists():
            print(f"  [skip] eval set missing: {eval_path}")
            continue
        row = run_one_bet(bet_name, eval_path, ckpt, out_dir, args.max_tokens)
        rows.append(row)

    summary_md = render_summary(rows, args.run_name)
    summary_path = out_dir / "summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")

    summary_json = out_dir / "summary.json"
    summary_json.write_text(
        json.dumps([
            {k: v for k, v in r.items() if k != "parse_ci" and k != "fields_ci"} | {
                "parse_ci_lo": r["parse_ci"][0], "parse_ci_hi": r["parse_ci"][1],
                "fields_ci_lo": r["fields_ci"][0], "fields_ci_hi": r["fields_ci"][1],
            }
            for r in rows
        ], indent=2),
        encoding="utf-8",
    )

    print()
    print(summary_md)
    print(f"Summary md:   {summary_path}")
    print(f"Summary json: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
