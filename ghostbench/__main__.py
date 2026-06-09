"""GhostBench CLI: ``python -m ghostbench [score|compare|summary]``.

Usage:

    # Score a single run (predictions JSONL + eval JSONL).
    python -m ghostbench score \\
        --eval data/raw/format_aware_eval.jsonl \\
        --predictions logs/baselines_v09_chat/bet6_format_aware.jsonl \\
        --bench-name bet6_format_aware \\
        --run-name v09_chat \\
        --out logs/baselines_v09_chat/bet6_score_gb.md

    # Suite-level summary across all benches.
    python -m ghostbench summary \\
        --eval-dir data/raw \\
        --predictions-dir logs/baselines_v09_chat \\
        --run-name v09_chat \\
        --out logs/baselines_v09_chat/suite_summary.md

    # Paired comparison: two checkpoints scored on the same eval.
    python -m ghostbench compare \\
        --eval data/raw/format_aware_eval.jsonl \\
        --a-predictions logs/baselines_v09_chat/bet6_format_aware.jsonl \\
        --a-name v09_chat \\
        --b-predictions logs/baselines_ghost_base_v1/bet6_format_aware.jsonl \\
        --b-name ghost_base_v1 \\
        --bench-name bet6_format_aware \\
        --out logs/comparisons/bet6_v09_vs_ghost_base.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from .behavioral import BEHAVIORAL_VALIDATORS
from .bench import Bench, Prediction, Suite
from .parsers import DEFAULT_PARSERS
from .reports import (
    render_paired_comparison,
    render_per_format_breakdown,
    render_run_report,
    render_suite_paired_comparison,
    render_suite_summary,
)
from .scoring import RunReport


def _load_predictions(path: Path) -> List[Prediction]:
    out: List[Prediction] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.append(Prediction.from_dict(rec))
    return out


def _cmd_score(args: argparse.Namespace) -> int:
    eval_path = Path(args.eval)
    pred_path = Path(args.predictions)
    if not eval_path.exists():
        sys.exit(f"eval file not found: {eval_path}")
    if not pred_path.exists():
        sys.exit(f"predictions file not found: {pred_path}")
    bench = Bench.from_jsonl(
        name=args.bench_name, description=args.bench_name,
        path=eval_path, parsers=DEFAULT_PARSERS,
    )
    preds = _load_predictions(pred_path)
    report = bench.score(
        preds, run_name=args.run_name,
        behavioral_validators=BEHAVIORAL_VALIDATORS if args.behavioral else None,
        force_behavioral=args.behavioral,
    )
    out_md = render_run_report(report)
    if any(s.fmt for s in report.scores):
        out_md += "\n## Per-format breakdown\n\n"
        out_md += render_per_format_breakdown(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out_md, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(out_md)
    return 0


def _cmd_summary(args: argparse.Namespace) -> int:
    eval_dir = Path(args.eval_dir)
    pred_dir = Path(args.predictions_dir)
    suite = Suite.from_dir(eval_dir, parsers=DEFAULT_PARSERS)
    if len(suite) == 0:
        sys.exit(f"no benches discovered in {eval_dir}")

    reports: List[RunReport] = []
    for bench in suite:
        # Convention: prediction file lives at <pred_dir>/<bench.name>.jsonl
        pred_path = pred_dir / f"{bench.name}.jsonl"
        if not pred_path.exists():
            print(f"  [skip] predictions missing: {pred_path}")
            continue
        preds = _load_predictions(pred_path)
        reports.append(bench.score(
            preds, run_name=args.run_name,
            behavioral_validators=BEHAVIORAL_VALIDATORS if args.behavioral else None,
            force_behavioral=args.behavioral,
        ))

    out_md = render_suite_summary(reports, args.run_name)
    out_md += "\n## Per-bench detail\n\n"
    for r in reports:
        out_md += "\n" + render_run_report(r) + "\n"

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out_md, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(out_md)
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    eval_path = Path(args.eval)
    a_path = Path(args.a_predictions)
    b_path = Path(args.b_predictions)
    if not (eval_path.exists() and a_path.exists() and b_path.exists()):
        sys.exit("one of the input files is missing; check paths")
    bench = Bench.from_jsonl(
        name=args.bench_name, description=args.bench_name,
        path=eval_path, parsers=DEFAULT_PARSERS,
    )
    a_preds = _load_predictions(a_path)
    b_preds = _load_predictions(b_path)
    bv = BEHAVIORAL_VALIDATORS if args.behavioral else None
    a_report = bench.score(a_preds, run_name=args.a_name,
                            behavioral_validators=bv,
                            force_behavioral=args.behavioral)
    b_report = bench.score(b_preds, run_name=args.b_name,
                            behavioral_validators=bv,
                            force_behavioral=args.behavioral)
    out_md = render_paired_comparison(a_report, b_report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out_md, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(out_md)
    return 0


def _cmd_suite_compare(args: argparse.Namespace) -> int:
    eval_dir = Path(args.eval_dir)
    a_dir = Path(args.a_predictions_dir)
    b_dir = Path(args.b_predictions_dir)
    suite = Suite.from_dir(eval_dir, parsers=DEFAULT_PARSERS)

    a_reports: List[RunReport] = []
    b_reports: List[RunReport] = []
    bv = BEHAVIORAL_VALIDATORS if args.behavioral else None
    for bench in suite:
        a_path = a_dir / f"{bench.name}.jsonl"
        b_path = b_dir / f"{bench.name}.jsonl"
        if not (a_path.exists() and b_path.exists()):
            print(f"  [skip] missing predictions for {bench.name}")
            continue
        a_reports.append(bench.score(_load_predictions(a_path), args.a_name,
                                      behavioral_validators=bv,
                                      force_behavioral=args.behavioral))
        b_reports.append(bench.score(_load_predictions(b_path), args.b_name,
                                      behavioral_validators=bv,
                                      force_behavioral=args.behavioral))

    out_md = render_suite_paired_comparison(a_reports, b_reports)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out_md, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(out_md)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="ghostbench", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    # Common --behavioral flag for every subcommand: opts every record
    # into the behavioural tier at score time. When absent, behavioural
    # is only run for records that explicitly set ``behavioral: true``
    # in the eval JSONL.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--behavioral", action="store_true",
                        help="Force the behavioural tier on for every "
                             "record. Lazy-imports stix2 / yara-python / "
                             "pysigma / jsonschema if installed; falls "
                             "back to enhanced-structural validators.")

    p_score = sub.add_parser("score", help="Score a single run",
                              parents=[common])
    p_score.add_argument("--eval", required=True)
    p_score.add_argument("--predictions", required=True)
    p_score.add_argument("--bench-name", required=True)
    p_score.add_argument("--run-name", required=True)
    p_score.add_argument("--out")
    p_score.set_defaults(func=_cmd_score)

    p_sum = sub.add_parser("summary", help="Suite-level summary",
                            parents=[common])
    p_sum.add_argument("--eval-dir", required=True)
    p_sum.add_argument("--predictions-dir", required=True)
    p_sum.add_argument("--run-name", required=True)
    p_sum.add_argument("--out")
    p_sum.set_defaults(func=_cmd_summary)

    p_cmp = sub.add_parser("compare",
                           help="Paired comparison of two runs on one bench",
                           parents=[common])
    p_cmp.add_argument("--eval", required=True)
    p_cmp.add_argument("--a-predictions", required=True)
    p_cmp.add_argument("--a-name", required=True)
    p_cmp.add_argument("--b-predictions", required=True)
    p_cmp.add_argument("--b-name", required=True)
    p_cmp.add_argument("--bench-name", required=True)
    p_cmp.add_argument("--out")
    p_cmp.set_defaults(func=_cmd_compare)

    p_scmp = sub.add_parser("suite-compare",
                             help="Paired comparison across all benches",
                             parents=[common])
    p_scmp.add_argument("--eval-dir", required=True)
    p_scmp.add_argument("--a-predictions-dir", required=True)
    p_scmp.add_argument("--a-name", required=True)
    p_scmp.add_argument("--b-predictions-dir", required=True)
    p_scmp.add_argument("--b-name", required=True)
    p_scmp.add_argument("--out")
    p_scmp.set_defaults(func=_cmd_suite_compare)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
