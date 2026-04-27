"""Render a cross-phase comparison table from saved eval_security JSON outputs.

Reads every `logs/eval_security_*_expanded.json` file (or a user-provided
list), infers a phase label from each filename, and prints a per-task and
overall accuracy table with mode-collapse (most-common-share) annotations.

Usage:
    PYTHONPATH=. python3 scripts/compare_phase_evals.py
    PYTHONPATH=. python3 scripts/compare_phase_evals.py logs/eval_security_phase2_expanded.json logs/eval_security_phase3.5_expanded.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Pretty labels for known phase keys; anything else falls back to the filename
# stem so the script keeps working as new phases are added.
KNOWN_PHASE_LABELS = {
    "phase1": "Phase 1 (early)",
    "phase2": "Phase 2 (v0.3.0)",
    "phase3": "Phase 3 (v0.3.3)",
    "phase3.5": "Phase 3.5 (v0.3.5)",
    "phase3.6": "Phase 3.6 (Exploit-DB+)",
    "phase4": "Phase 4 (ghost-small)",
}


def label_for(path: Path) -> str:
    """Derive a pretty phase label from an eval-security JSON filename."""
    m = re.search(r"eval_security_([a-z0-9.]+)_(?:expanded|pmi)", path.stem)
    if not m:
        return path.stem
    key = m.group(1)
    return KNOWN_PHASE_LABELS.get(key, key)


def _phase_sort_key(path: Path) -> Tuple[float, str]:
    """Sort phase JSONs by phase number (3 before 3.5), unknowns at the end.

    Lexical sort on filenames puts "phase3.5" before "phase3" because the
    dot sorts ahead of the underscore that starts "_expanded" — extracting
    the numeric phase fixes the column order.
    """
    m = re.search(r"eval_security_phase([0-9.]+)_(?:expanded|pmi)", path.stem)
    if not m:
        return (float("inf"), path.stem)
    try:
        return (float(m.group(1)), path.stem)
    except ValueError:
        return (float("inf"), path.stem)


def collect_default_inputs() -> List[Path]:
    """Find all expanded-eval JSONs in the standard logs directory."""
    logs = Path("logs")
    if not logs.is_dir():
        return []
    paths = sorted(logs.glob("eval_security_*_expanded.json"), key=_phase_sort_key)
    return paths


def load_runs(paths: List[Path]) -> List[Tuple[str, Path, Dict]]:
    """Load each JSON and pair it with a label."""
    runs = []
    for p in paths:
        with p.open() as f:
            data = json.load(f)
        runs.append((label_for(p), p, data))
    return runs


def print_table(runs: List[Tuple[str, Path, Dict]]) -> None:
    """Print an accuracy + most-common-share table across runs."""
    if not runs:
        print("No expanded-eval JSON files found.", file=sys.stderr)
        sys.exit(1)

    # Use the first run's task list as the canonical ordering. Bail out if
    # task lists drift between runs — that means the eval suite changed and
    # the comparison would be misleading.
    canonical_tasks = [t["task"] for t in runs[0][2]["tasks"]]
    for name, path, data in runs[1:]:
        if [t["task"] for t in data["tasks"]] != canonical_tasks:
            print(
                f"[warning] task list in {path} differs from {runs[0][1]} — "
                "skipping it from the comparison",
                file=sys.stderr,
            )
            runs = [r for r in runs if r[1] != path]

    label_w = 42
    cell_w = 19
    sep = "-" * label_w + ("-+-" + "-" * cell_w) * len(runs)

    print(f"{'Task':<{label_w}}", end="")
    for name, _, _ in runs:
        print(f" | {name:<{cell_w}}", end="")
    print()
    print(sep)

    for i, task in enumerate(canonical_tasks):
        print(f"{task:<{label_w}}", end="")
        for _, _, data in runs:
            t = data["tasks"][i]
            cell = (
                f"{t['correct']:>2}/{t['total']:<2} "
                f"({t['accuracy']:5.1%}) "
                f"[{t['most_common_share']:3.0%}]"
            )
            print(f" | {cell:<{cell_w}}", end="")
        print()

    print(sep)

    print(f"{'OVERALL':<{label_w}}", end="")
    for _, _, data in runs:
        correct = sum(t["correct"] for t in data["tasks"])
        total = sum(t["total"] for t in data["tasks"])
        cell = f"{correct:>3}/{total} ({data['overall_accuracy']:5.1%})"
        print(f" | {cell:<{cell_w}}", end="")
    print()
    print()
    print("Cell format: correct/total (accuracy) [most-common-share]")
    print("Most-common-share above ~60% indicates the task is mode-collapsing —")
    print("treat the accuracy as suspect even when it looks above-random.")


def main():
    parser = argparse.ArgumentParser(
        description="Compare expanded-eval JSON outputs across phases"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help=(
            "Eval-security JSON files to compare. If omitted, all "
            "logs/eval_security_*_expanded.json files are loaded in "
            "filename order."
        ),
    )
    args = parser.parse_args()

    paths = args.inputs if args.inputs else collect_default_inputs()
    if not paths:
        print(
            "No JSON files passed and none found under logs/eval_security_"
            "*_expanded.json. Run scripts/eval_security.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    runs = load_runs(paths)
    print_table(runs)


if __name__ == "__main__":
    main()
