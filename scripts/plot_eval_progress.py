#!/usr/bin/env python3
"""Chart a training run's benchmark scores and val loss over steps.

Reads the JSON lines written by ``scorecard.py --json-out`` and, optionally,
the trainer's ``training_log.json``, then writes a PNG and a markdown table.

Usage:
    python scripts/plot_eval_progress.py --evals .bg/evals.jsonl \
        --train-log logs/ghost_base_mac/training_log.json --out .bg/progress
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LINESTYLES = ["-", "--", ":", "-."]
MARKERS = ["o", "s", "^", "D", "v", "x"]


def load_evals(path: Path) -> list:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return sorted((r for r in rows if r.get("step") is not None), key=lambda r: r["step"])


def load_val_curve(path: Path) -> list:
    if not path or not path.exists():
        return []
    return [(e["step"], e["val_loss"]) for e in json.loads(path.read_text())
            if "val_loss" in e and e["val_loss"] != float("inf")]


def render_table(rows: list) -> str:
    benches = sorted({k for r in rows for k in r["results"]})
    head = "| step | " + " | ".join(benches) + " |"
    sep = "|---:|" + "---:|" * len(benches)
    body = []
    for r in rows:
        cells = [f"{r['results'][b]['acc']:.1f}" if b in r["results"] else "" for b in benches]
        body.append(f"| {r['step']:,} | " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *body]) + "\n"


def plot(rows: list, val_curve: list, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_panels = 2 if val_curve else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4), squeeze=False)
    ax = axes[0][0]
    benches = sorted({k for r in rows for k in r["results"]})
    for i, bench in enumerate(benches):
        pts = [(r["step"], r["results"][bench]["acc"]) for r in rows if bench in r["results"]]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color="black",
                linestyle=LINESTYLES[i % len(LINESTYLES)], marker=MARKERS[i % len(MARKERS)],
                markersize=4, linewidth=1, label=bench)
    ax.axhline(25, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("step")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Benchmarks (dotted gray: 4-choice chance)")
    ax.legend(fontsize=7, frameon=False)

    if val_curve:
        ax = axes[0][1]
        ax.plot([s for s, _ in val_curve], [v for _, v in val_curve], color="black", linewidth=1)
        ax.set_xlabel("step")
        ax.set_ylabel("val loss")
        ax.set_title("Validation loss")

    for a in axes[0]:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--evals", required=True)
    p.add_argument("--train-log", default=None)
    p.add_argument("--out", required=True, help="Output path prefix (writes .png and .md)")
    args = p.parse_args()

    rows = load_evals(Path(args.evals))
    val_curve = load_val_curve(Path(args.train_log) if args.train_log else None)
    if not rows and not val_curve:
        print("nothing to plot yet")
        return 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot(rows, val_curve, out.with_suffix(".png"))
    out.with_suffix(".md").write_text(render_table(rows) if rows else "no evals yet\n")
    print(f"wrote {out.with_suffix('.png')} and {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
