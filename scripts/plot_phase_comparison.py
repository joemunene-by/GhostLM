"""Plot a final-state phase comparison for ghost-tiny.

Auto-detects Phase 1, Phase 2, and (when present) Phase 3 outputs and
plots three panels: final val_loss, perplexity vs. the GPT-2 baseline,
and security-task accuracy. Phases without data for a given panel are
skipped in that panel only — re-running this script after each new
benchmark/eval round picks up the new data automatically.

Per-step training logs are too sparse to draw real curves, so this
plot summarizes the endpoint metrics that actually exist.
"""

import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Install matplotlib: pip install matplotlib")
    sys.exit(1)


PHASES = [
    {
        "name": "Phase 1",
        "label": "Phase 1\n(leaky split)",
        "color": "#888888",
        "log": Path("archive/logs_v1_pre_corpus_fix/training_log.json"),
        "bench": Path("archive/logs_v1_pre_corpus_fix/benchmark_step_10000.json"),
        "sec": Path("archive/logs_v1_pre_corpus_fix/eval_security.json"),
        "note": "leaky split — not directly comparable",
    },
    {
        "name": "Phase 2",
        "label": "Phase 2\n(clean split)",
        "color": "#E8943A",
        "log": Path("logs/training_log.json"),
        "bench": Path("logs/benchmark_phase2.json"),
        "sec": Path("logs/eval_security_phase2.json"),
        "note": "clean deterministic-hash split",
    },
    {
        "name": "Phase 3",
        "label": "Phase 3\n(12x corpus, released)",
        "color": "#6FB76F",
        "log": Path("logs/phase3_refresh/training_log.json"),
        "bench": Path("logs/benchmark_phase3.json"),
        "sec": Path("logs/eval_security_phase3.json"),
        "note": "v0.3.3 released ghost-tiny on the post-NVD-pull corpus",
    },
]

OUTPUT = Path("logs/phase_comparison.png")
GPT2_BASELINE_COLOR = "#4A90D9"


def load_json(path):
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def final_val_loss(log):
    if not log:
        return None
    completed = [e for e in log if e.get("status") == "complete"]
    return completed[-1]["val_loss"] if completed else log[-1]["val_loss"]


def total_security_correct(eval_data):
    if not eval_data:
        return None
    correct = sum(t["correct"] for t in eval_data["tasks"])
    total = sum(t["total"] for t in eval_data["tasks"])
    return correct, total


def collect_phase_metrics():
    """Read every phase's three files and pack metrics into a per-phase dict.

    Missing files are silently treated as ``None`` so the script keeps working
    after Phase 3 training finishes but before benchmark/security re-runs land.
    """
    metrics = []
    for phase in PHASES:
        log = load_json(phase["log"])
        bench = load_json(phase["bench"])
        sec = load_json(phase["sec"])
        metrics.append({
            **phase,
            "val_loss": final_val_loss(log),
            "perplexity": bench["ghostlm_perplexity"] if bench else None,
            "gpt2_perplexity": bench["gpt2_perplexity"] if bench else None,
            "security": total_security_correct(sec),
        })
    return metrics


def plot_val_loss(ax, phases):
    rows = [p for p in phases if p["val_loss"] is not None]
    if not rows:
        ax.set_visible(False)
        return
    labels = [p["label"] for p in rows]
    values = [p["val_loss"] for p in rows]
    colors = [p["color"] for p in rows]
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Final validation loss")
    ax.set_ylabel("val_loss (lower is better)")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.05, f"{v:.3f}",
                ha="center", fontsize=10)
    ax.set_ylim(0, max(values) * 1.25)


def plot_perplexity(ax, phases):
    rows = [p for p in phases if p["perplexity"] is not None]
    if not rows:
        ax.set_visible(False)
        return
    labels = [p["name"] for p in rows]
    values = [p["perplexity"] for p in rows]
    colors = [p["color"] for p in rows]
    # Append the GPT-2 baseline (use whichever phase carries it; they should agree)
    gpt2 = next((p["gpt2_perplexity"] for p in rows if p["gpt2_perplexity"] is not None), None)
    if gpt2 is not None:
        labels.append("GPT-2\n(124M baseline)")
        values.append(gpt2)
        colors.append(GPT2_BASELINE_COLOR)
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Perplexity on cyber-text benchmark")
    ax.set_ylabel("perplexity (lower is better, log scale)")
    ax.set_yscale("log")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v * 1.1, f"{v:.1f}",
                ha="center", fontsize=10)


def plot_security(ax, phases):
    rows = [p for p in phases if p["security"] is not None]
    if not rows:
        ax.set_visible(False)
        return
    labels = [p["name"] for p in rows]
    values = [100 * c / t for c, t in (p["security"] for p in rows)]
    raw = [f"{c}/{t}" for c, t in (p["security"] for p in rows)]
    colors = [p["color"] for p in rows]
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Security tasks (3 tasks, 30 questions)")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 100)
    ax.axhline(33.3, color="#666666", linestyle="--", linewidth=1, label="random (≈33%)")
    for bar, pct, r in zip(bars, values, raw):
        ax.text(bar.get_x() + bar.get_width() / 2, pct + 2, f"{pct:.1f}%\n({r})",
                ha="center", fontsize=9)
    ax.legend(loc="upper right", fontsize=8)


def print_summary(phases):
    for p in phases:
        bits = []
        if p["val_loss"] is not None:
            bits.append(f"val_loss={p['val_loss']:.4f}")
        if p["perplexity"] is not None:
            bits.append(f"ppl={p['perplexity']:.2f}")
        if p["security"] is not None:
            c, t = p["security"]
            bits.append(f"sec={c}/{t} ({100*c/t:.1f}%)")
        if not bits:
            print(f"{p['name']}: (no data yet)")
        else:
            print(f"{p['name']}: {' · '.join(bits)}  — {p['note']}")


def main():
    phases = collect_phase_metrics()

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    n_phases = sum(1 for p in phases if p["val_loss"] is not None)
    fig.suptitle(
        f"ghost-tiny — final-state comparison ({n_phases} phases)",
        fontsize=13,
        y=1.02,
    )

    plot_val_loss(axes[0], phases)
    plot_perplexity(axes[1], phases)
    plot_security(axes[2], phases)

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=120, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")
    print()
    print_summary(phases)


if __name__ == "__main__":
    main()
