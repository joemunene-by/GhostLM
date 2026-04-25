"""Plot a Phase 1 vs Phase 2 final-state comparison for ghost-tiny.

The per-step training logs are too sparse to draw real curves, so this
plot summarizes the endpoint metrics that actually exist: final val_loss,
GhostLM perplexity vs GPT-2 baseline, and security-task accuracy.
"""

import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Install matplotlib: pip install matplotlib")
    sys.exit(1)


PHASE1_LOG = Path("archive/logs_v1_pre_corpus_fix/training_log.json")
PHASE2_LOG = Path("logs/training_log.json")
PHASE1_BENCH = Path("archive/logs_v1_pre_corpus_fix/benchmark_step_10000.json")
PHASE2_BENCH = Path("logs/benchmark_phase2.json")
PHASE1_SEC = Path("archive/logs_v1_pre_corpus_fix/eval_security.json")
PHASE2_SEC = Path("logs/eval_security_phase2.json")
OUTPUT = Path("logs/phase_comparison.png")


def load(path):
    with open(path, "r") as f:
        return json.load(f)


def final_val_loss(log):
    completed = [e for e in log if e.get("status") == "complete"]
    return completed[-1]["val_loss"] if completed else log[-1]["val_loss"]


def total_security_correct(eval_data):
    return sum(t["correct"] for t in eval_data["tasks"]), sum(
        t["total"] for t in eval_data["tasks"]
    )


def main():
    p1_log = load(PHASE1_LOG)
    p2_log = load(PHASE2_LOG)
    p1_bench = load(PHASE1_BENCH)
    p2_bench = load(PHASE2_BENCH)
    p1_sec = load(PHASE1_SEC)
    p2_sec = load(PHASE2_SEC)

    p1_val = final_val_loss(p1_log)
    p2_val = final_val_loss(p2_log)
    p1_ppl = p1_bench["ghostlm_perplexity"]
    p2_ppl = p2_bench["ghostlm_perplexity"]
    gpt2_ppl = p2_bench["gpt2_perplexity"]
    p1_correct, p1_total = total_security_correct(p1_sec)
    p2_correct, p2_total = total_security_correct(p2_sec)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(
        "ghost-tiny — Phase 1 (leaky split) vs Phase 2 (clean split)",
        fontsize=13,
        y=1.02,
    )

    ax = axes[0]
    bars = ax.bar(
        ["Phase 1\n(leaky)", "Phase 2\n(clean)"],
        [p1_val, p2_val],
        color=["#888888", "#E8943A"],
    )
    ax.set_title("Final validation loss")
    ax.set_ylabel("val_loss (lower is better)")
    for bar, v in zip(bars, [p1_val, p2_val]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.05,
            f"{v:.3f}",
            ha="center",
            fontsize=10,
        )
    ax.set_ylim(0, max(p1_val, p2_val) * 1.25)

    ax = axes[1]
    bars = ax.bar(
        ["Phase 1", "Phase 2", "GPT-2\n(124M baseline)"],
        [p1_ppl, p2_ppl, gpt2_ppl],
        color=["#888888", "#E8943A", "#4A90D9"],
    )
    ax.set_title("Perplexity on cyber-text benchmark")
    ax.set_ylabel("perplexity (lower is better, log scale)")
    ax.set_yscale("log")
    for bar, v in zip(bars, [p1_ppl, p2_ppl, gpt2_ppl]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v * 1.1,
            f"{v:.1f}",
            ha="center",
            fontsize=10,
        )

    ax = axes[2]
    p1_pct = 100 * p1_correct / p1_total
    p2_pct = 100 * p2_correct / p2_total
    bars = ax.bar(
        ["Phase 1", "Phase 2"],
        [p1_pct, p2_pct],
        color=["#888888", "#E8943A"],
    )
    ax.set_title("Security tasks (3 tasks, 30 questions)")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 100)
    ax.axhline(33.3, color="#666666", linestyle="--", linewidth=1, label="random (≈33%)")
    for bar, pct, raw in zip(
        bars, [p1_pct, p2_pct], [f"{p1_correct}/{p1_total}", f"{p2_correct}/{p2_total}"]
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            pct + 2,
            f"{pct:.1f}%\n({raw})",
            ha="center",
            fontsize=9,
        )
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=120, bbox_inches="tight")
    print(f"Saved: {OUTPUT}")
    print()
    print(f"Phase 1 val_loss: {p1_val:.4f}  (leaky split — not directly comparable)")
    print(f"Phase 2 val_loss: {p2_val:.4f}  (clean deterministic-hash split)")
    print(f"Phase 1 perplexity: {p1_ppl:.2f}")
    print(f"Phase 2 perplexity: {p2_ppl:.2f}  ({p1_ppl / p2_ppl:.1f}x improvement)")
    print(f"GPT-2 perplexity:   {gpt2_ppl:.2f}")
    print(f"Phase 1 security:   {p1_correct}/{p1_total} ({p1_pct:.1f}%)")
    print(f"Phase 2 security:   {p2_correct}/{p2_total} ({p2_pct:.1f}%)")


if __name__ == "__main__":
    main()
