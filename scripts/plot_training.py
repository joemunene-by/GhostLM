"""GhostLM training visualizer — plots loss curves from training log."""

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Install matplotlib: pip install matplotlib")
    sys.exit(1)


def parse_args():
    """Parse command-line arguments for training visualization.

    Returns:
        argparse.Namespace with all parsed plotting arguments.
    """
    parser = argparse.ArgumentParser(description="Plot GhostLM training loss curves")

    parser.add_argument(
        "--log",
        type=str,
        default="logs/training_log.json",
        help="Path to training log JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/training_curve.png",
        help="Path to save the output plot PNG",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plot interactively",
    )

    return parser.parse_args()


def main():
    """Load training log and plot train/validation loss curves.

    Reads the JSON training log, extracts step and loss data,
    creates a styled matplotlib figure, and saves it as PNG.
    """
    args = parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Error: Log file not found: {log_path}")
        sys.exit(1)

    with open(log_path, "r") as f:
        log_data = json.load(f)

    steps = [entry["step"] for entry in log_data]
    train_loss = [entry["train_loss"] for entry in log_data]
    val_loss = [entry["val_loss"] for entry in log_data]

    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(steps, train_loss, color="#4A90D9", linewidth=1.5, label="Train Loss")
    ax.plot(steps, val_loss, color="#E8943A", linewidth=1.5, label="Val Loss")

    # Annotate final validation loss
    if val_loss:
        final_step = steps[-1]
        final_val = val_loss[-1]
        ax.annotate(
            f"Final val: {final_val:.4f}",
            xy=(final_step, final_val),
            xytext=(final_step * 0.7, final_val * 1.05),
            color="#E8943A",
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="#E8943A", lw=1.5),
        )

    ax.set_title("GhostLM Training Loss — ghost-tiny", fontsize=14, fontweight="bold")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

    if args.show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()
