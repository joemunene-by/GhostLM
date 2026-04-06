"""GhostLM evaluation script — compute perplexity and loss on validation data."""

import argparse
import math
import sys
from dataclasses import fields
from pathlib import Path

import torch

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.tokenizer import GhostTokenizer
from ghostlm.dataset import build_dataloaders


def parse_args():
    """Parse command-line arguments for evaluation.

    Returns:
        argparse.Namespace with all parsed evaluation arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate GhostLM")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default="data/processed/val.jsonl",
        help="Path to validation JSONL file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on: auto, cpu, cuda, mps",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=100,
        help="Number of validation batches to evaluate",
    )

    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, device: str) -> tuple:
    """Load a GhostLM model from a saved checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Target device string.

    Returns:
        Tuple of (model, config) with model in eval mode.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    saved_config = checkpoint["config"]
    config = GhostLMConfig(**{
        f.name: saved_config[f.name]
        for f in fields(GhostLMConfig)
        if f.name in saved_config
    })

    model = GhostLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model = model.to(device)

    return model, config


def main():
    """Run GhostLM evaluation on validation data.

    Computes average loss and perplexity over the validation set
    and prints a summary of results.
    """
    args = parse_args()

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # Load model
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    model, config = load_model_from_checkpoint(args.checkpoint, device)
    tokenizer = GhostTokenizer()

    # Check data exists
    val_path = Path(args.val_data)
    if not val_path.exists():
        print("Error: Validation data not found. Run: python data/collect.py")
        sys.exit(1)

    # Build validation loader
    train_path = Path("data/processed/train.jsonl")
    if not train_path.exists():
        train_path = val_path

    _, val_loader = build_dataloaders(
        str(train_path), str(val_path), tokenizer, config
    )

    # Evaluate
    print("=" * 50)
    print("GhostLM Evaluation")
    print("=" * 50)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device:     {device}")
    print(f"  Val data:   {args.val_data}")
    print("=" * 50)

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= args.num_batches:
                break

            x, y = batch
            x = x.to(device)
            y = y.to(device)

            logits, loss = model(x, targets=y)

            # Count non-ignored tokens (ignore_index = -1)
            valid_tokens = (y != -1).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(avg_loss)

    print(f"\nResults:")
    print(f"  Avg loss:     {avg_loss:.4f}")
    print(f"  Perplexity:   {perplexity:.2f}")
    print(f"  Tokens eval:  {total_tokens:,}")
    print(f"  Batches:      {min(i + 1, args.num_batches)}")
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
