"""GhostLM training entry point — initializes model, tokenizer, data, and runs the training loop."""

import argparse
import sys
from pathlib import Path

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.tokenizer import GhostTokenizer
from ghostlm.dataset import build_dataloaders
from ghostlm.trainer import GhostTrainer


def parse_args():
    """Parse command-line arguments for training.

    Returns:
        argparse.Namespace with all parsed training arguments.
    """
    parser = argparse.ArgumentParser(description="Train GhostLM")

    parser.add_argument(
        "--preset",
        type=str,
        choices=["ghost-tiny", "ghost-small", "ghost-medium"],
        default="ghost-small",
        help="Model preset configuration (default: ghost-small)",
    )
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/processed/train.jsonl",
        help="Path to training JSONL file",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default="data/processed/val.jsonl",
        help="Path to validation JSONL file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override config max_steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override config batch_size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override config learning_rate",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to train on: auto, cpu, cuda, mps",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )

    return parser.parse_args()


def main():
    """Run the full GhostLM training pipeline.

    Parses arguments, loads configuration, initializes the model and
    tokenizer, builds data loaders, and starts the training loop.
    Supports resuming from a checkpoint and CLI overrides for key
    hyperparameters.
    """
    args = parse_args()

    # Load preset config
    config = GhostLMConfig.from_preset(args.preset)

    # Apply CLI overrides
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    config.grad_accum_steps = args.grad_accum
    if args.device != "auto":
        config.device = args.device
    if args.no_wandb:
        config.use_wandb = False

    # Set vocab size to match GhostTokenizer (base 50257 + 4 special tokens)
    config.vocab_size = 50261
    config.context_length = 128  # Reduced for CPU training on low-RAM hardware

    # Print config
    print(repr(config))

    # Initialize tokenizer
    tokenizer = GhostTokenizer()

    # Verify data files exist
    train_path = Path(args.train_data)
    val_path = Path(args.val_data)

    if not train_path.exists() or not val_path.exists():
        print("Error: Data not found. Run: python data/collect.py")
        sys.exit(1)

    # Build data loaders
    print("Building data loaders...")
    train_loader, val_loader = build_dataloaders(
        str(train_path), str(val_path), tokenizer, config
    )

    # Initialize model
    print("Initializing model...")
    model = GhostLM(config)
    print(f"Model parameters: {model.num_params():,}")

    # Initialize trainer
    trainer = GhostTrainer(model, config)

    # Resume from checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    # Run training
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
