#!/usr/bin/env python3
"""GhostLM v0.5 from-scratch pretrain — RoPE + SwiGLU + RMSNorm + custom 32K BPE.

Spins up the full v0.5 stack and starts pretraining ghost-small-v0.5 on
the expanded v0.4.2 corpus (~58M tokens after the arxiv full-text pull).
This is a from-scratch run — no checkpoint, random init.

Defaults:
  - 60,000 steps, batch_size 8 × grad_accum 4 (effective 32)
  - Cosine LR schedule, peak 3e-4, 2,000-step warmup
  - Custom 32K BPE from ``data/tokenizer_v05/tokenizer.json``
  - Architecture: RoPE + SwiGLU + RMSNorm (the ``ghost-small-v0.5`` preset)

Wall-clock estimate on Mac M4 MPS at the existing 1.8s/step throughput:
about 30 hours for 60K steps. That's a real overnight + half-a-day
commitment — only kick this off when you can leave the Mac running.

Outputs land under ``checkpoints/phase6_v05_pretrain/`` with periodic
saves and the standard JSON training log.
"""

import argparse
from pathlib import Path

import torch

from ghostlm.config import GhostLMConfig
from ghostlm.dataset import build_dataloaders
from ghostlm.model import GhostLM
from ghostlm.tokenizer import GhostTokenizerV05, load_tokenizer
from ghostlm.trainer import GhostTrainer


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="GhostLM v0.5 from-scratch pretrain")
    p.add_argument("--train-data", default="data/processed/train.jsonl")
    p.add_argument("--val-data", default="data/processed/val.jsonl")
    p.add_argument("--tokenizer", default="data/tokenizer_v05/tokenizer.json",
                   help="Path to the v0.5 BPE tokenizer.json")
    p.add_argument("--run-name", default="phase6_v05_pretrain")
    p.add_argument("--max-steps", type=int, default=60_000)
    p.add_argument("--warmup-steps", type=int, default=2_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--save-interval", type=int, default=2_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    """Build everything and run the training loop."""
    args = parse_args()
    torch.manual_seed(args.seed)

    # ---- Tokenizer ----
    tokenizer_path = Path(args.tokenizer)
    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"v0.5 tokenizer not found at {tokenizer_path}. Run "
            "`scripts/train_tokenizer.py` first."
        )
    tokenizer = load_tokenizer(str(tokenizer_path))
    if not isinstance(tokenizer, GhostTokenizerV05):
        raise TypeError("Expected v0.5 tokenizer; got the legacy GhostTokenizer.")
    print(f"Tokenizer: {tokenizer}")

    # ---- Config: ghost-small-v0.5 architecture (RoPE + SwiGLU + RMSNorm) ----
    config = GhostLMConfig.from_preset("ghost-small-v0.5")
    config.vocab_size = tokenizer.vocab_size
    config.context_length = args.context_length
    config.batch_size = args.batch_size
    config.grad_accum_steps = args.grad_accum_steps
    config.learning_rate = args.learning_rate
    config.warmup_steps = args.warmup_steps
    config.max_steps = args.max_steps
    config.eval_interval = args.eval_interval
    config.save_interval = args.save_interval
    config.checkpoint_dir = f"checkpoints/{args.run_name}"
    config.log_dir = f"logs/{args.run_name}"
    config.device = args.device
    print(config)

    # ---- Model ----
    model = GhostLM(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ---- Data ----
    train_loader, val_loader = build_dataloaders(
        args.train_data, args.val_data, tokenizer, config,
    )

    # ---- Train ----
    trainer = GhostTrainer(model, config)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {config.checkpoint_dir}")
    print(f"Logs to:   {config.log_dir}")
    trainer.train(train_loader, val_loader)
    print()
    print("v0.5 pretrain complete.")


if __name__ == "__main__":
    main()
