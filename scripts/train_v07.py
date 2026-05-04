#!/usr/bin/env python3
"""GhostLM v0.7 from-scratch pretrain — 81M wide variant.

Tests whether scale matters in the 45M -> 81M range, given that the
debiased CTIBench eval revealed v0.4 (45M) and v0.6 (45M) both top
out at the same ~30% real capability ceiling. Per the SmolLM2 /
Phi-3.5-mini literature, factual recall on cybersec MCQ should start
emerging meaningfully around 130-300M params; 81M is the rung between
our existing 45M models and that emergence point.

Architecture: ghost-small-v0.5 (RoPE + SwiGLU + RMSNorm) but widened
from d_model 512 -> 768, d_ff 2048 -> 3072, n_heads 8 -> 12. Depth
stays at 6 layers (going deeper hurts MPS throughput
disproportionately; the 130M ghost-medium ran at 13.6s/step vs this
81M wide variant at 1.5s/step).

Tokenizer: GPT-2 50K BPE (proven in v0.6 to be on par with v0.4 base
under debiased eval, despite single-order letter scoring suggesting
otherwise).

Corpus: data/processed/train.jsonl (307K records including the
2026-05-03 MITRE full + CISA KEV expansion).

Sized for an overnight run: 15K steps × ctx 1024 × effective batch 16
(4 × accum 4) at ~1.5s/step ≈ 6.5h. Resume-safe via --resume.
"""

import argparse
from pathlib import Path

import torch

from ghostlm.config import GhostLMConfig
from ghostlm.dataset import build_dataloaders
from ghostlm.model import GhostLM
from ghostlm.tokenizer import GhostTokenizer
from ghostlm.trainer import GhostTrainer


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="GhostLM v0.7 from-scratch pretrain (81M wide)")
    p.add_argument("--train-data", default="data/processed/train.jsonl")
    p.add_argument("--val-data", default="data/processed/val.jsonl")
    p.add_argument("--run-name", default="phase14_v07_pretrain")
    p.add_argument("--max-steps", type=int, default=15_000)
    p.add_argument("--warmup-steps", type=int, default=1_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--save-interval", type=int, default=1_500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", default=None,
                   help="Resume from a checkpoint .pt file.")
    return p.parse_args()


def main() -> None:
    """Build everything and run the v0.7 training loop."""
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = GhostTokenizer()
    print(f"Tokenizer: {tokenizer}")

    # Start from the v0.5 architecture preset, then widen.
    config = GhostLMConfig.from_preset("ghost-small-v0.5")
    config.n_layers = 6
    config.d_model = 768
    config.n_heads = 12
    config.d_ff = 3072
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

    model = GhostLM(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

    train_loader, val_loader = build_dataloaders(
        args.train_data, args.val_data, tokenizer, config,
    )

    trainer = GhostTrainer(model, config)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {config.checkpoint_dir}")
    print(f"Logs to:   {config.log_dir}")

    if args.resume:
        if not Path(args.resume).exists():
            raise FileNotFoundError(f"--resume path not found: {args.resume}")
        print(f"Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)
        print(f"  Resumed at step {trainer.step}, best val so far: {trainer.best_val_loss:.4f}")

    trainer.train(train_loader, val_loader)
    print()
    print("v0.7 pretrain complete.")


if __name__ == "__main__":
    main()
