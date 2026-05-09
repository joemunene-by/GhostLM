#!/usr/bin/env python3
"""GhostLM v0.9.33 from-scratch pretrain — ghost-small (81M wide) on the
422M-token v0.9.32 corpus.

This is the truth experiment for the v0.9.30/31/32 push: did the open-source
code corpus expansion (code share 2.4% -> 11.6%) and the corpus growth (363M ->
422M tokens, 516K -> 768K records) actually move the val PPL needle? The
existing v0.4.0 ghost-small (81M wide, RoPE+SwiGLU+RMSNorm, GPT-2 50K BPE,
trained on the prior 273M-token corpus) hit val_loss 2.3535 / val PPL 11.12.
Same architecture + tokenizer, new corpus -> direct A/B.

Architecture parity with the v0.7 pretrain script:
  ghost-small-v0.5 preset (RoPE + SwiGLU + RMSNorm), widened
  d_model 512 -> 768, d_ff 2048 -> 3072, n_heads 8 -> 12, n_layers 6.
  GPT-2 50K BPE (no surgery on the embedding rows).

Steps budget: 30K (was 15K for the smaller v0.7 corpus). Roughly Chinchilla-
proportional to the new corpus volume — 422M tokens / 273M = 1.55x growth.
At ~1.5s/step on M4 MPS: ~12.5 hours for a clean run. Resume-safe via
--resume.

Outputs:
  checkpoints/phase21_v0933_pretrain/  — {best_model, checkpoint_step_*}.pt
  logs/phase21_v0933_pretrain/         — JSONL training log

After completion, compare against v0.4.0:
  python3 scripts/eval_chat.py \\
      --checkpoint checkpoints/phase21_v0933_pretrain/best_model.pt \\
      ...
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
    p = argparse.ArgumentParser(
        description="GhostLM v0.9.33 from-scratch pretrain on 422M-token corpus",
    )
    p.add_argument("--train-data", default="data/processed/train.jsonl")
    p.add_argument("--val-data", default="data/processed/val.jsonl")
    p.add_argument("--run-name", default="phase21_v0933_pretrain")
    p.add_argument("--max-steps", type=int, default=30_000)
    p.add_argument("--warmup-steps", type=int, default=1_500)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--save-interval", type=int, default=2_500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", default=None,
                   help="Resume from a checkpoint .pt file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = GhostTokenizer()
    print(f"Tokenizer: {tokenizer}")

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
        print(f"  Resumed at step {trainer.step}, "
              f"best val so far: {trainer.best_val_loss:.4f}")

    trainer.train(train_loader, val_loader)
    print()
    print("v0.9.33 pretrain complete.")
    print()
    print("Next: compare against v0.4.0 baseline (val 2.3535) via:")
    print("  python3 scripts/eval_chat.py "
          f"--checkpoint checkpoints/{args.run_name}/best_model.pt ...")


if __name__ == "__main__":
    main()
