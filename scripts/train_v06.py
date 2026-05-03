#!/usr/bin/env python3
"""GhostLM v0.6 from-scratch pretrain — RoPE + SwiGLU + RMSNorm + GPT-2 50K BPE.

Tests the leading hypothesis from the v0.5 postmortem: that v0.5 trails v0.4
on CTIBench MCQ because v0.5's custom 32K BPE fragments cybersec terms that
v0.4's GPT-2 50K BPE keeps whole. v0.6 keeps everything else from v0.5
(modern attention stack, expanded corpus) and swaps only the tokenizer.

Architecture: ghost-small-v0.5 preset (RoPE + SwiGLU + RMSNorm), but vocab
set to 50264 (GPT-2 50K + 7 special tokens) instead of the v0.5 32K BPE.
Tokenizer: the legacy ``GhostTokenizer`` wrapping tiktoken's GPT-2 BPE.
Corpus: ``data/processed/train.jsonl`` (307K records — includes the
2026-05-03 MITRE full + CISA KEV expansion).

Defaults sized for an overnight run: 15K steps × ctx 1024 × effective
batch 32 ≈ 500M tokens, ~7-8h on Mac M4 MPS. v0.4 base was Chinchilla-
undertrained at ~30K steps and still held the chat-tune ceiling, so
this should be enough signal to test the BPE-swap hypothesis without a
multi-day commitment.

Resume-safe via --resume.
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
    p = argparse.ArgumentParser(description="GhostLM v0.6 from-scratch pretrain (v0.5 arch + GPT-2 BPE)")
    p.add_argument("--train-data", default="data/processed/train.jsonl")
    p.add_argument("--val-data", default="data/processed/val.jsonl")
    p.add_argument("--run-name", default="phase9_v06_pretrain")
    p.add_argument("--max-steps", type=int, default=15_000)
    p.add_argument("--warmup-steps", type=int, default=1_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--save-interval", type=int, default=2_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", default=None,
                   help="Resume from a checkpoint .pt file. Restores model + "
                        "optimizer state + step counter.")
    return p.parse_args()


def main() -> None:
    """Build everything and run the v0.6 training loop."""
    args = parse_args()
    torch.manual_seed(args.seed)

    # ---- Tokenizer: legacy GPT-2 50K BPE (NOT the v0.5 custom 32K) ----
    tokenizer = GhostTokenizer()
    print(f"Tokenizer: {tokenizer}")

    # ---- Config: v0.5 architecture, GPT-2 vocab ----
    config = GhostLMConfig.from_preset("ghost-small-v0.5")
    config.vocab_size = tokenizer.vocab_size  # 50264
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
    print(f"Model parameters: {n_params:,}")

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
    print("v0.6 pretrain complete.")


if __name__ == "__main__":
    main()
