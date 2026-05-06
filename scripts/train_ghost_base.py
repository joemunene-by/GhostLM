#!/usr/bin/env python3
"""GhostLM ghost-base from-scratch pretrain (~360M, v1.0 target).

Eight ghost-small attempts (v0.4 through v0.9, plus the ctx-1024
variant) saturated at 27-29% on the full debiased CTIBench bench and
0-2% on free-form fact recall. The diagnosis from the v0.9.2
postmortem is that 81M params can't bind facts retrievably, regardless
of corpus density. Ghost-base lifts the model to the parameter rung
where SmolLM2-360M and Phi-3.5-mini report factual recall on cybersec
MCQ starting to emerge.

Architecture (ghost-base, matches SmolLM2-360M):
  layers   30   (5x v0.7's 6; deeper helps factual binding)
  d_model  960  (1.25x v0.7's 768; head_dim 64 with 15 heads)
  n_heads  15   (head_dim 64, head budget 64 unchanged)
  d_ff     3200 (~3.33x d_model; SwiGLU full width, sized to hit
                  exactly ~360M params matching SmolLM2-360M)
  vocab    50,264  (GPT-2 50K BPE + 7 special tokens, unchanged)
  context  1024 train, 2048 inference (RoPE base 10000)
  norm     RMSNorm (unchanged)
  ffn      SwiGLU (unchanged)
  pos      RoPE (unchanged)

Estimated params: ~360M (matches SmolLM2-360M, the literature
reference where factual recall on cybersec MCQ has been reported
to start emerging). The earlier "12L x 768d -> ~360M" estimate in
the spec was incorrect; an M4 smoke at 12L x 768d revealed the
actual count was 124M, so this launcher is updated to the deeper
SmolLM2-style shape that hits the target capacity.

Recipe defaults assume an H100 (or comparable bf16 GPU) with at
least 40GB. On M4 MPS this will OOM at the default batch size; for
M4 smoke-testing pass --batch-size 1 --grad-accum-steps 32 and
expect ~5s/step. The real runs are H100 territory.

Corpus: data/processed/train.jsonl after the v1.0 rebuild that folds
in security_code + fineweb_edu + nist_sp800 + security_blogs +
wikipedia_cyber. Target ~3-5B-token mix on 360M params is still
sub-Chinchilla but covers code + cybersec + general language.

Acceptance gate (per docs/ghost_base_spec.md):
  * >=40% per-perm avg on debiased CTIBench (n=2500), OR
  * >=65% on the in-repo CTF eval (n=30), OR
  * >=30% on the 50-question fact-recall set
Passing any one validates the rung. Fact-recall threshold is the
truth metric.
"""

import argparse
from pathlib import Path

import torch

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.tokenizer import GhostTokenizer
from ghostlm.dataset import build_dataloaders
from ghostlm.trainer import GhostTrainer


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="GhostLM ghost-base pretrain (12L x 768d, ~360M)")
    p.add_argument("--train-data", default="data/processed/train.jsonl")
    p.add_argument("--val-data", default="data/processed/val.jsonl")
    p.add_argument("--run-name", default="ghost_base_pretrain")
    p.add_argument("--max-steps", type=int, default=30_000)
    p.add_argument("--warmup-steps", type=int, default=2_000)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=16,
                   help="Per-device batch (H100 default; lower for M4 smoke)")
    p.add_argument("--grad-accum-steps", type=int, default=4)
    p.add_argument("--context-length", type=int, default=1024)
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--save-interval", type=int, default=1_500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto",
                   help="cuda / mps / cpu / auto")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"],
                   help="Mixed-precision dtype (bf16 for H100, fp32 for MPS smoke)")
    p.add_argument("--resume", default=None,
                   help="Path to a checkpoint .pt to resume from")
    return p.parse_args()


def main() -> None:
    """Build config + model + trainer, run."""
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = GhostTokenizer()
    print(f"Tokenizer: {tokenizer}")

    # Start from the v0.5 architecture preset (RoPE + SwiGLU + RMSNorm),
    # then deepen + widen to the SmolLM2-360M-style ghost-base shape.
    config = GhostLMConfig.from_preset("ghost-small-v0.5")
    config.n_layers = 30
    config.d_model = 960
    config.n_heads = 15
    config.d_ff = 3200
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
    config.dtype = args.dtype
    print(config)

    model = GhostLM(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params / 1e6:.1f}M)")
    if n_params < 300_000_000 or n_params > 420_000_000:
        print(f"WARNING: param count {n_params / 1e6:.0f}M outside expected "
              f"~300-420M range for ghost-base (12L x 768d x 12h). Check overrides.")

    train_loader, val_loader = build_dataloaders(
        args.train_data, args.val_data, tokenizer, config,
    )

    trainer = GhostTrainer(model, config)
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    print(f"Saving to:    {config.checkpoint_dir}")
    print(f"Logs to:      {config.log_dir}")

    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
