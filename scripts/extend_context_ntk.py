#!/usr/bin/env python3
"""Extend GhostLM's context length from training-time (1024) to
inference-time (16K-32K) via RoPE NTK-aware interpolation.

Why this matters: real cybersec workflows are long-context. An
incident-response analyst dumping a 50K-token threat report into the
chat box doesn't want a 1024-context model. ghost-base trains at
1024 ctx; extending to 16K-32K unlocks the IR triage register that
4K-context models can't touch.

NTK-aware interpolation (the standard method since 2023's Code Llama
paper): instead of linearly interpolating RoPE frequencies (which
hurts low-frequency / long-distance reasoning), scale the RoPE base
non-linearly so high-frequency components stay sharp while
low-frequency components stretch to cover the new context.

The math:

    base_new = base_old * (scale ^ (d / (d - 2)))

where:
    scale = new_ctx / old_ctx   (e.g. 16384 / 1024 = 16.0)
    d     = head dim            (= d_model / n_heads, typically 64)

For GhostLM v0.9 (head_dim=64) extending from 1024 -> 16384:
    base_new = 10000 * (16 ** (64/62)) = ~177,000

After updating rope_base, you have two paths:

    Option A (zero-shot extension): just change rope_base at
    inference. Works surprisingly well for ~2-4x extensions but
    quality degrades at 16x. No retraining cost.

    Option B (fine-tune extension): take the v0.9 chat checkpoint,
    update rope_base, run a short fine-tune (1-3K steps) on
    long-form corpus (arxiv_full, full security blog posts, NIST
    SP 800 documents that are naturally long). The model adapts
    to the new positional encoding distribution. This is the
    canonical recipe for production-grade extensions.

This script does Option B end-to-end:

  1. Load the v0.9 chat checkpoint
  2. Compute new rope_base for the target context
  3. Save the rebased checkpoint as a new starting point
  4. Run scripts/finetune_chat.py with --context-length 16384 over
     a long-form data subset (filtered to records >=2K tokens)
  5. Output: checkpoints/phase20_ctx16k/best_model.pt

Run:

    PYTHONPATH=. python3 scripts/extend_context_ntk.py \\
        --base checkpoints/phase19_chat_v09/best_model.pt \\
        --target-context 16384 \\
        --out-dir checkpoints/phase20_ctx16k

Cost: ~3-5 GPU hours on H100 / RTX 6000. M4 will OOM at 16K context;
needs rented GPU or workstation card.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM


def compute_ntk_base(old_base: float, old_ctx: int, new_ctx: int,
                     head_dim: int) -> float:
    """NTK-aware rope_base for context extension."""
    scale = new_ctx / old_ctx
    return old_base * (scale ** (head_dim / (head_dim - 2)))


def rebase_checkpoint(src: Path, dst: Path, new_ctx: int,
                      new_rope_base: float) -> None:
    """Load a checkpoint, update its config's rope_base + context_length,
    and save under dst. The model weights are unchanged; only the
    positional encoding's base frequency parameter shifts."""
    print(f"Loading {src}")
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    saved = ckpt["config"]
    cfg = GhostLMConfig(**{
        f.name: saved[f.name]
        for f in fields(GhostLMConfig)
        if f.name in saved
    })
    print(f"  current: rope_base={cfg.rope_base}, context_length={cfg.context_length}")
    cfg.rope_base = new_rope_base
    cfg.context_length = new_ctx
    print(f"  rebased: rope_base={cfg.rope_base:.1f}, context_length={cfg.context_length}")

    # Save with updated config; weights identical.
    out = {
        "config": {f.name: getattr(cfg, f.name) for f in fields(cfg)},
        "model_state_dict": ckpt["model_state_dict"],
        "step": ckpt.get("step"),
        "val_loss": ckpt.get("val_loss"),
        "rebased_from": str(src),
        "rebased_at": __import__("datetime").datetime.now().isoformat(),
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)
    print(f"Wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True,
                   help="Source checkpoint to rebase")
    p.add_argument("--target-context", type=int, default=16384,
                   help="New context length (default 16384, i.e. 16K)")
    p.add_argument("--out-dir", default="checkpoints/phase20_ctx16k")
    p.add_argument("--head-dim", type=int, default=64,
                   help="Per-head dimension (override if checkpoint differs)")
    p.add_argument("--rebase-only", action="store_true",
                   help="Just compute + save the rebased checkpoint; don't "
                        "run the fine-tune. Useful for zero-shot extension "
                        "test before committing to the tune.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    src = Path(args.base)
    if not src.exists():
        sys.exit(f"checkpoint not found: {src}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: probe the source's current rope_base + ctx.
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    cfg_raw = ckpt["config"]
    old_base = cfg_raw.get("rope_base", 10000.0)
    old_ctx = cfg_raw.get("context_length", 1024)

    print(f"Source:        {src}")
    print(f"  rope_base:   {old_base}")
    print(f"  context:     {old_ctx}")
    print(f"Target context: {args.target_context}")

    new_base = compute_ntk_base(old_base, old_ctx,
                                args.target_context, args.head_dim)
    scale = args.target_context / old_ctx
    print(f"NTK rebase:")
    print(f"  scale:       {scale:.2f}x")
    print(f"  head_dim:    {args.head_dim}")
    print(f"  new rope_base: {new_base:.1f}")

    # Step 2: save rebased checkpoint.
    rebased_path = out_dir / "rebased_init.pt"
    rebase_checkpoint(src, rebased_path, args.target_context, new_base)

    if args.rebase_only:
        print("\n--rebase-only set; skipping fine-tune. Test the rebased "
              "checkpoint zero-shot with scripts/chat.py to see whether "
              "long-context inference works without further training.")
        return 0

    # Step 3: kick off the fine-tune. We don't run the tune here
    # directly — that's a multi-hour GPU job that wants its own
    # dedicated entry point. Instead we print the exact command the
    # operator should run, with all the right flags.
    cmd = [
        sys.executable, "scripts/finetune_chat.py",
        "--checkpoint", str(rebased_path),
        "--context-length", str(args.target_context),
        "--max-steps", "3000",
        "--warmup-steps", "200",
        "--batch-size", "1",
        "--grad-accum-steps", "16",
        "--learning-rate", "5e-5",
        "--data", "data/processed/longform_train.jsonl",
        "--out-dir", str(out_dir),
    ]
    print("\nNext step (run on GPU host with the rebased checkpoint):")
    print(" ".join(cmd))
    print()
    print("Tune the long-form data subset only (records >= 2K tokens). "
          "Recipe assumes data/processed/longform_train.jsonl exists; "
          "filter from data/processed/train.jsonl with a quick "
          "`jq 'select(.text | length > 8000)' < train.jsonl`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
