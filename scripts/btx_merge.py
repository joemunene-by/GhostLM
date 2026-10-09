#!/usr/bin/env python3
"""Merge domain-branched dense checkpoints into one Mixture-of-Experts model (BTX).

Branch-Train-MiX (Sukhbaatar et al., 2024): copies of one seed model are trained
separately on different domains, then their feed-forward layers become the
experts of a single MoE and every other parameter (attention, norms,
embeddings) is averaged. A freshly initialised router then learns, on mixed
data, which expert each token should use.

    python scripts/btx_merge.py \\
        --branch cybersec=checkpoints/btx_cybersec/best_model.pt \\
        --branch code=checkpoints/btx_code/best_model.pt \\
        --branch math=checkpoints/btx_math/best_model.pt \\
        --branch general=checkpoints/btx_general/best_model.pt \\
        --top-k 2 --out checkpoints/ghost_base_moe/merged.pt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

FFN = re.compile(r"^(blocks\.\d+)\.ffn\.(fc[123])\.weight$")


def merge(branches: list[tuple[str, dict]], top_k: int, seed: int = 0) -> dict:
    """Return a torch-format MoE checkpoint built from (domain, checkpoint) pairs."""
    names = [name for name, _ in branches]
    states = [ckpt["model_state_dict"] for _, ckpt in branches]
    keys = set(states[0])
    for name, state in zip(names[1:], states[1:]):
        if set(state) != keys:
            raise ValueError(f"branch {name!r} has different parameters from {names[0]!r}")

    merged = {}
    gen = torch.Generator().manual_seed(seed)
    for key in sorted(keys):
        m = FFN.match(key)
        if m:
            for e, state in enumerate(states):
                merged[f"{m.group(1)}.ffn.experts.{e}.{m.group(2)}.weight"] = state[key].float().clone()
        elif key != "lm_head.weight":
            merged[key] = torch.stack([s[key].float() for s in states]).mean(dim=0)
    d_model = merged["token_embedding.weight"].shape[1]
    layers = sorted({int(k.split(".")[1]) for k in merged if k.startswith("blocks.")})
    for layer in layers:
        merged[f"blocks.{layer}.ffn.gate.weight"] = torch.randn(len(states), d_model, generator=gen) * 0.02
    merged["lm_head.weight"] = merged["token_embedding.weight"]

    config = dict(branches[0][1]["config"])
    config.update(use_moe=True, n_experts=len(states), n_experts_active=min(top_k, len(states)))
    return {
        "step": 0,
        "val_loss": float("inf"),
        "best_val_loss": float("inf"),
        "model_state_dict": merged,
        "config": config,
        "btx": {"experts": names, "branch_steps": [ckpt.get("step") for _, ckpt in branches]},
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--branch", action="append", required=True, metavar="DOMAIN=CHECKPOINT",
                   help="one per expert, in expert order")
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    branches = []
    for spec in args.branch:
        name, _, path = spec.partition("=")
        # mmap: branch checkpoints may carry multi-GB optimizer state the merge never reads.
        branches.append((name, torch.load(path, map_location="cpu", weights_only=False, mmap=True)))
    out = merge(branches, args.top_k, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    n = sum(v.numel() for k, v in out["model_state_dict"].items() if k != "lm_head.weight")
    print(f"merged {len(branches)} branches ({', '.join(out['btx']['experts'])}) -> {args.out}: "
          f"{n / 1e6:.0f}M parameters, top-{out['config']['n_experts_active']} routing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
