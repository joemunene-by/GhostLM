#!/usr/bin/env python3
"""Average the weights of several checkpoints into one model-only checkpoint.

Weight averaging over the tail of a run (LAWA; post-hoc EMA as in IMU-1)
usually beats any single checkpoint for free. Pass checkpoints oldest first.

Usage:
    python scripts/average_checkpoints.py ckpt_a.pt ckpt_b.pt ckpt_c.pt --out avg.pt
    python scripts/average_checkpoints.py snapshots/*.pt --ema 0.5 --out ema.pt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch


def _step_key(path: str) -> tuple:
    m = re.search(r"(\d+)", Path(path).stem)
    return (int(m.group(1)) if m else -1, path)


def average(paths: list[str], ema: float | None = None) -> dict:
    """Uniform mean of model weights, or an EMA over ``paths`` in order when ``ema`` is set."""
    avg = None
    first = None
    for i, path in enumerate(paths):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
        if first is None:
            first = ckpt
            avg = {k: v.detach().clone().float() for k, v in state.items()}
            continue
        if state.keys() != avg.keys():
            raise ValueError(f"{path} has different parameters from {paths[0]}")
        w = (1 - ema) if ema is not None else 1.0 / (i + 1)
        for k, v in state.items():
            if avg[k].is_floating_point():
                avg[k].lerp_(v.float(), w)
    dtypes = {k: v.dtype for k, v in first.get("model_state_dict", first.get("model", first)).items()}
    return {
        "step": first.get("step") if len(paths) == 1 else torch.load(
            paths[-1], map_location="cpu", weights_only=False, mmap=True).get("step"),
        "val_loss": float("inf"),
        "model_state_dict": {k: v.to(dtypes[k]) for k, v in avg.items()},
        "config": first.get("config"),
        "averaged_from": list(paths),
        "ema": ema,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoints", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--ema", type=float, default=None,
                   help="EMA decay (0-1) instead of a uniform mean; later checkpoints weigh more.")
    p.add_argument("--sort", action="store_true", help="Sort inputs by the step number in their names.")
    args = p.parse_args()

    paths = sorted(args.checkpoints, key=_step_key) if args.sort else args.checkpoints
    result = average(paths, args.ema)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.out)
    print(f"averaged {len(paths)} checkpoints -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
