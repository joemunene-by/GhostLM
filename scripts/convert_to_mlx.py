#!/usr/bin/env python3
"""Convert a PyTorch GhostLM checkpoint to MLX format.

Output is a single ``.safetensors`` file containing the model weights with
MLX-friendly key names, plus a ``config.json`` describing the architecture
so ``scripts/mlx_chat.py`` can rebuild the model. Optional 4-bit quantization
of the linear weights happens in-process via ``mlx.core.quantize``.

The PyTorch checkpoint is read via ``torch.load`` (CPU only, no MPS / CUDA
required at conversion time). The resulting MLX weights run on Apple Silicon
through the Metal backend with no external dependencies.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Dict

try:
    import mlx.core as mx
    import mlx.nn as mlx_nn
except ImportError as e:  # pragma: no cover
    raise SystemExit("mlx not installed: pip install mlx mlx-lm") from e

import torch

from ghostlm.config import GhostLMConfig


def torch_state_to_mlx(state: Dict[str, torch.Tensor]) -> Dict[str, "mx.array"]:
    """Convert a PyTorch state_dict to MLX arrays in float32.

    Key names are kept identical — the MLX model class in ``scripts/mlx_chat.py``
    mirrors the PyTorch module names exactly so no renaming is necessary.

    Args:
        state: Standard PyTorch ``model.state_dict()`` mapping.

    Returns:
        Dict of MLX float32 arrays with the same keys.
    """
    out: Dict[str, mx.array] = {}
    for k, v in state.items():
        # torch.float32 → mx.float32; bfloat16 / float16 are upcast at conversion.
        arr = v.detach().to(torch.float32).cpu().numpy()
        out[k] = mx.array(arr)
    return out


def quantize_linears(
    weights: Dict[str, "mx.array"],
    *,
    bits: int = 4,
    group_size: int = 64,
) -> Dict[str, "mx.array"]:
    """Quantize every Linear's ``.weight`` in place to ``bits`` precision.

    MLX's quantize returns three arrays per weight: ``{name}.weight`` (the
    packed quantized values), ``{name}.scales``, and ``{name}.biases``. The
    inference path in mlx_chat.py knows how to dequantize on the fly.

    Args:
        weights: MLX state dict from ``torch_state_to_mlx``.
        bits: 2/3/4/6/8 are supported by mlx; 4 is the typical sweet spot.
        group_size: Quantization group; 64 matches mlx-lm defaults.

    Returns:
        Mutated copy of ``weights`` with linear-weight rows replaced.
    """
    out: Dict[str, mx.array] = dict(weights)
    # Identify linear weights by name pattern — every weight ending in
    # ``.weight`` whose corresponding module is a Linear in the original
    # architecture. We match by suffix; this covers c_qkv, proj, fc1, fc2,
    # and lm_head. ``token_embedding.weight`` and ``pos_embedding.weight`` are
    # NOT quantized (embeddings are kept full-precision per common practice).
    skip_prefixes = ("token_embedding.", "pos_embedding.", "ln_")
    for k in list(out.keys()):
        if not k.endswith(".weight"):
            continue
        if any(k.startswith(s) or f".{s}" in k for s in skip_prefixes):
            continue
        # LayerNorm weights are 1-D; skip them too.
        if out[k].ndim != 2:
            continue
        # lm_head.weight is tied to token_embedding.weight in the source —
        # we still quantize it because at inference the lm_head is a separate
        # quantized Linear and the embedding stays full-precision.
        w_q, scales, biases = mx.quantize(out[k], group_size=group_size, bits=bits)
        out[k] = w_q
        out[k.replace(".weight", ".scales")] = scales
        out[k.replace(".weight", ".biases")] = biases
        # Annotate so the loader knows this is quantized.
        out.setdefault(f"_quantized.{k}", mx.array([bits, group_size]))
    return out


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    p = argparse.ArgumentParser(description="GhostLM PyTorch → MLX converter")
    p.add_argument("--checkpoint", required=True,
                   help="Path to PyTorch .pt checkpoint")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for the MLX safetensors + config")
    p.add_argument("--quantize", type=int, default=0, choices=[0, 4, 8],
                   help="If non-zero, quantize linear weights to this bit-depth "
                        "(4 is the typical sweet spot; 0 disables)")
    p.add_argument("--group-size", type=int, default=64,
                   help="Quantization group size (mlx-lm default 64)")
    return p.parse_args()


def main() -> None:
    """Run the conversion."""
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg_raw = ckpt["config"]
    if isinstance(cfg_raw, dict):
        cfg = GhostLMConfig(**{
            f.name: cfg_raw[f.name]
            for f in fields(GhostLMConfig)
            if f.name in cfg_raw
        })
    else:
        cfg = cfg_raw

    state = ckpt.get("model_state_dict", ckpt.get("model"))
    print(f"  {len(state)} tensors, "
          f"params={sum(v.numel() for v in state.values()):,}")
    print(f"  Architecture: vocab={cfg.vocab_size} d_model={cfg.d_model} "
          f"n_heads={cfg.n_heads} n_layers={cfg.n_layers} ctx={cfg.context_length}")

    print("Converting tensors → MLX...")
    mlx_state = torch_state_to_mlx(state)

    if args.quantize:
        print(f"Quantizing to {args.quantize}-bit (group_size={args.group_size})...")
        mlx_state = quantize_linears(
            mlx_state, bits=args.quantize, group_size=args.group_size,
        )
        print(f"  Now {len(mlx_state)} tensors after quantization split")

    # Use mlx.core.save_safetensors when available; fall back to numpy save.
    weights_path = out_dir / "weights.safetensors"
    try:
        mx.save_safetensors(str(weights_path), mlx_state)
    except AttributeError:
        # Older mlx — write a directory of .npy files instead.
        weights_path = out_dir / "weights"
        weights_path.mkdir(exist_ok=True)
        for k, v in mlx_state.items():
            mx.save(str(weights_path / k.replace(".", "_") + ".npy"), v)

    print(f"  Wrote {weights_path}")

    config_path = out_dir / "config.json"
    cfg_dict = asdict(cfg) if hasattr(cfg, "__dataclass_fields__") else vars(cfg)
    cfg_dict["_quantization_bits"] = args.quantize
    cfg_dict["_quantization_group_size"] = args.group_size if args.quantize else None
    config_path.write_text(json.dumps(cfg_dict, indent=2))
    print(f"  Wrote {config_path}")

    if args.quantize:
        size_mb = sum(v.nbytes for v in mlx_state.values()) / 1024 / 1024
        print(f"\n  Quantized weights: ~{size_mb:.1f} MB total")
    print("\nLoad with: scripts/mlx_chat.py --weights-dir " + str(out_dir))


if __name__ == "__main__":
    main()
