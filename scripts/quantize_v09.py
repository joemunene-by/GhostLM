#!/usr/bin/env python3
"""Quantize the v0.9 chat checkpoint to fp16 and int8, ship as
inference-only artifacts alongside the bf16 weights.

Why: the bf16 v0.9 chat checkpoint at 324 MB (slim, optimizer-state
stripped) hits two friction points:

  1. The HF free-Space LFS budget is 1 GB total per repo. Hosting
     just the bf16 weights inside the Space repo is feasible but
     leaves no headroom for the RAG index, chat history, etc; we
     already pushed the bf16 weights to a Models repo for that
     reason.
  2. Anyone wanting to run GhostLM locally on a small CPU host (e.g.
     a Raspberry Pi, a 4 GB VPS) wants a smaller artifact. Even on
     M4-class workstations, fp16/int8 inference is 1.5-3x faster
     than fp32 for a model this size on CPU.

This script produces two new artifacts:

  best_model_fp16.pt   ~162 MB. Same architecture, half-precision
                        weights. Lossless for inference quality at
                        81M parameters; PyTorch CPU fp16 path is
                        well-supported.

  best_model_int8.pt   ~80-110 MB. Per-layer dynamic int8 quantization
                        of the linear layers via torch.ao.quantization.
                        Some quality loss (not measured here; quantize
                        + bench separately to confirm). Inference in
                        int8 on CPU is meaningfully faster than fp16
                        on hardware with vectorized int8 dot-product
                        (most modern x86 + Apple Silicon).

A third artifact, GGUF for llama.cpp / Ollama, is non-trivial because
GhostLM uses SwiGLU + RMSNorm + RoPE in a layout that does not
exactly match LLaMA-2's canonical GGUF format. The skeleton of that
exporter lives here as ``export_gguf_attempt`` but is currently
unimplemented; doing it right is its own project (~1 week's work).
For Ollama-style local inference today, the int8 .pt artifact is the
shipped path.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM


def load_full_checkpoint(path: Path) -> tuple[GhostLM, GhostLMConfig, dict]:
    """Load weights + config from a slim or full checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    saved = ckpt["config"]
    cfg = GhostLMConfig(**{
        f.name: saved[f.name]
        for f in fields(GhostLMConfig)
        if f.name in saved
    })
    model = GhostLM(cfg)
    state = ckpt.get("model_state_dict", ckpt.get("model"))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, cfg, ckpt


def save_fp16(model: GhostLM, ckpt: dict, dst: Path) -> None:
    """Cast weights to fp16 and save as a slim inference checkpoint."""
    fp16 = {}
    for k, v in model.state_dict().items():
        if v.dtype.is_floating_point:
            fp16[k] = v.to(torch.float16)
        else:
            fp16[k] = v
    out = {
        "config": ckpt["config"],
        "model_state_dict": fp16,
        "step": ckpt.get("step"),
        "val_loss": ckpt.get("val_loss"),
        "dtype": "float16",
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)


def quantize_int8(model: GhostLM) -> GhostLM:
    """Apply dynamic int8 quantization to all nn.Linear layers in the
    model. PyTorch's torch.ao.quantization.quantize_dynamic is the
    cleanest option here: it leaves the model architecturally identical
    (the GhostLM class still works for inference) but the linear layer
    weights are int8 with per-channel scales. No calibration pass
    needed (dynamic quantization computes activation scales on the
    fly per inference call).

    Note: this does NOT quantize embedding weights or norm parameters.
    Linear layers are ~95% of the model's parameter count at this
    architecture, so the savings still approach 4x on the dominant
    tensors. fp16 fallback weights for non-linear modules."""
    quantized = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},  # only quantize linear layers
        dtype=torch.qint8,
    )
    return quantized


def save_int8(model: GhostLM, ckpt: dict, dst: Path) -> None:
    """Save the int8-quantized model. The state_dict here contains
    QInt8 packed tensors that PyTorch reloads via the same
    quantize_dynamic call at load time; the consumer needs to apply
    the same wrapping before load_state_dict."""
    qmodel = quantize_int8(model)
    out = {
        "config": ckpt["config"],
        "model_state_dict": qmodel.state_dict(),
        "step": ckpt.get("step"),
        "val_loss": ckpt.get("val_loss"),
        "dtype": "qint8",
        "quantization_recipe": "torch.ao.quantization.quantize_dynamic({nn.Linear})",
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst)


def export_gguf_attempt(model: GhostLM, cfg: GhostLMConfig, dst: Path) -> None:
    """Placeholder for a future GGUF exporter compatible with
    llama.cpp / Ollama.

    GGUF has no native architecture matching GhostLM exactly:
    LLaMA-2 GGUF expects (Q, K, V) merged into wq/wk/wv tensors with
    specific naming, SwiGLU's three-projection FFN at gate/down/up,
    RMSNorm, RoPE base 10000. GhostLM's SwiGLU is named fc1/fc2/fc3
    rather than gate/up/down; the attention is split into separate
    wq/wk/wv linear layers; the RMSNorm scale is named .weight.
    Mapping these to llama.cpp's expected tensor names is
    mechanically straightforward but error-prone.

    Right approach when this happens: study huggingface/transformers'
    LlamaForCausalLM state_dict conventions, write a translator that
    renames + reshapes, write a GGUF v3 writer (or use gguf-py from
    llama.cpp). Validate with llama.cpp's quantize tool. ~1 week of
    careful work + benchmarks against the PyTorch reference.

    Until that lands, the int8 .pt artifact above is the shipped
    quantized format; consumers run it via plain PyTorch."""
    raise NotImplementedError(
        "GGUF export for the GhostLM SwiGLU/RoPE/RMSNorm architecture is "
        "not yet implemented. See docstring for the conversion recipe and "
        "use scripts/quantize_v09.py --skip-gguf to produce fp16 + int8 .pt "
        "artifacts only."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True,
                   help="Source bf16/fp32 checkpoint to quantize")
    p.add_argument("--out-dir", default="checkpoints/quantized",
                   help="Where to write the *_fp16.pt and *_int8.pt artifacts")
    p.add_argument("--name", default="ghost-small-v0.9-chat",
                   help="Name prefix for output files")
    p.add_argument("--skip-fp16", action="store_true")
    p.add_argument("--skip-int8", action="store_true")
    p.add_argument("--skip-gguf", action="store_true",
                   help="Skip the (currently NotImplemented) GGUF attempt")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    src = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    print(f"Loading {src}")
    model, cfg, ckpt = load_full_checkpoint(src)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params:,} parameters ({n_params / 1e6:.1f}M)")

    if not args.skip_fp16:
        dst = out_dir / f"{args.name}_fp16.pt"
        print(f"\nfp16 cast -> {dst}")
        save_fp16(model, ckpt, dst)
        size_mb = dst.stat().st_size / 1e6
        print(f"  wrote {size_mb:.1f} MB")

    if not args.skip_int8:
        dst = out_dir / f"{args.name}_int8.pt"
        print(f"\nint8 dynamic quantize -> {dst}")
        save_int8(model, ckpt, dst)
        size_mb = dst.stat().st_size / 1e6
        print(f"  wrote {size_mb:.1f} MB")

    if not args.skip_gguf:
        dst = out_dir / f"{args.name}.gguf"
        try:
            export_gguf_attempt(model, cfg, dst)
            print(f"\nGGUF -> {dst}")
        except NotImplementedError as e:
            print(f"\nGGUF export skipped: {e}")

    print("\nDone. Reload either artifact with the standard GhostLM loader; "
          "for int8 wrap with quantize_dynamic({nn.Linear}, qint8) before "
          "load_state_dict so the QInt8 packed tensors deserialize.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
