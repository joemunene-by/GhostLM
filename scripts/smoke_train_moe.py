#!/usr/bin/env python3
"""MoE training smoke test: 100 steps of ghost-tiny+MoE on synthetic
random-token data. Validates that bet 5's aux-loss wiring actually
trains, not just compiles.

Why this exists: the aux-loss wiring landed in commit b1015b3 with
single-forward-pass smoke tests (does loss = CE + coef*aux, do gates
get gradients). That doesn't tell us whether 100 backward-and-step
iterations stay stable: whether the router learns to spread expert
load, whether expert weights drift, whether the aux-loss term
converges or diverges. This script answers those questions on M4
in about a minute, with no real training data needed.

The point isn't to learn anything language-y. Random tokens forced
into a ghost-tiny shape will memorise toward zero loss eventually;
what matters here is the SHAPE of the curves:

  - cross-entropy: should monotonically decrease (model fits the
    random target).
  - per-layer aux loss: should drift down or stabilise near
    equilibrium (uniform routing across experts gives aux ~= 1.0
    in this scaled formulation; collapse to one expert gives aux
    closer to n_experts = 4). Initial aux ~= 2.0 with cold gates;
    if it stays flat or climbs, the router isn't learning.
  - gradient norms on the gate: should be non-zero throughout.

The output is logs/moe_smoke_<timestamp>.md with the full curve, a
pass/fail summary, and a copy of the per-step numeric log so future
runs can be diffed against this one.

Run:

    PYTHONPATH=. python3 scripts/smoke_train_moe.py
    PYTHONPATH=. python3 scripts/smoke_train_moe.py --steps 200 --device cpu

Cost: ~1-2 minutes on M4 MPS (default 100 steps), 30s on CUDA.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ghostlm.config import GhostLMConfig  # noqa: E402
from ghostlm.model import GhostLM, SparseMoE  # noqa: E402


def build_tiny_moe_config() -> GhostLMConfig:
    """Smallest config that exercises the MoE path without burning RAM."""
    return GhostLMConfig(
        vocab_size=256, context_length=32, d_model=64, n_heads=4,
        n_layers=4, d_ff=128,
        use_rope=True, use_swiglu=True, use_rmsnorm=True,
        use_moe=True, n_experts=4, n_experts_active=2,
        moe_aux_loss_coef=0.01,
        learning_rate=1e-3, weight_decay=0.0,
        warmup_steps=10, max_steps=10_000,
    )


def resolve_device(arg: str) -> str:
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    cfg = build_tiny_moe_config()
    model = GhostLM(cfg).to(device)
    model.train()
    optimizer = model.configure_optimizers(cfg)

    moe_layers = [m for m in model.modules() if isinstance(m, SparseMoE)]
    if not moe_layers:
        sys.exit("MoE layers not found; check use_moe wiring")
    print(f"device={device}, MoE layers={len(moe_layers)}, "
          f"experts={cfg.n_experts}, top-K={cfg.n_experts_active}")

    # Fixed synthetic dataset: same random sequences every batch so
    # the model can fit if and only if training is healthy.
    rng = torch.Generator().manual_seed(args.seed)
    fixed = torch.randint(0, cfg.vocab_size, (args.batch, cfg.context_length + 1),
                          generator=rng, device=device)
    # ``.contiguous()`` so the flat-view in cross_entropy doesn't hit the
    # non-contiguous-slice path inside model.forward.
    inputs = fixed[:, :-1].contiguous()
    targets = fixed[:, 1:].contiguous()

    log = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        _, total_loss = model(inputs, targets=targets)
        total_loss.backward()

        # Capture aux losses + gate-grad norms BEFORE the optimizer step
        # so the numbers reflect what the optimizer is about to act on.
        aux_per_layer = [m.last_aux_loss.item() for m in moe_layers]
        gate_grad_norms = [
            m.gate.weight.grad.norm().item() if m.gate.weight.grad is not None else 0.0
            for m in moe_layers
        ]

        # CE component = total - coef * sum(aux). Recover it for the curve.
        ce = total_loss.item() - cfg.moe_aux_loss_coef * sum(aux_per_layer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        log.append({
            "step": step,
            "total": total_loss.item(),
            "ce": ce,
            "aux_mean": sum(aux_per_layer) / len(aux_per_layer),
            "aux_per_layer": aux_per_layer,
            "gate_grad_norm_mean": sum(gate_grad_norms) / len(gate_grad_norms),
        })
        if step == 1 or step % max(1, args.steps // 10) == 0 or step == args.steps:
            print(f"  step {step:4d} | total={total_loss.item():.4f} "
                  f"| ce={ce:.4f} | aux_mean={log[-1]['aux_mean']:.3f} "
                  f"| gate_grad={log[-1]['gate_grad_norm_mean']:.3e}")

    elapsed = time.time() - t0
    print(f"\ndone in {elapsed:.1f}s")

    # Pass/fail evaluation.
    first = log[0]
    last = log[-1]
    ce_dropped = last["ce"] < first["ce"] - 0.1
    aux_finite = all(0.0 < r["aux_mean"] < 100.0 for r in log)
    grads_nonzero = all(r["gate_grad_norm_mean"] > 0.0 for r in log)
    aux_at_end_lower = last["aux_mean"] <= first["aux_mean"] + 0.5

    print("\nVerdict:")
    print(f"  CE decreased ({first['ce']:.3f} -> {last['ce']:.3f}): "
          f"{'PASS' if ce_dropped else 'FAIL'}")
    print(f"  aux finite all steps: {'PASS' if aux_finite else 'FAIL'}")
    print(f"  gate gradients non-zero all steps: "
          f"{'PASS' if grads_nonzero else 'FAIL'}")
    print(f"  aux mean did not blow up "
          f"({first['aux_mean']:.3f} -> {last['aux_mean']:.3f}): "
          f"{'PASS' if aux_at_end_lower else 'FAIL'}")

    overall = ce_dropped and aux_finite and grads_nonzero and aux_at_end_lower

    # Persist a markdown report next to the rest of the bet 5 docs.
    report_dir = REPO_ROOT / "logs"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"moe_smoke_{int(time.time())}.md"
    lines = [
        f"# MoE training smoke ({args.steps} steps)",
        "",
        f"- Device: `{device}`",
        f"- Seed: `{args.seed}`",
        f"- Config: ghost-tiny shape (2-layer would be enough, this is "
        f"4-layer for stability), MoE 4 experts top-{cfg.n_experts_active}, "
        f"coef={cfg.moe_aux_loss_coef}",
        f"- Wall clock: {elapsed:.1f}s",
        "",
        "| step | total | ce | aux_mean | gate_grad_mean |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in log[::max(1, args.steps // 20)]:
        lines.append(
            f"| {r['step']} | {r['total']:.4f} | {r['ce']:.4f} | "
            f"{r['aux_mean']:.3f} | {r['gate_grad_norm_mean']:.3e} |"
        )
    lines.append(
        f"| {last['step']} | {last['total']:.4f} | {last['ce']:.4f} | "
        f"{last['aux_mean']:.3f} | {last['gate_grad_norm_mean']:.3e} |"
    )
    lines.extend([
        "",
        "## Pass/fail",
        "",
        f"- CE decreased: **{'PASS' if ce_dropped else 'FAIL'}** "
        f"({first['ce']:.3f} -> {last['ce']:.3f})",
        f"- aux finite all steps: **{'PASS' if aux_finite else 'FAIL'}**",
        f"- gate gradients non-zero all steps: "
        f"**{'PASS' if grads_nonzero else 'FAIL'}**",
        f"- aux did not blow up: **{'PASS' if aux_at_end_lower else 'FAIL'}**",
        "",
        f"**Overall: {'PASS' if overall else 'FAIL'}**",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport: {report_path}")

    # Also dump the raw per-step JSON so curves can be plotted later.
    json_path = report_path.with_suffix(".json")
    json_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"Per-step JSON: {json_path}")

    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
