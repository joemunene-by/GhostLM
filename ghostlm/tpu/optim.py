"""Learning-rate schedule and optimizers for the JAX trainer.

Mirrors ``scripts/train_ghost_base_mlx.py``: warmup then cosine or WSD, AdamW with
bias correction, and Muon / NorMuon on the 2D Linear weights with AdamW elsewhere.
Weight decay is decoupled and, with ``cautious_wd``, only applied where the update
agrees in sign with the weight. Flax kernels are stored (in, out), the transpose of
torch's (out, in); Muon's per-neuron maths runs on the torch orientation so the
numbers match.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict

from ghostlm.config import GhostLMConfig

MIN_LR = 1e-5
Path = Tuple[str, ...]


def lr_at(step: int, cfg: GhostLMConfig) -> float:
    base, warmup, max_steps = cfg.learning_rate, cfg.warmup_steps, cfg.max_steps
    if step < warmup:
        return base * (step + 1) / warmup
    if cfg.lr_schedule == "wsd":
        decay_start = int(max_steps * (1 - cfg.wsd_decay_frac))
        if step < decay_start:
            return base
        frac = min(1.0, (step - decay_start) / max(1, max_steps - decay_start))
        return base + (MIN_LR - base) * frac
    ratio = min((step - warmup) / max(1, max_steps - warmup), 1.0)
    return MIN_LR + (base - MIN_LR) * 0.5 * (1.0 + math.cos(math.pi * ratio))


def decay_mask(params: dict) -> Dict[Path, bool]:
    """Linear kernels are decayed; embeddings, norms, biases and scalars are not."""
    return {path: path[-1] == "kernel" for path in flatten_dict(params)}


def newton_schulz5(g: jax.Array, steps: int = 5) -> jax.Array:
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.astype(jnp.float32)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.swapaxes(-1, -2)
    x = x / (jnp.sqrt((x * x).sum(axis=(-2, -1), keepdims=True)) + 1e-7)
    for _ in range(steps):
        A = x @ x.swapaxes(-1, -2)
        B = b * A + c * (A @ A)
        x = a * x + B @ x
    return x.swapaxes(-1, -2) if transposed else x


class Optimizer:
    """Pure-function optimizer: ``init(params)`` makes the state, ``update`` returns new params and state."""

    def __init__(self, cfg: GhostLMConfig, params: dict):
        self.cfg = cfg
        self.decay = decay_mask(params)
        use_muon = cfg.optimizer in ("muon", "normuon")
        self.muon = {path for path, d in self.decay.items() if d and use_muon} if use_muon else set()
        self.normuon = cfg.optimizer == "normuon"

    def init(self, params: dict) -> dict:
        state = {}
        for path, p in flatten_dict(params).items():
            if path in self.muon:
                st = {"momentum_buffer": jnp.zeros_like(p, dtype=jnp.float32)}
                if self.normuon:
                    # torch orientation (out, in): one running mean per output neuron
                    st["row_sq"] = jnp.zeros((p.shape[1], 1), jnp.float32)
                state[path] = st
            else:
                state[path] = {"exp_avg": jnp.zeros_like(p, dtype=jnp.float32),
                               "exp_avg_sq": jnp.zeros_like(p, dtype=jnp.float32)}
        state[("step",)] = jnp.zeros((), jnp.int32)
        return state

    def _decay(self, p, update, amount):
        if self.cfg.cautious_wd:
            return p - p * ((update * p) > 0) * amount
        return p * (1 - amount)

    def update(self, params: dict, grads: dict, state: dict, lr) -> Tuple[dict, dict]:
        cfg = self.cfg
        b1, b2, eps = cfg.beta1, cfg.beta2, 1e-8
        step = state[("step",)] + 1
        flat_p, flat_g = flatten_dict(params), flatten_dict(grads)
        new_p, new_s = {}, {("step",): step}
        for path, p in flat_p.items():
            g = flat_g[path].astype(jnp.float32)
            st = state[path]
            wd = lr * cfg.weight_decay if self.decay[path] else 0.0
            if path in self.muon:
                gt = g.T
                buf = st["momentum_buffer"].T * cfg.muon_momentum + gt
                update = newton_schulz5(gt + cfg.muon_momentum * buf)
                rows, cols = update.shape
                out_state = {"momentum_buffer": buf.T}
                if self.normuon:
                    row = st["row_sq"]
                    row = row + (1 - b2) * (jnp.mean(update * update, axis=-1, keepdims=True) - row)
                    out_state["row_sq"] = row
                    update = update / (jnp.sqrt(row) + eps)
                    norm = jnp.sqrt((update * update).sum())
                    update = update * (0.2 * (rows * cols) ** 0.5 / (norm + eps))
                else:
                    update = update * (0.2 * max(rows, cols) ** 0.5)
                update = update.T
            else:
                m = st["exp_avg"] + (1 - b1) * (g - st["exp_avg"])
                v = st["exp_avg_sq"] * b2 + (1 - b2) * g * g
                out_state = {"exp_avg": m, "exp_avg_sq": v}
                update = (m / (1 - b1 ** step)) / (jnp.sqrt(v / (1 - b2 ** step)) + eps)
            new_p[path] = (self._decay(p, update, wd) - lr * update).astype(p.dtype)
            new_s[path] = out_state
        return unflatten_dict(new_p), new_s


def clip_by_global_norm(grads: dict, max_norm: float) -> Tuple[dict, jax.Array]:
    leaves = jax.tree_util.tree_leaves(grads)
    norm = jnp.sqrt(sum(jnp.sum(jnp.square(g.astype(jnp.float32))) for g in leaves))
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-6))
    return jax.tree_util.tree_map(lambda g: g * scale, grads), norm
