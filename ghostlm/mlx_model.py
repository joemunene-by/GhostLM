"""MLX port of the GhostLM decoder for fast training on Apple Silicon.

Covers the ghost-base feature set: RoPE (rotate-half), SwiGLU, RMSNorm, GQA,
QK-norm, intra-document masking, attention gate and value residual, with
tied embeddings and no dropout. Parameter names mirror ``ghostlm.model`` so
weights convert between frameworks by name (``to_torch_state`` /
``from_torch_state``). On a 16GB M4 this trains the 349M shape well over
twice as fast as PyTorch MPS.
"""

from __future__ import annotations

import math
import re
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ghostlm.config import GhostLMConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)


class Attention(nn.Module):
    def __init__(self, config: GhostLMConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads or config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.rope_base = getattr(config, "rope_base", 10000.0)
        qkv_dim = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
        self.c_qkv = nn.Linear(config.d_model, qkv_dim, bias=config.bias)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        self.use_attn_gate = getattr(config, "attn_gate", False)
        if self.use_attn_gate:
            self.gate = nn.Linear(config.d_model, self.n_heads, bias=True)
        self.value_residual = getattr(config, "value_residual", False)
        if self.value_residual:
            self.v_mix = mx.array(0.5)

    def __call__(self, x, mask, v_first=None):
        B, T, C = x.shape
        hd = self.head_dim
        q, k, v = mx.split(self.c_qkv(x), [self.n_heads * hd, (self.n_heads + self.n_kv_heads) * hd], axis=-1)
        q = q.reshape(B, T, self.n_heads, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.n_kv_heads, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_kv_heads, hd).transpose(0, 2, 1, 3)
        if self.value_residual:
            if v_first is None:
                v_first = v
            else:
                v = (1 - self.v_mix) * v + self.v_mix * v_first
        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        q = mx.fast.rope(q, hd, traditional=False, base=self.rope_base, scale=1.0, offset=0)
        k = mx.fast.rope(k, hd, traditional=False, base=self.rope_base, scale=1.0, offset=0)
        # MLX attention groups consecutive query heads per KV head, matching
        # torch's repeat_interleave in ghostlm.model._repeat_kv.
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(hd), mask=mask)
        if self.use_attn_gate:
            y = y * mx.sigmoid(self.gate(x)).transpose(0, 2, 1)[..., None]
        y = self.proj(y.transpose(0, 2, 1, 3).reshape(B, T, C))
        return y, v_first


class SwiGLU(nn.Module):
    def __init__(self, config: GhostLMConfig):
        super().__init__()
        hidden = (int(config.d_ff * 2 / 3) + 63) // 64 * 64
        self.fc1 = nn.Linear(config.d_model, hidden, bias=False)
        self.fc2 = nn.Linear(config.d_model, hidden, bias=False)
        self.fc3 = nn.Linear(hidden, config.d_model, bias=False)

    def __call__(self, x):
        return self.fc3(nn.silu(self.fc1(x)) * self.fc2(x))


class MoE(nn.Module):
    """Top-k routed SwiGLU experts, numerically matching ghostlm.model.SparseMoE.

    Expert weights are stacked (n_experts, out, in) so ``mx.gather_mm`` runs each
    token through only its chosen experts. ``to_torch_state``/``from_torch_state``
    map them to torch's per-expert ``experts.{i}.fc*.weight`` names.
    """

    def __init__(self, config: GhostLMConfig):
        super().__init__()
        self.n_experts = int(config.n_experts)
        self.top_k = int(config.n_experts_active)
        hidden = (int(config.d_ff * 2 / 3) + 63) // 64 * 64
        d, e = config.d_model, self.n_experts
        self.gate = nn.Linear(d, e, bias=False)
        self.fc1 = mx.random.normal((e, hidden, d)) * 0.02
        self.fc2 = mx.random.normal((e, hidden, d)) * 0.02
        self.fc3 = mx.random.normal((e, d, hidden)) * 0.02

    def __call__(self, x):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        logits = self.gate(flat).astype(mx.float32)
        idx = mx.stop_gradient(mx.argpartition(-logits, kth=self.top_k - 1, axis=-1)[:, :self.top_k])
        weights = mx.softmax(mx.take_along_axis(logits, idx, axis=-1), axis=-1)

        # Switch-style load balancing, as in SparseMoE: routed fraction x mean router probability.
        routed = (idx[:, :, None] == mx.arange(self.n_experts)).sum(axis=1).astype(mx.float32)
        aux = (routed.mean(axis=0) * mx.softmax(logits, axis=-1).mean(axis=0)).sum() * self.n_experts

        xe = mx.expand_dims(flat, (-2, -3))
        h = (nn.silu(mx.gather_mm(xe, self.fc1.swapaxes(-1, -2), rhs_indices=idx))
             * mx.gather_mm(xe, self.fc2.swapaxes(-1, -2), rhs_indices=idx))
        out = mx.gather_mm(h, self.fc3.swapaxes(-1, -2), rhs_indices=idx).squeeze(-2)
        out = (out * weights[..., None].astype(out.dtype)).sum(axis=1)
        return out.reshape(shape), aux


class Block(nn.Module):
    def __init__(self, config: GhostLMConfig):
        super().__init__()
        self.ln_1 = RMSNorm(config.d_model)
        self.attn = Attention(config)
        self.ln_2 = RMSNorm(config.d_model)
        self.ffn = MoE(config) if getattr(config, "use_moe", False) else SwiGLU(config)

    def __call__(self, x, mask, v_first=None):
        a, v_first = self.attn(self.ln_1(x), mask, v_first)
        x = x + a
        out = self.ffn(self.ln_2(x))
        if isinstance(out, tuple):
            out, aux = out
        else:
            aux = mx.array(0.0)
        return x + out, v_first, aux


def intra_doc_mask(idx: mx.array, eos_token_id: int) -> mx.array:
    """Boolean (B, 1, T, T) mask: causal and same-document, EOS kept with the doc it ends."""
    T = idx.shape[1]
    is_eos = (idx == eos_token_id).astype(mx.int32)
    seg = mx.cumsum(is_eos, axis=1) - is_eos
    same_doc = seg[:, :, None] == seg[:, None, :]
    causal = mx.tril(mx.ones((T, T), dtype=mx.bool_))
    return (same_doc & causal)[:, None, :, :]


class GhostLMMLX(nn.Module):
    def __init__(self, config: GhostLMConfig, gradient_checkpointing: bool = False):
        super().__init__()
        unsupported = [name for name, ok in [
            ("use_rope", config.use_rope), ("use_swiglu", config.use_swiglu),
            ("use_rmsnorm", config.use_rmsnorm), ("dropout == 0", config.dropout == 0),
        ] if not ok]
        if unsupported:
            raise ValueError(f"MLX port requires: {', '.join(unsupported)}")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = [Block(config) for _ in range(config.n_layers)]
        self.ln_f = RMSNorm(config.d_model)
        self.gradient_checkpointing = gradient_checkpointing
        # nn.utils.checkpoint (not mx.checkpoint) treats the block's own
        # parameters as inputs; mx.checkpoint silently drops their gradients.
        self._ckpt_calls = [nn.utils.checkpoint(b) for b in self.blocks] if gradient_checkpointing else None

    def __call__(self, idx: mx.array) -> mx.array:
        return self.forward(idx)[0]

    def forward(self, idx: mx.array):
        """Logits plus the summed MoE load-balancing loss (0 for dense models)."""
        cfg = self.config
        if getattr(cfg, "intra_doc_mask", False) and cfg.eos_token_id is not None:
            mask = intra_doc_mask(idx, cfg.eos_token_id)
        else:
            mask = "causal"
        x = self.token_embedding(idx)
        v_first, aux = None, mx.array(0.0)
        for i, block in enumerate(self.blocks):
            call = self._ckpt_calls[i] if self._ckpt_calls else block
            x, v_first, layer_aux = call(x, mask, v_first)
            aux = aux + layer_aux
        return self.token_embedding.as_linear(self.ln_f(x)), aux

    def loss(self, idx: mx.array, targets: mx.array) -> mx.array:
        logits, aux = self.forward(idx)
        logits = logits.astype(mx.float32)
        valid = targets != -1
        ce = nn.losses.cross_entropy(logits, mx.where(valid, targets, 0), reduction="none")
        loss = (ce * valid).sum() / mx.maximum(valid.sum(), 1)
        if getattr(self.config, "use_moe", False):
            loss = loss + self.config.moe_aux_loss_coef * aux
        return loss


def _flatten(tree, prefix=""):
    out = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            out.update(_flatten(v, f"{prefix}{k}."))
    elif isinstance(tree, list):
        for i, v in enumerate(tree):
            out.update(_flatten(v, f"{prefix}{i}."))
    else:
        out[prefix[:-1]] = tree
    return out


_STACKED = re.compile(r"^(blocks\.\d+\.ffn)\.(fc[123])$")
_PER_EXPERT = re.compile(r"^(blocks\.\d+\.ffn)\.experts\.(\d+)\.(fc[123])\.weight$")


def to_torch_state(model: GhostLMMLX) -> dict:
    """MLX parameters as a torch-style state dict (numpy float32), incl. the tied lm_head."""
    state = {}
    for k, v in _flatten(model.parameters()).items():
        arr = np.array(v.astype(mx.float32))
        m = _STACKED.match(k)
        if m:
            for e in range(arr.shape[0]):
                state[f"{m.group(1)}.experts.{e}.{m.group(2)}.weight"] = arr[e]
        else:
            state[k] = arr
    state["lm_head.weight"] = state["token_embedding.weight"]
    return state


def from_torch_state(model: GhostLMMLX, state: dict, dtype=mx.float32) -> None:
    """Load a torch state dict (tensors or arrays) into the MLX model by name."""
    expected = set(_flatten(model.parameters()))
    weights, stacks = [], {}
    for name, value in state.items():
        arr = value.detach().float().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        m = _PER_EXPERT.match(name)
        if m:
            stacks.setdefault(f"{m.group(1)}.{m.group(3)}", {})[int(m.group(2))] = arr
            continue
        if name == "lm_head.weight" or name not in expected:
            continue
        weights.append((name, mx.array(arr, dtype=dtype)))
    for name, experts in stacks.items():
        if name in expected:
            weights.append((name, mx.array(np.stack([experts[e] for e in sorted(experts)]), dtype=dtype)))
    missing = expected - {n for n, _ in weights}
    if missing:
        raise ValueError(f"missing parameters: {sorted(missing)[:5]}")
    model.load_weights(weights)


def config_from_torch(cfg: GhostLMConfig, dropout: Optional[float] = 0.0) -> GhostLMConfig:
    """Copy of ``cfg`` usable by the MLX model (dropout forced to 0 by default)."""
    from dataclasses import replace
    return replace(cfg, dropout=dropout if dropout is not None else cfg.dropout)
