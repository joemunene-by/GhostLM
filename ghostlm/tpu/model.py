"""JAX/Flax port of the GhostLM decoder for TPU training.

Covers the dense ghost-base feature set: RoPE (rotate-half), SwiGLU, RMSNorm, GQA,
QK-norm, intra-document masking, attention gate and value residual, tied embeddings,
no dropout. Parameters convert to and from the PyTorch state dict by name
(``to_torch_state`` / ``from_torch_state``), so checkpoints move between the MLX,
PyTorch and JAX trainers. MoE is not ported yet.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict

from ghostlm.config import GhostLMConfig

IGNORE_INDEX = -1


def _normal(std: float):
    return nn.initializers.normal(stddev=std)


class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        weight = self.param("weight", nn.initializers.ones, (self.dim,))
        xf = x.astype(jnp.float32)
        y = xf * jax.lax.rsqrt(jnp.mean(xf * xf, axis=-1, keepdims=True) + self.eps)
        return (y * weight).astype(x.dtype)


def rope_tables(T: int, head_dim: int, base: float):
    inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / head_dim))
    freqs = jnp.outer(jnp.arange(T, dtype=jnp.float32), inv_freq)
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb), jnp.sin(emb)


def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    xf = x.astype(jnp.float32)
    rotated = jnp.concatenate([-xf[..., half:], xf[..., :half]], axis=-1)
    return (xf * cos + rotated * sin).astype(x.dtype)


def intra_doc_mask(idx, eos_token_id: int):
    """Boolean (B, 1, T, T): causal and same-document, EOS kept with the doc it ends."""
    T = idx.shape[1]
    is_eos = (idx == eos_token_id).astype(jnp.int32)
    seg = jnp.cumsum(is_eos, axis=1) - is_eos
    same_doc = seg[:, :, None] == seg[:, None, :]
    causal = jnp.tril(jnp.ones((T, T), dtype=bool))
    return (same_doc & causal)[:, None, :, :]


class Attention(nn.Module):
    config: GhostLMConfig
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, mask, v_first: Optional[jax.Array]):
        cfg = self.config
        H, KV = cfg.n_heads, cfg.n_kv_heads or cfg.n_heads
        hd = cfg.d_model // H
        B, T, C = x.shape
        resid_std = 0.02 / math.sqrt(2 * cfg.n_layers)
        qkv = nn.Dense((H + 2 * KV) * hd, use_bias=cfg.bias, dtype=self.dtype, name="c_qkv",
                       kernel_init=_normal(0.02), bias_init=nn.initializers.zeros)(x)
        q, k, v = jnp.split(qkv, [H * hd, (H + KV) * hd], axis=-1)
        q = q.reshape(B, T, H, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, KV, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, KV, hd).transpose(0, 2, 1, 3)
        if cfg.value_residual:
            v_mix = self.param("v_mix", lambda key: jnp.array(0.5, jnp.float32))
            if v_first is None:
                v_first = v
            else:
                v = ((1 - v_mix) * v.astype(jnp.float32) + v_mix * v_first.astype(jnp.float32)).astype(v.dtype)
        if cfg.use_qk_norm:
            q = RMSNorm(hd, name="q_norm")(q)
            k = RMSNorm(hd, name="k_norm")(k)
        cos, sin = rope_tables(T, hd, cfg.rope_base)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if KV != H:
            # Consecutive query heads share a KV head, as torch's repeat_interleave does.
            k = jnp.repeat(k, H // KV, axis=1)
            v = jnp.repeat(v, H // KV, axis=1)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32)) / math.sqrt(hd)
        scores = jnp.where(mask, scores, jnp.finfo(jnp.float32).min)
        probs = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
        y = jnp.einsum("bhqk,bhkd->bhqd", probs, v)
        if cfg.attn_gate:
            gate = nn.Dense(H, use_bias=True, dtype=self.dtype, name="gate",
                            kernel_init=_normal(0.02), bias_init=nn.initializers.zeros)(x)
            y = y * jax.nn.sigmoid(gate.astype(jnp.float32)).astype(y.dtype).transpose(0, 2, 1)[..., None]
        y = y.transpose(0, 2, 1, 3).reshape(B, T, C)
        y = nn.Dense(C, use_bias=cfg.bias, dtype=self.dtype, name="proj",
                     kernel_init=_normal(resid_std), bias_init=nn.initializers.zeros)(y)
        return y, v_first


class SwiGLU(nn.Module):
    config: GhostLMConfig
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        cfg = self.config
        hidden = (int(cfg.d_ff * 2 / 3) + 63) // 64 * 64
        resid_std = 0.02 / math.sqrt(2 * cfg.n_layers)
        h1 = nn.Dense(hidden, use_bias=False, dtype=self.dtype, name="fc1", kernel_init=_normal(0.02))(x)
        h2 = nn.Dense(hidden, use_bias=False, dtype=self.dtype, name="fc2", kernel_init=_normal(0.02))(x)
        return nn.Dense(cfg.d_model, use_bias=False, dtype=self.dtype, name="fc3",
                        kernel_init=_normal(resid_std))(jax.nn.silu(h1) * h2)


class Block(nn.Module):
    config: GhostLMConfig
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, mask, v_first):
        a, v_first = Attention(self.config, self.dtype, name="attn")(RMSNorm(self.config.d_model, name="ln_1")(x), mask, v_first)
        x = x + a
        return x + SwiGLU(self.config, self.dtype, name="ffn")(RMSNorm(self.config.d_model, name="ln_2")(x)), v_first


class GhostLMFlax(nn.Module):
    """Logits for a (B, T) batch of token ids. ``dtype`` is the compute dtype; params stay fp32."""

    config: GhostLMConfig
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, idx):
        cfg = self.config
        if cfg.use_moe:
            raise NotImplementedError("the JAX port does not cover MoE yet")
        embed = nn.Embed(cfg.vocab_size, cfg.d_model, dtype=self.dtype, name="token_embedding",
                         embedding_init=_normal(0.02))
        x = embed(idx)
        if cfg.intra_doc_mask and cfg.eos_token_id is not None:
            mask = intra_doc_mask(idx, cfg.eos_token_id)
        else:
            T = idx.shape[1]
            mask = jnp.tril(jnp.ones((T, T), dtype=bool))[None, None]
        block_cls = nn.remat(Block, static_argnums=()) if cfg.gradient_checkpointing else Block
        v_first = None
        for i in range(cfg.n_layers):
            x, v_first = block_cls(cfg, self.dtype, name=f"blocks_{i}")(x, mask, v_first)
        x = RMSNorm(cfg.d_model, name="ln_f")(x)
        return embed.attend(x.astype(embed.embedding.dtype))


def cross_entropy(logits, targets):
    """Mean token loss in fp32, ignoring IGNORE_INDEX targets."""
    logits = logits.astype(jnp.float32)
    valid = targets != IGNORE_INDEX
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    picked = jnp.take_along_axis(log_probs, jnp.where(valid, targets, 0)[..., None], axis=-1)[..., 0]
    return -(picked * valid).sum() / jnp.maximum(valid.sum(), 1)


def init_params(config: GhostLMConfig, seed: int = 0) -> dict:
    model = GhostLMFlax(config)
    idx = jnp.zeros((1, min(8, config.context_length)), dtype=jnp.int32)
    return model.init(jax.random.PRNGKey(seed), idx)["params"]


def param_count(params: dict) -> int:
    return sum(int(np.prod(v.shape)) for v in jax.tree_util.tree_leaves(params))


def _torch_name(path: tuple) -> str:
    """Flax param path to the torch state-dict key (``blocks_3/attn/c_qkv/kernel`` -> ``blocks.3.attn.c_qkv.weight``)."""
    parts = list(path)
    if parts[0].startswith("blocks_"):
        parts[0] = "blocks." + parts[0][len("blocks_"):]
    leaf = parts[-1]
    parts[-1] = {"kernel": "weight", "embedding": "weight"}.get(leaf, leaf)
    return ".".join(parts)


def to_torch_state(params: dict) -> dict:
    """Flax params as a torch-style state dict of numpy fp32 arrays, including the tied lm_head."""
    state = {}
    for path, value in flatten_dict(params).items():
        arr = np.asarray(value, dtype=np.float32)
        if path[-1] == "kernel":
            arr = arr.T
        state[_torch_name(path)] = arr
    state["lm_head.weight"] = state["token_embedding.weight"]
    return state


def from_torch_state(params: dict, state: dict) -> dict:
    """A copy of ``params`` with values loaded from a torch state dict (tensors or arrays) by name."""
    flat = flatten_dict(params)
    out = {}
    for path, value in flat.items():
        name = _torch_name(path)
        if name not in state:
            raise KeyError(f"missing parameter in state dict: {name}")
        src = state[name]
        arr = src.detach().float().cpu().numpy() if hasattr(src, "detach") else np.asarray(src, dtype=np.float32)
        if path[-1] == "kernel":
            arr = arr.T
        if arr.shape != tuple(value.shape):
            raise ValueError(f"{name}: expected shape {tuple(value.shape)}, got {arr.shape}")
        out[path] = jnp.asarray(arr, dtype=value.dtype)
    return unflatten_dict(out)
