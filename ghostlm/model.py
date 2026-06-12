"""GhostLM transformer model — decoder-only architecture built from scratch in PyTorch."""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from ghostlm.config import GhostLMConfig

# One transformer layer's cached keys and values, each of shape
# (B, n_kv_heads, T_past, head_dim) — n_kv_heads == n_heads unless
# grouped-query attention is configured. A full-model cache is a list
# with one entry per block, in block order.
LayerKVCache = Tuple[torch.Tensor, torch.Tensor]


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE).

    Encodes relative position information directly into the attention
    computation by rotating query and key vectors. Used by LLaMA, Mistral,
    and most modern transformer architectures.
    """

    def __init__(self, head_dim: int, context_length: int, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin for all positions
        t = torch.arange(context_length).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, offset: int = 0):
        """Return cos/sin tables for positions [offset, offset + seq_len).

        ``offset`` is the number of tokens already in the KV cache, so
        incremental decoding rotates new queries/keys with their true
        absolute positions.
        """
        return (
            self.cos_cached[offset : offset + seq_len],
            self.sin_cached[offset : offset + seq_len],
        )


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand grouped KV heads to match the query head count.

    (B, n_kv_heads, T, head_dim) -> (B, n_kv_heads * n_rep, T, head_dim),
    a no-op when n_rep == 1. The KV cache stays at n_kv_heads (that is
    the memory win of GQA); the expansion only exists inside the
    attention computation.
    """
    if n_rep == 1:
        return x
    return x.repeat_interleave(n_rep, dim=1)


def _rotate_half(x):
    """Rotate the second half of the last dimension and negate it."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to query and key tensors.

    Args:
        q: Query tensor of shape (B, n_heads, T, head_dim).
        k: Key tensor of shape (B, n_heads, T, head_dim).
        cos: Cosine frequencies of shape (T, head_dim).
        sin: Sine frequencies of shape (T, head_dim).

    Returns:
        Tuple of (rotated_q, rotated_k).
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (LLaMA-style, no mean subtraction).

    Used by Llama-2 / Llama-3 / Mistral / Gemma — half the params of LayerNorm
    and matches its quality at this scale per the 2024 - 2026 small-LM
    literature. Toggled via ``GhostLMConfig.use_rmsnorm``.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        """Initialize a learned scale vector of shape (dim,)."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        """Normalize by RMS along the last dim, then scale."""
        norm = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight).to(x.dtype)


def make_norm(config: GhostLMConfig, dim: int) -> nn.Module:
    """Return RMSNorm or LayerNorm based on ``config.use_rmsnorm``."""
    if getattr(config, "use_rmsnorm", False):
        return RMSNorm(dim)
    return nn.LayerNorm(dim)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with autoregressive masking.

    Uses a single combined QKV projection for efficiency, then splits
    the result into separate query, key, and value tensors. Supports
    optional RoPE (Rotary Position Embeddings), Flash Attention,
    grouped-query attention (``n_kv_heads``), and QK-norm
    (``use_qk_norm``).
    """

    def __init__(self, config: GhostLMConfig):
        """Initialize causal self-attention.

        Args:
            config: GhostLMConfig containing d_model, n_heads, dropout,
                    context_length, bias, use_rope, and use_flash_attention.
        """
        super().__init__()
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        # Grouped-query attention: fewer KV heads than query heads, each
        # shared by n_rep query heads. n_kv_heads == n_heads (None in the
        # config) is plain MHA and keeps the historical checkpoint shapes.
        self.n_kv_heads = config.n_kv_heads if getattr(config, "n_kv_heads", None) else config.n_heads
        assert config.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({config.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.context_length = config.context_length
        self.use_rope = config.use_rope
        self.use_flash_attention = config.use_flash_attention
        self.dropout_p = config.dropout

        # Single combined QKV projection. Q keeps the full head count;
        # K and V are projected at n_kv_heads each.
        qkv_dim = (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
        self.c_qkv = nn.Linear(config.d_model, qkv_dim, bias=config.bias)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        # QK-norm: per-head RMSNorm on queries and keys before RoPE,
        # guarding against attention-logit growth on long pretrains.
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # Dropout applied to attention weights (manual path only)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # RoPE
        if self.use_rope:
            self.rope = RotaryEmbedding(
                self.head_dim, config.context_length,
                base=getattr(config, "rope_base", 10000.0),
            )

        # Causal mask buffer (only needed for manual attention path)
        if not self.use_flash_attention:
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(config.context_length, config.context_length))
                .view(1, 1, config.context_length, config.context_length),
                persistent=False,
            )

    def _attention_bias(
        self,
        B: int,
        T: int,
        past_len: int,
        attn_mask: Optional[torch.Tensor],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Build an additive attention bias combining the (offset) causal
        mask with an optional padding mask.

        Query positions are [past_len, past_len + T); key positions are
        [0, past_len + T). ``attn_mask`` is (B, past_len + T) with 1 for
        real tokens and 0 for padding. The diagonal is always kept
        unmasked so fully-padded query rows (left padding) self-attend
        instead of softmaxing over an all--inf row and producing NaNs.
        """
        total_len = past_len + T
        # keep[i, j] == True when query past_len + i may attend to key j.
        keep = torch.ones(T, total_len, dtype=torch.bool, device=device)
        keep = keep.tril(diagonal=past_len)
        keep = keep.unsqueeze(0).unsqueeze(0)  # (1, 1, T, total_len)
        if attn_mask is not None:
            pad_keep = attn_mask.to(torch.bool).view(B, 1, 1, total_len)
            keep = keep & pad_keep
            # Re-open the diagonal (query attends to itself) to avoid
            # fully-masked rows; padded positions produce garbage that
            # downstream losses/readers must ignore anyway.
            diag = torch.zeros(T, total_len, dtype=torch.bool, device=device)
            idx = torch.arange(T, device=device)
            diag[idx, idx + past_len] = True
            keep = keep | diag.view(1, 1, T, total_len)
        bias = torch.zeros(keep.shape, dtype=dtype, device=device)
        bias.masked_fill_(~keep, float("-inf"))
        return bias

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[LayerKVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[LayerKVCache]]:
        """Forward pass through causal self-attention.

        Args:
            x: Input tensor of shape (B, T, d_model).
            attn_mask: Optional (B, past_len + T) padding mask; 1 for real
                tokens, 0 for padding.
            past_kv: Optional (k, v) tensors from previous steps, each of
                shape (B, n_kv_heads, past_len, head_dim).
            use_cache: If True, also return the updated (k, v) cache.

        Returns:
            Tuple of (output of shape (B, T, d_model), updated cache or None).
        """
        B, T, C = x.size()
        past_len = past_kv[0].size(2) if past_kv is not None else 0

        # Combined QKV projection and split (K/V at n_kv_heads)
        qkv = self.c_qkv(x)
        kv_dim = self.n_kv_heads * self.head_dim
        q, k, v = qkv.split([self.n_heads * self.head_dim, kv_dim, kv_dim], dim=-1)

        # Reshape to (B, heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # QK-norm before RoPE, per head.
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply RoPE to Q and K (not V) at their absolute positions.
        # Cached keys were already rotated when first computed.
        if self.use_rope:
            cos, sin = self.rope(T, offset=past_len)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Prepend cached keys/values from previous decode steps. The
        # cache holds compact n_kv_heads tensors; expansion to the full
        # query head count happens after, per attention call.
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        present: Optional[LayerKVCache] = (k, v) if use_cache else None

        k = _repeat_kv(k, self.n_rep)
        v = _repeat_kv(v, self.n_rep)

        if self.use_flash_attention:
            if attn_mask is None and past_len == 0:
                # Fast path: plain causal attention, backend-selected.
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=True,
                )
            else:
                bias = self._attention_bias(
                    B, T, past_len, attn_mask, q.dtype, q.device,
                )
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=bias,
                    dropout_p=self.dropout_p if self.training else 0.0,
                    is_causal=False,
                )
        else:
            # Manual attention path
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale
            if attn_mask is None and past_len == 0:
                causal = self.causal_mask[:, :, :T, :T]
                att = att.masked_fill(causal == 0, float("-inf"))
            else:
                att = att + self._attention_bias(
                    B, T, past_len, attn_mask, att.dtype, x.device,
                )
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # Reassemble heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))

        return y, present


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation.

    Two linear layers with an intermediate GELU non-linearity:
    d_model -> d_ff -> d_model, with dropout after the second layer.
    """

    def __init__(self, config: GhostLMConfig):
        """Initialize the feed-forward network.

        Args:
            config: GhostLMConfig containing d_model, d_ff, dropout, and bias.
        """
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """Forward pass through the feed-forward network.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class SwiGLU(nn.Module):
    """SwiGLU feed-forward — Llama / Mistral / Gemma style gated FFN.

    Two parallel projections from d_model to a 2/3 d_ff hidden, gated through
    SiLU. Matches GELU's parameter budget (we shrink the hidden dim by 2/3 to
    compensate for the extra projection) but reliably wins by 1-2 nat-loss in
    sub-1B comparisons. Toggled via ``GhostLMConfig.use_swiglu``.
    """

    def __init__(self, config: GhostLMConfig):
        """Initialize the gated FFN with three linear projections (no bias)."""
        super().__init__()
        # Shrink hidden dim to keep total parameter count comparable to the
        # GELU FeedForward at the same d_ff (which has 2 projections vs our 3).
        hidden = int(config.d_ff * 2 / 3)
        # Round to a multiple of 64 so MPS / CUDA matmul shapes stay friendly.
        hidden = (hidden + 63) // 64 * 64
        self.fc1 = nn.Linear(config.d_model, hidden, bias=False)
        self.fc2 = nn.Linear(config.d_model, hidden, bias=False)
        self.fc3 = nn.Linear(hidden, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        """fc3(SiLU(fc1(x)) * fc2(x))."""
        return self.dropout(self.fc3(F.silu(self.fc1(x)) * self.fc2(x)))


class SparseMoE(nn.Module):
    """Sparse Mixture-of-Experts feed-forward layer.

    Differentiator for ghost-1B+: dense FFN at 1B params is the
    obvious move, but going MoE (4 experts, 2 active per token, with
    a learned router) gives effectively 2B parameters of capacity at
    the inference cost of ~1B dense. From-scratch MoE at this scale
    is rare; most cybersec LMs that ever reach 1B graft MoE in late
    or skip it entirely.

    Architecture (matches the standard Mixtral / DeepSeek-MoE shape):
      - Router: linear projection from d_model to n_experts
      - Top-K gating: softmax over top_k logits, zero elsewhere
      - Each expert is its own SwiGLU FFN (parallel parameter pools)
      - Output: sum of expert(x) weighted by gate(x), summed over
                the top_k experts the router selected

    Config flags (additions to GhostLMConfig):
      use_moe (bool, default False)
      n_experts (int, default 4)
      n_experts_active (int, default 2; the K in top-K routing)
      moe_aux_loss_coef (float, default 0.01; load-balancing weight)

    The auxiliary load-balancing loss (returned alongside the FFN
    output) penalizes routing collapse to a single expert. The
    trainer needs to add it to the cross-entropy loss with the
    configured weight.
    """

    def __init__(self, config: GhostLMConfig):
        super().__init__()
        self.n_experts = int(getattr(config, "n_experts", 4))
        self.top_k = int(getattr(config, "n_experts_active", 2))
        if self.top_k > self.n_experts:
            raise ValueError(
                f"n_experts_active ({self.top_k}) cannot exceed "
                f"n_experts ({self.n_experts})"
            )
        self.gate = nn.Linear(config.d_model, self.n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(config) for _ in range(self.n_experts)])
        self.dropout = nn.Dropout(config.dropout)
        # Most recently computed aux loss (load-balancing term), set on
        # every forward. ``GhostLM.forward`` sums it across all MoE
        # layers and adds ``moe_aux_loss_coef * sum`` to the objective.
        # Plain attribute (not a buffer): it is a per-forward scratch
        # value carrying autograd history, not model state.
        self.last_aux_loss: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Top-K routing + weighted expert sum + load-balancing loss."""
        b, t, d = x.shape
        flat = x.reshape(b * t, d)

        # Router: pick top_k experts per token.
        gate_logits = self.gate(flat)                          # [b*t, n_experts]
        top_w, top_idx = gate_logits.topk(self.top_k, dim=-1)  # [b*t, top_k]
        top_w = F.softmax(top_w, dim=-1)                       # gate weights

        # Aux loss: encourage uniform expert utilization.
        # Equation 4 from the Switch Transformer paper, adapted for
        # top-K (the per-expert mass + per-expert routing fraction
        # should both be near 1/n_experts).
        gate_probs = F.softmax(gate_logits, dim=-1)
        expert_mask = torch.zeros_like(gate_probs).scatter_(
            -1, top_idx, 1.0,
        )
        f_i = expert_mask.mean(dim=0)            # fraction routed to each expert
        p_i = gate_probs.mean(dim=0)             # routing prob mass per expert
        self.last_aux_loss = (f_i * p_i).sum() * self.n_experts

        # Dispatch: compute expert(x) for each routed token, weighted.
        out = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            # Tokens routed to this expert (may be empty).
            mask = (top_idx == i).any(dim=-1)
            if not mask.any():
                continue
            # Per-token weight = sum of weights where this expert was
            # in the top_k (typically 0 or 1 entries; rarely 2 if
            # n_experts == top_k).
            w = (top_w * (top_idx == i).float()).sum(dim=-1, keepdim=True)
            tokens = flat[mask]
            expert_out = expert(tokens)
            out[mask] += expert_out * w[mask]

        return self.dropout(out).reshape(b, t, d)


def make_ffn(config: GhostLMConfig) -> nn.Module:
    """Return SparseMoE, SwiGLU, or FeedForward based on config flags.
    Order: MoE wins if use_moe is set; SwiGLU wins if use_swiglu is set
    (and MoE is off); plain GELU FeedForward otherwise."""
    if getattr(config, "use_moe", False):
        return SparseMoE(config)
    if getattr(config, "use_swiglu", False):
        return SwiGLU(config)
    return FeedForward(config)


class TransformerBlock(nn.Module):
    """Single transformer decoder block with pre-normalization.

    Applies LayerNorm before both the self-attention and feed-forward
    sub-layers (pre-norm architecture), with residual connections
    around each sub-layer.
    """

    def __init__(self, config: GhostLMConfig):
        """Initialize the transformer block.

        Args:
            config: GhostLMConfig passed to sub-modules. Switches between
                LayerNorm / RMSNorm and FeedForward / SwiGLU based on the
                ``use_rmsnorm`` and ``use_swiglu`` flags.
        """
        super().__init__()
        self.ln_1 = make_norm(config, config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = make_norm(config, config.d_model)
        self.ffn = make_ffn(config)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        past_kv: Optional[LayerKVCache] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[LayerKVCache]]:
        """Forward pass through the transformer block.

        Args:
            x: Input tensor of shape (B, T, d_model).
            attn_mask: Optional (B, past_len + T) padding mask.
            past_kv: Optional cached (k, v) for this layer.
            use_cache: If True, return the updated layer cache.

        Returns:
            Tuple of (output of shape (B, T, d_model), updated cache or None).
        """
        # Pre-norm + self-attention with residual
        attn_out, present = self.attn(
            self.ln_1(x), attn_mask=attn_mask,
            past_kv=past_kv, use_cache=use_cache,
        )
        x = x + attn_out
        # Pre-norm + feed-forward with residual
        x = x + self.ffn(self.ln_2(x))
        return x, present


class GhostLM(nn.Module):
    """GhostLM decoder-only transformer language model.

    Built from scratch in PyTorch with learned positional embeddings,
    stacked transformer blocks, and weight-tied output projection.
    """

    def __init__(self, config: GhostLMConfig):
        """Initialize the GhostLM model.

        Args:
            config: GhostLMConfig with all model hyperparameters.
        """
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        if not config.use_rope:
            self.pos_embedding = nn.Embedding(config.context_length, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Final layer norm — RMSNorm or LayerNorm depending on config.
        self.ln_f = make_norm(config, config.d_model)

        # Output head with weight tying (no bias)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Apply scaled residual initialization for deeper models. Only
        # the projections that feed the residual stream get the
        # depth-scaled std: attention output (proj), GELU FFN output
        # (fc2), and SwiGLU output (fc3 — NOT fc2, which is the gate).
        resid_std = 0.02 / math.sqrt(2 * config.n_layers)
        for module in self.modules():
            if isinstance(module, CausalSelfAttention):
                torch.nn.init.normal_(module.proj.weight, mean=0.0, std=resid_std)
            elif isinstance(module, FeedForward):
                torch.nn.init.normal_(module.fc2.weight, mean=0.0, std=resid_std)
            elif isinstance(module, SwiGLU):
                torch.nn.init.normal_(module.fc3.weight, mean=0.0, std=resid_std)

    def _init_weights(self, module):
        """Initialize module weights with a normal distribution.

        Args:
            module: nn.Module to initialize.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, attn_mask=None, past_kv=None, use_cache=False):
        """Forward pass of the model.

        Args:
            idx: Input token ids of shape (B, T).
            targets: Optional target token ids of shape (B, T) for loss computation.
            attn_mask: Optional padding mask of shape (B, past_len + T);
                1 for real tokens, 0 for padding (see
                ``GhostTokenizer.pad_batch``). Outputs at padded positions
                are undefined and must be ignored by the caller.
            past_kv: Optional per-layer KV cache from a previous call made
                with ``use_cache=True``. When provided, ``idx`` should
                contain only the new (not yet cached) tokens.
            use_cache: If True, additionally return the updated KV cache
                for incremental decoding.

        Returns:
            Tuple of (logits, loss), or (logits, loss, kv_cache) when
            ``use_cache`` is True. Logits have shape (B, T, vocab_size).
            Loss is returned only if targets are provided.

        Raises:
            AssertionError: If cached + new sequence length exceeds
                context_length.
        """
        B, T = idx.size()
        past_len = past_kv[0][0].size(2) if past_kv else 0
        assert past_len + T <= self.config.context_length, (
            f"Sequence length {past_len + T} exceeds context length "
            f"{self.config.context_length}"
        )

        # Token + positional embeddings
        tok_emb = self.token_embedding(idx)
        if self.config.use_rope:
            x = self.dropout(tok_emb)
        else:
            pos = torch.arange(past_len, past_len + T, dtype=torch.long, device=idx.device)
            pos_emb = self.pos_embedding(pos)
            x = self.dropout(tok_emb + pos_emb)

        # Transformer blocks. Gradient checkpointing recomputes each
        # block's activations during backward instead of storing them —
        # a large activation-memory cut for ~25-30% extra step time.
        # Only sensible while training (and pointless with a KV cache).
        use_ckpt = (
            getattr(self.config, "gradient_checkpointing", False)
            and self.training
            and torch.is_grad_enabled()
            and not use_cache
        )
        presents: Optional[List[LayerKVCache]] = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_past = past_kv[i] if past_kv else None
            if use_ckpt:
                x, present = torch.utils.checkpoint.checkpoint(
                    block, x, attn_mask, layer_past, use_cache,
                    use_reentrant=False,
                )
            else:
                x, present = block(
                    x, attn_mask=attn_mask, past_kv=layer_past, use_cache=use_cache,
                )
            if use_cache:
                presents.append(present)

        # Final layer norm
        x = self.ln_f(x)

        # Output logits
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            # MoE auxiliary load-balancing loss. Each SparseMoE layer
            # stashes its per-batch aux term in self.last_aux_loss
            # during forward; sum them here and weight by the configured
            # coefficient. Keeping the wiring inside model.forward keeps
            # the trainer architecture-agnostic.
            if getattr(self.config, "use_moe", False):
                aux = sum(
                    m.last_aux_loss for m in self.modules()
                    if isinstance(m, SparseMoE) and m.last_aux_loss is not None
                )
                loss = loss + self.config.moe_aux_loss_coef * aux

        if use_cache:
            return logits, loss, presents
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressively generate new tokens with KV caching.

        The prompt is prefilled once; every subsequent step feeds only
        the newly sampled token and reuses the cached keys/values, so
        each step costs O(T) attention instead of a full O(T^2)
        recompute. If the sequence outgrows the context window, the
        cache is rebuilt from the cropped tail (sliding window), which
        matches the pre-cache behaviour.

        Args:
            idx: Input token ids of shape (B, T) serving as the prompt.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_k: If set, only sample from the top-k most likely tokens.

        Returns:
            Tensor of shape (B, T + max_new_tokens) with generated tokens.
        """
        ctx = self.config.context_length
        past_kv = None
        # First step prefills the (cropped) prompt; later steps feed one token.
        input_ids = idx[:, -ctx:]

        for _ in range(max_new_tokens):
            past_len = past_kv[0][0].size(2) if past_kv else 0
            if past_len + input_ids.size(1) > ctx:
                # Cache full: slide the window and re-prefill.
                past_kv = None
                input_ids = idx[:, -ctx:]

            logits, _, past_kv = self(input_ids, past_kv=past_kv, use_cache=True)

            # Take logits at the last position
            logits = logits[:, -1, :] / temperature

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Apply softmax and sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence; next step only feeds the new token.
            idx = torch.cat((idx, idx_next), dim=1)
            input_ids = idx_next

        return idx

    def num_params(self) -> int:
        """Return the total number of trainable parameters.

        Returns:
            Integer count of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def configure_optimizers(self, config: GhostLMConfig):
        """Create an AdamW optimizer with weight decay separation.

        Separates parameters into two groups: those that should receive
        weight decay (linear weights) and those that should not
        (biases, LayerNorm weights, embeddings).

        Args:
            config: GhostLMConfig containing learning_rate, betas, and weight_decay.

        Returns:
            torch.optim.AdamW optimizer with properly configured parameter groups.
        """
        decay = set()
        no_decay = set()

        whitelist = (nn.Linear,)
        # RMSNorm is custom (defined in this module), so we include it in the
        # blacklist by class — its `.weight` should be no-decay just like
        # LayerNorm's. Without this, the v0.5 ghost-small-v0.5 preset
        # crashes at optimizer setup with every block's ln_*.weight
        # uncategorized.
        blacklist = (nn.LayerNorm, nn.Embedding, RMSNorm)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist):
                    no_decay.add(fpn)

        # Remove lm_head.weight from decay if present — it is tied to token_embedding.weight
        decay.discard("lm_head.weight")
        no_decay.discard("lm_head.weight")

        # Validate all parameters are accounted for (excluding tied weight)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        all_params = decay | no_decay
        uncategorized = {k for k in param_dict.keys() if k not in all_params and k != "lm_head.weight"}
        assert len(uncategorized) == 0, f"Parameters {uncategorized} not categorized"

        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(decay)],
                "weight_decay": config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
        )

        return optimizer
