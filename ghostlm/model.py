"""GhostLM transformer model — decoder-only architecture built from scratch in PyTorch."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ghostlm.config import GhostLMConfig


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

    def forward(self, seq_len: int):
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


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


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with autoregressive masking.

    Uses a single combined QKV projection for efficiency, then splits
    the result into separate query, key, and value tensors. Supports
    optional RoPE (Rotary Position Embeddings) and Flash Attention.
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
        self.context_length = config.context_length
        self.use_rope = config.use_rope
        self.use_flash_attention = config.use_flash_attention
        self.dropout_p = config.dropout

        # Single combined QKV projection
        self.c_qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)

        # Dropout applied to attention weights (manual path only)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # RoPE
        if self.use_rope:
            self.rope = RotaryEmbedding(self.head_dim, config.context_length)

        # Causal mask buffer (only needed for manual attention path)
        if not self.use_flash_attention:
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(config.context_length, config.context_length))
                .view(1, 1, config.context_length, config.context_length),
                persistent=False,
            )

    def forward(self, x):
        """Forward pass through causal self-attention.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        B, T, C = x.size()

        # Combined QKV projection and split
        qkv = self.c_qkv(x)
        q, k, v = qkv.split(self.n_heads * self.head_dim, dim=-1)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K (not V)
        if self.use_rope:
            cos, sin = self.rope(T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self.use_flash_attention:
            # PyTorch 2.0+ scaled_dot_product_attention with automatic backend selection
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Manual attention path
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale
            att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        # Reassemble heads and project
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y))

        return y


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


class TransformerBlock(nn.Module):
    """Single transformer decoder block with pre-normalization.

    Applies LayerNorm before both the self-attention and feed-forward
    sub-layers (pre-norm architecture), with residual connections
    around each sub-layer.
    """

    def __init__(self, config: GhostLMConfig):
        """Initialize the transformer block.

        Args:
            config: GhostLMConfig passed to sub-modules.
        """
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x):
        """Forward pass through the transformer block.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        # Pre-norm + self-attention with residual
        x = x + self.attn(self.ln_1(x))
        # Pre-norm + feed-forward with residual
        x = x + self.ffn(self.ln_2(x))
        return x


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

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)

        # Output head with weight tying (no bias)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Apply scaled residual initialization for deeper models
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight") or pn.endswith("fc2.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

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

    def forward(self, idx, targets=None):
        """Forward pass of the model.

        Args:
            idx: Input token ids of shape (B, T).
            targets: Optional target token ids of shape (B, T) for loss computation.

        Returns:
            Tuple of (logits, loss). Logits have shape (B, T, vocab_size).
            Loss is returned only if targets are provided.

        Raises:
            AssertionError: If sequence length exceeds context_length.
        """
        B, T = idx.size()
        assert T <= self.config.context_length, (
            f"Sequence length {T} exceeds context length {self.config.context_length}"
        )

        # Token + positional embeddings
        tok_emb = self.token_embedding(idx)
        if self.config.use_rope:
            x = self.dropout(tok_emb)
        else:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            pos_emb = self.pos_embedding(pos)
            x = self.dropout(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Output logits
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Autoregressively generate new tokens.

        Args:
            idx: Input token ids of shape (B, T) serving as the prompt.
            max_new_tokens: Number of tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_k: If set, only sample from the top-k most likely tokens.

        Returns:
            Tensor of shape (B, T + max_new_tokens) with generated tokens.
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx[:, -self.config.context_length:]

            # Forward pass
            logits, _ = self(idx_cond)

            # Take logits at the last position
            logits = logits[:, -1, :] / temperature

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Apply softmax and sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)

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
        blacklist = (nn.LayerNorm, nn.Embedding)

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
