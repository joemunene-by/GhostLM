"""GhostLM configuration — all model and training hyperparameters live here."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GhostLMConfig:
    """Configuration dataclass for the GhostLM transformer language model.

    Holds all hyperparameters for model architecture, training, data paths,
    and system settings. Supports preset configurations and parameter counting.
    """

    # Model architecture
    vocab_size: int = 50257
    context_length: int = 1024
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    bias: bool = True
    use_rope: bool = False
    rope_base: float = 10000.0
    use_swiglu: bool = False
    use_rmsnorm: bool = False
    use_flash_attention: bool = False
    # Grouped-query attention: number of key/value heads. None means
    # n_kv_heads == n_heads (plain multi-head attention, the historical
    # behaviour). Setting it lower (n_heads must be divisible by it)
    # shares each KV head across n_heads / n_kv_heads query heads —
    # Llama-3 / Qwen / Gemma style — shrinking the KV cache by the same
    # factor at near-zero quality cost.
    n_kv_heads: Optional[int] = None
    # RMSNorm on per-head queries and keys before RoPE (Qwen3 / Gemma-3 /
    # OLMo-2 style). Cheap insurance against attention-logit blowups on
    # long bf16 pretrains.
    use_qk_norm: bool = False
    # Intra-document attention masking for packed pretraining. The dataset
    # concatenates documents (EOS-separated) and slices fixed blocks, so a
    # block usually straddles document boundaries; with plain causal
    # attention, tokens attend back into the previous, unrelated document.
    # When True (and ``eos_token_id`` is set), each token only attends
    # within its own document, the "no cross-contamination" packing used by
    # GPT-3 / Llama / OLMo. Off by default so existing checkpoints are
    # bit-for-bit unchanged; only affects the full-sequence training path
    # (skipped under KV-cache decoding and when a padding mask is present).
    intra_doc_mask: bool = False
    # Token id that terminates a document (the tokenizer's EOS). Required
    # for ``intra_doc_mask``; the trainer sets it from the tokenizer.
    eos_token_id: Optional[int] = None

    # Mixture-of-Experts (sparse FFN). Off by default; enable for
    # ghost-1B and beyond. With use_moe = True the FFN at every
    # transformer block becomes a SparseMoE layer with n_experts
    # parallel SwiGLU pools and top-K routing. Effective parameter
    # count = n_experts * (per-expert SwiGLU params), inference cost
    # = n_experts_active * (per-expert SwiGLU compute).
    # See ghostlm.model.SparseMoE for routing + load-balancing detail.
    use_moe: bool = False
    n_experts: int = 4
    n_experts_active: int = 2     # the K in top-K routing
    moe_aux_loss_coef: float = 0.01  # load-balancing loss weight

    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    grad_accum_steps: int = 4
    warmup_steps: int = 2000
    max_steps: int = 100000
    eval_interval: int = 500
    save_interval: int = 1000

    # Paths
    data_dir: str = "data/processed"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # System
    device: str = "auto"
    dtype: str = "float32"
    seed: int = 42
    use_wandb: bool = False
    wandb_project: str = "ghostlm"
    # torch.compile the model before training. Typically a 1.3-1.8x
    # throughput win on CUDA; first step pays a compilation stall.
    use_compile: bool = False
    # Recompute block activations in backward instead of storing them.
    # Trades ~25-30% step time for a large activation-memory cut —
    # what makes ghost-1b/3b shapes fit on a single card.
    gradient_checkpointing: bool = False

    def num_params(self) -> int:
        """Return the exact trainable parameter count for this config.

        Instantiates the model on the meta device (no memory is
        allocated, no weights are initialized on real storage), so the
        count is exact for every architecture switch — RoPE vs learned
        positions, GELU vs SwiGLU vs MoE FFN, bias on/off, weight tying.
        """
        import torch  # local imports avoid a config <-> model cycle
        from ghostlm.model import GhostLM

        with torch.device("meta"):
            model = GhostLM(self)
        # parameters() yields tied tensors once, so the tied lm_head
        # is not double-counted.
        return sum(p.numel() for p in model.parameters())

    def model_size(self) -> str:
        """Return the exact parameter count as a human-readable string.

        Returns:
            A string like "124M" or "1.2B".
        """
        total = self.num_params()

        if total >= 1e9:
            return f"{total / 1e9:.1f}B"
        elif total >= 1e6:
            return f"{total / 1e6:.0f}M"
        else:
            return f"{total:.0f}K"

    @classmethod
    def from_preset(cls, preset: str) -> "GhostLMConfig":
        """Return a GhostLMConfig instance from a named preset.

        Args:
            preset: One of "ghost-tiny", "ghost-small", or "ghost-medium".

        Returns:
            A GhostLMConfig configured with the preset's hyperparameters.

        Raises:
            ValueError: If the preset name is not recognized.
        """
        presets = {
            "ghost-tiny": {
                "n_layers": 2,
                "d_model": 256,
                "n_heads": 4,
                "d_ff": 1024,
            },
            "ghost-small": {
                "n_layers": 6,
                "d_model": 512,
                "n_heads": 8,
                "d_ff": 2048,
            },
            "ghost-medium": {
                "n_layers": 12,
                "d_model": 768,
                "n_heads": 12,
                "d_ff": 3072,
            },
            # v0.5 preset — same param shape as ghost-small but flips on
            # the modern-arch switches. Use this for the v0.4.2 retrain.
            "ghost-small-v0.5": {
                "n_layers": 6,
                "d_model": 512,
                "n_heads": 8,
                "d_ff": 2048,
                "use_rope": True,
                "use_swiglu": True,
                "use_rmsnorm": True,
            },
            # ghost-1b — first MoE preset. Target shape: ~1.2B active,
            # ~2.1B total (4 experts of SwiGLU, top-2 routing). This is
            # the bet 5 differentiator from docs/differentiation.md:
            # going MoE at 1B from-scratch is rare; most cybersec LMs
            # that ever reach 1B graft MoE in late or skip it. Defaults
            # to v1 BPE (vocab 32000) so the cybersec-native tokenizer
            # benefit lands in the same artifact.
            "ghost-1b": {
                "vocab_size": 32000,
                "context_length": 2048,
                "d_model": 1536,
                "n_layers": 24,
                "n_heads": 24,
                "d_ff": 6144,
                "use_rope": True,
                "use_swiglu": True,
                "use_rmsnorm": True,
                "use_flash_attention": True,
                "use_moe": True,
                "n_experts": 4,
                "n_experts_active": 2,
                "moe_aux_loss_coef": 0.01,
            },
            # ghost-3b — second MoE preset. Target shape: ~3.3B active,
            # ~6B total. Same MoE recipe as ghost-1b, scaled to a
            # SmolLM2-Mistral-7B-style dense backbone.
            "ghost-3b": {
                "vocab_size": 32000,
                "context_length": 2048,
                "d_model": 2048,
                "n_layers": 32,
                "n_heads": 32,
                "d_ff": 10240,
                "use_rope": True,
                "use_swiglu": True,
                "use_rmsnorm": True,
                "use_flash_attention": True,
                "use_moe": True,
                "n_experts": 4,
                "n_experts_active": 2,
                "moe_aux_loss_coef": 0.01,
            },
        }

        if preset not in presets:
            raise ValueError(
                f"Unknown preset '{preset}'. "
                f"Available presets: {', '.join(presets.keys())}"
            )

        return cls(**presets[preset])

    def __repr__(self) -> str:
        """Return a clean, grouped string summary of all config values.

        Returns:
            A formatted multi-line string with config values grouped by
            category: Architecture, Training, Paths, and System.
        """
        lines = [
            "GhostLMConfig",
            "=" * 40,
            "Architecture:",
            f"  vocab_size:      {self.vocab_size}",
            f"  context_length:  {self.context_length}",
            f"  d_model:         {self.d_model}",
            f"  n_heads:         {self.n_heads}",
            f"  n_layers:        {self.n_layers}",
            f"  d_ff:            {self.d_ff}",
            f"  dropout:         {self.dropout}",
            f"  bias:            {self.bias}",
            f"  use_rope:        {self.use_rope}",
            f"  use_swiglu:      {self.use_swiglu}",
            f"  use_rmsnorm:     {self.use_rmsnorm}",
            f"  use_flash_attn:  {self.use_flash_attention}",
            f"  n_kv_heads:      {self.n_kv_heads if self.n_kv_heads is not None else f'{self.n_heads} (MHA)'}",
            f"  use_qk_norm:     {self.use_qk_norm}",
            f"  use_moe:         {self.use_moe}"
            + (f" ({self.n_experts} experts, top-{self.n_experts_active})"
               if self.use_moe else ""),
            "Training:",
            f"  batch_size:      {self.batch_size}",
            f"  learning_rate:   {self.learning_rate}",
            f"  weight_decay:    {self.weight_decay}",
            f"  beta1:           {self.beta1}",
            f"  beta2:           {self.beta2}",
            f"  grad_clip:       {self.grad_clip}",
            f"  warmup_steps:    {self.warmup_steps}",
            f"  max_steps:       {self.max_steps}",
            f"  eval_interval:   {self.eval_interval}",
            f"  save_interval:   {self.save_interval}",
            "Paths:",
            f"  data_dir:        {self.data_dir}",
            f"  checkpoint_dir:  {self.checkpoint_dir}",
            f"  log_dir:         {self.log_dir}",
            "System:",
            f"  device:          {self.device}",
            f"  dtype:           {self.dtype}",
            f"  seed:            {self.seed}",
            f"  use_wandb:       {self.use_wandb}",
            f"  use_compile:     {self.use_compile}",
            f"  grad_checkpoint: {self.gradient_checkpointing}",
            "=" * 40,
            f"Estimated size: {self.model_size()}",
        ]
        return "\n".join(lines)
