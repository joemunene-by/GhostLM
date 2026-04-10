"""GhostLM configuration — all model and training hyperparameters live here."""

from dataclasses import dataclass, field


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
    use_flash_attention: bool = False

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

    def model_size(self) -> str:
        """Estimate total parameter count and return a human-readable string.

        Computes the approximate number of trainable parameters based on
        vocab_size, d_model, n_heads, n_layers, and d_ff.

        Returns:
            A string like "124M" or "1.2B" representing the estimated size.
        """
        embedding_params = self.vocab_size * self.d_model
        attention_params = self.n_layers * (
            4 * self.d_model * self.d_model + 2 * self.d_model
        )
        ffn_params = self.n_layers * (
            2 * self.d_model * self.d_ff + self.d_model + self.d_ff
        )
        layer_norm_params = self.n_layers * 4 * self.d_model
        output_head_params = self.d_model * self.vocab_size

        total = embedding_params + attention_params + ffn_params + layer_norm_params + output_head_params

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
            "=" * 40,
            f"Estimated size: {self.model_size()}",
        ]
        return "\n".join(lines)
