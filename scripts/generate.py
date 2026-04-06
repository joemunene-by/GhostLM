"""GhostLM inference script — load a trained model and generate text from a prompt."""

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import torch

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.tokenizer import GhostTokenizer


def parse_args():
    """Parse command-line arguments for text generation.

    Returns:
        argparse.Namespace with all parsed generation arguments.
    """
    parser = argparse.ArgumentParser(description="Generate text with GhostLM")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain the following CVE vulnerability:",
        help="Text prompt to start generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (higher = more random)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling (0 to disable)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on: auto, cpu, cuda, mps",
    )

    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, device: str) -> tuple:
    """Load a GhostLM model from a saved checkpoint.

    Reconstructs the model configuration from the checkpoint metadata,
    instantiates the model, and loads the saved weights.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Target device string ("cpu", "cuda", "mps").

    Returns:
        Tuple of (model, config) where model is in eval mode on the target device.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct config from saved dict
    saved_config = checkpoint["config"]
    config = GhostLMConfig(**{
        f.name: saved_config[f.name]
        for f in fields(GhostLMConfig)
        if f.name in saved_config
    })

    # Build model and load weights
    model = GhostLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model = model.to(device)

    return model, config


def main():
    """Run the GhostLM text generation pipeline.

    Loads a trained model from checkpoint, encodes a user prompt,
    generates new tokens autoregressively, and prints the result.
    """
    args = parse_args()

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # Load model
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    model, config = load_model_from_checkpoint(args.checkpoint, device)

    # Initialize tokenizer
    tokenizer = GhostTokenizer()

    # Print header
    print("=" * 50)
    print("GhostLM Generation")
    print("=" * 50)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device:     {device}")
    print(f"  Preset:     {config.n_layers} layers, {config.d_model} dim")
    print(f"  Temp:       {args.temperature}")
    print(f"  Top-k:      {args.top_k if args.top_k > 0 else 'disabled'}")
    print(f"  Max tokens: {args.max_tokens}")
    print("=" * 50)

    # Encode prompt
    prompt_ids = tokenizer.encode(args.prompt)
    input_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    # Generate
    top_k = args.top_k if args.top_k > 0 else None
    generated = model.generate(
        input_tensor,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=top_k,
    )

    # Decode and print
    output_text = tokenizer.decode(generated[0].tolist())
    print(f"\nPrompt:\n  {args.prompt}\n")
    print(f"Generated:\n  {output_text}")
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
