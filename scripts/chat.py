"""GhostLM interactive chat — terminal interface for real-time generation."""

import argparse
import sys
from dataclasses import fields
from pathlib import Path

import torch

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.tokenizer import GhostTokenizer


def parse_args():
    """Parse command-line arguments for interactive chat.

    Returns:
        argparse.Namespace with all parsed chat arguments.
    """
    parser = argparse.ArgumentParser(description="GhostLM Interactive Chat")

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (uses random init if not provided)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling (0 to disable)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=150,
        help="Maximum tokens to generate per response",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on: auto, cpu, cuda, mps",
    )

    return parser.parse_args()


def load_model(checkpoint_path, device) -> tuple:
    """Load GhostLM model from checkpoint or initialize randomly.

    If no checkpoint is provided or the path doesn't exist, falls back
    to a randomly initialized ghost-tiny model for testing.

    Args:
        checkpoint_path: Path to .pt checkpoint file, or None.
        device: Target device string.

    Returns:
        Tuple of (model, config) with model in eval mode.
    """
    if checkpoint_path is None or not Path(checkpoint_path).exists():
        print("  No checkpoint provided — using random ghost-tiny weights.")
        config = GhostLMConfig.from_preset("ghost-tiny")
        config.vocab_size = 50261
        config.context_length = 128
        model = GhostLM(config)
        model.eval()
        model = model.to(device)
        return model, config

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    saved_config = checkpoint["config"]
    config = GhostLMConfig(**{
        f.name: saved_config[f.name]
        for f in fields(GhostLMConfig)
        if f.name in saved_config
    })

    model = GhostLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model = model.to(device)

    return model, config


def format_output(prompt: str, generated_text: str) -> str:
    """Strip the prompt from the beginning of generated text.

    Args:
        prompt: The original input prompt.
        generated_text: The full decoded output including prompt.

    Returns:
        Only the newly generated portion, stripped of whitespace.
    """
    if generated_text.startswith(prompt):
        return generated_text[len(prompt):].strip()
    return generated_text.strip()


def main():
    """Run the GhostLM interactive terminal chat loop.

    Loads the model, initializes the tokenizer, and enters an infinite
    loop accepting user prompts. Supports quit/exit commands, screen
    clearing, and graceful keyboard interrupt handling.
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

    # Load model and tokenizer
    model, config = load_model(args.checkpoint, device)
    tokenizer = GhostTokenizer()

    # Print header
    print()
    print("╔══════════════════════════════════════╗")
    print("║         GhostLM Chat v0.1.0          ║")
    print("║   Cybersecurity Language Model       ║")
    print("╚══════════════════════════════════════╝")
    print("Type your prompt and press Enter.")
    print("Commands: 'quit' or 'exit' to stop, 'clear' to reset")
    print()

    top_k = args.top_k if args.top_k > 0 else None

    while True:
        try:
            user_input = input("Ghost > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye.")
            break

        if user_input.lower() == "clear":
            print("\n" * 50)
            continue

        if not user_input:
            continue

        # Encode and generate
        ids = tokenizer.encode(user_input)
        input_tensor = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            output = model.generate(
                input_tensor,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=top_k,
            )

        generated_text = tokenizer.decode(output[0].tolist())
        new_text = format_output(user_input, generated_text)

        print(f"\nGhostLM > {new_text}\n")


if __name__ == "__main__":
    main()
