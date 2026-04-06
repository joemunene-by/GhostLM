"""GhostLM evaluation script — benchmarks a trained model on cybersecurity reasoning tasks."""

import argparse
import json
import math
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import List

import torch

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.tokenizer import GhostTokenizer


def parse_args():
    """Parse command-line arguments for evaluation.

    Returns:
        argparse.Namespace with all parsed evaluation arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate GhostLM on cybersecurity tasks")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on: auto, cpu, cuda, mps",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/eval_results.json",
        help="Where to save evaluation results",
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


def compute_perplexity(
    model: GhostLM,
    tokenizer: GhostTokenizer,
    texts: List[str],
    device: str,
    context_length: int,
) -> float:
    """Compute perplexity of the model over a list of cybersecurity texts.

    Encodes each text, chunks it into context-length windows, runs forward
    passes to accumulate loss, and returns the exponential of average loss.

    Args:
        model: GhostLM model in eval mode.
        tokenizer: GhostTokenizer instance.
        texts: List of text strings to evaluate on.
        device: Device to run computation on.
        context_length: Maximum sequence length for the model.

    Returns:
        Perplexity as a float (exp of average cross-entropy loss).
    """
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            ids = tokenizer.encode(text)

            if len(ids) < 10:
                continue

            # Chunk into context-length windows
            for i in range(0, len(ids) - 1, context_length):
                chunk = ids[i : i + context_length]

                if len(chunk) < 2:
                    continue

                x = torch.tensor(chunk[:-1], dtype=torch.long, device=device).unsqueeze(0)
                y = torch.tensor(chunk[1:], dtype=torch.long, device=device).unsqueeze(0)

                _, loss = model(x, targets=y)

                total_loss += loss.item() * y.size(1)
                total_tokens += y.size(1)

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


CYBERSEC_PROMPTS = [
    "A SQL injection attack works by",
    "The CVE scoring system rates vulnerabilities based on",
    "To perform a buffer overflow exploit, an attacker",
    "XSS (Cross-Site Scripting) allows attackers to",
    "A man-in-the-middle attack intercepts",
    "Ransomware encrypts victim files by",
    "In penetration testing, reconnaissance involves",
    "Zero-day vulnerabilities are dangerous because",
]


def evaluate_generation(
    model: GhostLM,
    tokenizer: GhostTokenizer,
    prompts: List[str],
    device: str,
    max_tokens: int = 80,
) -> List[dict]:
    """Generate text responses for each cybersecurity prompt.

    Runs autoregressive generation for each prompt and collects
    the generated text along with token counts.

    Args:
        model: GhostLM model in eval mode.
        tokenizer: GhostTokenizer instance.
        prompts: List of prompt strings.
        device: Device to run generation on.
        max_tokens: Maximum number of tokens to generate per prompt.

    Returns:
        List of dicts with keys: prompt, generated, tokens.
    """
    results = []

    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

        generated = model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=0.8,
            top_k=40,
        )

        generated_text = tokenizer.decode(generated[0].tolist())
        token_count = generated.size(1) - input_tensor.size(1)

        results.append({
            "prompt": prompt,
            "generated": generated_text,
            "tokens": token_count,
        })

    return results


def main():
    """Run the full GhostLM evaluation pipeline.

    Loads a trained model, computes perplexity on cybersecurity texts,
    generates responses to security prompts, and saves all results to JSON.
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
    tokenizer = GhostTokenizer()

    # Print header
    print("=" * 50)
    print("GhostLM Evaluation")
    print("=" * 50)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device:     {device}")
    print(f"  Model:      {config.n_layers} layers, {config.d_model} dim")
    print(f"  Parameters: {model.num_params():,}")
    print("=" * 50)

    # Perplexity evaluation texts
    PERPLEXITY_TEXTS = [
        "SQL injection is a code injection technique that exploits security vulnerabilities in an application's database layer. It occurs when user input is incorrectly filtered or not strongly typed, allowing malicious SQL statements to be executed. This can lead to unauthorized data access, data modification, or even complete database compromise.",
        "A buffer overflow occurs when a program writes more data to a buffer than it can hold. This excess data overwrites adjacent memory locations, potentially corrupting data, crashing the program, or executing arbitrary code. Attackers exploit this by carefully crafting input that overwrites the return address on the stack with a pointer to their own shellcode.",
        "Cross-Site Scripting (XSS) is a web security vulnerability that allows an attacker to inject malicious scripts into web pages viewed by other users. The injected script executes in the victim's browser, potentially stealing session cookies, redirecting to malicious sites, or performing actions on behalf of the user without their knowledge.",
        "Phishing is a social engineering attack where attackers impersonate legitimate organizations through email, SMS, or fake websites to trick victims into revealing sensitive information. Common targets include login credentials, credit card numbers, and personal identification data. Successful phishing attacks often exploit urgency and fear to bypass critical thinking.",
        "Network scanning is a technique used to identify active hosts, open ports, and running services on a network. Tools like Nmap send specially crafted packets to target systems and analyze the responses to build a map of network topology and service versions. This reconnaissance step is essential for both security auditing and penetration testing.",
    ]

    # Compute perplexity
    print("\nComputing perplexity...")
    t0 = time.time()
    perplexity = compute_perplexity(
        model, tokenizer, PERPLEXITY_TEXTS, device, config.context_length
    )
    elapsed = time.time() - t0
    print(f"  Perplexity: {perplexity:.2f} ({elapsed:.1f}s)")

    # Generation evaluation
    print("\nGenerating responses...")
    generations = evaluate_generation(
        model, tokenizer, CYBERSEC_PROMPTS, device, max_tokens=80
    )

    for i, gen in enumerate(generations):
        print(f"\n  [{i + 1}] Prompt: {gen['prompt']}")
        print(f"      Generated: {gen['generated']}")
        print(f"      Tokens: {gen['tokens']}")

    # Build results
    checkpoint_data = torch.load(args.checkpoint, map_location=device, weights_only=False)
    step = checkpoint_data.get("step", 0)

    results = {
        "checkpoint": args.checkpoint,
        "step": step,
        "perplexity": round(perplexity, 4),
        "model_size": f"{config.n_layers} layers, {config.d_model} dim",
        "parameters": model.num_params(),
        "generations": generations,
    }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation complete. Results saved to {output_path}")


if __name__ == "__main__":
    main()
