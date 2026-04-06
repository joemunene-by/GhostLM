"""GhostLM benchmark — compares perplexity of GhostLM vs GPT-2 baseline on cybersecurity texts."""

import argparse
import json
import math
import sys
import time
from dataclasses import fields
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.tokenizer import GhostTokenizer


BENCHMARK_TEXTS = [
    "SQL injection is a code injection technique that exploits security vulnerabilities in an application's database layer. It occurs when user input is incorrectly filtered or not strongly typed, allowing malicious SQL statements to be executed. This can lead to unauthorized data access, data modification, or complete database compromise.",
    "A buffer overflow occurs when a program writes more data to a buffer than it can hold, overwriting adjacent memory locations. Attackers exploit this vulnerability by carefully crafting input that overwrites the return address on the stack with a pointer to their own shellcode. Modern mitigations include stack canaries, ASLR, and DEP.",
    "Cross-Site Scripting (XSS) is a web security vulnerability that allows an attacker to inject malicious scripts into web pages viewed by other users. The injected script executes in the victim's browser, potentially stealing session cookies or performing actions on behalf of the user. Reflected, stored, and DOM-based XSS are the three main categories.",
    "Ransomware is a type of malicious software that encrypts a victim's files and demands payment for the decryption key. Modern ransomware campaigns often use double extortion, where attackers also exfiltrate data and threaten to publish it if the ransom is not paid. Prevention strategies include regular backups, network segmentation, and endpoint detection.",
    "Zero-day vulnerabilities are security flaws that are unknown to the software vendor and have no available patch. Attackers actively exploit these vulnerabilities before defenders can develop and deploy a fix. The market for zero-day exploits includes both legitimate security researchers and malicious threat actors, with prices ranging from thousands to millions of dollars.",
    "Network scanning is a technique used to identify active hosts, open ports, and running services on a network. Tools like Nmap send specially crafted packets to target systems and analyze the responses to build a map of network topology. This reconnaissance step is essential for both security auditing and the initial phases of penetration testing.",
    "Privilege escalation is the act of exploiting a bug or misconfiguration to gain elevated access to resources that are normally protected. Vertical privilege escalation grants higher-level permissions, while horizontal escalation allows access to another user's resources. Common vectors include SUID binaries, misconfigured services, and kernel exploits.",
    "Phishing is a social engineering attack where attackers impersonate legitimate organizations through email, SMS, or fake websites. The goal is to trick victims into revealing sensitive information such as login credentials or credit card numbers. Spear phishing targets specific individuals with personalized messages, while whaling targets high-level executives.",
    "A man-in-the-middle attack occurs when an attacker secretly intercepts and possibly alters communication between two parties who believe they are directly communicating with each other. Common techniques include ARP spoofing, DNS spoofing, and SSL stripping. Encryption protocols like TLS and certificate pinning are primary defenses against MITM attacks.",
    "The Common Vulnerability Scoring System (CVSS) provides a standardized method for rating the severity of security vulnerabilities. Scores range from 0.0 to 10.0 and are based on metrics including attack vector, complexity, privileges required, and impact on confidentiality, integrity, and availability. CVSS v3.1 introduced additional metrics for scope and user interaction.",
]


def compute_ghostlm_perplexity(
    model: GhostLM,
    tokenizer: GhostTokenizer,
    texts: list,
    device: str,
    context_length: int,
) -> float:
    """Compute perplexity of GhostLM over a list of cybersecurity texts.

    Encodes each text, chunks it into context-length windows, runs forward
    passes to accumulate loss, and returns the exponential of average loss.

    Args:
        model: GhostLM model in eval mode.
        tokenizer: GhostTokenizer instance.
        texts: List of text strings to evaluate.
        device: Device to run computation on.
        context_length: Maximum sequence length for the model.

    Returns:
        Perplexity as a float.
    """
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            ids = tokenizer.encode(text)

            if len(ids) < 10:
                continue

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

    return math.exp(total_loss / total_tokens)


def compute_gpt2_perplexity(texts: list, device: str) -> float:
    """Compute perplexity of GPT-2 over a list of cybersecurity texts.

    Loads the pretrained GPT-2 model and tokenizer, computes cross-entropy
    loss for each text, and returns the exponential of average loss.

    Args:
        texts: List of text strings to evaluate.
        device: Device to run computation on.

    Returns:
        Perplexity as a float.
    """
    print("  Loading GPT-2 baseline...")
    gpt2_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_model.eval()
    gpt2_model = gpt2_model.to(device)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            encoded = gpt2_tokenizer(text, return_tensors="pt", truncation=True)
            input_ids = encoded.input_ids.to(device)

            if input_ids.size(1) < 2:
                continue

            outputs = gpt2_model(input_ids, labels=input_ids)
            loss = outputs.loss

            total_loss += loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)

    if total_tokens == 0:
        return float("inf")

    return math.exp(total_loss / total_tokens)


def load_ghostlm(checkpoint_path: str, device: str) -> tuple:
    """Load a GhostLM model from a saved checkpoint.

    Reconstructs the model configuration from the checkpoint metadata,
    instantiates the model, and loads the saved weights.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Target device string.

    Returns:
        Tuple of (model, config) with model in eval mode.
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


def print_results(
    ghostlm_ppl: float,
    gpt2_ppl: float,
    ghostlm_step: int,
    elapsed: float,
) -> None:
    """Print a clean comparison table of GhostLM vs GPT-2 perplexity.

    Args:
        ghostlm_ppl: GhostLM perplexity score.
        gpt2_ppl: GPT-2 perplexity score.
        ghostlm_step: Training step of the GhostLM checkpoint.
        elapsed: Total benchmark time in seconds.
    """
    print("\n" + "=" * 40)
    print("GhostLM Benchmark Results")
    print("=" * 40)
    print(f"{'Model':<16} {'Perplexity':<12}")
    print("-" * 40)
    print(f"{'GPT-2 (117M)':<16} {gpt2_ppl:<12.2f}")
    print(f"{'GhostLM':<16} {ghostlm_ppl:<12.2f}")
    print("-" * 40)
    print(f"Step:  {ghostlm_step}")
    print(f"Time:  {elapsed:.1f}s")
    print("=" * 40)

    if ghostlm_ppl < gpt2_ppl:
        print("GhostLM BEATS GPT-2 baseline!")
    else:
        print("GPT-2 still ahead — keep training.")


def main():
    """Run the GhostLM vs GPT-2 perplexity benchmark.

    Loads GhostLM from checkpoint (or random init for testing), computes
    perplexity on cybersecurity texts, compares against GPT-2, and saves
    results to JSON.
    """
    parser = argparse.ArgumentParser(description="Benchmark GhostLM vs GPT-2")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to GhostLM checkpoint (optional, uses random init if omitted)",
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
        default="logs/benchmark.json",
        help="Where to save benchmark results",
    )
    args = parser.parse_args()

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

    print("=" * 40)
    print("GhostLM vs GPT-2 Benchmark")
    print("=" * 40)
    print(f"Device: {device}")
    print(f"Texts:  {len(BENCHMARK_TEXTS)} cybersecurity samples")

    t0 = time.time()

    # Load GhostLM
    ghostlm_step = 0
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"\nLoading GhostLM from {args.checkpoint}...")
        model, config = load_ghostlm(args.checkpoint, device)
        checkpoint_data = torch.load(args.checkpoint, map_location=device, weights_only=False)
        ghostlm_step = checkpoint_data.get("step", 0)
    else:
        print("\nNo checkpoint provided — using random ghost-small init...")
        config = GhostLMConfig.from_preset("ghost-small")
        config.vocab_size = 50261
        model = GhostLM(config)
        model.eval()
        model = model.to(device)

    tokenizer = GhostTokenizer()

    # Compute GhostLM perplexity
    print("\nComputing GhostLM perplexity...")
    ghostlm_ppl = compute_ghostlm_perplexity(
        model, tokenizer, BENCHMARK_TEXTS, device, config.context_length
    )
    print(f"  GhostLM perplexity: {ghostlm_ppl:.2f}")

    # Compute GPT-2 perplexity
    print("\nComputing GPT-2 perplexity...")
    gpt2_ppl = compute_gpt2_perplexity(BENCHMARK_TEXTS, device)
    print(f"  GPT-2 perplexity:   {gpt2_ppl:.2f}")

    elapsed = time.time() - t0

    # Print results
    print_results(ghostlm_ppl, gpt2_ppl, ghostlm_step, elapsed)

    # Save results
    results = {
        "ghostlm_perplexity": round(ghostlm_ppl, 4),
        "gpt2_perplexity": round(gpt2_ppl, 4),
        "ghostlm_step": ghostlm_step,
        "elapsed_seconds": round(elapsed, 1),
        "device": device,
        "num_texts": len(BENCHMARK_TEXTS),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
