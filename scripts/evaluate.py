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


# --- Domain-specific evaluation tasks ---

CVE_TO_CWE_SAMPLES = [
    {
        "cve_text": "A vulnerability in the web interface allows an authenticated attacker to inject arbitrary SQL commands via the search parameter, leading to unauthorized data access.",
        "expected_cwe": "CWE-89",
        "cwe_name": "SQL Injection",
    },
    {
        "cve_text": "The application does not properly sanitize user input in the comment field, allowing attackers to inject malicious JavaScript that executes in other users' browsers.",
        "expected_cwe": "CWE-79",
        "cwe_name": "Cross-site Scripting",
    },
    {
        "cve_text": "A stack-based buffer overflow in the parsing function allows remote attackers to execute arbitrary code via a crafted input file that exceeds the allocated buffer size.",
        "expected_cwe": "CWE-787",
        "cwe_name": "Out-of-bounds Write",
    },
    {
        "cve_text": "The application constructs OS commands using unsanitized user input, allowing attackers to inject additional commands via shell metacharacters in the filename parameter.",
        "expected_cwe": "CWE-78",
        "cwe_name": "OS Command Injection",
    },
    {
        "cve_text": "The web application allows users to upload files without verifying the file type, enabling attackers to upload executable scripts that run on the server.",
        "expected_cwe": "CWE-434",
        "cwe_name": "Unrestricted Upload of File with Dangerous Type",
    },
    {
        "cve_text": "The application uses external input to construct file paths without properly neutralizing '../' sequences, allowing attackers to read arbitrary files outside the intended directory.",
        "expected_cwe": "CWE-22",
        "cwe_name": "Path Traversal",
    },
    {
        "cve_text": "A use-after-free vulnerability in the memory management routine allows attackers to execute arbitrary code by triggering specific allocation patterns after a free operation.",
        "expected_cwe": "CWE-416",
        "cwe_name": "Use After Free",
    },
    {
        "cve_text": "The application deserializes untrusted data from user requests without integrity checks, allowing attackers to achieve remote code execution by crafting malicious serialized objects.",
        "expected_cwe": "CWE-502",
        "cwe_name": "Deserialization of Untrusted Data",
    },
]

ATTACK_TECHNIQUE_SAMPLES = [
    {
        "text": "The adversary used PowerShell scripts to download and execute additional malware payloads on the compromised system.",
        "expected_technique": "T1059",
        "technique_name": "Command and Scripting Interpreter",
        "tactic": "Execution",
    },
    {
        "text": "The attackers obtained valid employee credentials through a previous breach and used them to access the corporate VPN without triggering alerts.",
        "expected_technique": "T1078",
        "technique_name": "Valid Accounts",
        "tactic": "Initial Access",
    },
    {
        "text": "The threat actor exploited a known vulnerability in the public-facing Apache Struts web server to gain initial access to the network.",
        "expected_technique": "T1190",
        "technique_name": "Exploit Public-Facing Application",
        "tactic": "Initial Access",
    },
    {
        "text": "The malware communicated with its command and control server by embedding data in DNS TXT record queries to avoid detection by network security tools.",
        "expected_technique": "T1071",
        "technique_name": "Application Layer Protocol",
        "tactic": "Command and Control",
    },
    {
        "text": "The ransomware encrypted all files on the victim's system using AES-256 and demanded payment in cryptocurrency for the decryption key.",
        "expected_technique": "T1486",
        "technique_name": "Data Encrypted for Impact",
        "tactic": "Impact",
    },
    {
        "text": "The attacker created a scheduled task that executed the backdoor every time the system booted, ensuring persistent access to the compromised machine.",
        "expected_technique": "T1053",
        "technique_name": "Scheduled Task/Job",
        "tactic": "Persistence",
    },
    {
        "text": "The malware injected its code into the address space of a legitimate running process to evade detection by endpoint security products.",
        "expected_technique": "T1055",
        "technique_name": "Process Injection",
        "tactic": "Defense Evasion",
    },
    {
        "text": "The payload was heavily obfuscated using multiple layers of base64 encoding and XOR encryption to avoid detection by antivirus software.",
        "expected_technique": "T1027",
        "technique_name": "Obfuscated Files or Information",
        "tactic": "Defense Evasion",
    },
]


def evaluate_cve_to_cwe(
    model: GhostLM,
    tokenizer: GhostTokenizer,
    device: str,
    max_tokens: int = 60,
) -> dict:
    """Evaluate the model's ability to classify CVE descriptions to CWE categories.

    Prompts the model with CVE descriptions and checks if the generated
    text contains the expected CWE identifier or related keywords.

    Args:
        model: GhostLM model in eval mode.
        tokenizer: GhostTokenizer instance.
        device: Device string.
        max_tokens: Max tokens to generate per sample.

    Returns:
        Dict with accuracy, total, correct, and per-sample results.
    """
    results = []
    correct = 0

    for sample in CVE_TO_CWE_SAMPLES:
        prompt = f"Classify this vulnerability: {sample['cve_text']}\nCWE category:"
        prompt_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

        generated = model.generate(input_tensor, max_new_tokens=max_tokens, temperature=0.5, top_k=20)
        generated_text = tokenizer.decode(generated[0].tolist())

        # Check if expected CWE ID or name appears in output
        output_text = generated_text[len(prompt):].lower()
        expected_id = sample["expected_cwe"].lower()
        expected_name = sample["cwe_name"].lower()
        hit = expected_id in output_text or expected_name in output_text

        if hit:
            correct += 1

        results.append({
            "cve_text": sample["cve_text"][:100] + "...",
            "expected": f"{sample['expected_cwe']} ({sample['cwe_name']})",
            "generated": generated_text[len(prompt):][:200],
            "correct": hit,
        })

    accuracy = correct / len(CVE_TO_CWE_SAMPLES) if CVE_TO_CWE_SAMPLES else 0
    return {
        "task": "cve_to_cwe_classification",
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": len(CVE_TO_CWE_SAMPLES),
        "samples": results,
    }


def evaluate_attack_tagging(
    model: GhostLM,
    tokenizer: GhostTokenizer,
    device: str,
    max_tokens: int = 60,
) -> dict:
    """Evaluate the model's ability to tag ATT&CK techniques from threat descriptions.

    Prompts the model with threat activity descriptions and checks
    if the generated text identifies the correct ATT&CK technique.

    Args:
        model: GhostLM model in eval mode.
        tokenizer: GhostTokenizer instance.
        device: Device string.
        max_tokens: Max tokens to generate per sample.

    Returns:
        Dict with accuracy, total, correct, and per-sample results.
    """
    results = []
    correct = 0

    for sample in ATTACK_TECHNIQUE_SAMPLES:
        prompt = f"Identify the MITRE ATT&CK technique: {sample['text']}\nTechnique:"
        prompt_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

        generated = model.generate(input_tensor, max_new_tokens=max_tokens, temperature=0.5, top_k=20)
        generated_text = tokenizer.decode(generated[0].tolist())

        output_text = generated_text[len(prompt):].lower()
        expected_id = sample["expected_technique"].lower()
        expected_name = sample["technique_name"].lower()
        hit = expected_id in output_text or expected_name in output_text

        if hit:
            correct += 1

        results.append({
            "text": sample["text"][:100] + "...",
            "expected": f"{sample['expected_technique']} ({sample['technique_name']})",
            "tactic": sample["tactic"],
            "generated": generated_text[len(prompt):][:200],
            "correct": hit,
        })

    accuracy = correct / len(ATTACK_TECHNIQUE_SAMPLES) if ATTACK_TECHNIQUE_SAMPLES else 0
    return {
        "task": "attack_technique_tagging",
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": len(ATTACK_TECHNIQUE_SAMPLES),
        "samples": results,
    }


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

    # Domain-specific evaluations
    print("\nRunning CVE-to-CWE classification...")
    cve_cwe_results = evaluate_cve_to_cwe(model, tokenizer, device)
    print(f"  Accuracy: {cve_cwe_results['accuracy']:.1%} ({cve_cwe_results['correct']}/{cve_cwe_results['total']})")

    print("\nRunning ATT&CK technique tagging...")
    attack_results = evaluate_attack_tagging(model, tokenizer, device)
    print(f"  Accuracy: {attack_results['accuracy']:.1%} ({attack_results['correct']}/{attack_results['total']})")

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
        "cve_to_cwe": cve_cwe_results,
        "attack_tagging": attack_results,
    }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation complete. Results saved to {output_path}")


if __name__ == "__main__":
    main()
