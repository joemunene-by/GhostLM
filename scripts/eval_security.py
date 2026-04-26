"""GhostLM cybersecurity evaluation — tests the model on security-specific classification tasks."""

import argparse
import json
import math
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.tokenizer import GhostTokenizer


# ---------------------------------------------------------------------------
# Task 1: CVE Severity Classification
#   Given a CVE-style description, classify as Critical / High / Medium / Low.
# ---------------------------------------------------------------------------
CVE_SEVERITY_SAMPLES = [
    {
        "description": (
            "A remote code execution vulnerability exists in the HTTP protocol stack (http.sys) "
            "that allows an unauthenticated attacker to send specially crafted packets to a "
            "targeted server to execute arbitrary code with SYSTEM privileges. No user interaction "
            "is required and the attack can be carried out over the network without authentication."
        ),
        "label": "Critical",
    },
    {
        "description": (
            "A heap-based buffer overflow in the SIP module of a firewall appliance allows an "
            "unauthenticated remote attacker to achieve remote code execution by sending crafted "
            "SIP packets. The vulnerability exists due to insufficient validation of SIP message "
            "headers. CVSS base score 9.8."
        ),
        "label": "Critical",
    },
    {
        "description": (
            "An SQL injection vulnerability in the admin panel of a web application allows an "
            "authenticated administrator to execute arbitrary SQL commands via the user search "
            "parameter. Successful exploitation requires valid administrator credentials and "
            "could lead to data exfiltration from the backend database."
        ),
        "label": "High",
    },
    {
        "description": (
            "A privilege escalation vulnerability in the kernel allows a local attacker with "
            "low-privilege user access to gain root privileges by exploiting a race condition "
            "in the filesystem mount handling code. The attacker must have local access to the "
            "system and can exploit the vulnerability to gain full control."
        ),
        "label": "High",
    },
    {
        "description": (
            "A stored cross-site scripting vulnerability exists in the user profile page of the "
            "web application. An authenticated user can inject malicious JavaScript code via the "
            "bio field that executes in other users' browsers when they view the profile. The "
            "impact is limited to session hijacking within the application context."
        ),
        "label": "Medium",
    },
    {
        "description": (
            "An information disclosure vulnerability in the API endpoint allows an authenticated "
            "user to enumerate valid usernames by observing differences in response times between "
            "valid and invalid usernames. The vulnerability does not directly expose sensitive data "
            "but could assist in targeted attacks."
        ),
        "label": "Medium",
    },
    {
        "description": (
            "A denial-of-service vulnerability exists in the XML parser that can be triggered by "
            "sending a specially crafted XML document with deeply nested entities. The parser "
            "consumes excessive memory and CPU but the service automatically recovers after the "
            "malicious request is processed. No data is exposed or modified."
        ),
        "label": "Medium",
    },
    {
        "description": (
            "The application discloses its software version number in HTTP response headers. "
            "While this information could help an attacker identify known vulnerabilities for "
            "the specific version, it does not directly enable exploitation. The information "
            "is limited to the server software name and version string."
        ),
        "label": "Low",
    },
    {
        "description": (
            "A self-XSS vulnerability exists where a user can inject JavaScript code that only "
            "executes in their own browser session. The attack cannot be triggered remotely or "
            "against other users and requires the victim to manually paste the malicious code "
            "into their own browser console or form field."
        ),
        "label": "Low",
    },
    {
        "description": (
            "The application uses a cookie without the Secure flag set, which means the cookie "
            "could be transmitted over an unencrypted HTTP connection if the user navigates to "
            "the HTTP version of the site. The cookie does not contain sensitive session data."
        ),
        "label": "Low",
    },
]

# ---------------------------------------------------------------------------
# Task 2: Vulnerability Type Detection
#   Given a description, identify the vulnerability class.
# ---------------------------------------------------------------------------
VULN_TYPE_SAMPLES = [
    {
        "description": (
            "The application constructs SQL queries by directly concatenating user input from "
            "the search field without parameterization. An attacker can inject UNION SELECT "
            "statements to extract data from other tables. The input field was not properly "
            "sanitized allowing malicious SQL statements to be executed against the database."
        ),
        "label": "SQL Injection",
    },
    {
        "description": (
            "The web application reflects user-supplied input from the URL parameter directly "
            "into the HTML response without encoding. An attacker can craft a URL containing "
            "JavaScript code that executes in the victim's browser when they click the link, "
            "potentially stealing session cookies or performing actions on behalf of the user."
        ),
        "label": "Cross-Site Scripting (XSS)",
    },
    {
        "description": (
            "The C program uses the gets() function to read user input into a fixed-size 64-byte "
            "character array on the stack. An attacker can provide input exceeding 64 bytes to "
            "overwrite the saved return address and redirect execution to shellcode placed in the "
            "buffer. The binary lacks stack canaries and ASLR protections."
        ),
        "label": "Buffer Overflow",
    },
    {
        "description": (
            "The web application sends a request to a user-supplied URL to fetch remote content "
            "for preview functionality. An attacker can provide an internal IP address such as "
            "169.254.169.254 to access the cloud metadata service and retrieve IAM credentials, "
            "or scan internal network services not accessible from the internet."
        ),
        "label": "Server-Side Request Forgery (SSRF)",
    },
    {
        "description": (
            "The application accepts serialized Java objects from untrusted sources and "
            "deserializes them without validation. An attacker can craft a malicious serialized "
            "object using gadget chains from common libraries like Apache Commons Collections "
            "to achieve remote code execution when the object is deserialized by the application."
        ),
        "label": "Insecure Deserialization",
    },
    {
        "description": (
            "The application's password reset functionality uses a predictable token generated "
            "from the current timestamp. An attacker can predict valid reset tokens by knowing "
            "the approximate time the reset was requested, allowing them to reset any user's "
            "password without access to their email account."
        ),
        "label": "Broken Authentication",
    },
    {
        "description": (
            "The application allows users to upload files but only checks the file extension on "
            "the client side. An attacker can bypass this by intercepting the request and changing "
            "the filename to include a .php extension, uploading a web shell that provides remote "
            "command execution on the server when accessed via the web."
        ),
        "label": "Unrestricted File Upload",
    },
    {
        "description": (
            "The API endpoint /api/users/123/profile returns the full user profile including "
            "sensitive fields when accessed with any valid authentication token. There is no "
            "check to verify that the authenticated user has permission to view user 123's "
            "profile, allowing any authenticated user to access any other user's data."
        ),
        "label": "Broken Access Control (IDOR)",
    },
    {
        "description": (
            "The application stores user passwords using the MD5 hashing algorithm without salt. "
            "The MD5 algorithm is computationally fast and vulnerable to rainbow table attacks. "
            "An attacker who gains access to the password database can quickly recover plaintext "
            "passwords for the majority of users using precomputed hash tables."
        ),
        "label": "Cryptographic Failure",
    },
    {
        "description": (
            "The application parses user-supplied XML input with external entity processing "
            "enabled. An attacker can define an external entity referencing a local file such as "
            "/etc/passwd and include it in the XML document. The parser resolves the entity and "
            "returns the file contents in the response, enabling arbitrary file reading."
        ),
        "label": "XML External Entity (XXE)",
    },
]

# ---------------------------------------------------------------------------
# Task 3: Attack Technique Identification
#   Given a scenario, identify the ATT&CK-style technique being used.
# ---------------------------------------------------------------------------
ATTACK_TECHNIQUE_SAMPLES = [
    {
        "description": (
            "The adversary sent a targeted email to an employee in the finance department "
            "containing a malicious Excel attachment with an embedded macro. When the employee "
            "opened the attachment and enabled macros, a PowerShell command was executed that "
            "downloaded and ran a second-stage payload from a remote server."
        ),
        "label": "Spearphishing Attachment",
    },
    {
        "description": (
            "After gaining initial access, the attacker used Mimikatz to extract plaintext "
            "passwords and NTLM hashes from the LSASS process memory on the compromised "
            "workstation. These credentials were then used to authenticate to other systems "
            "on the network without triggering password brute-force detection."
        ),
        "label": "Credential Dumping",
    },
    {
        "description": (
            "The malware established persistence by creating a new Windows Registry Run key "
            "at HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run pointing to the malicious "
            "executable in the user's AppData directory. This ensured the malware would "
            "automatically execute every time the user logged into the system."
        ),
        "label": "Registry Run Keys / Startup Folder",
    },
    {
        "description": (
            "The attacker used PsExec to remotely execute commands on multiple systems within "
            "the network using the domain administrator credentials obtained earlier. The tool "
            "connected to the target systems' ADMIN$ share and deployed a service executable "
            "to run the attacker's commands with SYSTEM-level privileges."
        ),
        "label": "Lateral Movement via Remote Services",
    },
    {
        "description": (
            "The threat actor exfiltrated collected data by encoding it in DNS TXT record "
            "queries sent to a domain they controlled. Each query contained a chunk of "
            "base64-encoded stolen data as a subdomain label, effectively tunneling data out "
            "of the network through the DNS protocol which was allowed through the firewall."
        ),
        "label": "Exfiltration Over DNS",
    },
    {
        "description": (
            "The adversary replaced a legitimate DLL in the application's installation directory "
            "with a malicious version. When the application launched, it loaded the attacker's "
            "DLL instead of the legitimate one due to the Windows DLL search order, executing "
            "malicious code in the context of the trusted application."
        ),
        "label": "DLL Search Order Hijacking",
    },
    {
        "description": (
            "The attacker scheduled a Windows Task Scheduler job to run a PowerShell script "
            "every 6 hours that beaconed to the command and control server. The scheduled task "
            "was given a name similar to a legitimate Windows maintenance task to avoid detection "
            "during routine system administration."
        ),
        "label": "Scheduled Task / Job",
    },
    {
        "description": (
            "After compromising the build server, the attacker modified the CI/CD pipeline "
            "configuration to inject a malicious dependency into the software build process. "
            "Every subsequent build included a backdoor that was distributed to all customers "
            "through the normal software update mechanism."
        ),
        "label": "Supply Chain Compromise",
    },
    {
        "description": (
            "The malware disabled Windows Defender by modifying the registry key "
            "DisableAntiSpyware and terminated the MsMpEng.exe process. It also cleared "
            "Windows Event Logs using wevtutil to remove evidence of its activity, and "
            "disabled the Windows Firewall to allow unrestricted network communication."
        ),
        "label": "Defense Evasion / Impair Defenses",
    },
    {
        "description": (
            "The attacker used a compromised web server as a proxy to relay commands to "
            "internal systems that had no direct internet access. HTTPS traffic to the web "
            "server appeared as normal web traffic to network monitoring tools. The encrypted "
            "channel carried command and control instructions disguised as standard API calls."
        ),
        "label": "Proxy / Web Service C2",
    },
]


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------

SEVERITY_LEVELS = ["Critical", "High", "Medium", "Low"]
VULN_TYPES = [s["label"] for s in VULN_TYPE_SAMPLES]
ATTACK_TECHNIQUES = [s["label"] for s in ATTACK_TECHNIQUE_SAMPLES]


def load_model(checkpoint_path: str, device: str) -> Tuple[GhostLM, GhostLMConfig]:
    """Load a GhostLM model from checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Target device string.

    Returns:
        Tuple of (model in eval mode, config).
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


def score_candidate(
    model: GhostLM,
    tokenizer: GhostTokenizer,
    prompt_ids: List[int],
    candidate: str,
    device: str,
    context_length: int,
    aggregate: str = "mean",
) -> float:
    """Compute log-probability of a candidate completion given a prompt.

    Concatenates the prompt token IDs with the candidate token IDs, runs the
    model forward pass, and aggregates the log-probabilities over the
    candidate token positions only.

    Args:
        model: GhostLM model in eval mode.
        tokenizer: GhostTokenizer instance.
        prompt_ids: Pre-encoded token IDs for the prompt.
        candidate: Candidate text string to score.
        device: Device string.
        context_length: Maximum sequence length for the model.
        aggregate: ``"mean"`` averages log-prob over candidate tokens
            (length-normalized; the historical default). ``"sum"`` returns
            the total log-prob and is the right choice for PMI scoring,
            where the same candidate is scored under two prompts of equal
            candidate length and length normalization would cancel out.

    Returns:
        Aggregated log-probability (higher is better, less negative).
    """
    cand_ids = tokenizer.encode(candidate)
    if not cand_ids:
        return float("-inf")

    full_ids = prompt_ids + cand_ids
    # Truncate from the left if too long, keeping the end (candidate) intact
    if len(full_ids) > context_length:
        full_ids = full_ids[-context_length:]
        cand_len = len(cand_ids)
    else:
        cand_len = len(cand_ids)

    input_ids = full_ids[:-1]
    target_ids = full_ids[1:]

    x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(target_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        logits, _ = model(x, targets=y)

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    start = max(0, len(target_ids) - cand_len)
    total_logp = 0.0
    count = 0
    for i in range(start, len(target_ids)):
        token_id = target_ids[i]
        total_logp += log_probs[0, i, token_id].item()
        count += 1

    if count == 0:
        return float("-inf")

    if aggregate == "sum":
        return total_logp
    return total_logp / count


def classify(
    model: GhostLM,
    tokenizer: GhostTokenizer,
    description: str,
    candidates: List[str],
    task_prompt: str,
    device: str,
    context_length: int,
    scoring: str = "pmi",
) -> Tuple[str, Dict[str, float]]:
    """Classify a description by scoring each candidate label.

    Two scoring modes:

    * ``"pmi"`` (default) — pointwise mutual information. Score is
      log P(candidate | task_prompt + description) - log P(candidate | task_prompt).
      Subtracting the task-prompt-only baseline cancels the model's
      unconditional prior toward common labels (the failure mode where
      the model picks "High" or "XSS" for every sample regardless of
      input). Both terms use ``aggregate="sum"`` so length normalization
      does not interfere.

    * ``"logp"`` — historical length-normalized log-probability of
      candidate given full prompt. Mode-collapses when one candidate's
      tokens are unconditionally more likely than the others, which is
      exactly what we observed across Phases 1-3 (4/30 = 13.3%, below
      the 15% random baseline).

    Args:
        model: GhostLM model in eval mode.
        tokenizer: GhostTokenizer instance.
        description: The text to classify.
        candidates: List of candidate label strings.
        task_prompt: Task-specific instruction prefix.
        device: Device string.
        context_length: Maximum sequence length for the model.
        scoring: ``"pmi"`` or ``"logp"``.

    Returns:
        ``(best_label, scores)`` — the winning label plus the per-candidate
        score dict (useful for debugging and per-sample inspection).
    """
    full_prompt = f"{task_prompt}\n\nDescription: {description}\n\nClassification:"
    full_ids = tokenizer.encode(full_prompt)

    baseline_ids = None
    if scoring == "pmi":
        baseline_prompt = f"{task_prompt}\n\nClassification:"
        baseline_ids = tokenizer.encode(baseline_prompt)

    aggregate = "sum" if scoring == "pmi" else "mean"

    scores: Dict[str, float] = {}
    for cand in candidates:
        cand_text = f" {cand}"
        conditional = score_candidate(
            model, tokenizer, full_ids, cand_text, device, context_length,
            aggregate=aggregate,
        )
        if scoring == "pmi":
            unconditional = score_candidate(
                model, tokenizer, baseline_ids, cand_text, device, context_length,
                aggregate=aggregate,
            )
            scores[cand] = conditional - unconditional
        else:
            scores[cand] = conditional

    best_label = max(scores.items(), key=lambda kv: kv[1])[0]
    return best_label, scores


def run_task(
    task_name: str,
    samples: List[Dict],
    candidates: List[str],
    task_prompt: str,
    model: GhostLM,
    tokenizer: GhostTokenizer,
    device: str,
    context_length: int,
    scoring: str = "pmi",
) -> Dict:
    """Run a classification task over a set of samples and return accuracy metrics."""
    correct = 0
    total = len(samples)
    details = []

    for sample in samples:
        predicted, scores = classify(
            model, tokenizer, sample["description"], candidates,
            task_prompt, device, context_length, scoring=scoring,
        )
        is_correct = predicted == sample["label"]
        if is_correct:
            correct += 1

        details.append({
            "expected": sample["label"],
            "predicted": predicted,
            "correct": is_correct,
            # Round for clean JSON without losing useful precision
            "scores": {k: round(v, 3) for k, v in scores.items()},
        })

    accuracy = correct / total if total > 0 else 0.0
    # Distribution check — flags mode-collapse, the failure mode that
    # killed the previous eval. If one label was predicted >70% of the
    # time the eval is not actually discriminating, regardless of
    # accuracy.
    pred_counts: Dict[str, int] = {}
    for d in details:
        pred_counts[d["predicted"]] = pred_counts.get(d["predicted"], 0) + 1
    most_common_share = max(pred_counts.values()) / total if total > 0 else 0.0

    return {
        "task": task_name,
        "correct": correct,
        "total": total,
        "accuracy": accuracy,
        "scoring": scoring,
        "prediction_distribution": pred_counts,
        "most_common_share": round(most_common_share, 3),
        "details": details,
    }


def print_scorecard(results: List[Dict], elapsed: float) -> None:
    """Print a formatted score card summarizing all evaluation tasks.

    Args:
        results: List of task result dicts from run_task.
        elapsed: Total evaluation time in seconds.
    """
    print("\n" + "=" * 60)
    print("GhostLM Cybersecurity Evaluation Score Card")
    print("=" * 60)
    print(f"{'Task':<40} {'Correct':>8} {'Total':>6} {'Accuracy':>10}")
    print("-" * 60)

    total_correct = 0
    total_samples = 0

    for r in results:
        print(f"{r['task']:<40} {r['correct']:>8} {r['total']:>6} {r['accuracy']:>9.1%}")
        total_correct += r["correct"]
        total_samples += r["total"]

    overall = total_correct / total_samples if total_samples > 0 else 0.0

    print("-" * 60)
    print(f"{'OVERALL':<40} {total_correct:>8} {total_samples:>6} {overall:>9.1%}")
    print("=" * 60)
    print(f"Time: {elapsed:.1f}s")
    print()

    # Print per-sample details
    for r in results:
        print(f"\n--- {r['task']} ---")
        for i, d in enumerate(r["details"]):
            status = "PASS" if d["correct"] else "FAIL"
            print(f"  [{status}] Expected: {d['expected']:<35} Predicted: {d['predicted']}")


def main():
    """Run the full cybersecurity evaluation suite against a GhostLM checkpoint.

    Evaluates three tasks: CVE severity classification, vulnerability type
    detection, and attack technique identification. Prints a score card and
    optionally saves results to JSON.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate GhostLM on cybersecurity classification tasks"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to GhostLM checkpoint (uses random init if omitted)",
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
        default="logs/eval_security.json",
        help="Where to save evaluation results",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        choices=["pmi", "logp"],
        default="pmi",
        help=(
            "Candidate-scoring strategy. 'pmi' (default) subtracts the "
            "unconditional log-prob from the conditional log-prob — fixes "
            "the mode-collapse failure mode where the model picked the "
            "same label for every sample. 'logp' is the historical "
            "length-normalized scorer kept for back-compat / regression "
            "comparison."
        ),
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

    print("=" * 60)
    print("GhostLM Cybersecurity Evaluation")
    print("=" * 60)
    print(f"Device: {device}")

    t0 = time.time()

    # Load model
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading model from {args.checkpoint}...")
        model, config = load_model(args.checkpoint, device)
    else:
        print("No checkpoint provided -- using random ghost-small init...")
        config = GhostLMConfig.from_preset("ghost-small")
        config.vocab_size = 50261
        model = GhostLM(config)
        model.eval()
        model = model.to(device)

    tokenizer = GhostTokenizer()
    context_length = config.context_length

    # Run evaluation tasks
    results = []

    print(f"Scoring: {args.scoring}")

    print("\n[1/3] CVE Severity Classification...")
    results.append(run_task(
        task_name="CVE Severity Classification",
        samples=CVE_SEVERITY_SAMPLES,
        candidates=SEVERITY_LEVELS,
        task_prompt=(
            "Classify the severity of the following security vulnerability as one of: "
            "Critical, High, Medium, or Low."
        ),
        model=model,
        tokenizer=tokenizer,
        device=device,
        context_length=context_length,
        scoring=args.scoring,
    ))

    r = results[-1]
    print(f"  Accuracy: {r['accuracy']:.1%}  (most-common-share: {r['most_common_share']:.0%})")

    print("\n[2/3] Vulnerability Type Detection...")
    results.append(run_task(
        task_name="Vulnerability Type Detection",
        samples=VULN_TYPE_SAMPLES,
        candidates=VULN_TYPES,
        task_prompt=(
            "Identify the type of security vulnerability described below. Choose from: "
            + ", ".join(VULN_TYPES) + "."
        ),
        model=model,
        tokenizer=tokenizer,
        device=device,
        context_length=context_length,
        scoring=args.scoring,
    ))

    r = results[-1]
    print(f"  Accuracy: {r['accuracy']:.1%}  (most-common-share: {r['most_common_share']:.0%})")

    print("\n[3/3] Attack Technique Identification...")
    results.append(run_task(
        task_name="Attack Technique Identification",
        samples=ATTACK_TECHNIQUE_SAMPLES,
        candidates=ATTACK_TECHNIQUES,
        task_prompt=(
            "Identify the attack technique being used in the following scenario. Choose from: "
            + ", ".join(ATTACK_TECHNIQUES) + "."
        ),
        model=model,
        tokenizer=tokenizer,
        device=device,
        context_length=context_length,
        scoring=args.scoring,
    ))

    r = results[-1]
    print(f"  Accuracy: {r['accuracy']:.1%}  (most-common-share: {r['most_common_share']:.0%})")

    elapsed = time.time() - t0

    # Print score card
    print_scorecard(results, elapsed)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "device": device,
        "checkpoint": args.checkpoint,
        "scoring": args.scoring,
        "elapsed_seconds": round(elapsed, 1),
        "tasks": [
            {
                "task": r["task"],
                "scoring": r["scoring"],
                "correct": r["correct"],
                "total": r["total"],
                "accuracy": round(r["accuracy"], 4),
                "prediction_distribution": r["prediction_distribution"],
                "most_common_share": r["most_common_share"],
                "details": r["details"],
            }
            for r in results
        ],
        "overall_accuracy": round(
            sum(r["correct"] for r in results) / sum(r["total"] for r in results), 4
        ),
    }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
