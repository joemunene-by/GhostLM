#!/usr/bin/env python3
"""Build MCQ-format training data for GhostLM chat-tuning.

Produces multiple-choice questions in the same JSONL chat format used by
``build_chat_dataset.py`` so the output can be concatenated with the regular
small_talk + cybersec Q&A. The assistant turn is intentionally a bare letter
(A / B / C / D), occasionally followed by a one-sentence reason — this
teaches the model to output a single letter after an "Answer:" cue, which
is exactly the format CTIBench / CyberMetric / similar benchmarks expect.

Three template families:

- **NVD CWE-class MCQ.** Extract the vulnerability class (XSS, SQLi, buffer
  overflow, use-after-free, etc.) from each NVD description via keyword
  matching. Build an MCQ where the correct answer is that class and the
  distractors are 3 other classes drawn from the same pool.
- **MITRE tactic MCQ.** Each ATT&CK technique has a ``Tactic:`` line. Build
  "Which tactic does {technique-id} belong to?" with the correct tactic and
  3 distractor tactics.
- **Definition matching MCQ.** Common acronyms (XSS, SSRF, CSRF, RCE, …) get
  "What does {acronym} stand for?" with the correct expansion and 3 plausible
  distractors.

Each MCQ is a single-turn chat record with assistant content = "A" / "B" /
"C" / "D" most of the time, plus a 30% mix where the letter is followed by a
brief justification (e.g. "B. Buffer overflow — the description mentions a
stack write past a fixed-size array.").
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Vulnerability class taxonomy — keyword → canonical name
# ---------------------------------------------------------------------------

VULN_CLASSES: List[Tuple[str, List[str]]] = [
    ("Cross-Site Scripting (XSS)", [
        "cross-site scripting", "cross site scripting", "xss",
        "stored xss", "reflected xss", "dom-based xss",
    ]),
    ("SQL Injection", [
        "sql injection", "sqli", "sql-injection",
    ]),
    ("Cross-Site Request Forgery (CSRF)", [
        "csrf", "cross-site request forgery", "cross site request forgery",
    ]),
    ("Server-Side Request Forgery (SSRF)", [
        "ssrf", "server-side request forgery", "server side request forgery",
    ]),
    ("XML External Entity (XXE)", [
        "xxe", "xml external entity",
    ]),
    ("Remote Code Execution (RCE)", [
        "remote code execution", " rce ", "arbitrary code execution",
    ]),
    ("Buffer Overflow", [
        "buffer overflow", "stack-based buffer overflow", "stack buffer overflow",
        "heap-based buffer overflow", "heap buffer overflow",
    ]),
    ("Use-After-Free", [
        "use-after-free", "use after free", " uaf ",
    ]),
    ("Out-of-Bounds Read", [
        "out-of-bounds read", "out of bounds read",
    ]),
    ("Out-of-Bounds Write", [
        "out-of-bounds write", "out of bounds write",
    ]),
    ("Integer Overflow", [
        "integer overflow", "integer wraparound",
    ]),
    ("Denial of Service (DoS)", [
        "denial of service", " dos ", "denial-of-service",
    ]),
    ("Path Traversal", [
        "path traversal", "directory traversal", "../",
    ]),
    ("Open Redirect", [
        "open redirect",
    ]),
    ("Authentication Bypass", [
        "authentication bypass", "auth bypass", "bypasses authentication",
    ]),
    ("Privilege Escalation", [
        "privilege escalation", "elevation of privilege",
    ]),
    ("Information Disclosure", [
        "information disclosure", "information leak", "sensitive information",
    ]),
    ("Race Condition", [
        "race condition", "toctou",
    ]),
    ("Insecure Deserialization", [
        "deserialization", "unsafe deserialization",
    ]),
    ("Memory Corruption", [
        "memory corruption", "heap corruption",
    ]),
]

VULN_CLASS_NAMES = [name for name, _ in VULN_CLASSES]


def classify_nvd(text: str) -> Optional[str]:
    """Return the canonical vulnerability class for an NVD description, or None.

    Matches the first taxonomy entry whose keywords appear in the text. We
    iterate in order so more specific terms (e.g. "stack-based buffer overflow")
    bind before more general ones.
    """
    lower = text.lower()
    for name, keywords in VULN_CLASSES:
        for kw in keywords:
            if kw in lower:
                return name
    return None


# ---------------------------------------------------------------------------
# MITRE tactics
# ---------------------------------------------------------------------------

MITRE_TACTICS = [
    "Initial Access",
    "Execution",
    "Persistence",
    "Privilege Escalation",
    "Defense Evasion",
    "Credential Access",
    "Discovery",
    "Lateral Movement",
    "Collection",
    "Command and Control",
    "Exfiltration",
    "Impact",
    "Reconnaissance",
    "Resource Development",
]

# Map the slug form ("defense-evasion") to display form ("Defense Evasion")
TACTIC_SLUG_TO_NAME = {t.lower().replace(" ", "-"): t for t in MITRE_TACTICS}


def parse_mitre_tactic(text: str) -> Optional[str]:
    """Extract the canonical tactic name from a MITRE technique description."""
    m = re.search(r"Tactic:\s*([\w\-,\s]+)", text)
    if not m:
        return None
    raw = m.group(1).strip().splitlines()[0].strip()
    # Some entries list multiple tactics separated by commas — take the first.
    first = raw.split(",")[0].strip().lower()
    return TACTIC_SLUG_TO_NAME.get(first) or TACTIC_SLUG_TO_NAME.get(first.replace(" ", "-"))


# ---------------------------------------------------------------------------
# Acronym definition pool
# ---------------------------------------------------------------------------

ACRONYMS: Dict[str, str] = {
    "XSS": "Cross-Site Scripting",
    "SQLi": "SQL Injection",
    "SSRF": "Server-Side Request Forgery",
    "CSRF": "Cross-Site Request Forgery",
    "XXE": "XML External Entity",
    "RCE": "Remote Code Execution",
    "DoS": "Denial of Service",
    "DDoS": "Distributed Denial of Service",
    "MITM": "Man-in-the-Middle attack",
    "UAF": "Use-After-Free",
    "TOCTOU": "Time-of-Check to Time-of-Use",
    "CVE": "Common Vulnerabilities and Exposures",
    "CWE": "Common Weakness Enumeration",
    "CVSS": "Common Vulnerability Scoring System",
    "CAPEC": "Common Attack Pattern Enumeration and Classification",
    "ATT&CK": "Adversarial Tactics, Techniques, and Common Knowledge",
    "OWASP": "Open Worldwide Application Security Project",
    "CTF": "Capture The Flag",
    "SOC": "Security Operations Center",
    "SIEM": "Security Information and Event Management",
    "EDR": "Endpoint Detection and Response",
    "IDS": "Intrusion Detection System",
    "IPS": "Intrusion Prevention System",
    "WAF": "Web Application Firewall",
    "PII": "Personally Identifiable Information",
    "MFA": "Multi-Factor Authentication",
    "JWT": "JSON Web Token",
    "API": "Application Programming Interface",
    "VPN": "Virtual Private Network",
    "TLS": "Transport Layer Security",
    "SSL": "Secure Sockets Layer",
    "DNS": "Domain Name System",
    "TCP": "Transmission Control Protocol",
    "UDP": "User Datagram Protocol",
    "ICMP": "Internet Control Message Protocol",
    "DHCP": "Dynamic Host Configuration Protocol",
    "FTP": "File Transfer Protocol",
    "SMTP": "Simple Mail Transfer Protocol",
    "ARP": "Address Resolution Protocol",
    "BIOS": "Basic Input/Output System",
    "UEFI": "Unified Extensible Firmware Interface",
    "ASLR": "Address Space Layout Randomization",
    "DEP": "Data Execution Prevention",
    "NX": "No-Execute (memory protection)",
    "PAM": "Pluggable Authentication Modules",
    "LDAP": "Lightweight Directory Access Protocol",
    "SAML": "Security Assertion Markup Language",
    "OAuth": "Open Authorization",
    "OIDC": "OpenID Connect",
}


# ---------------------------------------------------------------------------
# MCQ assembly helpers
# ---------------------------------------------------------------------------

LETTERS = ["A", "B", "C", "D"]


def make_mcq(
    question: str,
    correct: str,
    distractors: List[str],
    rng: random.Random,
) -> Dict:
    """Shuffle correct + 3 distractors into A/B/C/D, return the chat record.

    The assistant turn is just the letter — that's the format we want the
    model to learn. A 30% subset gets a one-line justification appended so
    the model still sees the "letter, then explanation" pattern occasionally.
    """
    options = [correct] + list(distractors)
    rng.shuffle(options)
    correct_idx = options.index(correct)
    correct_letter = LETTERS[correct_idx]

    body = (
        "Pick the best answer (A, B, C, or D) for this multiple-choice "
        "cybersecurity question.\n\n"
        f"Question: {question}\n\n"
        + "\n".join(f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options))
        + "\n\nAnswer:"
    )

    # 30% get a one-sentence justification appended after the letter.
    if rng.random() < 0.30:
        assistant = f"{correct_letter}. {correct}."
    else:
        assistant = correct_letter

    return {
        "turns": [
            {"role": "user", "content": body},
            {"role": "assistant", "content": assistant},
        ],
        "source": "mcq",
    }


# ---------------------------------------------------------------------------
# Per-source builders
# ---------------------------------------------------------------------------


def build_nvd_mcqs(records: List[dict], target: int, rng: random.Random) -> List[Dict]:
    """Sample NVD records and produce CWE-class MCQs.

    Only includes records where ``classify_nvd`` returns a class.
    """
    classified: List[Tuple[str, str]] = []
    for r in records:
        cls = classify_nvd(r.get("text", ""))
        if cls and r.get("id"):
            classified.append((r["id"], cls))
    if not classified:
        return []

    # Balance across classes — sample more from underrepresented buckets.
    by_class: Dict[str, List[str]] = {}
    for cve, cls in classified:
        by_class.setdefault(cls, []).append(cve)

    # Round-robin sample
    out: List[Dict] = []
    classes_list = list(by_class.keys())
    rng.shuffle(classes_list)
    while len(out) < target:
        progressed = False
        for cls in classes_list:
            if not by_class[cls]:
                continue
            cve = by_class[cls].pop()
            distractor_pool = [c for c in VULN_CLASS_NAMES if c != cls]
            distractors = rng.sample(distractor_pool, k=3)
            question = f"What type of vulnerability is {cve}?"
            out.append(make_mcq(question, cls, distractors, rng))
            if len(out) >= target:
                break
            progressed = True
        if not progressed:
            break
    return out


def build_mitre_mcqs(records: List[dict], rng: random.Random) -> List[Dict]:
    """Build "which tactic does {technique} belong to?" MCQs."""
    out: List[Dict] = []
    for r in records:
        text = r.get("text", "")
        # First line: "MITRE ATT&CK Technique T1234: Title"
        m = re.match(r"MITRE ATT&CK Technique (T[\d.]+):\s*(.+)", text)
        if not m:
            continue
        tid, tname = m.group(1), m.group(2).strip()
        tactic = parse_mitre_tactic(text)
        if not tactic:
            continue
        distractor_pool = [t for t in MITRE_TACTICS if t != tactic]
        distractors = rng.sample(distractor_pool, k=3)
        question = f"Which MITRE ATT&CK tactic does {tid} ({tname}) belong to?"
        out.append(make_mcq(question, tactic, distractors, rng))
    return out


def build_acronym_mcqs(rng: random.Random, copies: int = 2) -> List[Dict]:
    """One MCQ per acronym, replicated ``copies`` times with fresh shuffles."""
    out: List[Dict] = []
    items = list(ACRONYMS.items())
    distractor_pool = list(ACRONYMS.values())
    for _ in range(copies):
        for acronym, expansion in items:
            distractors = rng.sample(
                [d for d in distractor_pool if d != expansion],
                k=3,
            )
            question = f"What does {acronym} stand for in cybersecurity?"
            out.append(make_mcq(question, expansion, distractors, rng))
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="Build MCQ-format chat training data")
    p.add_argument("--corpus", default="data/processed/train.jsonl")
    p.add_argument("--out", default="data/raw/chat/mcq.jsonl")
    p.add_argument("--nvd-target", type=int, default=1000,
                   help="Number of NVD CWE-class MCQs to emit")
    p.add_argument("--acronym-copies", type=int, default=3,
                   help="How many shuffles of the acronym pool")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    """Build and write the MCQ dataset."""
    args = parse_args()
    rng = random.Random(args.seed)

    records: List[dict] = []
    with open(args.corpus, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    by_source: Dict[str, List[dict]] = {}
    for r in records:
        by_source.setdefault(r.get("source", "?"), []).append(r)

    print("Source counts:")
    for s, rs in sorted(by_source.items(), key=lambda kv: -len(kv[1])):
        print(f"  {s}: {len(rs):,}")

    mcqs: List[Dict] = []

    nvd_mcqs = build_nvd_mcqs(by_source.get("nvd", []), args.nvd_target, rng)
    print(f"\nNVD CWE-class MCQs: {len(nvd_mcqs)}")
    mcqs.extend(nvd_mcqs)

    mitre_mcqs = build_mitre_mcqs(by_source.get("mitre_attack", []), rng)
    print(f"MITRE tactic MCQs: {len(mitre_mcqs)}")
    mcqs.extend(mitre_mcqs)

    acro_mcqs = build_acronym_mcqs(rng, copies=args.acronym_copies)
    print(f"Acronym MCQs: {len(acro_mcqs)}")
    mcqs.extend(acro_mcqs)

    rng.shuffle(mcqs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for m in mcqs:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"\nTotal: {len(mcqs):,} MCQs → {out_path}")
    answer_dist = Counter(m["turns"][1]["content"][0] for m in mcqs)
    print(f"Answer-letter distribution: {dict(answer_dist)}")


if __name__ == "__main__":
    main()
