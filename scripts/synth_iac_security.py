#!/usr/bin/env python3
"""Templated synthesis of cloud-IaC-security training records (bet 11).

Bet 11 ([docs/differentiation.md](differentiation.md) §"Bet 11: cloud
IaC security") trains ghost-base on the analyst-developer overlap:
read a Terraform / CloudFormation / Kubernetes manifest, identify
the misconfiguration, propose the secure version. This is where
security shifts left, where the SRE / DevSecOps / cloud-engineer
audience overlaps with the SOC analyst, and where a small LM that
knows the patterns is genuinely useful at PR-review time.

Seed: ``data/raw/iac_security_patterns.jsonl``, a hand-curated
bank of 15 (platform, rule_id, name, vulnerable, patched,
explanation, severity, cwe_examples) tuples covering AWS S3 / IAM
/ Security Group / RDS / EBS / CloudFront / ALB+WAF, Kubernetes
Pod / Network / RBAC / Secret resources.

Output formats per pattern:

  1. ``pretrain_prose``    flat markdown article with the platform,
                            rule, vulnerable code, patched code,
                            explanation, severity. Right shape for
                            pretrain mixing.
  2. ``identify_and_fix``  chat Q&A. USER pastes the vulnerable
                            manifest, asks 'what is wrong + how to
                            fix'; ASSISTANT names the issue, shows
                            the patched version, explains.
  3. ``explain_the_diff``  chat Q&A. USER pastes both versions, asks
                            'why is the second safer'; ASSISTANT
                            explains the security property delta.
  4. ``severity_mapping``  chat Q&A. USER pastes the vulnerable code,
                            asks 'how serious is this and which CWE';
                            ASSISTANT cites severity + CWE + rationale.

15 patterns x 4 variants = 60 records.

Run:

    PYTHONPATH=. python3 scripts/synth_iac_security.py \\
        --bank data/raw/iac_security_patterns.jsonl \\
        --out data/processed/synth_iac_security.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_record(seed_id: str, variant: str, text: str) -> Dict[str, str]:
    h = hashlib.sha1(
        f"{seed_id}\n{variant}\n{text}".encode("utf-8")
    ).hexdigest()[:10]
    return {
        "id": f"synth_iac_security#{seed_id}#{variant}#{h}",
        "source": "synth_iac_security",
        "teacher": "templated",
        "seed_source": variant,
        "seed_id": seed_id,
        "text": text,
    }


def fence(code: str, lang: str = "") -> str:
    return f"```{lang}\n{code}\n```"


def cwe_phrase(cwes: List[str]) -> str:
    if not cwes:
        return ""
    if len(cwes) == 1:
        return f" Maps to {cwes[0]}."
    return f" Maps to {', '.join(cwes)}."


def lang_for(platform: str) -> str:
    if platform.startswith("terraform"):
        return "hcl"
    if platform.startswith("kubernetes"):
        return "yaml"
    if platform.startswith("cloudformation"):
        return "yaml"
    return ""


def pretrain_prose(p: Dict) -> str:
    lang = lang_for(p["platform"])
    return (
        f"# IaC misconfig: {p['name']} ({p['rule_id']})\n\n"
        f"Platform: `{p['platform']}`. Severity: **{p['severity']}**."
        f"{cwe_phrase(p.get('cwe_examples', []))}\n\n"
        f"## Vulnerable\n\n{fence(p['vulnerable'], lang)}\n\n"
        f"## Patched\n\n{fence(p['patched'], lang)}\n\n"
        f"## Why this matters\n\n{p['explanation']}\n"
    )


def identify_and_fix(p: Dict) -> str:
    lang = lang_for(p["platform"])
    return (
        f"USER: Look at this {p['platform']} snippet. What is the "
        f"security misconfiguration, and how would you fix it?\n\n"
        f"{fence(p['vulnerable'], lang)}\n\n"
        f"ASSISTANT: This is {p['name']} ({p['rule_id']}, severity "
        f"{p['severity']}). {p['explanation']}\n\n"
        f"Patched version:\n\n"
        f"{fence(p['patched'], lang)}"
        f"{cwe_phrase(p.get('cwe_examples', []))}\n"
    )


def explain_the_diff(p: Dict) -> str:
    lang = lang_for(p["platform"])
    return (
        f"USER: Here are two versions of the same {p['platform']} "
        f"resource. Why is the second one safer?\n\n"
        f"Version A:\n\n{fence(p['vulnerable'], lang)}\n\n"
        f"Version B:\n\n{fence(p['patched'], lang)}\n\n"
        f"ASSISTANT: Version A is vulnerable to {p['name']} "
        f"({p['rule_id']}). {p['explanation']}\n"
    )


def severity_mapping(p: Dict) -> str:
    lang = lang_for(p["platform"])
    sentences = [s.strip() for s in p["explanation"].split(".") if s.strip()]
    rationale = ". ".join(sentences[:2]) + "."
    return (
        f"USER: How serious is the misconfiguration in this "
        f"{p['platform']} snippet, and what CWE class does it map to?\n\n"
        f"{fence(p['vulnerable'], lang)}\n\n"
        f"ASSISTANT: This is {p['name']} ({p['rule_id']}). "
        f"Severity: **{p['severity']}**."
        f"{cwe_phrase(p.get('cwe_examples', []))} {rationale}\n"
    )


VARIANTS = [
    ("pretrain_prose", pretrain_prose),
    ("identify_and_fix", identify_and_fix),
    ("explain_the_diff", explain_the_diff),
    ("severity_mapping", severity_mapping),
]


def quality_ok(text: str, min_words: int = 60, max_words: int = 1800) -> bool:
    words = text.split()
    if not (min_words <= len(words) <= max_words):
        return False
    if "```" not in text:
        return False
    return True


def stream_bank(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", default="data/raw/iac_security_patterns.jsonl")
    p.add_argument("--out", default="data/processed/synth_iac_security.jsonl")
    p.add_argument("--variants", default=",".join(v[0] for v in VARIANTS))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bank_path = REPO_ROOT / args.bank if not Path(args.bank).is_absolute() \
                else Path(args.bank)
    out_path = REPO_ROOT / args.out if not Path(args.out).is_absolute() \
               else Path(args.out)
    if not bank_path.exists():
        sys.exit(f"pattern bank not found: {bank_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wanted = {v.strip() for v in args.variants.split(",") if v.strip()}
    variants = [(name, fn) for name, fn in VARIANTS if name in wanted]
    if not variants:
        sys.exit(f"no valid variants selected; available: "
                 f"{[v[0] for v in VARIANTS]}")

    counts: Dict[str, int] = {}
    rejects: Dict[str, int] = {}
    n_total = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for pattern in stream_bank(bank_path):
            for variant_name, fn in variants:
                text = fn(pattern)
                if not quality_ok(text):
                    rejects[variant_name] = rejects.get(variant_name, 0) + 1
                    continue
                rec = build_record(pattern["id"], variant_name, text)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[variant_name] = counts.get(variant_name, 0) + 1
                n_total += 1

    print(f"Wrote {n_total} records to {out_path}")
    print(f"  by variant: {counts}")
    print(f"  rejects:    {rejects}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
