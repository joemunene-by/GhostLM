#!/usr/bin/env python3
"""Templated synthesis of code-for-security training records (bet 7).

Bet 7 ([docs/differentiation.md](differentiation.md) §"Bet 7: code-
for-security") aims to give ghost-base a concrete reasoning signal
on real vulnerability patterns: read code, identify the vulnerability
class, propose a fix. The hypothesis is that big general-purpose
small models do this poorly because their pretrain mix dilutes
security-relevant code with general code, and their RLHF often
filters out exploit-shaped content. A small from-scratch LM trained
natively on security-context code is a different artifact.

Seed: ``data/raw/code_security_patterns.jsonl``, a hand-curated bank
of (CWE, name, language, vulnerable_code, patched_code, explanation,
cve_examples) tuples covering OWASP Top 10 and common CWEs across
Python / JavaScript / C.

Output formats per pattern:

  1. **pretrain prose**: a flat document presenting the pattern
     as a markdown article. Right shape for pretrain-corpus mixing.
  2. **identify-and-fix Q&A**: USER shows vulnerable code, asks
     'what is wrong + how to fix'; ASSISTANT identifies the CWE,
     explains the bug, shows the patched version, references CVEs.
  3. **explain-the-diff Q&A**: USER shows both versions, asks why
     the second is safer; ASSISTANT explains the security property
     each version has / lacks.
  4. **CWE-mapping Q&A**: USER shows vulnerable code, asks 'what
     CWE does this map to'; ASSISTANT names the CWE and gives a
     one-paragraph rationale.

12 patterns x 4 record types = 48 records minimum, all parser-clean
because the templates are deterministic. Each record is shaped as
a ``DistillRecord`` so it drops into the corpus identically to
synth_format_aware / synth_tool_use output.

Run:

    PYTHONPATH=. python3 scripts/synth_code_security.py \\
        --bank data/raw/code_security_patterns.jsonl \\
        --out data/processed/synth_code_security.jsonl

Cost: zero. Deterministic. Same bank + same script produces
byte-identical output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_record(seed_id: str, variant: str, text: str) -> Dict[str, str]:
    """Assemble a DistillRecord-shaped dict ready to write."""
    h = hashlib.sha1(
        f"{seed_id}\n{variant}\n{text}".encode("utf-8")
    ).hexdigest()[:10]
    return {
        "id": f"synth_code_security#{seed_id}#{variant}#{h}",
        "source": "synth_code_security",
        "teacher": "templated",
        "seed_source": variant,
        "seed_id": seed_id,
        "text": text,
    }


def fence(code: str, lang: str) -> str:
    """Wrap code in a markdown fence with the given language tag."""
    return f"```{lang}\n{code}\n```"


def cve_phrase(cves: List[str]) -> str:
    """One-line 'real-world examples' suffix; empty string if no CVEs."""
    if not cves:
        return ""
    if len(cves) == 1:
        return f"\n\nReal-world example: {cves[0]}."
    return f"\n\nReal-world examples: {', '.join(cves)}."


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


def pretrain_prose(p: Dict) -> str:
    """Variant 1: pretrain-shaped flat markdown article."""
    return (
        f"# Vulnerability pattern: {p['name']} ({p['cwe']})\n\n"
        f"Language: {p['language']}.\n\n"
        f"## Vulnerable version\n\n"
        f"{fence(p['vulnerable'], p['language'])}\n\n"
        f"## Patched version\n\n"
        f"{fence(p['patched'], p['language'])}\n\n"
        f"## Why the vulnerable version is exploitable\n\n"
        f"{p['explanation']}"
        f"{cve_phrase(p.get('cve_examples', []))}\n"
    )


def identify_and_fix(p: Dict) -> str:
    """Variant 2: USER shows vulnerable code, asks for vuln + fix."""
    return (
        f"USER: Look at this {p['language']} snippet. What is wrong "
        f"with it from a security standpoint, and how would you fix "
        f"it?\n\n"
        f"{fence(p['vulnerable'], p['language'])}\n\n"
        f"ASSISTANT: This is {p['name']} ({p['cwe']}). "
        f"{p['explanation']}\n\n"
        f"Fixed version:\n\n"
        f"{fence(p['patched'], p['language'])}"
        f"{cve_phrase(p.get('cve_examples', []))}\n"
    )


def explain_the_diff(p: Dict) -> str:
    """Variant 3: USER shows both versions, asks why patched is safer."""
    return (
        f"USER: Here are two versions of the same {p['language']} "
        f"function. Why is the second one safer than the first?\n\n"
        f"Version A:\n\n"
        f"{fence(p['vulnerable'], p['language'])}\n\n"
        f"Version B:\n\n"
        f"{fence(p['patched'], p['language'])}\n\n"
        f"ASSISTANT: Version A is vulnerable to {p['name']} ({p['cwe']}). "
        f"{p['explanation']}\n"
    )


def cwe_mapping(p: Dict) -> str:
    """Variant 4: USER shows vulnerable code, asks which CWE."""
    # First 2 sentences of explanation; many start with the diagnosis
    # then move into the fix discussion, so 2 sentences captures the
    # 'why this is the right CWE' rationale.
    sentences = [s.strip() for s in p["explanation"].split(".") if s.strip()]
    rationale = ". ".join(sentences[:2]) + "."
    return (
        f"USER: Which CWE class does the following {p['language']} "
        f"code fall under, and why does that classification fit?\n\n"
        f"{fence(p['vulnerable'], p['language'])}\n\n"
        f"ASSISTANT: {p['cwe']} ({p['name']}). {rationale}"
        f"{cve_phrase(p.get('cve_examples', []))}\n"
    )


VARIANTS = [
    ("pretrain_prose", pretrain_prose),
    ("identify_and_fix", identify_and_fix),
    ("explain_the_diff", explain_the_diff),
    ("cwe_mapping", cwe_mapping),
]


# ---------------------------------------------------------------------------
# Quality filter
# ---------------------------------------------------------------------------


def quality_ok(text: str, min_words: int = 50, max_words: int = 1500) -> bool:
    """Light filter: word-count bound and required-content sanity."""
    words = text.split()
    if not (min_words <= len(words) <= max_words):
        return False
    # Code fence appears at least once.
    if "```" not in text:
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
    p.add_argument("--bank", default="data/raw/code_security_patterns.jsonl",
                   help="Path to the hand-curated pattern bank")
    p.add_argument("--out", default="data/processed/synth_code_security.jsonl",
                   help="Output JSONL path")
    p.add_argument("--variants", default=",".join(v[0] for v in VARIANTS),
                   help="Comma-separated subset of variants to emit")
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
