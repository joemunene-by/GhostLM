#!/usr/bin/env python3
"""Templated synthesis of log-analysis training records (bet 10).

Bet 10 ([docs/differentiation.md](differentiation.md) §"Bet 10: log
analysis & event reasoning") trains ghost-base on the SOC-analyst
core workflow: read a log line, name the ATT&CK technique it most
likely represents, cite the field that justifies the call, decide
whether to alert.

This is the bet that makes GhostLM a SIEM-shaped artifact rather
than just a chatbot. Every alert in a security operations centre
boils down to a chain of (log -> technique -> action), and no
small LM is good at any link in that chain today.

Seed: ``data/raw/log_analysis_patterns.jsonl``, a hand-curated
bank of 30 (technique_id, log_source, sample_log_line,
characteristic_fields, detection_signature, explanation, false_
positive_examples) tuples covering Windows Sysmon, Windows
Security, Linux auditbeat, network proxy / webserver / DNS,
and email gateway logs.

Output formats per pattern:

  1. ``pretrain_prose``  flat markdown article presenting the
                          technique, the canonical log line, the
                          detection signature, and analyst-next-step
                          guidance. Right shape for pretrain corpus
                          mixing.
  2. ``identify_technique`` chat Q&A. USER pastes a log line, asks
                            'what ATT&CK technique does this look
                            like?'. ASSISTANT names the technique
                            with a one-paragraph rationale citing
                            the characteristic fields.
  3. ``explain_detection`` chat Q&A. USER asks 'how would you
                            alert on technique <T-code>?'.
                            ASSISTANT describes the detection
                            signature with the false-positive
                            caveats.
  4. ``field_citation``    chat Q&A. USER pastes a log, asks
                            'which field is the most diagnostic?'.
                            ASSISTANT names the characteristic
                            fields and explains why each is
                            informative. Pairs naturally with bet
                            9 (provenance / cite tags).

30 patterns x 4 variants = 120 records, all parser-clean.

Run:

    PYTHONPATH=. python3 scripts/synth_log_analysis.py \\
        --bank data/raw/log_analysis_patterns.jsonl \\
        --out data/processed/synth_log_analysis.jsonl

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


def build_record(seed_id: str, variant: str, text: str) -> Dict[str, str]:
    h = hashlib.sha1(
        f"{seed_id}\n{variant}\n{text}".encode("utf-8")
    ).hexdigest()[:10]
    return {
        "id": f"synth_log_analysis#{seed_id}#{variant}#{h}",
        "source": "synth_log_analysis",
        "teacher": "templated",
        "seed_source": variant,
        "seed_id": seed_id,
        "text": text,
    }


def fp_note(false_positive_examples: List[str]) -> str:
    if not false_positive_examples:
        return ""
    if len(false_positive_examples) == 1:
        return (f"\n\nFalse-positive caveat: "
                f"{false_positive_examples[0]}.")
    return ("\n\nFalse-positive caveats: "
            + "; ".join(false_positive_examples) + ".")


def fence_log(log_line: str) -> str:
    """Wrap a sample log line in a code fence for readability."""
    return f"```\n{log_line}\n```"


# ---------------------------------------------------------------------------
# Variant templates
# ---------------------------------------------------------------------------


def pretrain_prose(p: Dict) -> str:
    return (
        f"# Detection pattern: {p['technique_id']} "
        f"({p['technique_name']})\n\n"
        f"Log source: `{p['log_source']}`. "
        f"Event ID: `{p.get('event_id', '(varies)')}`.\n\n"
        f"## Sample log line\n\n{fence_log(p['sample_log_line'])}\n\n"
        f"## Characteristic fields\n\n"
        + "".join(f"- `{f}`\n" for f in p['characteristic_fields'])
        + f"\n## Detection signature\n\n{p['detection_signature']}\n\n"
        f"## Why this maps to {p['technique_id']}\n\n"
        f"{p['explanation']}"
        f"{fp_note(p.get('false_positive_examples', []))}\n"
    )


def identify_technique(p: Dict) -> str:
    return (
        f"USER: I see this log line on a {p['log_source']} feed:\n\n"
        f"{fence_log(p['sample_log_line'])}\n\n"
        f"What ATT&CK technique does this most likely represent, "
        f"and which field tells you?\n\n"
        f"ASSISTANT: This is {p['technique_id']} "
        f"({p['technique_name']}). "
        f"{p['explanation']}\n\n"
        f"The most diagnostic fields here are "
        + ", ".join(f"`{f}`" for f in p['characteristic_fields'])
        + f"."
        f"{fp_note(p.get('false_positive_examples', []))}\n"
    )


def explain_detection(p: Dict) -> str:
    return (
        f"USER: How would you alert on {p['technique_id']} "
        f"({p['technique_name']}) in {p['log_source']} logs?\n\n"
        f"ASSISTANT: The detection signature is: "
        f"{p['detection_signature']}.\n\n"
        f"The fields you key on are "
        + ", ".join(f"`{f}`" for f in p['characteristic_fields'])
        + f". A canonical sample of what the malicious version "
        f"looks like:\n\n{fence_log(p['sample_log_line'])}\n\n"
        f"{p['explanation']}"
        f"{fp_note(p.get('false_positive_examples', []))}\n"
    )


def field_citation(p: Dict) -> str:
    return (
        f"USER: For this {p['log_source']} log, which fields are "
        f"the most diagnostic for {p['technique_id']} detection, "
        f"and why?\n\n"
        f"{fence_log(p['sample_log_line'])}\n\n"
        f"ASSISTANT: The fields that carry the {p['technique_id']} "
        f"({p['technique_name']}) signal here are: "
        + "".join(
            f"\n- `{f}` (informative because the {p['detection_signature']} "
            f"detection keys on it)"
            for f in p['characteristic_fields']
        )
        + f"\n\n{p['explanation']}\n"
    )


VARIANTS = [
    ("pretrain_prose", pretrain_prose),
    ("identify_technique", identify_technique),
    ("explain_detection", explain_detection),
    ("field_citation", field_citation),
]


def quality_ok(text: str, min_words: int = 60, max_words: int = 1500) -> bool:
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
    p.add_argument("--bank", default="data/raw/log_analysis_patterns.jsonl")
    p.add_argument("--out", default="data/processed/synth_log_analysis.jsonl")
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
