#!/usr/bin/env python3
"""Distill STRIDE-style threat models from OWASP + CWE seeds into a
pretrain-ready JSONL.

Companion to ``distill_ctf_walkthroughs.py``. The two cover different
content registers: CTF walkthroughs are step-by-step exploitation
narratives (offensive register), threat models are STRIDE-structured
defender-side architectural analysis (defensive register). A
ghost-3B+ corpus needs both to cover the security writing the
existing 363M-token corpus is thin on.

Each output record is a self-contained STRIDE threat model for a
hypothetical web/cloud/IoT component derived from one OWASP or CWE
seed. The output names assets, identifies threats per STRIDE
category (Spoofing / Tampering / Repudiation / Info disclosure /
Denial of service / Elevation of privilege), proposes mitigations,
and ends with a residual-risk paragraph.

Recommended scale for ghost-3B (per ``docs/distillation.md``):
30K threat models, ~500 tokens each = 15M synthetic tokens.

Run (Ollama, free, slow):

    PYTHONPATH=. python3 scripts/distill_threat_modeling.py \\
        --provider ollama --model qwen2.5:14b \\
        --max-records 100   # smoke

Run (Together Llama-3.3-70B, paid, ~$200 for the 30K target):

    TOGETHER_API_KEY=... PYTHONPATH=. python3 scripts/distill_threat_modeling.py \\
        --provider openai \\
        --base-url https://api.together.xyz \\
        --api-key-env TOGETHER_API_KEY \\
        --model meta-llama/Llama-3.3-70B-Instruct-Turbo \\
        --max-records 30000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.distill_common import (  # noqa: E402
    DistillRecord, ProviderConfig, ResumeIndex, StreamingWriter,
    call_provider, content_dedup, load_jsonl_source, quality_ok,
)


SYSTEM_PROMPT = """You are a senior application-security architect
producing STRIDE-style threat models for a security-engineering
training corpus. You take a single OWASP weakness category or CWE
entry and produce a self-contained threat model for a hypothetical
component that exhibits that weakness.

Required output structure (use exactly these section headers):

  Component: <one-line description of the asset, e.g. "Customer-facing
              web API exposing /v1/orders with JWT auth">

  Trust boundaries: <numbered list of trust boundaries, 2-5 entries>

  STRIDE threats:
    Spoofing: <2-3 specific threats with attacker capability + impact>
    Tampering: <2-3 specific threats>
    Repudiation: <2-3 specific threats>
    Information disclosure: <2-3 specific threats>
    Denial of service: <2-3 specific threats>
    Elevation of privilege: <2-3 specific threats>

  Mitigations: <bulleted list, mapped back to specific threats above>

  Residual risk: <one paragraph describing what risk remains after
                  mitigations, what monitoring/detection should
                  catch the residue, and what compensating controls
                  apply>

Style:
- Be concrete. Real header names, real protocol behaviors, real
  CVE-style examples where relevant. Do not invent CVE IDs.
- Defender-side perspective throughout. The audience is a security
  architect deciding what to build, not an attacker deciding what
  to exploit.
- No moralizing, no AI-disclaimers. The corpus is for training a
  defensive-security model; refusal text actively poisons it."""


PROMPT_TEMPLATE = """Source weakness or category:

{seed_text}

Produce a complete STRIDE threat model for one specific hypothetical
component that exhibits this weakness. Use the section structure
described in the system prompt; output nothing except the threat
model itself (no preamble, no postscript).
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", choices=["ollama", "anthropic", "openai"],
                   default="ollama")
    p.add_argument("--model", default="qwen2.5:14b")
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key-env", default=None)
    p.add_argument("--seed-files", nargs="+",
                   default=[
                       "data/raw/owasp_top10.jsonl",
                       "data/raw/owasp_asvs.jsonl",
                       "data/raw/owasp_cheatsheets.jsonl",
                       "data/raw/owasp_wstg.jsonl",
                       "data/raw/cwe.jsonl",
                   ])
    p.add_argument("--out", default="data/raw/distill_threat_modeling.jsonl")
    p.add_argument("--max-records", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--max-tokens", type=int, default=2200)
    return p.parse_args()


def threat_model_quality_ok(text: str) -> bool:
    """STRIDE-specific filter on top of the base quality_ok. Requires
    all six STRIDE category headers AND the four section headers to
    appear; rejects truncated outputs that only got partway through."""
    if not quality_ok(text, min_words=120, max_words=2000):
        return False
    required = [
        "Component:", "Trust boundaries", "STRIDE threats",
        "Spoofing", "Tampering", "Repudiation",
        "Information disclosure", "Denial of service",
        "Elevation of privilege",
        "Mitigations", "Residual risk",
    ]
    lower = text.lower()
    for req in required:
        if req.lower() not in lower:
            return False
    return True


def main() -> int:
    args = parse_args()
    cfg = ProviderConfig(
        name=args.provider, model=args.model,
        base_url=args.base_url, api_key_env=args.api_key_env,
        temperature=args.temperature, max_tokens=args.max_tokens,
    )

    out_path = Path(args.out)
    resume = ResumeIndex(out_path)
    writer = StreamingWriter(out_path)
    print(f"Provider: {cfg.name}/{cfg.model}")
    print(f"Output:   {out_path}")
    print(f"Resuming: {len(resume.seen)} records already done")

    seeds = []
    for sf in args.seed_files:
        sp = Path(sf)
        if not sp.exists():
            print(f"  skip missing seed file: {sf}")
            continue
        loaded = load_jsonl_source(sp)
        for s in loaded:
            s["seed_source"] = sp.stem
        seeds.extend(loaded)
    print(f"Total seed records: {len(seeds)}")

    written = 0
    skipped_resume = 0
    skipped_quality = 0
    failures = 0
    pending: list = []

    for s in seeds:
        if args.max_records and written >= args.max_records:
            break
        if resume.already_done(s["seed_source"], s["seed_id"]):
            skipped_resume += 1
            continue

        prompt = PROMPT_TEMPLATE.format(seed_text=s["seed_text"][:5000])
        text = call_provider(cfg, prompt, system=SYSTEM_PROMPT)
        if not text:
            failures += 1
            continue
        if not threat_model_quality_ok(text):
            skipped_quality += 1
            continue
        rec = DistillRecord.make(
            source="distill_threat_modeling",
            teacher=f"{cfg.name}/{cfg.model}",
            seed_source=s["seed_source"], seed_id=s["seed_id"], text=text,
        )
        pending.append(rec)
        if len(pending) >= 64:
            for kept in content_dedup(pending):
                writer.write(kept)
                written += 1
            pending = []

        if (skipped_resume + written + failures + skipped_quality) % 25 == 0:
            print(f"  progress: written={written}  resume_skip={skipped_resume}  "
                  f"quality_skip={skipped_quality}  fail={failures}")

    for kept in content_dedup(pending):
        writer.write(kept)
        written += 1

    writer.close()
    print(f"\nDone. Wrote {written} records to {out_path}")
    print(f"  resume-skipped: {skipped_resume}  quality-rejected: {skipped_quality}  "
          f"provider-failed: {failures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
