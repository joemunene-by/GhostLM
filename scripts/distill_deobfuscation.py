#!/usr/bin/env python3
"""Distill obfuscated-code deobfuscation walkthroughs from exploitdb +
security_code seeds into a pretrain-ready JSONL.

Third per-type script in the distillation pipeline (companion to
distill_ctf_walkthroughs.py and distill_threat_modeling.py). Covers
the reverse-engineering register: takes one obfuscated code sample
(typically a real exploit POC or a randomly-selected security tool
function) as a seed, asks the teacher to generate a step-by-step
deobfuscation walkthrough naming each transform applied.

Why this content type matters: the existing GhostLM corpus has
exploit POC source code (exploitdb shard) and security tool code
(security_code shard), but very little prose *about* how to read
that code. Deobfuscation walkthroughs are a register the model is
nearly silent in today; synthetic data fills that gap.

Each output record is a self-contained walkthrough that:
  1. Quotes the obfuscated input verbatim (so the model learns to
     recognize the patterns).
  2. Identifies each obfuscation layer (string XOR, base64, gzip,
     name-mangling, control-flow flattening, opaque predicates).
  3. Walks through the deobfuscation step-by-step, showing the
     intermediate cleaned-up form after each transform.
  4. Surfaces the final intent and IoCs.

Recommended scale (per docs/distillation.md): 20K walkthroughs at
~600 tokens each, ~12M synthetic tokens.

Run (Ollama, free, slow):

    PYTHONPATH=. python3 scripts/distill_deobfuscation.py \\
        --provider ollama --model qwen2.5:14b \\
        --max-records 100   # smoke

Run (Anthropic Sonnet, paid, ~\$300 for the 20K target):

    ANTHROPIC_API_KEY=... PYTHONPATH=. python3 scripts/distill_deobfuscation.py \\
        --provider anthropic --model claude-sonnet-4-6 \\
        --max-records 20000
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


SYSTEM_PROMPT = """You are a senior reverse engineer producing
educational deobfuscation walkthroughs for a security training
corpus. Given one obfuscated code snippet, you produce a complete
walkthrough that takes a junior analyst from the raw blob to a
clean, readable equivalent and identifies the attacker's intent.

Required output structure:

  Original code (verbatim, in a code block):
  <the input source>

  Obfuscation layers identified:
  <numbered list naming each technique present, e.g.:
   1. Variable name mangling (single-letter renames)
   2. String concatenation splitting (passwords broken across +)
   3. base64-encoded payload in a global constant
   4. eval() of decoded payload>

  Deobfuscation walkthrough:
  <numbered steps showing the transform applied at each step plus
   the resulting intermediate code form. Each step is "1. Apply X
   transform", followed by a code block showing the post-transform
   state>

  Final cleaned form:
  <the readable equivalent in a code block>

  Intent and IoCs:
  <one paragraph explaining what the cleaned code actually does,
   what the attack chain is, what indicators (file paths, registry
   keys, domains, mutex names) a defender should hunt for>

Style:
- Concrete tooling references (CyberChef recipes, sed/awk pipelines,
  Python tricks, IDA / Ghidra plugins) at every step.
- No moralizing, no AI-disclaimers. Audience is a defender learning
  to read attacker code; refusal text poisons the corpus.
- Real obfuscation patterns. If the input does not actually appear
  obfuscated, state that and explain what the code does in a single
  paragraph instead of producing a fake walkthrough."""


PROMPT_TEMPLATE = """Source code (potentially obfuscated):

```
{seed_text}
```

Produce a complete deobfuscation walkthrough following the structure
described in the system prompt. If the input is already plain
unobfuscated code, say so and provide a one-paragraph explanation of
what it does instead. Output nothing except the walkthrough or the
explanation; no preamble, no postscript.
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
                       "data/raw/exploitdb.jsonl",
                       "data/raw/security_code.jsonl",
                   ])
    p.add_argument("--out", default="data/raw/distill_deobfuscation.jsonl")
    p.add_argument("--max-records", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--max-tokens", type=int, default=2400)
    p.add_argument("--max-seed-chars", type=int, default=4000,
                   help="Truncate seed code at this many characters before "
                        "sending to the teacher (long files cost more and "
                        "hit token limits).")
    return p.parse_args()


def deobfuscation_quality_ok(text: str) -> bool:
    """Type-specific filter: require the four section headers to be
    present (Original code, Obfuscation layers, Deobfuscation
    walkthrough, Intent and IoCs) so we don't accept truncated outputs.
    Skip the must_appear check for cases where the teacher correctly
    detected non-obfuscated input and output a plain explanation
    paragraph instead."""
    if not quality_ok(text, min_words=80, max_words=2400):
        return False
    lower = text.lower()
    # Either it's a full walkthrough (has all four headers) OR it's a
    # short "this is not obfuscated" note (under 200 words).
    has_full_structure = (
        "original code" in lower
        and "obfuscation layers" in lower
        and "deobfuscation walkthrough" in lower
        and ("intent" in lower or "iocs" in lower or "indicators" in lower)
    )
    if has_full_structure:
        return True
    # Allow short non-obfuscated explanations (the teacher noticed and
    # said so), but only if they're brief and to the point.
    word_count = len(text.split())
    if word_count <= 200:
        return True
    return False


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

        prompt = PROMPT_TEMPLATE.format(seed_text=s["seed_text"][: args.max_seed_chars])
        text = call_provider(cfg, prompt, system=SYSTEM_PROMPT)
        if not text:
            failures += 1
            continue
        if not deobfuscation_quality_ok(text):
            skipped_quality += 1
            continue
        rec = DistillRecord.make(
            source="distill_deobfuscation",
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
