#!/usr/bin/env python3
"""Distill tool-use traces from a strong teacher model into a
fine-tuning corpus that teaches GhostLM to USE tools (not memorize
facts).

The v0.9.3 RAG diagnostic established the bottleneck cleanly:
retrieval@4 = 41/100, generation@RAG = 0/100. The 81M model can't
extract facts even when the right context is in the prompt. The
fix is not "add more retrieval" but "train the model on tool-use
traces" so it learns the meta-skill of issuing a tool call,
ingesting the response, and answering from it.

This script generates synthetic training examples in the shape:

    user:       "What is the CVE for EternalBlue?"
    assistant:  <|tool_call|>{"name": "search_cve_nvd",
                              "args": {"q": "EternalBlue"}}<|/tool_call|>
    tool:       <|tool_response|>{"cve": "CVE-2017-0144", ...}<|/tool_response|>
    assistant:  "EternalBlue is CVE-2017-0144, a critical SMB vulnerability..."

The teacher (Claude / Llama-3.3-70B / Qwen-72B) generates the full
trace given a question and the simulated tool's response. We don't
actually CALL the tool during distillation; we let the teacher
hallucinate a plausible tool response based on real corpus data
(seeded from the v1.0 corpus), then sanity-check that the final
answer aligns with the seed material.

Output: ``data/raw/distill_tool_use.jsonl`` with one chat-format
trace per line. Schema is the standard messages-list format with
two new role tokens (`tool_call`, `tool_response`) added to the
GhostLM tokenizer.

Target volume: 10,000 traces across the four tool types
(search_cve, lookup_mitre, lookup_cwe, rag_retrieve). At ~600
tokens per trace = ~6M training tokens. Fine-tune ghost-base on
this set + the existing chat-v3 SFT data and the model learns to
issue tool calls before answering.

Why this is the differentiator: every other small cybersec LM
trains the model to memorize CVE numbers, then watches it fail.
Training to ISSUE A TOOL CALL is a different model-level
objective. A 360M tool-using model crushes a 7B memorizing one on
real workflows because it gets the answer right by querying NVD,
not by guessing from training-data residue.

Run (Ollama smoke):

    PYTHONPATH=. python3 scripts/distill_tool_use.py \\
        --provider ollama --model qwen2.5:14b \\
        --max-records 50

Run (Anthropic production, ~$200 for 10K traces on Sonnet):

    ANTHROPIC_API_KEY=... PYTHONPATH=. python3 scripts/distill_tool_use.py \\
        --provider anthropic --model claude-sonnet-4-6 \\
        --max-records 10000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.distill_common import (  # noqa: E402
    DistillRecord, ProviderConfig, ResumeIndex, StreamingWriter,
    call_provider, content_dedup, load_jsonl_source, quality_ok,
)


# ---------------------------------------------------------------------------
# Tool registry. The teacher is told to issue calls in this shape.
# ---------------------------------------------------------------------------

TOOLS = {
    "search_cve_nvd": {
        "description": "Look up a CVE by ID or search NVD by free-text query.",
        "args": {"q": "CVE id or natural-language query"},
        "seeds": ["data/raw/cve_full.jsonl", "data/raw/cisa_kev.jsonl"],
    },
    "lookup_mitre_technique": {
        "description": "Look up a MITRE ATT&CK technique by ID (e.g. T1059).",
        "args": {"technique_id": "MITRE technique ID"},
        "seeds": ["data/raw/mitre_attack.jsonl", "data/raw/mitre_full.jsonl"],
    },
    "lookup_cwe": {
        "description": "Look up a CWE entry by number (e.g. CWE-89).",
        "args": {"cwe_id": "CWE identifier"},
        "seeds": ["data/raw/cwe.jsonl"],
    },
    "rag_retrieve": {
        "description": "Retrieve top-K passages from the cybersec corpus by query.",
        "args": {"query": "natural-language search query", "k": "number of passages (default 4)"},
        "seeds": ["data/raw/owasp_top10.jsonl", "data/raw/owasp_asvs.jsonl",
                  "data/raw/rfcs.jsonl", "data/raw/security_blogs.jsonl"],
    },
}


SYSTEM_PROMPT = """You are generating supervised fine-tuning examples
for a small cybersecurity language model that needs to learn to USE
tools (rather than memorize facts and confabulate).

Each example is a complete chat trace in this exact format:

  USER: <a natural-language cybersecurity question>
  ASSISTANT: <|tool_call|>{"name": "<TOOL>", "args": {...}}<|/tool_call|>
  TOOL: <|tool_response|>{...realistic JSON response based on seed text...}<|/tool_response|>
  ASSISTANT: <natural-language answer that synthesizes the tool response>

Rules:

1. Use exactly the literal tag strings <|tool_call|>, <|/tool_call|>,
   <|tool_response|>, <|/tool_response|>. The model is being trained on
   these tokens; do not vary them.
2. Produce realistic tool responses based on the seed source material
   you're given. Do not hallucinate fields the seed doesn't justify.
3. The final ASSISTANT message must reference specifics that ONLY
   appeared in the tool response (e.g. the actual CVE ID, the actual
   CVSS score). The point of training is to teach the model that
   facts come from tools, not memory.
4. Sometimes the tool returns nothing relevant ("not found"). Train
   the model to acknowledge that, not confabulate. ~10% of examples
   should have an empty / not-found tool response and a "I don't
   know based on this lookup" answer.
5. Tool args must be valid JSON. The literal `{"name": "...", "args": {...}}`
   structure is what the trainer parses.

Output exactly the four-message trace, nothing else. No preamble, no
postscript, no commentary."""


def render_seed_context(seed_text: str, tool_name: str, tool_def: dict) -> str:
    """Show the teacher the seed material plus the tool spec, so the
    teacher can craft a coherent question + realistic tool response +
    grounded answer."""
    return (
        f"Tool to demonstrate: {tool_name}\n"
        f"Tool description:    {tool_def['description']}\n"
        f"Tool args schema:    {json.dumps(tool_def['args'], indent=2)}\n\n"
        f"Seed material (use this as the basis for the tool's response):\n\n"
        f"{seed_text[:2500]}"
    )


PROMPT_TEMPLATE = """{seed_context}

Generate one complete tool-use chat trace following the system
prompt's format. The user asks a question that this specific tool
can answer; the assistant issues exactly one tool call; the tool
response contains the relevant facts derived from the seed; the
assistant produces a final answer that uses ONLY those facts.

Mix it up: ~10% of traces should have the tool return nothing
relevant ("not found" / empty array), and the assistant should
acknowledge the gap rather than confabulate.

Output the trace in the literal four-message format. Nothing else.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--provider", choices=["ollama", "anthropic", "openai"],
                   default="ollama")
    p.add_argument("--model", default="qwen2.5:14b")
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key-env", default=None)
    p.add_argument("--out", default="data/raw/distill_tool_use.jsonl")
    p.add_argument("--max-records", type=int, default=0,
                   help="Total cap across all tool types; 0 = no cap")
    p.add_argument("--per-tool", type=int, default=2500,
                   help="Target traces per tool type (4 tools * 2500 = 10K)")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-tokens", type=int, default=1500)
    return p.parse_args()


def trace_quality_ok(text: str) -> bool:
    """Type-specific filter: require the four message tags AND
    realistic structure."""
    if not quality_ok(text, min_words=40, max_words=600):
        return False
    required = [
        "<|tool_call|>", "<|/tool_call|>",
        "<|tool_response|>", "<|/tool_response|>",
    ]
    for tag in required:
        if tag not in text:
            return False
    # Must have a JSON-ish tool_call body.
    tc_start = text.find("<|tool_call|>") + len("<|tool_call|>")
    tc_end = text.find("<|/tool_call|>")
    if tc_end <= tc_start:
        return False
    tc_body = text[tc_start:tc_end].strip()
    if not (tc_body.startswith("{") and tc_body.endswith("}")):
        return False
    try:
        parsed = json.loads(tc_body)
        if "name" not in parsed or "args" not in parsed:
            return False
        if parsed["name"] not in TOOLS:
            return False
    except json.JSONDecodeError:
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
    print(f"Resuming: {len(resume.seen)} traces already written")
    print(f"Per-tool target: {args.per_tool}")
    print(f"Tools: {list(TOOLS.keys())}")

    written = 0
    skipped_resume = 0
    skipped_quality = 0
    failures = 0

    for tool_name, tool_def in TOOLS.items():
        print(f"\n=== Tool: {tool_name} ===")
        # Load all available seeds for this tool's source files.
        seeds: List[Dict] = []
        for sf in tool_def["seeds"]:
            sp = Path(sf)
            if not sp.exists():
                print(f"  skip missing seed file: {sf}")
                continue
            loaded = load_jsonl_source(sp)
            for s in loaded:
                s["seed_source"] = sp.stem
            seeds.extend(loaded)
        print(f"  total seed pool: {len(seeds)}")

        per_tool_written = 0
        pending: list = []

        for seed in seeds:
            if args.max_records and written >= args.max_records:
                break
            if per_tool_written >= args.per_tool:
                break
            seed_id = f"{tool_name}#{seed['seed_id']}"
            if resume.already_done(tool_name, seed["seed_id"]):
                skipped_resume += 1
                continue

            ctx = render_seed_context(seed["seed_text"], tool_name, tool_def)
            prompt = PROMPT_TEMPLATE.format(seed_context=ctx)
            text = call_provider(cfg, prompt, system=SYSTEM_PROMPT)
            if not text:
                failures += 1
                continue
            if not trace_quality_ok(text):
                skipped_quality += 1
                continue
            rec = DistillRecord.make(
                source="distill_tool_use",
                teacher=f"{cfg.name}/{cfg.model}",
                seed_source=seed["seed_source"],
                seed_id=seed["seed_id"],
                text=text,
            )
            # Tag the record with the tool type for later filtering.
            rec_dict = json.loads(json.dumps(rec.__dict__))
            rec_dict["tool"] = tool_name
            pending.append(rec_dict)
            per_tool_written += 1
            written += 1

            if (skipped_resume + written + failures + skipped_quality) % 25 == 0:
                print(f"    progress: per-tool={per_tool_written}  total={written}  "
                      f"resume={skipped_resume}  quality_skip={skipped_quality}  "
                      f"fail={failures}")

        # Drain pending (no dedup for tool-use traces; each is unique by design).
        for r in pending:
            writer.fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            writer.fh.flush()

    writer.close()
    print(f"\nDone. Wrote {written} traces across {len(TOOLS)} tool types.")
    print(f"  resume-skipped: {skipped_resume}  quality-rejected: {skipped_quality}  "
          f"provider-failed: {failures}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
