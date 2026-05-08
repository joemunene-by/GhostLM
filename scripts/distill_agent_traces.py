#!/usr/bin/env python3
"""Distill bet 1 + bet 9 agent traces from an OpenAI-compatible teacher.

The 850 templated synth traces produced by scripts/synth_tool_use.py
+ scripts/synth_tool_use_provenance.py are structurally correct but
come from a fixed template bank. This script generates fresh traces
by driving any OpenAI-compatible teacher (Ollama running Qwen-14B
locally, vLLM on rented compute, the real OpenAI API, anything) through
the GhostAgent runtime and capturing the resulting traces.

Pipeline:
  1. Read a prompts JSONL (one prompt per line; can be the
     held-out eval files or a curated prompt bank).
  2. For each prompt, run GhostAgent with an OpenAICompatGenerator
     pointed at the teacher's chat-completions endpoint. The
     teacher emits <|tool_call|> blocks; the runtime parses them,
     dispatches against the canonical tool registry, and feeds
     results back. The teacher emits the final cite-tagged answer.
  3. Validate the trace through parse_agent_output; skip traces
     that don't have at least one parseable tool call AND a final
     answer with cite tags (we want bet 9-quality output, not just
     bet 1).
  4. Convert each valid trace to the bet-1 4-message text format
     (USER / ASSISTANT / TOOL / ASSISTANT) and write to output
     JSONL with the same record shape as scripts/synth_tool_use.py.

The output is drop-in compatible with scripts/prep_tool_use_sft.py.

CLI:

    PYTHONPATH=. python3 scripts/distill_agent_traces.py \\
        --teacher-base-url http://localhost:11434/v1 \\
        --teacher-model qwen2.5:14b \\
        --teacher-api-key ollama \\
        --prompts data/raw/provenance_eval.jsonl \\
        --out data/processed/distilled_tool_use.jsonl

    PYTHONPATH=. python3 scripts/distill_agent_traces.py \\
        --teacher-base-url https://api.openai.com/v1 \\
        --teacher-model gpt-4o-mini \\
        --teacher-api-key "$OPENAI_API_KEY" \\
        --prompts data/raw/curated_prompts.jsonl \\
        --out data/processed/distilled_openai.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ghostlm.agent import (  # noqa: E402
    AgentTrace,
    GhostAgent,
    MessageRole,
    RuntimeConfig,
    parse_agent_output,
)
from ghostlm.agent.teacher import OpenAICompatGenerator  # noqa: E402


def stream_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def trace_to_bet1_text(trace: AgentTrace) -> Optional[str]:
    """Convert a trace to the 4-message USER/ASSISTANT/TOOL/ASSISTANT
    text format that scripts/synth_tool_use.py emits and that
    scripts/prep_tool_use_sft.py parses.

    Returns None if the trace doesn't fit the 4-message shape (no
    tool call, multiple tool calls, missing final answer, etc.).
    """
    user_text: Optional[str] = None
    asst1_text: Optional[str] = None  # the tool-call message
    tool_text: Optional[str] = None  # the tool response
    asst2_text: Optional[str] = None  # the final answer

    for m in trace.history:
        if m.role == MessageRole.USER and user_text is None:
            user_text = m.content
        elif m.role == MessageRole.ASSISTANT:
            parsed = parse_agent_output(m.content)
            if parsed.tool_calls and asst1_text is None:
                # First assistant message with a tool call.
                asst1_text = m.content.strip()
            elif not parsed.tool_calls and asst1_text is not None:
                # Final assistant message without tool calls.
                if asst2_text is None:
                    asst2_text = parsed.plain_text or m.content.strip()
        elif m.role == MessageRole.TOOL and tool_text is None:
            tool_text = m.content.strip()

    if not (user_text and asst1_text and tool_text and asst2_text):
        return None

    return (
        f"USER: {user_text}\n"
        f"ASSISTANT: {asst1_text}\n"
        f"TOOL: {tool_text}\n"
        f"ASSISTANT: {asst2_text}\n"
    )


def trace_has_cite_tag(trace: AgentTrace) -> bool:
    """True iff any assistant message in the trace contains a parseable
    cite tag, which is the bet 9 quality bar."""
    for m in trace.history:
        if m.role != MessageRole.ASSISTANT:
            continue
        parsed = parse_agent_output(m.content)
        if parsed.cites:
            return True
    return False


def build_record(trace_text: str, source_id: str, teacher: str) -> Dict:
    h = hashlib.sha1(
        f"{teacher}\n{source_id}\n{trace_text}".encode("utf-8")
    ).hexdigest()[:10]
    return {
        "id": f"distilled_tool_use#{source_id}#{h}",
        "source": "distilled_tool_use",
        "teacher": teacher,
        "seed_source": "distilled",
        "seed_id": source_id,
        "text": trace_text,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="scripts/distill_agent_traces.py")
    p.add_argument("--teacher-base-url", required=True,
                    help="OpenAI-compatible base URL "
                         "(e.g. http://localhost:11434/v1 for Ollama, "
                         "https://api.openai.com/v1 for OpenAI).")
    p.add_argument("--teacher-model", required=True,
                    help="Teacher model identifier.")
    p.add_argument("--teacher-api-key", default="anything",
                    help="Bearer credential; local servers accept "
                         "any non-empty value.")
    p.add_argument("--prompts", required=True,
                    help="Input JSONL with one record per prompt; "
                         "each record must carry a 'prompt' field.")
    p.add_argument("--out", required=True,
                    help="Output JSONL of distilled records.")
    p.add_argument("--max-records", type=int, default=None,
                    help="Cap on number of prompts to process.")
    p.add_argument("--max-iters", type=int, default=4,
                    help="Agent loop cap per prompt.")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--require-cite", action="store_true",
                    help="Skip traces that do not contain a parseable "
                         "<|cite|> tag (bet 9 quality bar).")
    p.add_argument("--offline", action="store_true",
                    help="Force tool backends to use offline caches "
                         "(deterministic teacher tool responses).")
    p.add_argument("--seed-source", default=None,
                    help="Optional override for seed_source field "
                         "(otherwise 'distilled').")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.offline:
        os.environ["GHOST_AGENT_OFFLINE"] = "1"

    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        print(f"[error] prompts file not found: {prompts_path}",
              file=sys.stderr)
        return 1
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    teacher = OpenAICompatGenerator(
        base_url=args.teacher_base_url,
        api_key=args.teacher_api_key,
        model=args.teacher_model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )
    cfg = RuntimeConfig(
        max_iters=args.max_iters,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    agent = GhostAgent(teacher, cfg)
    teacher_tag = f"{args.teacher_model}@{args.teacher_base_url}"

    print(f"  Teacher:        {args.teacher_model}")
    print(f"  Endpoint:       {args.teacher_base_url}")
    print(f"  Prompts:        {prompts_path}")
    print(f"  Output:         {out_path}")
    print(f"  Require cite:   {args.require_cite}")
    print()

    n_total = 0
    n_kept = 0
    n_skipped_shape = 0
    n_skipped_cite = 0
    n_errors = 0
    t0 = time.time()

    with out_path.open("w", encoding="utf-8") as fout:
        for raw in stream_jsonl(prompts_path):
            if args.max_records and n_total >= args.max_records:
                break
            n_total += 1
            prompt = raw.get("prompt") or raw.get("query") or raw.get("text")
            if not prompt:
                continue
            seed_id = (raw.get("seed_id")
                        or raw.get("id")
                        or hashlib.sha1(prompt.encode()).hexdigest()[:12])

            try:
                trace = agent.run(prompt)
            except Exception as e:  # noqa: BLE001
                n_errors += 1
                print(f"  [{n_total}] ERROR: {e}", file=sys.stderr)
                continue

            if args.require_cite and not trace_has_cite_tag(trace):
                n_skipped_cite += 1
                continue

            text = trace_to_bet1_text(trace)
            if text is None:
                n_skipped_shape += 1
                continue

            rec = build_record(text, seed_id, teacher_tag)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_kept += 1

    elapsed = time.time() - t0
    print()
    print(f"Total prompts:          {n_total}")
    print(f"  kept (4-msg trace):   {n_kept}")
    print(f"  skipped (shape):      {n_skipped_shape}")
    print(f"  skipped (no cite):    {n_skipped_cite}")
    print(f"  errors:               {n_errors}")
    print(f"Elapsed:                {elapsed:.1f}s")
    print(f"Output:                 {out_path}")
    if n_kept > 0:
        print()
        print(f"Distilled traces are drop-in compatible with "
              f"scripts/prep_tool_use_sft.py:")
        print(f"  PYTHONPATH=. python3 scripts/prep_tool_use_sft.py \\")
        print(f"    --in-tool-use {out_path} \\")
        print(f"    --in-provenance data/processed/synth_tool_use_provenance.jsonl \\")
        print(f"    --base-train data/processed/chat_train.jsonl \\")
        print(f"    --out-train data/processed/chat_train_distilled.jsonl \\")
        print(f"    --out-val data/processed/chat_val_distilled.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
