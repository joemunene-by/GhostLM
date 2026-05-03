#!/usr/bin/env python3
"""Generate CoT-templated MCQ training data via Qwen-7B (Ollama).

Takes the existing letter-only MCQs in ``data/raw/chat/mcq.jsonl`` and
augments each with a 1-3 sentence justification synthesized by a local
Qwen-2.5-7B (or compatible) model running via Ollama. The output keeps
the letter answer up front (so the run_bench.py logprob-of-letter
scoring still works) and adds reasoning after the period — the model
gets supervised on both signals.

Output schema is the same as ``build_mcq_data.py`` — a chat record
with ``{"turns": [{"role": "user", "content": ...},
{"role": "assistant", "content": "B. <reasoning>"}]}`` — so it drops
into ``build_chat_dataset.py`` via the ``--mcq-jsonl`` flag without any
other plumbing changes.

Why this fix (per research agent, 2025-2026 references):
- The Phi-3.5-mini and OpenMath-Mini reports both show CoT-templated MCQ
  at 1× outperforms raw letter-only MCQ at 5× for sub-200M models.
- The mechanism: letter-only records teach the model "after Answer:
  emit a letter" — a parlor trick that doesn't transfer to rephrased
  questions. CoT records teach the underlying knowledge connections.

Prerequisite: Ollama running locally with a Qwen pulled. Default
expects ``qwen2.5:7b`` — pull via ``ollama pull qwen2.5:7b`` (~5 GB).
On the Mac M4 a 7B model runs at ~30 tokens/sec, so 1.8K records ×
~80 tokens of reasoning ≈ 80 minutes. Resume-safe — re-run continues
where it left off.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional


OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT_TEMPLATE = """You are helping label a cybersecurity multiple-choice training dataset. You will be given a question, four options, and the correct letter. Write a 1-2 sentence justification for why the correct answer is right. Be specific — reference the technical reason, not just "this is correct".

Format your response as exactly one line:
JUSTIFICATION: <your 1-2 sentence reason>

Do not repeat the question. Do not list other options. Do not add disclaimers.

Question: {question}
{options}

Correct answer: {letter}) {correct_text}

Now write the JUSTIFICATION line."""


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="Build CoT-templated MCQ training data via Ollama")
    p.add_argument("--in-mcq", default="data/raw/chat/mcq.jsonl",
                   help="Input letter-only MCQ JSONL from build_mcq_data.py")
    p.add_argument("--out", default="data/raw/chat/mcq_cot.jsonl",
                   help="Output augmented JSONL")
    p.add_argument("--model", default="qwen2.5:14b",
                   help="Ollama model tag — qwen2.5:14b chosen as default "
                        "because the 14B general variant beats both qwen2.5:7b "
                        "and qwen2.5-coder:14b for narrative-style "
                        "justifications. Already cached on Joe's Mac.")
    p.add_argument("--temperature", type=float, default=0.4,
                   help="Lower = more focused justifications")
    p.add_argument("--request-delay", type=float, default=0.0,
                   help="Seconds between Ollama calls")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap records (smoke testing)")
    return p.parse_args()


def parse_existing_mcq(record: Dict) -> Optional[Dict]:
    """Recover (question, options A/B/C/D, correct letter, correct text) from
    an existing letter-only MCQ record.

    The prompt format from ``build_mcq_data.py`` is::

        Pick the best answer (A, B, C, or D) for this multiple-choice ...

        Question: <q>

        A) <opt_a>
        B) <opt_b>
        C) <opt_c>
        D) <opt_d>

        Answer:

    Returns None if the record can't be parsed cleanly (defensive — skip
    rather than crash).
    """
    user_text = record["turns"][0]["content"]
    assistant_text = record["turns"][1]["content"]

    q_match = re.search(r"Question:\s*(.+?)\n", user_text)
    if not q_match:
        return None
    question = q_match.group(1).strip()

    options: Dict[str, str] = {}
    for letter in "ABCD":
        opt_match = re.search(rf"^{letter}\)\s*(.+?)$", user_text, re.MULTILINE)
        if opt_match:
            options[letter] = opt_match.group(1).strip()
    if len(options) != 4:
        return None

    # Assistant turn is "B" or "B. <text>." — first char is the letter.
    correct_letter = assistant_text.strip()[:1].upper()
    if correct_letter not in options:
        return None

    return {
        "question": question,
        "options": options,
        "letter": correct_letter,
        "correct_text": options[correct_letter],
    }


def call_ollama(model: str, prompt: str, temperature: float, timeout: int = 120) -> Optional[str]:
    """POST to Ollama's /api/generate and return the response text, or None on error."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 256},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read())
            return body.get("response", "").strip()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"  ollama error: {e}")
        return None


def extract_justification(raw: str) -> Optional[str]:
    """Pull the JUSTIFICATION line out of Qwen's response."""
    if not raw:
        return None
    for line in raw.splitlines():
        line = line.strip()
        if line.upper().startswith("JUSTIFICATION:"):
            return line.split(":", 1)[1].strip()
    # Fallback: take the first non-empty line.
    for line in raw.splitlines():
        line = line.strip()
        if line:
            return line[:400]
    return None


def render_options_block(options: Dict[str, str]) -> str:
    """Render A/B/C/D options for the prompt."""
    return "\n".join(f"{letter}) {options[letter]}" for letter in "ABCD")


def main() -> None:
    """Generate one CoT-augmented record per input MCQ, resume-safe."""
    args = parse_args()
    in_path = Path(args.in_mcq)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        records = records[: args.limit]

    seen_questions: set = set()
    if out_path.exists():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    user = rec.get("turns", [{}])[0].get("content", "")
                    qm = re.search(r"Question:\s*(.+?)\n", user)
                    if qm:
                        seen_questions.add(qm.group(1).strip())
        print(f"  resume: {len(seen_questions)} records already done")

    out_fh = out_path.open("a", encoding="utf-8", buffering=1)
    written = 0
    skipped = 0
    failed = 0

    for i, rec in enumerate(records):
        parsed = parse_existing_mcq(rec)
        if parsed is None:
            skipped += 1
            continue
        if parsed["question"] in seen_questions:
            continue

        prompt = PROMPT_TEMPLATE.format(
            question=parsed["question"],
            options=render_options_block(parsed["options"]),
            letter=parsed["letter"],
            correct_text=parsed["correct_text"],
        )
        raw = call_ollama(args.model, prompt, args.temperature)
        justification = extract_justification(raw or "")

        if not justification or len(justification) < 12:
            failed += 1
            if args.request_delay:
                time.sleep(args.request_delay)
            continue

        # Letter-first format: scorer still hits the letter logprob; reasoning
        # is appended after the period for knowledge supervision.
        new_assistant = f"{parsed['letter']}. {justification}"
        out_record = {
            "turns": [
                {"role": "user", "content": rec["turns"][0]["content"]},
                {"role": "assistant", "content": new_assistant},
            ],
            "source": "mcq_cot",
        }
        out_fh.write(json.dumps(out_record, ensure_ascii=False) + "\n")
        out_fh.flush()
        written += 1

        if written % 50 == 0:
            print(f"  {written} records written ({i + 1}/{len(records)} processed)")

        if args.request_delay:
            time.sleep(args.request_delay)

    out_fh.close()
    print()
    print(f"Done. Wrote {written} CoT records to {out_path}")
    if skipped:
        print(f"  Skipped {skipped} unparseable inputs")
    if failed:
        print(f"  Failed (empty / short justification) on {failed}")


if __name__ == "__main__":
    main()
