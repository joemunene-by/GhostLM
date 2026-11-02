#!/usr/bin/env python3
"""Pre-review judge: a teacher model grades unreviewed chat conversations against docs/chat_review.md.

    python scripts/judge_chat.py run [--loop] [--batch 4]
    python scripts/judge_chat.py status

The judge does not approve anything. It writes one line per conversation to
data/chat_judge/judgements.jsonl, ``{"id", "flag": "clean" | "suspect" | "drop", "issues": [...]}``,
and ``review_chat.py next`` attaches that to each record and puts the flagged ones first,
so a human reviewer knows where to look hardest. Every line still gets read by a person
before it trains. With ``--loop`` the judge waits for new conversations and keeps going.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from review_chat import done_ids, raw_records  # noqa: E402
from teacher_http import DEFAULT_MODELS, ask_http, last_model  # noqa: E402

OUT_DIR = ROOT / "data/chat_judge"
OUT = OUT_DIR / "judgements.jsonl"
STANDARD = ROOT / "docs/chat_review.md"
FLAGS = ("clean", "suspect", "drop")

PROMPT = """You are the pre-review judge for GhostLM's chat fine-tuning data. Below is the review standard, then {n} conversations as JSON, each with an id. Read every turn of every conversation completely.

For each conversation decide:
- "clean": every check in the standard passes. You verified the facts, redid any arithmetic, and traced any code or command.
- "suspect": something may be wrong but you are not certain, or a small fix would be needed (a wrong number, a broken flag, an odd user turn).
- "drop": a clear factual error, wrong maths, code that does not do what it claims, filler, identity leak, dashes, a forced persona, or a safety problem.

Be strict: a conversation with one wrong claim is "drop", not "suspect". Do not reward length or confidence.

Output exactly {n} lines and nothing else, one JSON object per conversation, in the same order:
{{"id": "...", "flag": "clean|suspect|drop", "issues": ["short, specific reason", ...]}}
"issues" is an empty list for clean. Quote the exact wrong claim or line in each issue, briefly.

=== Review standard ===
{standard}

=== Conversations ===
{convs}"""


def judged_ids() -> set[str]:
    if not OUT.exists():
        return set()
    return {json.loads(line)["id"] for line in OUT.read_text().splitlines() if line.strip()}


def load_judgements() -> dict:
    if not OUT.exists():
        return {}
    out = {}
    for line in OUT.read_text().splitlines():
        if line.strip():
            j = json.loads(line)
            out[j["id"]] = j
    return out


def parse(text: str, wanted: set[str]) -> list[dict]:
    decoder, pos, out = json.JSONDecoder(), 0, []
    while (pos := text.find("{", pos)) != -1:
        try:
            obj, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            pos += 1
            continue
        pos = end
        if isinstance(obj, dict) and obj.get("id") in wanted and obj.get("flag") in FLAGS:
            issues = obj.get("issues") or []
            out.append({"id": obj["id"], "flag": obj["flag"],
                        "issues": [str(i)[:300] for i in issues if str(i).strip()][:6]})
    return out


def judge_batch(batch: list[dict], model: str) -> list[dict]:
    standard = STANDARD.read_text()
    standard = standard[standard.index("## What to check"):standard.index("## Verdicts")].strip()
    convs = "\n".join(json.dumps({"id": r["id"], "turns": r["turns"]}, ensure_ascii=False) for r in batch)
    prompt = PROMPT.format(n=len(batch), standard=standard, convs=convs)
    text = ask_http("groq", model, prompt, timeout=600, max_tokens=2500)
    rows = parse(text, {r["id"] for r in batch})
    seen = set()
    out = []
    for row in rows:
        if row["id"] not in seen:
            seen.add(row["id"])
            row["judge"] = last_model()
            out.append(row)
    return out


def run_once(batch_size: int, model: str, limit: int | None) -> int:
    done, judged = done_ids(), judged_ids()
    todo = [r for r in raw_records() if r["id"] not in done and r["id"] not in judged]
    if limit:
        todo = todo[:limit]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for i in range(0, len(todo), batch_size):
        batch = todo[i:i + batch_size]
        rows = judge_batch(batch, model)
        if not rows:
            print(f"{time.strftime('%H:%M:%S')} batch {i // batch_size}: no usable output", flush=True)
            continue
        with OUT.open("a") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        written += len(rows)
        if (i // batch_size) % 25 == 0:
            flags = {k: sum(1 for r in rows if r["flag"] == k) for k in FLAGS}
            print(f"{time.strftime('%H:%M:%S')} {i + len(batch)}/{len(todo)} judged, last batch {flags}", flush=True)
    return written


def cmd_status() -> None:
    judgements = load_judgements()
    done = done_ids()
    raw = {r["id"] for r in raw_records()}
    pending = raw - done
    flags = {k: sum(1 for j in judgements.values() if j["flag"] == k and j["id"] in pending) for k in FLAGS}
    print(f"unreviewed {len(pending)}, judged {len(pending & set(judgements))}, flags on unreviewed {flags}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--batch", type=int, default=4)
    r.add_argument("--model", default=DEFAULT_MODELS["groq"])
    r.add_argument("--limit", type=int, default=None, help="judge at most this many conversations, then stop")
    r.add_argument("--loop", action="store_true", help="keep polling for new conversations every 30 minutes")
    sub.add_parser("status")
    a = p.parse_args()
    if a.cmd == "status":
        return cmd_status()
    while True:
        n = run_once(a.batch, a.model, a.limit)
        print(f"{time.strftime('%H:%M:%S')} judged {n} conversations this pass", flush=True)
        if not a.loop or a.limit:
            break
        time.sleep(1800)


if __name__ == "__main__":
    main()
