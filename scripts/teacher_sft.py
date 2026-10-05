#!/usr/bin/env python3
"""Grounded chat SFT data written by an external teacher model, one batch at a time.

    python scripts/teacher_sft.py queue             # sample passages into data/teacher/queue/
    python scripts/teacher_sft.py validate N        # check data/teacher/out/batch_N.jsonl
    python scripts/teacher_sft.py status            # batches done / pending
    python scripts/teacher_sft.py merge             # accepted examples -> data/processed/teacher_sft.jsonl

The teacher (any chat model or coding agent) reads a queue batch of corpus
passages and writes question/answer pairs grounded in them. ``validate``
enforces the format and checks that answers actually draw on the passage;
``merge`` emits ``{"turns": [...]}`` records in the same "Reference passages /
Question" layout the web app and RAFT use. See docs/teacher_sft.md.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TEACHER = ROOT / "data/teacher"
QUEUE, OUT = TEACHER / "queue", TEACHER / "out"
RAG = ROOT / "data/rag"
BATCH_SIZE = 10
PASSAGE_CHARS = 1400
UNANSWERABLE = "My sources don't cover that."

# Curated sources are a sliver of the index but the most valuable to ground on.
SOURCE_WEIGHTS = {
    "nvd": 15, "primus_seed": 10, "primus_fineweb": 10, "wikipedia": 12, "mitre_full": 8,
    "cwe": 6, "cisa_kev": 5, "nist_sp800": 5, "capec": 4, "exploitdb": 4, "ctftime": 4,
    "security_blogs": 4, "owasp_cheatsheets": 3, "owasp_wstg": 2, "owasp_asvs": 1,
    "vendor_research": 3, "wikipedia_cyber": 2, "security_code": 2,
}
STOPWORDS = set("""about above after again against because before being below between could does doing during
each from further having here into itself just more most other over same should some such than that their them
then there these they this those through under until very were what when where which while will with would your
also using used uses based allow allows within""".split())
BANNED = ("as an ai", "the passage", "the text above", "the provided", "according to the passage",
          "the reference", "reference passage", "this passage")


def content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9][a-z0-9\-_.]{3,}", text.lower()) if w not in STOPWORDS}


def grounding(answer: str, passage: str) -> float:
    words = content_words(re.sub(r"\[\d+\]", "", answer))
    return len(words & content_words(passage)) / max(1, len(words))


def build_queue(n_batches: int, seed: int) -> None:
    offsets = np.load(RAG / "chunk_offsets.npy")
    by_source: dict = {}
    with (RAG / "chunks.jsonl").open("rb") as f:
        for i, off in enumerate(offsets):
            f.seek(int(off))
            src = json.loads(f.readline()).get("source", "")
            if src in SOURCE_WEIGHTS:
                by_source.setdefault(src, []).append(i)
    rng = random.Random(seed)
    total = n_batches * BATCH_SIZE
    weight_sum = sum(w for s, w in SOURCE_WEIGHTS.items() if s in by_source)
    picks = []
    for src, w in SOURCE_WEIGHTS.items():
        pool = by_source.get(src, [])
        picks += rng.sample(pool, min(len(pool), round(total * w / weight_sum)))
    rng.shuffle(picks)
    QUEUE.mkdir(parents=True, exist_ok=True)
    existing = len(list(QUEUE.glob("batch_*.jsonl")))
    rows, written = [], 0
    with (RAG / "chunks.jsonl").open("rb") as f:
        for i in picks:
            f.seek(int(offsets[i]))
            c = json.loads(f.readline())
            text = " ".join(c["text"].split())
            if len(text) < 300:
                continue
            rows.append({"pid": f"p{i}", "source": c["source"], "ref": str(c.get("ref", "")),
                         "text": text[:PASSAGE_CHARS]})
            if len(rows) == BATCH_SIZE:
                written += 1
                (QUEUE / f"batch_{existing + written:04d}.jsonl").write_text(
                    "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
                rows = []
    print(f"wrote {written} batches of {BATCH_SIZE} to {QUEUE}")
    print("sources:", dict(Counter(json.loads(l)["source"] for p in QUEUE.glob("batch_*.jsonl")
                                   for l in p.read_text().splitlines()).most_common()))


def validate(batch: str) -> tuple[list, list]:
    """Return (accepted records, error strings) for one output batch."""
    name = f"batch_{int(batch):04d}.jsonl"
    queue_path, out_path = QUEUE / name, OUT / name
    if not queue_path.exists():
        return [], [f"no queue file {queue_path}"]
    if not out_path.exists():
        return [], [f"no output file yet: write {out_path}"]
    passages = {json.loads(l)["pid"]: json.loads(l) for l in queue_path.read_text().splitlines() if l.strip()}
    accepted, errors, covered, unanswerable, skipped = [], [], set(), 0, 0
    for n, line in enumerate(out_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {n}: not valid JSON ({e.msg}); write one JSON object per line")
            continue
        if r.get("type") == "skip" and r.get("pid") in passages:
            covered.add(r["pid"])
            skipped += 1
            continue
        missing = {"pid", "type", "question", "answer"} - set(r)
        if missing:
            errors.append(f"line {n}: missing keys {sorted(missing)}")
            continue
        p = passages.get(r["pid"])
        q, a, kind = str(r["question"]).strip(), str(r["answer"]).strip(), r["type"]
        problems = []
        if p is None:
            problems.append(f"pid {r['pid']!r} is not in {name}")
        if kind not in ("answer", "unanswerable"):
            problems.append('type must be "answer", "unanswerable" or "skip"')
        if not 10 <= len(q) <= 300:
            problems.append("question must be 10-300 characters")
        if any(b in (q + " " + a).lower() for b in BANNED):
            problems.append("do not mention 'the passage/text/reference'; write as if answering a user")
        if p is not None and kind == "answer":
            if not 40 <= len(a) <= 900:
                problems.append("answer must be 40-900 characters")
            if "[1]" not in a:
                problems.append("answer must cite the passage as [1]")
            g = grounding(a, p["text"])
            if g < 0.5:
                problems.append(f"answer is not grounded enough in the passage ({g:.0%} of its key words "
                                f"appear there, need 50%); use the passage's facts and wording")
        if p is not None and kind == "unanswerable":
            if a != UNANSWERABLE:
                problems.append(f'unanswerable answers must be exactly "{UNANSWERABLE}"')
            if grounding(q, p["text"]) > 0.3:
                problems.append("unanswerable question overlaps the passage; ask about something it does not cover")
        if problems:
            errors.append(f"line {n}: " + "; ".join(problems))
            continue
        covered.add(r["pid"])
        unanswerable += kind == "unanswerable"
        accepted.append({**r, "question": q, "answer": a})
    uncovered = set(passages) - covered
    if uncovered:
        errors.append(f"no accepted example for pids {sorted(uncovered)}")
    if skipped > 3:
        errors.append(f"{skipped} passages skipped; skip only junk (navigation, bare code), at most 3 per batch")
    if unanswerable > 3:
        errors.append(f"{unanswerable} unanswerable examples; keep it to 1-3 per batch")
    return accepted, errors


ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _rules() -> str:
    """The question/answer rules from docs/teacher_prompt.md, without the agent loop."""
    text = (ROOT / "docs/teacher_prompt.md").read_text()
    return text[text.index("Each output line is:"):text.index("Start now")].strip()


def _ask(model: str, prompt: str, timeout: int) -> str:
    import subprocess
    try:
        out = subprocess.run(["opencode", "run", "-m", model, prompt], capture_output=True,
                             text=True, timeout=timeout, cwd=ROOT).stdout
    except subprocess.TimeoutExpired:
        return ""
    return ANSI.sub("", out)


def _json_lines(text: str) -> list:
    rows = []
    for line in text.splitlines():
        line = line.strip().strip("`")
        if line.startswith("{") and line.endswith("}"):
            try:
                rows.append(json.dumps(json.loads(line), ensure_ascii=False))
            except json.JSONDecodeError:
                pass
    return rows


def run(model: str, batches: int, retries: int = 2, timeout: int = 600) -> None:
    """Drive the teacher non-interactively: one `opencode run` call per batch, validated, with retries."""
    OUT.mkdir(parents=True, exist_ok=True)
    rules = _rules()
    todo = [p for p in sorted(QUEUE.glob("batch_*.jsonl")) if not (OUT / p.name).exists()][:batches]
    for qp in todo:
        num = qp.stem.split("_")[1]
        prompt = (f"{rules}\n\nHere are the 10 passages, one JSON object per line:\n{qp.read_text()}\n"
                  "Reply with only the output JSON lines, nothing else.")
        errors = ["no attempt"]
        for attempt in range(retries + 1):
            rows = _json_lines(_ask(model, prompt, timeout))
            if not rows:
                errors = ["model returned no JSON lines"]
                continue
            (OUT / qp.name).write_text("\n".join(rows) + "\n")
            accepted, errors = validate(num)
            if not errors:
                break
            prompt += ("\n\nYour previous output had these problems; reply again with the full corrected "
                       "set of JSON lines:\n" + "\n".join(errors[:15]))
        if errors and (OUT / qp.name).exists():
            # Move the partial file aside so the batch is retried on the next run.
            (OUT / qp.name).rename(OUT / f"{qp.stem}.rejected.jsonl")
        print(f"batch {num}: {'OK' if not errors else 'FAILED: ' + errors[0]}", flush=True)


def status() -> None:
    queue = sorted(QUEUE.glob("batch_*.jsonl"))
    done = [p for p in queue if (OUT / p.name).exists() and not validate(p.stem.split("_")[1])[1]]
    nxt = next((p.stem.split("_")[1] for p in queue if not (OUT / p.name).exists()), None)
    print(f"{len(done)}/{len(queue)} batches valid; next batch to do: {nxt or 'none'}")


def merge() -> None:
    records, rejected = [], 0
    for qp in sorted(QUEUE.glob("batch_*.jsonl")):
        num = qp.stem.split("_")[1]
        if not (OUT / qp.name).exists():
            continue
        accepted, errors = validate(num)
        rejected += len(errors)
        passages = {json.loads(l)["pid"]: json.loads(l) for l in qp.read_text().splitlines() if l.strip()}
        for r in accepted:
            p = passages[r["pid"]]
            user = (f"Reference passages:\n[1] ({p['source']} {p['ref']}) {p['text']}\n\n"
                    f"Question: {r['question']}")
            records.append({"source": "teacher_rag", "kind": r["type"],
                            "turns": [{"role": "user", "content": user},
                                      {"role": "assistant", "content": r["answer"]}]})
    out = ROOT / "data/processed/teacher_sft.jsonl"
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    print(f"merged {len(records)} examples -> {out} ({rejected} validation issues skipped)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("queue")
    q.add_argument("--batches", type=int, default=300)
    q.add_argument("--seed", type=int, default=0)
    v = sub.add_parser("validate")
    v.add_argument("batch")
    sub.add_parser("status")
    sub.add_parser("merge")
    r = sub.add_parser("run", help="generate batches unattended through `opencode run`")
    r.add_argument("--model", default="opencode/mimo-v2.6-flash-free")
    r.add_argument("--batches", type=int, default=10)
    args = p.parse_args()
    if args.cmd == "queue":
        build_queue(args.batches, args.seed)
    elif args.cmd == "validate":
        accepted, errors = validate(args.batch)
        for e in errors:
            print("ERROR", e)
        print(f"{'OK' if not errors else 'FIX THE ERRORS ABOVE'}: {len(accepted)} examples accepted")
        return 0 if not errors else 1
    elif args.cmd == "status":
        status()
    elif args.cmd == "run":
        run(args.model, args.batches)
    else:
        merge()
    return 0


if __name__ == "__main__":
    sys.exit(main())
