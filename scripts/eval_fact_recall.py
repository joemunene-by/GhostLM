#!/usr/bin/env python3
"""Free-form fact-recall benchmark with substring grading.

Multiple-choice text-scoring rewards "this option's words match the
question topic" more than "the model knows the fact." This bench is
the free-form complement: hand-written ``data/raw/fact_recall_bench.jsonl``
asks 50 short factual questions where the right answer is a single
identifier (CVE id, CWE number, MITRE technique, port, header name,
etc.). The grader checks whether any of a question's ``alternates``
appears as a normalized substring in the model's first-N-token
completion.

Substring match is permissive: it credits the model for surfacing the
right token even if it doesn't structurally answer the question. That
matches the intent — we want to know if v0.9's "magic numbers near
the surface" pattern from the qualitative comparison reproduces at
scale.

Output: per-checkpoint pass rate, per-topic breakdown, and a JSONL
log of every (question, completion, matched_alternate?) row so we
can read the actual completions back later.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.tokenizer import GhostTokenizer, load_tokenizer


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="Free-form fact-recall bench")
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Format: label1=path1 label2=path2 ...")
    p.add_argument("--bench-jsonl", default="data/raw/fact_recall_bench.jsonl")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--device", default="mps")
    p.add_argument("--max-tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-jsonl", default=None,
                   help="Per-question completion log for offline review")
    return p.parse_args()


def parse_checkpoints(specs: list[str]) -> list[tuple[str, str]]:
    """Parse 'label=path' specs."""
    out = []
    for s in specs:
        if "=" not in s:
            raise SystemExit(f"--checkpoints expects label=path, got: {s}")
        label, path = s.split("=", 1)
        out.append((label.strip(), path.strip()))
    return out


def load_bench(path: str) -> list[dict]:
    """Load fact_recall_bench.jsonl."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("prompt") and r.get("answer"):
                out.append(r)
    return out


def normalize(s: str) -> str:
    """Lowercase, single-spaces, strip non-alphanum-or-dash-or-slash."""
    s = s.lower()
    # Keep alphanum, hyphen, slash, dot, colon, equals (for headers).
    s = re.sub(r"[^a-z0-9\-./:= ]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def matched(completion: str, alternates: list[str]) -> str | None:
    """Return the first alternate found as a substring (normalized), or None."""
    norm_comp = normalize(completion)
    for alt in alternates:
        norm_alt = normalize(alt)
        if not norm_alt:
            continue
        if norm_alt in norm_comp:
            return alt
    return None


def load_model(path: str, device: str) -> GhostLM:
    """Load a checkpoint with its embedded config."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    saved = ckpt["config"]
    cfg = GhostLMConfig(**{
        f.name: saved[f.name] for f in fields(GhostLMConfig) if f.name in saved
    })
    cfg.device = device
    model = GhostLM(cfg).to(device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt["model"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def chat_completion(model: GhostLM, tokenizer: GhostTokenizer,
                    prompt: str, *, device: str,
                    max_tokens: int, temperature: float, top_k: int) -> str:
    """One-shot chat completion."""
    ids = tokenizer.format_chat_prompt([{"role": "user", "content": prompt}])
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        out_ids = model.generate(
            x,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
        )
    new_ids = out_ids[0, len(ids):].tolist()
    text = tokenizer.decode(new_ids)
    return text.split("<|ghost_end|>")[0].strip()


def main() -> None:
    """Run the bench."""
    args = parse_args()
    torch.manual_seed(args.seed)

    pairs = parse_checkpoints(args.checkpoints)
    tokenizer = load_tokenizer(args.tokenizer) if args.tokenizer else GhostTokenizer()

    bench = load_bench(args.bench_jsonl)
    print(f"Loaded {len(bench)} fact-recall questions from {args.bench_jsonl}")
    print(f"Comparing {len(pairs)} checkpoints.")
    print()

    summary: dict[str, Any] = {
        "bench_jsonl": args.bench_jsonl,
        "n_questions": len(bench),
        "checkpoints": [{"label": l, "path": p} for l, p in pairs],
        "sampling": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "results": {},
    }
    detailed: list[dict] = []

    for label, path in pairs:
        print(f"=== {label} ===")
        model = load_model(path, args.device)

        topic_correct: dict[str, int] = {}
        topic_total: dict[str, int] = {}
        correct = 0
        for rec in bench:
            completion = chat_completion(
                model, tokenizer, rec["prompt"],
                device=args.device,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            alts = [rec["answer"]] + list(rec.get("alternates", []))
            hit = matched(completion, alts)
            topic = rec.get("topic", "misc")
            topic_total[topic] = topic_total.get(topic, 0) + 1
            if hit is not None:
                correct += 1
                topic_correct[topic] = topic_correct.get(topic, 0) + 1
            detailed.append({
                "checkpoint": label,
                "id": rec["id"],
                "topic": topic,
                "prompt": rec["prompt"],
                "answer": rec["answer"],
                "completion": completion,
                "hit": hit,
            })

        del model
        if args.device == "mps":
            torch.mps.empty_cache()

        n = len(bench)
        acc = correct / n if n else 0.0
        print(f"  overall: {correct}/{n} = {acc:.3f}")
        topic_summary = {}
        for t in sorted(topic_total):
            tc = topic_correct.get(t, 0)
            tt = topic_total[t]
            topic_summary[t] = {"correct": tc, "total": tt, "acc": tc / tt if tt else 0.0}
            print(f"    {t:<10s} {tc:>2d}/{tt:<2d} = {tc / tt:.3f}")
        print()

        summary["results"][label] = {
            "checkpoint_path": path,
            "correct": correct,
            "total": n,
            "acc": acc,
            "by_topic": topic_summary,
        }

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"saved summary: {out_path}")
    if args.out_jsonl:
        out_path = Path(args.out_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for row in detailed:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"saved per-question log: {out_path}")


if __name__ == "__main__":
    main()
