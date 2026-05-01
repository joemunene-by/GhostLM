#!/usr/bin/env python3
"""GhostLM cyber-LLM benchmarks — run open eval suites against a checkpoint.

Two suites, both multiple-choice:

- **CyberMetric** (Tihanyi et al., 2024) — 80/500/2000/10000-question MCQ over
  general security topics. We default to the 500-question split, which runs in
  ~5 minutes on a 45M model on M4 MPS.
- **CTIBench** (Alam et al., NeurIPS 2024) — multiple subtasks (MCQ, RCM, ATT&CK
  mapping). We run the MCQ subset by default since it's the directly-comparable
  one across small open models.

Scoring works by likelihood: encode the prompt, encode each answer choice
("A"/"B"/"C"/"D"), and compare the model's log-probability of each choice token
at the next position. Highest-probability choice = the model's answer.

Adds a row to ``RESULTS.md`` so progress is tracked across checkpoints.

Datasets are pulled from HuggingFace at first run; subsequent runs use the
cached copies under ``~/.cache/huggingface/``.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.tokenizer import GhostTokenizer


CHOICES = ["A", "B", "C", "D"]


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def load_cybermetric(split: str = "500") -> List[Dict]:
    """Load the CyberMetric MCQ benchmark.

    Args:
        split: One of "80", "500", "2000", "10000".

    Returns:
        List of records ``{"question": str, "choices": dict, "answer": "A".."D"}``.
    """
    from datasets import load_dataset
    name = f"CyberMetric-{split}-v1"
    ds = load_dataset("Tihanyi/CyberMetric", name=name, split="test")
    out: List[Dict] = []
    for r in ds:
        # Schema: question, answers (dict A-D), solution
        out.append({
            "question": r["question"],
            "choices": r["answers"] if isinstance(r["answers"], dict) else r["answers"],
            "answer": (r.get("solution") or "").strip().upper()[:1],
        })
    return out


def load_ctibench_mcq() -> List[Dict]:
    """Load the CTIBench multiple-choice subset (2500 records)."""
    from datasets import load_dataset
    ds = load_dataset("AI4Sec/cti-bench", "cti-mcq", split="test")
    out: List[Dict] = []
    for r in ds:
        # CTIBench schema: Question / Option A / Option B / Option C / Option D / GT
        choices = {
            "A": r.get("Option A") or r.get("option_a") or r.get("A"),
            "B": r.get("Option B") or r.get("option_b") or r.get("B"),
            "C": r.get("Option C") or r.get("option_c") or r.get("C"),
            "D": r.get("Option D") or r.get("option_d") or r.get("D"),
        }
        ans = (r.get("GT") or r.get("gt") or r.get("answer") or "").strip().upper()[:1]
        question = r.get("Question") or r.get("question") or r.get("prompt") or ""
        out.append({
            "question": question,
            "choices": choices,
            "answer": ans,
        })
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def format_mcq_prompt(record: Dict, tokenizer: GhostTokenizer, *, chat_format: bool) -> List[int]:
    """Build the prompt token ids for one MCQ.

    Uses the chat format when the model was chat-tuned; otherwise emits a
    plain "Question: ... Answer:" completion prompt.
    """
    question = record["question"]
    choices = record["choices"]
    body_lines = [f"{k}) {v}" for k, v in choices.items() if v]
    body = (
        f"Pick the best answer (A, B, C, or D) for this multiple-choice "
        f"cybersecurity question.\n\nQuestion: {question}\n\n"
        + "\n".join(body_lines)
        + "\n\nAnswer:"
    )
    if chat_format:
        return tokenizer.format_chat_prompt([{"role": "user", "content": body}])
    return tokenizer.encode(body)


def score_record(
    model: GhostLM,
    tokenizer: GhostTokenizer,
    record: Dict,
    *,
    chat_format: bool,
    device: str,
) -> Tuple[str, Dict[str, float]]:
    """Return the predicted choice + per-choice logits for one record.

    The choice with the highest single-token log-probability at the position
    immediately after the prompt is selected.
    """
    prompt_ids = format_mcq_prompt(record, tokenizer, chat_format=chat_format)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
    # Crop if needed
    ctx = model.config.context_length
    x = x[:, -ctx:]
    with torch.no_grad():
        logits, _ = model(x)
    next_logits = logits[0, -1, :]
    log_probs = F.log_softmax(next_logits, dim=-1)

    # Get token ids for " A", " B", " C", " D" (with leading space — matches
    # natural completion). Fall back to no-space if the leading-space token
    # is missing.
    scores: Dict[str, float] = {}
    for ch in CHOICES:
        ids_space = tokenizer._encoder.encode(f" {ch}")
        ids_plain = tokenizer._encoder.encode(ch)
        candidates = [ids_space[0]] if ids_space else []
        if ids_plain:
            candidates.append(ids_plain[0])
        scores[ch] = max(log_probs[c].item() for c in candidates)

    pred = max(scores.items(), key=lambda kv: kv[1])[0]
    return pred, scores


def evaluate(
    model: GhostLM,
    tokenizer: GhostTokenizer,
    dataset: List[Dict],
    *,
    chat_format: bool,
    device: str,
    limit: Optional[int] = None,
) -> Dict:
    """Run the model over a benchmark and return aggregate accuracy."""
    correct = 0
    total = 0
    examples: List[Dict] = []
    for i, rec in enumerate(dataset):
        if limit and i >= limit:
            break
        if not rec["answer"] or rec["answer"] not in CHOICES:
            continue
        pred, scores = score_record(model, tokenizer, rec, chat_format=chat_format, device=device)
        ok = pred == rec["answer"]
        correct += int(ok)
        total += 1
        if i < 3:
            examples.append({
                "question": rec["question"][:200],
                "gold": rec["answer"],
                "pred": pred,
                "scores": {k: round(v, 3) for k, v in scores.items()},
                "correct": ok,
            })
    return {
        "n": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "examples": examples,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def load_model(checkpoint_path: str, device: str) -> Tuple[GhostLM, GhostLMConfig]:
    """Load a GhostLM checkpoint into eval mode."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_raw = ckpt["config"]
    if isinstance(cfg_raw, dict):
        cfg = GhostLMConfig(**{
            f.name: cfg_raw[f.name]
            for f in fields(GhostLMConfig)
            if f.name in cfg_raw
        })
    else:
        cfg = cfg_raw
    model = GhostLM(cfg)
    state = ckpt.get("model_state_dict", ckpt.get("model"))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model.to(device), cfg


def resolve_device(arg: str) -> str:
    """Pick a device honoring ``auto``."""
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def update_results_md(rows: List[Dict], path: Path) -> None:
    """Append benchmark rows to RESULTS.md as a markdown table."""
    header = (
        "# GhostLM benchmark results\n\n"
        "Each row is one (checkpoint × benchmark) score. Updated by "
        "`scripts/run_bench.py`.\n\n"
        "| Checkpoint | Benchmark | n | Correct | Accuracy | Date |\n"
        "|---|---|---:|---:|---:|---|\n"
    )
    existing = ""
    if path.exists():
        existing = path.read_text(encoding="utf-8")
    if not existing.startswith("# GhostLM benchmark results"):
        existing = header
    appended = "\n".join(
        f"| {r['checkpoint']} | {r['benchmark']} | {r['n']} | {r['correct']} | "
        f"{r['accuracy']:.3f} | {r['date']} |"
        for r in rows
    ) + "\n"
    path.write_text(existing + appended, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="GhostLM cyber-LLM benchmark runner")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--label", default=None,
                   help="Human-readable label for the checkpoint in RESULTS.md")
    p.add_argument("--device", default="auto")
    p.add_argument("--bench", nargs="+",
                   default=["ctibench-mcq"],
                   choices=["cybermetric-80", "cybermetric-500",
                            "cybermetric-2000", "ctibench-mcq"],
                   help="CyberMetric splits require HuggingFace gated access; "
                        "CTIBench MCQ is open and the default.")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap evaluation to N records per benchmark (for smoke tests)")
    p.add_argument("--no-chat-format", action="store_true",
                   help="Force completion-style prompts (use for non-chat-tuned checkpoints)")
    p.add_argument("--results-md", default="RESULTS.md")
    p.add_argument("--out-json", default=None,
                   help="Optional path to write detailed JSON results")
    return p.parse_args()


def main() -> None:
    """Run the configured benchmarks and update RESULTS.md."""
    from datetime import date
    args = parse_args()
    device = resolve_device(args.device)
    model, cfg = load_model(args.checkpoint, device)
    tokenizer = GhostTokenizer()
    chat_format = not args.no_chat_format and cfg.vocab_size >= tokenizer.vocab_size

    label = args.label or Path(args.checkpoint).parent.name + "/" + Path(args.checkpoint).stem
    print(f"Checkpoint: {label}  device={device}  chat_format={chat_format}")

    rows: List[Dict] = []
    detailed: Dict[str, Dict] = {}
    today = date.today().isoformat()

    for bench in args.bench:
        print(f"\n--- {bench} ---")
        if bench.startswith("cybermetric-"):
            split = bench.split("-")[1]
            data = load_cybermetric(split)
        elif bench == "ctibench-mcq":
            data = load_ctibench_mcq()
        else:
            raise ValueError(f"Unknown bench: {bench}")
        print(f"  Loaded {len(data)} records")

        result = evaluate(
            model, tokenizer, data,
            chat_format=chat_format, device=device, limit=args.limit,
        )
        print(f"  Accuracy: {result['correct']}/{result['n']} = {result['accuracy']:.3f}")
        for ex in result["examples"]:
            print(f"    - gold={ex['gold']} pred={ex['pred']} ok={ex['correct']}")

        rows.append({
            "checkpoint": label, "benchmark": bench,
            "n": result["n"], "correct": result["correct"],
            "accuracy": result["accuracy"], "date": today,
        })
        detailed[bench] = result

    update_results_md(rows, Path(args.results_md))
    print(f"\nUpdated {args.results_md}")
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(detailed, indent=2), encoding="utf-8")
        print(f"Detailed results: {args.out_json}")


if __name__ == "__main__":
    main()
