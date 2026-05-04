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
from ghostlm.tokenizer import GhostTokenizer, load_tokenizer


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


def load_ctf_eval_bench(path: str = "data/raw/ctf_eval_bench.jsonl") -> List[Dict]:
    """Load the in-repo CTF MCQ evaluation benchmark (issue #6).

    30 hand-written questions across web / crypto / pwn / rev / forensics /
    stego / misc CTF categories. Schema matches CTIBench MCQ so the same
    scoring path works. Source: ``data/raw/ctf_eval_bench.jsonl``.
    """
    from pathlib import Path as _P
    p = _P(path)
    if not p.exists():
        raise FileNotFoundError(
            f"CTF benchmark not found at {p}. Run from the GhostLM repo root, "
            "or pass --ctf-bench-path."
        )
    out: List[Dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.append({
                "question": r["question"],
                "choices": r["choices"],
                "answer": r["answer"].strip().upper()[:1],
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


def format_mcq_prompt(
    record: Dict,
    tokenizer: GhostTokenizer,
    *,
    chat_format: bool,
    rag_passages: Optional[List[Dict]] = None,
) -> List[int]:
    """Build the prompt token ids for one MCQ.

    Uses the chat format when the model was chat-tuned; otherwise emits a
    plain "Question: ... Answer:" completion prompt. If ``rag_passages`` is
    provided, prepends a "Reference passages:" block ahead of the question —
    this is the RAG path.
    """
    question = record["question"]
    choices = record["choices"]
    body_lines = [f"{k}) {v}" for k, v in choices.items() if v]
    body_parts = []
    if rag_passages:
        ref_lines = []
        for i, p in enumerate(rag_passages):
            text = p["text"]
            if len(text) > 350:
                text = text[:350].rsplit(" ", 1)[0] + "…"
            ref_lines.append(f"[{i + 1}] ({p['source']} {p.get('ref', '')}) {text}")
        body_parts.append("Reference passages:\n\n" + "\n\n".join(ref_lines) + "\n")
    body_parts.append(
        f"Pick the best answer (A, B, C, or D) for this multiple-choice "
        f"cybersecurity question.\n\nQuestion: {question}\n\n"
        + "\n".join(body_lines)
        + "\n\nAnswer:"
    )
    body = "\n".join(body_parts)
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
    rag_passages: Optional[List[Dict]] = None,
) -> Tuple[str, Dict[str, float]]:
    """Return the predicted choice + per-choice logits for one record.

    The choice with the highest single-token log-probability at the position
    immediately after the prompt is selected. When ``rag_passages`` is
    provided, the prompt is prefixed with the retrieved context.
    """
    prompt_ids = format_mcq_prompt(
        record, tokenizer, chat_format=chat_format, rag_passages=rag_passages,
    )
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
        # Use the public encode() so this works for both the legacy tiktoken
        # backend (GhostTokenizer) and the v0.5 HF tokenizers backend
        # (GhostTokenizerV05) without special-casing the underlying engine.
        ids_space = tokenizer.encode(f" {ch}")
        ids_plain = tokenizer.encode(ch)
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
    retriever=None,
    top_k: int = 4,
) -> Dict:
    """Run the model over a benchmark and return aggregate accuracy.

    If ``retriever`` is non-None, it is called per record as
    ``retriever(record["question"], k=top_k) -> List[passage_dict]`` and the
    resulting passages are prepended to the MCQ prompt — i.e. RAG mode.
    """
    correct = 0
    total = 0
    examples: List[Dict] = []
    for i, rec in enumerate(dataset):
        if limit and i >= limit:
            break
        if not rec["answer"] or rec["answer"] not in CHOICES:
            continue
        passages = retriever(rec["question"], k=top_k) if retriever else None
        pred, scores = score_record(
            model, tokenizer, rec,
            chat_format=chat_format, device=device, rag_passages=passages,
        )
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


def build_retriever(rag_dir: str, device: str):
    """Return a closure ``query(text, k) -> List[chunk_dict]`` over the RAG index."""
    import numpy as np
    import torch.nn.functional as F
    from scripts.build_rag_index import load_embedder

    rag_path = Path(rag_dir)
    matrix = np.load(rag_path / "index.npy")
    chunks: List[Dict] = []
    with (rag_path / "chunks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    e_tok, e_model = load_embedder(device)

    def retrieve(query: str, k: int = 4) -> List[Dict]:
        """Return top-K passage dicts for ``query``."""
        text = "Represent this sentence for searching relevant passages: " + query
        enc = e_tok(text, padding=True, truncation=True, max_length=512,
                    return_tensors="pt").to(device)
        with torch.no_grad():
            out = e_model(**enc)
        emb = F.normalize(out.last_hidden_state[:, 0], p=2, dim=-1)
        emb = emb.cpu().to(torch.float32).numpy().reshape(-1)
        scores = matrix @ emb
        top = np.argsort(-scores)[:k]
        return [chunks[i] for i in top]

    return retrieve


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
                            "cybermetric-2000", "ctibench-mcq",
                            "ctf-eval"],
                   help="CyberMetric splits require HuggingFace gated access; "
                        "CTIBench MCQ is open and the default. ctf-eval is "
                        "the in-repo 30-question CTF benchmark "
                        "(data/raw/ctf_eval_bench.jsonl).")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap evaluation to N records per benchmark (for smoke tests)")
    p.add_argument("--no-chat-format", action="store_true",
                   help="Force completion-style prompts (use for non-chat-tuned checkpoints)")
    p.add_argument("--tokenizer", default=None,
                   help="Optional path to a v0.5 tokenizer.json. When provided, "
                        "uses the 32K BPE; otherwise legacy tiktoken GPT-2.")
    p.add_argument("--rag-dir", default=None,
                   help="If set, retrieve from data/rag and prepend top-K passages to each MCQ.")
    p.add_argument("--rag-top-k", type=int, default=4)
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
    tokenizer = load_tokenizer(args.tokenizer) if args.tokenizer else GhostTokenizer()
    chat_format = not args.no_chat_format and cfg.vocab_size >= tokenizer.vocab_size

    label = args.label or Path(args.checkpoint).parent.name + "/" + Path(args.checkpoint).stem
    rag_label = f" + RAG(top{args.rag_top_k})" if args.rag_dir else ""
    print(f"Checkpoint: {label}{rag_label}  device={device}  chat_format={chat_format}")

    retriever = None
    if args.rag_dir:
        print(f"Loading retriever from {args.rag_dir}...")
        retriever = build_retriever(args.rag_dir, device)

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
        elif bench == "ctf-eval":
            data = load_ctf_eval_bench()
        else:
            raise ValueError(f"Unknown bench: {bench}")
        print(f"  Loaded {len(data)} records")

        result = evaluate(
            model, tokenizer, data,
            chat_format=chat_format, device=device, limit=args.limit,
            retriever=retriever, top_k=args.rag_top_k,
        )
        print(f"  Accuracy: {result['correct']}/{result['n']} = {result['accuracy']:.3f}")
        for ex in result["examples"]:
            print(f"    - gold={ex['gold']} pred={ex['pred']} ok={ex['correct']}")

        rows.append({
            "checkpoint": label + rag_label, "benchmark": bench,
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
