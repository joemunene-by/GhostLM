#!/usr/bin/env python3
"""Build RAFT (Retrieval-Augmented Fine-Tuning) training data for GhostLM.

Takes the existing chat training set (built by ``build_chat_dataset.py``)
and the RAG index (built by ``build_rag_index.py``), retrieves top-K
passages per question, and emits a new dataset where each user turn is
prefixed with the retrieved passages. Assistant turns are unchanged —
they're already grounded in the same corpus.

The output JSONL is the standard ``{"turns": [...]}`` chat format, so
``scripts/finetune_chat.py`` can train on it without modification.

Why RAFT instead of vanilla retrieve-then-generate (the ``rag_chat.py``
baseline): a chat model that wasn't trained to read retrieved passages
treats them as noise — we measured 0pp lift on CTIBench from the baseline
(`bd95ada`). RAFT teaches the model to attend to the relevant passage
and ignore distractors. Per Zhang et al. (ICML 2024) and follow-ups,
typical lift is +10-25pp on factual MCQ benchmarks at the same model
size. The 83K-chunk RAG index we already built is the input.

Three augmentation modes per record:

- **Oracle (default ~70%)**: top-K passages from BGE retrieval. Some are
  highly relevant, some are weakly relevant. Standard RAG-shape data.
- **Distractor-only (~20%)**: replace the top-1 with a random unrelated
  passage. Forces the model to notice when none of the passages match
  the question and answer from prior knowledge instead of cargo-culting.
- **No-context (~10%)**: drop the passages entirely. Keeps the model
  capable of answering without retrieval (graceful degradation when
  the index is offline or empty).

Small-talk records skip retrieval entirely — pulling cybersec passages
for "hi" or "thanks" is noise.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


# Sources that pass through the RAFT augmentation unchanged. Started as
# {"small_talk"} alone in the v4 attempt; v4 regressed -12pp on CTIBench
# (36.9 -> 25.0 without retrieval, 21.6 with) because MCQ records were
# being augmented with retrieved CVE passages. The retrieved passages are
# noise for "What does XSS stand for? A/B/C/D" questions and dilute the
# crisp answer-letter signal. Adding "mcq" here keeps the v3 MCQ
# behavior intact while still teaching retrieval-aware reading on
# free-form Q&A.
SKIP_SOURCES = {"small_talk", "mcq"}


def load_jsonl(path: Path) -> List[dict]:
    """Read a JSONL file into a list of dicts."""
    out: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(records: List[dict], path: Path) -> int:
    """Write a list of dicts to a JSONL file. Returns the count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(records)


def resolve_device(arg: str) -> str:
    """Pick a torch device honoring ``auto``."""
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_embedder(device: str):
    """Load BGE-small (matches the index builder's choice)."""
    from transformers import AutoModel, AutoTokenizer
    name = "BAAI/bge-small-en-v1.5"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()
    return tok, model


def embed_query(tok, model, query: str, device: str) -> np.ndarray:
    """Encode a query as an L2-normalized FP32 vector."""
    text = "Represent this sentence for searching relevant passages: " + query
    enc = tok(text, padding=True, truncation=True, max_length=512,
              return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc)
    emb = F.normalize(out.last_hidden_state[:, 0], p=2, dim=-1)
    return emb.cpu().to(torch.float32).numpy().reshape(-1)


def trim_passage(text: str, max_chars: int = 350) -> str:
    """Cap a passage near a word boundary."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def format_passages_block(passages: List[Dict]) -> str:
    """Format passages as a numbered Reference passages block."""
    lines = []
    for i, p in enumerate(passages):
        body = trim_passage(p["text"])
        lines.append(f"[{i + 1}] ({p['source']} {p.get('ref','')}) {body}")
    return "Reference passages:\n\n" + "\n\n".join(lines)


def augment_record(
    record: Dict,
    chunks: List[Dict],
    matrix: np.ndarray,
    e_tok,
    e_model,
    device: str,
    *,
    top_k: int,
    mode: str,
    rng: random.Random,
) -> Dict:
    """Return a new record with retrieved passages folded into the user turn.

    Args:
        record: An existing chat record with ``turns``.
        chunks: List of all chunk metadata dicts.
        matrix: (N, dim) float32 L2-normalized embedding matrix.
        e_tok / e_model: BGE tokenizer + model.
        device: Torch device for embedding.
        top_k: Number of passages to attach.
        mode: One of "oracle", "distractor", "no_context".
        rng: Random source for distractor sampling.
    """
    turns = record["turns"]
    user_q = turns[0]["content"]

    if mode == "no_context":
        new_user = (
            "Answer the following from your own knowledge — no reference "
            "passages are provided.\n\n"
            f"Question: {user_q}"
        )
    else:
        q_vec = embed_query(e_tok, e_model, user_q, device)
        scores = matrix @ q_vec
        top_idx = np.argsort(-scores)[:top_k]
        passages = [chunks[i] for i in top_idx]

        if mode == "distractor":
            # Replace the top-1 with a random chunk from a different source —
            # most likely irrelevant to the question.
            other_sources = [
                i for i, c in enumerate(chunks)
                if c.get("source") != passages[0].get("source")
            ]
            if other_sources:
                passages[0] = chunks[rng.choice(other_sources)]

        ref_block = format_passages_block(passages)
        new_user = (
            f"{ref_block}\n\n"
            "Use the reference passages above to answer the question. If the "
            "passages don't contain the answer, say so rather than guessing.\n\n"
            f"Question: {user_q}"
        )

    new_turns = [
        {"role": "user", "content": new_user},
        {"role": "assistant", "content": turns[1]["content"]},
    ]
    out = dict(record)
    out["turns"] = new_turns
    out["raft_mode"] = mode
    return out


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="Build RAFT training data for GhostLM")
    p.add_argument("--in-train", default="data/processed/chat_train.jsonl")
    p.add_argument("--in-val", default="data/processed/chat_val.jsonl")
    p.add_argument("--rag-dir", default="data/rag")
    p.add_argument("--out-train", default="data/processed/raft_train.jsonl")
    p.add_argument("--out-val", default="data/processed/raft_val.jsonl")
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--oracle-frac", type=float, default=0.70)
    p.add_argument("--distractor-frac", type=float, default=0.20)
    p.add_argument("--no-context-frac", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap input records (smoke testing)")
    return p.parse_args()


def main() -> None:
    """Build raft_train.jsonl and raft_val.jsonl."""
    args = parse_args()
    assert abs(args.oracle_frac + args.distractor_frac + args.no_context_frac - 1.0) < 1e-6, (
        "oracle_frac + distractor_frac + no_context_frac must sum to 1.0"
    )
    rng = random.Random(args.seed)
    device = resolve_device(args.device)

    print(f"Device: {device}")
    print("Loading RAG index...")
    rag_dir = Path(args.rag_dir)
    matrix = np.load(rag_dir / "index.npy")
    chunks: List[Dict] = []
    with (rag_dir / "chunks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"  {len(chunks):,} chunks, dim={matrix.shape[1]}")

    print("Loading embedder...")
    e_tok, e_model = load_embedder(device)

    for split, in_path, out_path in [
        ("train", args.in_train, args.out_train),
        ("val", args.in_val, args.out_val),
    ]:
        print()
        print(f"=== {split}: {in_path} → {out_path} ===")
        records = load_jsonl(Path(in_path))
        if args.limit:
            records = records[: args.limit]
        print(f"  Loaded {len(records):,} records")

        out_records: List[Dict] = []
        mode_counts: Counter = Counter()
        skipped = 0

        for r in records:
            if r.get("source") in SKIP_SOURCES:
                # Pass through small_talk records unchanged — retrieving
                # cybersec passages for "hi" is noise.
                out_records.append(r)
                skipped += 1
                continue

            roll = rng.random()
            if roll < args.oracle_frac:
                mode = "oracle"
            elif roll < args.oracle_frac + args.distractor_frac:
                mode = "distractor"
            else:
                mode = "no_context"

            new_rec = augment_record(
                r, chunks, matrix, e_tok, e_model, device,
                top_k=args.top_k, mode=mode, rng=rng,
            )
            out_records.append(new_rec)
            mode_counts[mode] += 1

            if (mode_counts.total() if hasattr(mode_counts, "total") else sum(mode_counts.values())) % 500 == 0:
                done = sum(mode_counts.values())
                print(f"  augmented {done} / {len(records) - skipped}")

        rng.shuffle(out_records)
        n = write_jsonl(out_records, Path(out_path))
        print(f"  Wrote {n:,} records to {out_path}")
        print(f"  Mode distribution: {dict(mode_counts)} (passthrough small_talk: {skipped})")


if __name__ == "__main__":
    main()
