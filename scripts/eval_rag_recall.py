#!/usr/bin/env python3
"""Diagnostic: does the RAG layer surface the right passages?

Independent of whether the GhostLM model can extract facts from
retrieved context, this script asks a different question: for each
fact-recall v2 prompt, does the top-K retrieved passages set CONTAIN
the canonical answer (or one of its alternates) as a substring?

If retrieval@K is high (say >=70%) but the v0.9+RAG generation score
is still at floor (say 1-5%), the bottleneck is the LM's inability to
condition on retrieved context, not the retriever. That's the 81M
parameter scale showing through.

If retrieval@K is also low, the bottleneck is the index: the corpus
either doesn't contain the answer text or the BGE embedder doesn't
surface it for this query.

The two failure modes have different fixes (LM scaling vs corpus
coverage / embedder choice / chunking strategy), so distinguishing
them is worth ten lines of script.

Output: one line per question with the retrieval@K verdict, plus a
final summary.

Run on Mac alongside the RAG bench:

    PYTHONPATH=. python3 scripts/eval_rag_recall.py \\
        --rag-dir data/rag --bench data/raw/fact_recall_bench_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Allow running from any cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


WORD_CHARS = re.compile(r"[A-Za-z0-9]")
WS_RE = re.compile(r"\s+")


def normalize(s: str) -> str:
    return WS_RE.sub(" ", s.lower()).strip()


def appears_with_boundary(needle: str, haystack: str) -> bool:
    """Boundary-respecting substring match. Same semantics as the v2 grader."""
    if not needle:
        return False
    start = 0
    while True:
        i = haystack.find(needle, start)
        if i == -1:
            return False
        before_ok = (i == 0) or not WORD_CHARS.match(haystack[i - 1])
        after_idx = i + len(needle)
        after_ok = (after_idx == len(haystack)) or not WORD_CHARS.match(haystack[after_idx])
        if before_ok and after_ok:
            return True
        start = i + 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rag-dir", default="data/rag",
                   help="Directory with index.npy + chunks.jsonl + meta.json")
    p.add_argument("--bench", default="data/raw/fact_recall_bench_v2.jsonl")
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--embedder", default="BAAI/bge-small-en-v1.5")
    p.add_argument("--out", default="logs/rag_retrieval_at_k.jsonl")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    import numpy as np
    import torch
    import torch.nn.functional as F

    rag_dir = Path(args.rag_dir)
    bench_path = Path(args.bench)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Bench:    {bench_path}")
    print(f"RAG dir:  {rag_dir}")
    print(f"Top-K:    {args.top_k}")
    print()

    print("Loading RAG index...")
    idx = np.load(rag_dir / "index.npy")
    if idx.dtype != np.float32:
        idx = idx.astype(np.float32)
    chunks: List[dict] = []
    with (rag_dir / "chunks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    print(f"  {len(chunks)} chunks, dim {idx.shape[1]}")

    print("Loading embedder (CPU)...")
    from transformers import AutoModel, AutoTokenizer
    e_tok = AutoTokenizer.from_pretrained(args.embedder)
    e_model = AutoModel.from_pretrained(args.embedder).to("cpu").eval()

    print("Loading bench...")
    bench: List[dict] = []
    with bench_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                bench.append(json.loads(line))
    print(f"  {len(bench)} questions")
    print()

    out_fh = out_path.open("w", encoding="utf-8")
    hits = 0
    hits_per_topic: dict = {}
    total_per_topic: dict = {}

    for rec in bench:
        q = rec["prompt"]
        topic = rec.get("topic", "misc")
        total_per_topic[topic] = total_per_topic.get(topic, 0) + 1

        text = "Represent this sentence for searching relevant passages: " + q
        enc = e_tok(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            out = e_model(**enc)
        emb = out.last_hidden_state[:, 0]
        emb = F.normalize(emb, p=2, dim=-1)
        q_vec = emb.cpu().to(torch.float32).numpy().reshape(-1)
        scores = idx @ q_vec
        top_idxs = np.argsort(-scores)[: args.top_k]
        retrieved = [chunks[int(j)] for j in top_idxs]

        # Combine retrieved chunk text into one searchable haystack.
        haystack = normalize(" ".join((ch.get("text") or "") for ch in retrieved))

        # Answer set: must_appear (ALL), or alternates (ANY of), with boundary match.
        must = rec.get("must_appear") or []
        if must:
            ok = all(appears_with_boundary(normalize(p), haystack) for p in must)
            criterion = f"all of must_appear={must}"
        else:
            alts = [rec["answer"]] + list(rec.get("alternates", []) or [])
            ok = any(appears_with_boundary(normalize(a), haystack) for a in alts)
            criterion = f"any of alternates={alts}"

        if ok:
            hits += 1
            hits_per_topic[topic] = hits_per_topic.get(topic, 0) + 1

        out_fh.write(json.dumps({
            "id": rec.get("id"),
            "topic": topic,
            "prompt": q,
            "retrieved_sources": [c.get("source") for c in retrieved],
            "retrieved_refs": [c.get("ref") for c in retrieved],
            "retrieval_at_k": ok,
            "criterion": criterion,
        }, ensure_ascii=False) + "\n")

    out_fh.close()

    n = len(bench)
    print(f"\n=== Retrieval@{args.top_k} ===")
    print(f"  hits: {hits}/{n}  ({100 * hits / max(1, n):.1f}%)")
    print("  per-topic:")
    for t in sorted(total_per_topic):
        h = hits_per_topic.get(t, 0)
        tt = total_per_topic[t]
        print(f"    {t:10s}  {h:3d}/{tt:3d}  ({100*h/max(1,tt):.1f}%)")
    print(f"\nLog written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
