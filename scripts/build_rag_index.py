#!/usr/bin/env python3
"""Build the RAG index over the GhostLM cybersecurity corpus.

Chunks ``data/processed/train.jsonl`` into ~256-token passages, embeds each
with a small bi-encoder (``BAAI/bge-small-en-v1.5``, 30MB), and saves the
resulting FP32 matrix + chunk metadata as plain NumPy artifacts. At 75K
chunks × 384 dims, the index is ~115 MB — small enough to load into RAM and
use brute-force cosine similarity at query time (no LanceDB / FAISS needed
at this scale).

Output (under ``data/rag/``):
- ``index.npy`` — float32 array of shape (N, 384), L2-normalized
- ``chunks.jsonl`` — one record per chunk: {chunk_id, source, ref, text, ...}

Run once before ``scripts/rag_chat.py``. Re-run after corpus updates.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
import torch.nn.functional as F


def chunk_text(text: str, max_tokens: int = 256, overlap: int = 32) -> List[str]:
    """Split text into overlapping word-count chunks (proxy for tokens).

    Uses word count as a cheap proxy for token count — close enough for chunk
    boundaries. Real token counts are computed downstream by GhostTokenizer
    when assembling the final RAG prompt.
    """
    words = text.split()
    if len(words) <= max_tokens:
        return [text]
    chunks = []
    stride = max_tokens - overlap
    for i in range(0, len(words), stride):
        chunk = " ".join(words[i : i + max_tokens])
        chunks.append(chunk)
        if i + max_tokens >= len(words):
            break
    return chunks


def load_corpus(path: Path) -> Iterable[dict]:
    """Stream JSONL records."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def make_chunks(records: Iterable[dict], chunk_tokens: int) -> List[dict]:
    """Turn corpus records into per-passage chunks with metadata."""
    out: List[dict] = []
    for r in records:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        for j, body in enumerate(chunk_text(text, max_tokens=chunk_tokens)):
            out.append({
                "chunk_id": f"{r.get('id','?')}#{j}",
                "source": r.get("source", "unknown"),
                "ref": r.get("id", ""),
                "text": body,
            })
    return out


def load_embedder(device: str):
    """Load BGE-small with raw transformers (avoids the sentence-transformers dep)."""
    from transformers import AutoModel, AutoTokenizer
    name = "BAAI/bge-small-en-v1.5"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()
    return tok, model


def embed_batch(tok, model, texts: List[str], device: str, max_length: int = 512) -> np.ndarray:
    """Encode a batch of texts as L2-normalized float32 vectors."""
    # BGE wants a small instruction prefix on the query side; for the corpus side
    # we use plain text.
    enc = tok(texts, padding=True, truncation=True, max_length=max_length,
              return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc)
    # CLS-token pooling is what BGE uses by default.
    emb = out.last_hidden_state[:, 0]
    emb = F.normalize(emb, p=2, dim=-1)
    return emb.cpu().to(torch.float32).numpy()


def resolve_device(arg: str) -> str:
    """Pick a device honoring ``auto``."""
    if arg != "auto":
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="Build the GhostLM RAG index")
    p.add_argument("--corpus", default="data/processed/train.jsonl")
    p.add_argument("--out-dir", default="data/rag")
    p.add_argument("--chunk-tokens", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of source records (for smoke tests)")
    return p.parse_args()


def main() -> None:
    """Build the chunk list, embed each chunk, save index + metadata."""
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Device: {device}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading corpus + chunking...")
    records = list(load_corpus(Path(args.corpus)))
    if args.limit:
        records = records[: args.limit]
    chunks = make_chunks(records, args.chunk_tokens)
    print(f"  Built {len(chunks):,} chunks from {len(records):,} records")

    chunks_path = out / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"  Wrote {chunks_path}")

    print("Loading embedder...")
    tok, model = load_embedder(device)

    print("Embedding...")
    n = len(chunks)
    matrix = np.empty((n, 384), dtype=np.float32)
    for i in range(0, n, args.batch_size):
        batch = [c["text"] for c in chunks[i : i + args.batch_size]]
        matrix[i : i + len(batch)] = embed_batch(tok, model, batch, device)
        if (i // args.batch_size) % 25 == 0:
            print(f"  {i:>6} / {n}")

    index_path = out / "index.npy"
    np.save(index_path, matrix)
    print(f"  Wrote {index_path}  ({matrix.shape[0]} × {matrix.shape[1]})")

    meta_path = out / "meta.json"
    meta = {
        "n_chunks": len(chunks),
        "embedding_dim": int(matrix.shape[1]),
        "embedder": "BAAI/bge-small-en-v1.5",
        "chunk_tokens": args.chunk_tokens,
        "corpus": args.corpus,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  Wrote {meta_path}")


if __name__ == "__main__":
    main()
