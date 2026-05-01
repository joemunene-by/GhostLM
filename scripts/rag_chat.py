#!/usr/bin/env python3
"""Retrieval-augmented chat over GhostLM.

At inference time:
1. Encode the user query with the same BGE bi-encoder used to build the index.
2. Brute-force cosine similarity against the in-memory passage matrix
   (~115 MB at 75K passages × 384 dims — trivially fast on M4).
3. Take the top-K passages (default 4), join them as a "Reference passages"
   prefix, and prepend to the user turn.
4. Run the chat-tuned GhostLM as usual; the model is not RAFT-trained yet so
   it just sees retrieved context as part of the user message — no new tokens
   or special handling needed.

The full RAFT-style retrieval-aware fine-tune is a separate session — this
script is the working RAG baseline that proves the plumbing and gives an
honest "did retrieval help?" measurement.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.tokenizer import GhostTokenizer
from scripts.chat import generate_until_end, resolve_device


def load_index(rag_dir: Path) -> Tuple[np.ndarray, List[dict], dict]:
    """Load the embedding matrix, chunk metadata, and meta.json."""
    matrix = np.load(rag_dir / "index.npy")
    chunks = []
    with (rag_dir / "chunks.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    meta = json.loads((rag_dir / "meta.json").read_text())
    return matrix, chunks, meta


def load_embedder(device: str):
    """Load BGE-small via transformers."""
    from transformers import AutoModel, AutoTokenizer
    name = "BAAI/bge-small-en-v1.5"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()
    return tok, model


def embed_query(tok, model, query: str, device: str) -> np.ndarray:
    """Encode a single query as an L2-normalized float32 vector.

    BGE recommends prefixing queries with an instruction string for retrieval.
    """
    text = "Represent this sentence for searching relevant passages: " + query
    enc = tok(text, padding=True, truncation=True, max_length=512,
              return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**enc)
    emb = out.last_hidden_state[:, 0]
    emb = F.normalize(emb, p=2, dim=-1)
    return emb.cpu().to(torch.float32).numpy().reshape(-1)


def top_k(query_vec: np.ndarray, matrix: np.ndarray, k: int) -> List[int]:
    """Return the indices of the top-K most similar rows (cosine)."""
    scores = matrix @ query_vec  # rows are pre-normalized, so dot = cosine
    return np.argsort(-scores)[:k].tolist()


def format_rag_prompt(query: str, passages: List[dict]) -> str:
    """Wrap query + retrieved passages into a single user turn."""
    refs = []
    for i, p in enumerate(passages):
        # Trim each passage to ~400 chars so the budget stays manageable.
        text = p["text"]
        if len(text) > 400:
            text = text[:400].rsplit(" ", 1)[0] + "…"
        refs.append(f"[{i + 1}] ({p['source']} {p.get('ref','')}) {text}")
    refs_block = "\n\n".join(refs)
    return (
        "Reference passages from the cybersecurity corpus:\n\n"
        f"{refs_block}\n\n"
        "Use the reference passages above to answer the question. If the "
        "passages don't contain the answer, say so rather than guessing.\n\n"
        f"Question: {query}"
    )


def load_ghost(checkpoint_path: str, device: str) -> Tuple[GhostLM, GhostLMConfig]:
    """Load a GhostLM checkpoint."""
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


def parse_args() -> argparse.Namespace:
    """CLI args."""
    p = argparse.ArgumentParser(description="GhostLM RAG chat")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--rag-dir", default="data/rag")
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k-sample", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--device", default="auto")
    p.add_argument("--show-passages", action="store_true",
                   help="Print the retrieved passages before each reply")
    return p.parse_args()


def main() -> None:
    """REPL with retrieval pulled before each user turn."""
    args = parse_args()
    device = resolve_device(args.device)

    print("Loading RAG index...")
    matrix, chunks, meta = load_index(Path(args.rag_dir))
    print(f"  {meta['n_chunks']} chunks, dim={meta['embedding_dim']}, "
          f"embedder={meta['embedder']}")

    print("Loading embedder...")
    e_tok, e_model = load_embedder(device)

    print("Loading GhostLM...")
    model, cfg = load_ghost(args.checkpoint, device)
    tokenizer = GhostTokenizer()
    end_id = tokenizer._special_tokens[tokenizer.END]

    print()
    print("RAG chat ready. Commands: 'quit', 'exit'.")
    print()

    while True:
        try:
            query = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            return
        if query.lower() in ("quit", "exit"):
            return
        if not query:
            continue

        q_vec = embed_query(e_tok, e_model, query, device)
        idx = top_k(q_vec, matrix, args.top_k)
        passages = [chunks[i] for i in idx]
        if args.show_passages:
            print("\n  Retrieved:")
            for p in passages:
                print(f"    [{p['source']} {p.get('ref','')}] "
                      f"{p['text'][:120]}…")
            print()

        prompt = format_rag_prompt(query, passages)
        ids = tokenizer.format_chat_prompt([{"role": "user", "content": prompt}])
        new_ids = generate_until_end(
            model, ids, end_id=end_id, max_new_tokens=args.max_tokens,
            temperature=args.temperature, top_k=args.top_k_sample, top_p=args.top_p,
            device=device,
        )
        reply = tokenizer.decode(new_ids).strip()
        print(f"\nGhostLM > {reply}\n")


if __name__ == "__main__":
    main()
