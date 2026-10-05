#!/usr/bin/env python3
"""Retrieval-augmented chat over GhostLM.

At inference time:
1. Look up any CVE/CWE/CAPEC/ATT&CK IDs in the query exactly, then fill the
   rest by BGE cosine similarity against the in-memory passage matrix
   (~115 MB at 75K passages × 384 dims — trivially fast on M4).
2. Take the top-K passages (default 4), join them as a "Reference passages"
   prefix, and prepend to the user turn.
3. Run the chat-tuned GhostLM as usual; the model is not RAFT-trained yet so
   it just sees retrieved context as part of the user message — no new tokens
   or special handling needed.

The full RAFT-style retrieval-aware fine-tune is a separate session — this
script is the working RAG baseline that proves the plumbing and gives an
honest "did retrieval help?" measurement.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from typing import List, Tuple

import torch

from ghostlm.config import GhostLMConfig
from ghostlm.model import GhostLM
from ghostlm.rag import HybridRetriever, bge_embedder
from ghostlm.tokenizer import GhostTokenizer
from scripts.chat import generate_until_end, resolve_device


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

    print("Loading RAG index and embedder...")
    retriever = HybridRetriever.load(args.rag_dir, embed=bge_embedder(device=device))
    print(f"  {len(retriever.chunks)} chunks, {len(retriever.id_index)} exact IDs")

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

        passages = retriever.search(query, k=args.top_k)
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
