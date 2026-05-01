# GhostLM RAG (retrieval-augmented generation)

A 45M-parameter model can't reliably memorize 64K CVE descriptions, 655 MITRE
ATT&CK techniques, 4.7K Exploit-DB entries, and 451 CTF writeups all at once.
What it *can* do is condition fluently on retrieved context, if that context
is supplied at query time. RAG is how we close the factual-recall gap without
scaling the model up.

This is the **scaffolding** version — vanilla retrieve-then-generate. The
RAFT-style retrieval-aware fine-tune (the actual quality unlock per the
2024-2026 literature) is a separate phase.

## Pipeline

1. **Index build** (`scripts/build_rag_index.py`) — chunks the pretrain
   corpus into ~256-word passages, embeds each with `BAAI/bge-small-en-v1.5`
   (a 30 MB bi-encoder), and saves the float32 embedding matrix as a single
   `.npy` file plus a JSONL of chunk metadata. At ~75K chunks × 384 dims the
   index is ~115 MB — small enough to load into RAM and use brute-force
   cosine similarity at query time. No vector database needed at this scale.

2. **Query time** (`scripts/rag_chat.py`) — embed the user query, brute-force
   top-K against the index, format the retrieved passages as a "Reference
   passages: …" prefix in front of the user's question, and run the chat-tuned
   model normally. The model is not RAFT-trained yet so it just sees retrieved
   context as part of the user message — no new tokens or special handling.

## Build the index

One-time, takes ~10-20 min on M4 MPS for the full corpus::

    PYTHONPATH=. python3 scripts/build_rag_index.py \
        --corpus data/processed/train.jsonl \
        --out-dir data/rag

Output:
- `data/rag/index.npy`     — float32 (N, 384) L2-normalized
- `data/rag/chunks.jsonl`  — one record per chunk: id / source / ref / text
- `data/rag/meta.json`     — embedder name, dims, chunk-token budget

Re-run after corpus updates (e.g. v0.4.2 corpus expansion).

## Query

```bash
PYTHONPATH=. python3 scripts/rag_chat.py \
    --checkpoint checkpoints/phase5_chat_v2/best_model.pt \
    --top-k 4 \
    --show-passages
```

`--show-passages` prints the retrieved chunks before each reply — useful for
understanding when retrieval helps vs hurts.

## What the baseline gets you

For factual queries (specific CVE numbers, technique IDs, exploit-db entries),
RAG should sharply improve correctness because the relevant text is now
directly in the prompt. For open-ended security questions ("how does XSS
work"), RAG helps less — those are already well-covered by the chat tune.

The honest measurement is to run `scripts/run_bench.py` twice — once with
the chat-tuned model alone, once via `rag_chat.py` with the same prompts —
and compare. That landed alongside the standing CI in `bench.yml`.

## Next: RAFT-style retraining

The RAG baseline above puts retrieved context in the user prompt and hopes
the model uses it. RAFT (Zhang et al., ICML 2024, refreshed 2025) trains the
model to *cite* the right passage and *ignore* distractors. Expected lift
over the baseline is +10-25 pts on factual MCQ benchmarks per the cited
papers. Implementation is a separate phase; this scaffolding is the working
baseline that proves the plumbing.
