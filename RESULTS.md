# GhostLM benchmark results

Each row is one (checkpoint × benchmark) score. Updated by `scripts/run_bench.py`.

| Checkpoint | Benchmark | n | Correct | Accuracy | Date |
|---|---|---:|---:|---:|---|
| ghost-small-v0.5 chat-v2 | ctibench-mcq | 2500 | 475 | 0.190 | 2026-05-01 |
| ghost-small-v0.4 (pretrain, no chat) | ctibench-mcq | 2500 | 446 | 0.178 | 2026-05-01 |
| ghost-small-v0.5 chat-v2 + RAG(top4) | ctibench-mcq | 2500 | 476 | 0.190 | 2026-05-01 |
| ghost-small-v0.5 chat-v3 (MCQ-tuned) | ctibench-mcq | 2500 | 922 | 0.369 | 2026-05-01 |
| ghost-small-v0.5 chat-v4 (RAFT) + RAG(top4) | ctibench-mcq | 2500 | 540 | 0.216 | 2026-05-02 |
| ghost-small-v0.5 chat-v4 (RAFT, no retrieval) | ctibench-mcq | 2500 | 626 | 0.250 | 2026-05-02 |
| ghost-small-v0.5 chat (v0.5 base) | ctibench-mcq | 2500 | 813 | 0.325 | 2026-05-02 |
| ghost-small-v0.5 chat-long (v0.5 base, 4K steps) | ctibench-mcq | 2500 | 428 | 0.171 | 2026-05-02 |
| ghost-small-v0.5 chat-recovered (extended pretrain + CoT MCQ + tok surgery) | ctibench-mcq | 2500 | 771 | 0.308 | 2026-05-03 |
| ghost-small-v0.5 chat-v5 (hybrid raw×5+CoT×2 + small-talk×8 + lr5e-5 + mean-init) | ctibench-mcq | 2500 | 871 | 0.348 | 2026-05-03 |
| ghost-small-v0.4 chat-v3 (MCQ-tuned, canonical) + RAG(top4) + RAG(top4) | ctibench-mcq | 2500 | 913 | 0.365 | 2026-05-03 |
| ghost-small-v0.5 chat-v5 + RAG(top4) + RAG(top4) | ctibench-mcq | 2500 | 844 | 0.338 | 2026-05-03 |
| ghost-small-v0.4 chat-v3 (MCQ-tuned, canonical) + RAG(top2) + RAG(top2) | ctibench-mcq | 2500 | 923 | 0.369 | 2026-05-03 |
| ghost-small-v0.4 chat-v6 (v0.4 base, expanded SFT: +MITRE-full +CISA-KEV) | ctibench-mcq | 2500 | 465 | 0.186 | 2026-05-03 |
| ghost-small-v0.4 chat-v3-repro (baseline data, ctx 1024) | ctibench-mcq | 2500 | 816 | 0.326 | 2026-05-03 |
| ghost-small-v0.4 chat-v3-repro2 (canonical recipe: lr 3e-5, 1800 steps, batch 8 × accum 4, ctx 1024) | ctibench-mcq | 2500 | 780 | 0.312 | 2026-05-03 |
| ghost-small-v0.6 chat (v0.6 base: v0.5 arch + GPT-2 BPE + expanded corpus, canonical chat-v3 recipe) | ctibench-mcq | 2500 | 745 | 0.298 | 2026-05-03 |
| ghost-small-v0.6 chat-hybrid (v0.6 base + chat-v5 hybrid recipe: raw×5 + CoT×2 + small-talk×8) | ctibench-mcq | 2500 | 374 | 0.150 | 2026-05-03 |
