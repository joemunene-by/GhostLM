# GhostLM benchmark results

Each row is one (checkpoint × benchmark) score. Updated by `scripts/run_bench.py`.

| Checkpoint | Benchmark | n | Correct | Accuracy | Date |
|---|---|---:|---:|---:|---|
| ghost-small-v0.5 chat-v2 | ctibench-mcq | 2500 | 475 | 0.190 | 2026-05-01 |
| ghost-small-v0.4 (pretrain, no chat) | ctibench-mcq | 2500 | 446 | 0.178 | 2026-05-01 |
| ghost-small-v0.5 chat-v2 + RAG(top4) | ctibench-mcq | 2500 | 476 | 0.190 | 2026-05-01 |
| ghost-small-v0.5 chat-v3 (MCQ-tuned) | ctibench-mcq | 2500 | 922 | 0.369 | 2026-05-01 |
