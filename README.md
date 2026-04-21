![CI](https://github.com/joemunene-by/GhostLM/actions/workflows/ci.yml/badge.svg) ![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg) ![Status](https://img.shields.io/badge/status-Phase%201%20Complete-green.svg)

# GhostLM

> An open-source cybersecurity-focused language model built entirely from scratch in PyTorch.

GhostLM is a decoder-only transformer language model trained on CVE vulnerability descriptions, CTF writeups, and cybersecurity research. Built from scratch — no pretrained weights, no wrappers, every component written by hand.

---

## Why GhostLM?

Security researchers currently rely on generic models (GPT-4, Llama) that weren't trained with security context. GhostLM is purpose-built for:

- CVE analysis and vulnerability explanation
- CTF challenge reasoning
- Penetration testing assistance
- Exploit and attack pattern understanding
- Security concept explanation

### Why from scratch and not a fine-tune?

Two reasons. **First**, most offensive-security content that the best general models have seen was filtered or RLHF-nudged away during alignment — a fine-tune on top fights that prior. Training the tokenizer and weights from zero with security text in the mix lets the model treat CVE IDs, shell one-liners, and exploit technique names as first-class tokens rather than something to refuse. **Second**, GhostLM is also a study project. Every layer — attention, positional encoding, LR schedule, BPE — is hand-written so the codebase doubles as a readable reference for how a transformer is actually put together. A fine-tune hides that behind `AutoModel.from_pretrained`.

It is explicitly *not* trying to beat Llama on general benchmarks. It's trying to be the right tool for one narrow job, and a transparent one.

---

## Architecture

| Parameter | Value |
|---|---|
| Architecture | Decoder-only Transformer |
| Parameters (ghost-small) | ~55M |
| Context Length | 1024 tokens |
| Layers | 6 |
| Attention Heads | 8 |
| Embedding Dim | 512 |
| Tokenizer | GPT-2 BPE (50,261 tokens) |

Built with:
- Multi-head causal self-attention (manual implementation)
- **RoPE** (Rotary Position Embeddings) — opt-in via `use_rope=True`, replaces learned positional embeddings with the relative-position encoding used by LLaMA / Mistral
- **Flash Attention** — opt-in via `use_flash_attention=True`, routes through PyTorch 2.0+ `scaled_dot_product_attention` for `O(n)` memory
- Pre-norm transformer blocks with residual connections
- Cosine LR schedule with linear warmup
- Weight-tied output projection
- AdamW with weight decay separation
- **Safetensors** export for safe, arbitrary-code-free weight distribution (see `scripts/export.py`)

## Model Variants

| Variant | Layers | Dim | Params | Status |
|---|---|---|---|---|
| ghost-tiny | 2 | 256 | ~14.5M | Phase 1 complete (10K steps) |
| ghost-small | 6 | 512 | ~55M | Planned |
| ghost-medium | 12 | 768 | ~160M | Future |

---

## Quick Start

### Installation
```bash
git clone https://github.com/joemunene-by/GhostLM.git
cd GhostLM
make install
```

### Prepare Training Data
```bash
make data
```

### Train
```bash
# CPU-friendly (ghost-tiny)
make train-tiny

# GPU (ghost-small)
make train-small
```

### Generate Text
```bash
make generate
```

### Interactive Chat
```bash
make chat
```

### Run Web Demo
```bash
pip install gradio
python demo/app.py
```

### Benchmark vs GPT-2
```bash
make benchmark
```

### Export Weights (safetensors or PyTorch)
```bash
# Safe, pickle-free weights for HuggingFace Hub distribution
python scripts/export.py --format safetensors

# Classic PyTorch checkpoint
python scripts/export.py --format pt
```

### Plot Training Curves
```bash
make plot
```

---

## Training Data

| Source | Records | Type | Coverage |
|---|---|---|---|
| NVD CVE Database | 19,925 | Real | 1999–2025 (27 years, balanced per-year cap) |
| arXiv cs.CR Abstracts | 1,000 | Real | Recent-first (submittedDate descending) |
| CTF Writeups | 21 | Synthetic | Unique templates only, no padding |
| **Total (pre-dedup)** | **20,946** | | |
| **Total (post-dedup, training)** | **20,070** | | ~1.52M tokens |

The pipeline produces a deterministic, leakage-proof split — train and validation are assigned by content hash so identical texts always land in the same split. `scripts/data_audit.py` runs the diagnostics (length percentiles, dedup rate, CVE-year distribution, CTF category share, token share, leakage check) and writes a 4-panel chart to `logs/data_audit.png`.

---

## Training Progress

| Run | Steps | Train Loss | Val Loss | Status |
|---|---|---|---|---|
| ghost-tiny Phase 1 (initial) | 10,000 | 1.97 | 2.74 | Superseded — trained on pre-audit corpus (archived) |
| ghost-tiny Phase 1 (re-run) | 10,000 | — | — | Pending — rebalanced corpus, leakage-free split |
| ghost-tiny Phase 2 | 100,000 | — | — | Next (Mac Mini M4) |

> The initial Phase 1 run completed before the data audit surfaced heavy duplication (~98% in papers/CTF) and ~9% train/val leakage. Those checkpoints and logs are preserved under `archive/` for reference, and Phase 1 is being re-run on the rebalanced corpus before Phase 2 begins. Evaluation numbers will be refreshed when the re-run completes.

---

## Project Structure

```
GhostLM/
├── ghostlm/           # Core library
│   ├── model.py       # Transformer architecture (RoPE + Flash Attention toggles)
│   ├── config.py      # Hyperparameters + ghost-tiny/small/medium presets
│   ├── tokenizer.py   # GPT-2 BPE wrapper
│   ├── dataset.py     # PyTorch dataset
│   └── trainer.py     # Training loop
├── scripts/           # CLI tools
│   ├── train.py       # Training entry point
│   ├── generate.py    # Text generation
│   ├── chat.py        # Interactive chat
│   ├── evaluate.py    # Evaluation
│   ├── eval_security.py  # Security-specific evaluation
│   ├── benchmark.py   # GPT-2 comparison
│   ├── export.py      # Weights export (safetensors / pt) + SHA-256 + config.json
│   ├── api.py         # REST API server
│   ├── data_stats.py  # Training-data statistics
│   ├── plot_training.py  # Loss-curve plotter
│   ├── push_to_hub.py # HuggingFace Hub publisher
│   └── resume_train.sh   # Resume an interrupted training run
├── data/              # Data pipeline
├── demo/              # Gradio web demo (demo/app.py)
├── tests/             # 16 unit tests
└── Makefile           # One-command workflow
```

---

## Roadmap

### v0.1.0 — Architecture complete
- Full transformer from scratch
- Training pipeline verified
- Initial 10,925-record corpus (later rebalanced — see v0.2.2)

### v0.2.0 — Phase 1 training complete
- ghost-tiny trained to 10,000 steps on CPU
- Full evaluation suite with benchmark vs GPT-2
- MODEL_CARD with detailed results

### v0.2.1 — Phase 2 readiness
- RoPE (Rotary Position Embeddings) — config-toggled
- Flash Attention via `scaled_dot_product_attention` — config-toggled
- Safetensors export with config.json sidecar and SHA-256 checksum
- Pinned dependency versions + PEP 639 license metadata
- Test suite grown from 10 → 16 tests

### v0.2.2 — Data audit + corpus rebalancing
- New `scripts/data_audit.py` — length percentiles, dedup rate, CVE-year distribution, CTF category share, token share, train/val leakage check
- CVE collector rewritten to 119-day NVD windows with append mode — coverage extended from 1999–2005 to 1999–2025 (27 years)
- Paper collector switched from hand-written synthetic `× 50` padding to the arXiv cs.CR Atom API — 1,000 real abstracts
- Synthetic CTF generator emits unique templates only (fixed a rotation bug that limited output to 12 of ~22 templates)
- `merge_datasets` now uses a deterministic MD5-bucket split — identical texts always land in the same split, eliminating train/val leakage

### v0.3.0 — Phase 2 Training (in progress)
- 100K steps on Mac Mini M4 with RoPE + Flash Attention enabled
- HuggingFace Hub weights release (safetensors)
- Gradio web demo

### v1.0.0 — Release (planned)
- Public weights + REST API
- Fine-tuning scripts

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get involved.

---

## License

MIT — see [LICENSE](LICENSE)

---

## Author

**Joe Munene** — [Complex Developers](https://github.com/joemunene-by)

Built in Nairobi, Kenya.
