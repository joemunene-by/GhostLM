![CI](https://github.com/joemunene-by/GhostLM/actions/workflows/ci.yml/badge.svg) ![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg) ![Status](https://img.shields.io/badge/status-Phase%201%20Complete-green.svg)

# GhostLM 🔐

> An open-source cybersecurity-focused language model built entirely from scratch in PyTorch.

GhostLM is a decoder-only transformer language model trained on CVE vulnerability descriptions, CTF writeups, and cybersecurity research. Built from scratch — no pretrained weights, no wrappers, every component written by hand.

---

## Why GhostLM?

Security researchers currently rely on generic models (GPT-4, Llama) that weren't trained with security context. GhostLM is purpose-built for:

- 🛡️ CVE analysis and vulnerability explanation
- 🚩 CTF challenge reasoning
- 🔍 Penetration testing assistance
- 💀 Exploit and attack pattern understanding
- 📚 Security concept explanation

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
| ghost-tiny | 2 | 256 | ~14.5M | ✅ Phase 1 Complete (10K steps) |
| ghost-small | 6 | 512 | ~55M | 🔄 Planned |
| ghost-medium | 12 | 768 | ~160M | 🔜 Future |

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

| Source | Records | Type |
|---|---|---|
| NVD CVE Database | 9,925 | Real |
| Security Research Papers | 500 | Synthetic |
| CTF Writeups | 500 | Synthetic |
| **Total** | **10,925** | |

---

## Training Progress

| Run | Steps | Train Loss | Val Loss | Status |
|---|---|---|---|---|
| ghost-tiny Phase 1 | 10,000 | 1.97 | 2.74 | ✅ Complete |
| ghost-tiny Phase 2 | 100,000 | — | — | 🔄 Next (Mac Mini M4) |

## Evaluation Results (Phase 1)

| Metric | Score |
|---|---|
| Cybersecurity Perplexity | 2,183.94 |
| GPT-2 Baseline (117M) | 26.76 |
| CVE Severity Classification | 20.0% |
| Vulnerability Type Detection | 10.0% |
| Attack Technique ID | 10.0% |
| **Overall Security Score** | **13.3%** |

> The model generates security-domain text with correct vocabulary but can't reason yet at this scale. Phase 2 (100K steps) will close the gap.

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

### ✅ v0.1.0 — Architecture Complete
- Full transformer from scratch
- Training pipeline verified
- 10,925 cybersecurity records

### ✅ v0.2.0 — Phase 1 Training Complete
- ghost-tiny trained to 10,000 steps on CPU
- Full evaluation suite with benchmark vs GPT-2
- MODEL_CARD with detailed results

### ✅ v0.2.1 — Phase 2 Readiness
- RoPE (Rotary Position Embeddings) — config-toggled
- Flash Attention via `scaled_dot_product_attention` — config-toggled
- Safetensors export with config.json sidecar and SHA-256 checksum
- Pinned dependency versions + PEP 639 license metadata
- Test suite grown from 10 → 16 tests

### 🔄 v0.3.0 — Phase 2 Training
- 100K steps on Mac Mini M4 with RoPE + Flash Attention enabled
- HuggingFace Hub weights release (safetensors)
- Gradio web demo

### 🏁 v1.0.0 — Release
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

Built in Nairobi, Kenya 🇰🇪
