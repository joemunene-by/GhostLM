![CI](https://github.com/joemunene-by/GhostLM/actions/workflows/ci.yml/badge.svg) ![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg) ![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg) ![Status](https://img.shields.io/badge/status-pre--training-yellow.svg)

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
- Pre-norm transformer blocks with residual connections
- Cosine LR schedule with linear warmup
- Weight-tied output projection
- AdamW with weight decay separation

## Model Variants

| Variant | Layers | Dim | Params | Status |
|---|---|---|---|---|
| ghost-tiny | 2 | 256 | ~14.5M | ✅ Training |
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
| ghost-tiny v1 | 500 | 6.73 | 6.27 | ✅ Complete |
| ghost-tiny v2 | 2000 | TBD | TBD | 🔄 Running |

---

## Project Structure

```
GhostLM/
├── ghostlm/          # Core library
│   ├── model.py      # Transformer architecture
│   ├── config.py     # Hyperparameters
│   ├── tokenizer.py  # GPT-2 BPE wrapper
│   ├── dataset.py    # PyTorch dataset
│   └── trainer.py    # Training loop
├── scripts/          # CLI tools
│   ├── train.py      # Training entry point
│   ├── generate.py   # Text generation
│   ├── chat.py       # Interactive chat
│   ├── evaluate.py   # Evaluation
│   └── benchmark.py  # GPT-2 comparison
├── data/             # Data pipeline
├── demo/             # Gradio web demo
├── tests/            # 10 unit tests
└── Makefile          # One-command workflow
```

---

## Roadmap

### ✅ v0.1.0 — Architecture Complete
- Full transformer from scratch
- Training pipeline verified
- 10,925 cybersecurity records

### 🔄 v0.2.0 — Pre-training
- ghost-tiny trained to 10,000+ steps
- HuggingFace Hub weights release
- Gradio demo live

### 🔜 v0.3.0 — Scale
- ghost-small on GPU/TPU
- Benchmark vs GPT-2

### 🏁 v1.0.0 — Release
- Public weights + REST API

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get involved.

---

## License

Apache 2.0 — see [LICENSE](LICENSE)

---

## Author

**Joe Munene** — [Complex Developers](https://github.com/joemunene-by)

Built in Nairobi, Kenya 🇰🇪
