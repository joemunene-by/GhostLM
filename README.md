![CI](https://github.com/joemunene-by/GhostLM/actions/workflows/ci.yml/badge.svg) ![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg) ![Status](https://img.shields.io/badge/status-Phase%202%20Complete-green.svg)

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

The configuration below is for **ghost-tiny**, the current canonical variant. Larger variants share the same architecture with scaled layers / dim / heads — see the [Model Variants](#model-variants) table.

| Parameter | ghost-tiny |
|---|---|
| Architecture | Decoder-only Transformer |
| Parameters | 14.7M |
| Layers | 2 |
| Attention Heads | 4 |
| Embedding Dim | 256 |
| FFN Dim | 1024 |
| Context Length | 1024 tokens |
| Tokenizer | GPT-2 BPE (50,261 tokens — 50,257 base + 4 cyber special) |

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

GhostLM is a multi-year scale ladder. Each rung validates the recipe before climbing to the next:

| Variant | Layers | Dim | Params | Hardware target | Status |
|---|---|---|---|---|---|
| ghost-tiny | 2 | 256 | 14.7M | CPU | Phase 2 complete (10K steps, val_loss 3.78) |
| ghost-small | 6 | 512 | ~55M | M4 GPU/MPS | Planned |
| ghost-base | 12 | 768 | ~350M | Rented GPU (A/H100) | Planned |
| ghost-1B | 24 | 1024 | ~1B | Rented or owned GPU | Long-term goal |

ghost-tiny is the iteration vehicle and educational artifact. It is not — and at this scale will not become — a useful cyber-task model. The scale ladder above is the path to "useful." See [ROADMAP.md](ROADMAP.md) for phased milestones, corpus targets per rung, and honest compute estimates.

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
| CTF Writeups | 3,000 | Synthetic | Generated via local LLM, varied template + topic mix |
| **Total (pre-dedup)** | **23,925** | | |
| **Total (post-dedup, training)** | **23,049** | | ~2.66M tokens (train: 2,525,245 / val: 136,869) |

The pipeline produces a deterministic, leakage-proof split — train and validation are assigned by content hash so identical texts always land in the same split. `scripts/data_audit.py` runs the diagnostics (length percentiles, dedup rate, CVE-year distribution, CTF category share, token share, leakage check) and writes a 4-panel chart to `logs/data_audit.png`.

For where the corpus is heading — sources targeted (CTFtime archives, security research blogs, MITRE ATT&CK, tool docs) and licensing notes — see [CORPUS.md](CORPUS.md).

---

## Training Progress

| Run | Steps | Train Loss | Val Loss | Notes |
|---|---|---|---|---|
| ghost-tiny Phase 1 (pre-audit corpus) | 10,000 | 1.97 | 2.74 | Superseded — leaky train/val split, kept for reference under `archive/` |
| ghost-tiny Phase 2 (rebalanced corpus) | 10,000 | ~4.6 | **3.7813** | Current canonical model. Hardware-of-record: Mac Mini M4 (CPU) |

> Phase 1 and Phase 2 val_loss are **not directly comparable** — Phase 1's 2.74 was measured on a leaky split where many "validation" samples appeared in training. Phase 2's 3.78 is the trustworthy number on a clean, deterministic-hash split. Don't read the increase as regression; it's the first honest measurement.

The Phase 2 checkpoint is `checkpoints/best_model.pt`. Phase 1 is preserved as `checkpoints/best_model_phase1.pt` for archaeological reference.

## Sample Generations

These are real generations from `checkpoints/best_model.pt` at `temperature=0.7-0.8, top_k=50`. They reflect what a 14.7M-param model trained on ~2.7M tokens can actually do — and what it can't:

> **Prompt:** *A SQL injection attack works by*
> ...the login page. The login page is used to the login page's name of the login page does not properly sanitization of the password, which allows attackers to cause a denial of service via a long GET request. In this CTF challenge, participants are tasked with a classic vulnerability in exploiting a vulnerable to exploit a vulnerable to SQL injection vulnerability...

> **Prompt:** *CVE-2021-44228 is a vulnerability in*
> ...the C privileged user account, which allows local users to gain root privileges. Buffer overflow in x86 program in Oracle MySQL 5.1.2.5.0.0.2.1.0.1 and 7.1.2.1.0.x before 7.2.2.2.x before 7.1.x before 7.1.1.2.

> **Prompt:** *To enumerate Active Directory users from a Linux box, an attacker can*
> ...obtain the binary using the binary. The binary. The binary using UPX, the binary exploitation, we were a stack with a ret2libc base address of the stack with Ghidra's memory with pwndbg and function...

**What this shows:** the model has absorbed surface-level cyber vocabulary (CTF terminology, exploit techniques, CVE-string format, common vuln types) but has no semantic grounding — broken grammar, hallucinated version chains, can't bind topic (an AD-on-Linux prompt elicits binary exploitation tokens). This is exactly what a 14.7M-param model on ~22K records predicts. The fix is scale, not more steps at this scale — see the [Roadmap](#roadmap).

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

GhostLM is a multi-year effort. The honest framing is that ghost-tiny is a learning artifact and a working pipeline — *not* a useful cyber-task model. The path to "useful" is the scale ladder below, paired with a corpus that grows by ~100× from where it is today. See [ROADMAP.md](ROADMAP.md) for full milestones, compute estimates, and corpus targets.

**Where we are (Phase 2, complete):** ghost-tiny @ 10K steps, val_loss 3.78 on the rebalanced corpus. End-to-end training pipeline proven. ~2.7M training tokens.

**Where we're going:**

1. **Corpus expansion** — 10–100× the current corpus. NVD-at-scale, CTFtime archives, security research blogs (Project Zero, PortSwigger, Trail of Bits), MITRE ATT&CK, tool docs. This is the long-term moat and compounds even when compute is the bottleneck.
2. **ghost-small (~55M params)** — first scale-up rung. M4 GPU/MPS feasible. Validates whether the recipe scales.
3. **ghost-base (~350M params)** — first rung that needs rented GPU compute. Where domain-coherent generation should start to emerge.
4. **ghost-1B** — the long-term goal. The smallest scale at which a from-scratch cyber LM has a real shot at being genuinely useful. Will need either rented H100 hours or owned GPU.

**Realistic timeline:** 2–3 years of sustained work to a useful 1B from-scratch cyber LM. That is the actual shape of this work — there are no shortcuts for "from scratch" at scale. Detailed phase plan in [ROADMAP.md](ROADMAP.md).

For changelog history (v0.1.0 → v0.3.0), see [CHANGELOG.md](CHANGELOG.md).

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
