![CI](https://github.com/joemunene-by/GhostLM/actions/workflows/ci.yml/badge.svg) ![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg) ![Status](https://img.shields.io/badge/status-Phase%203%20Complete-green.svg)

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

The released v0.3.3 checkpoint was trained on the post-NVD-pull corpus:

| Source | Records | Type | Coverage |
|---|---|---|---|
| NVD CVE Database | 333,540 | Real | Full pull, 1999–2026 (28 years), `startIndex`-paginated |
| arXiv cs.CR Abstracts | 2,000 | Real | Recent-first (submittedDate descending) |
| Synthetic CTF Writeups | 3,000 | Synthetic | Generated via local LLM, varied template + topic mix |
| **Total (post-dedup)** | **~309K** | | **~30M tokens** (train: ~293K / val: ~15K) |

Token share is currently lopsided — NVD ~87%, CTF ~5%, papers ~2%. The next *corpus* track is diversity (CTFtime, MITRE ATT&CK), not deeper NVD. The pipeline produces a deterministic, leakage-proof split (content-hash bucketing, leakage check returns 0). `scripts/data_audit.py` runs the diagnostics and writes a 4-panel chart to `logs/data_audit.png`.

For where the corpus is heading — sources targeted (CTFtime archives, security research blogs, MITRE ATT&CK, tool docs) and licensing notes — see [CORPUS.md](CORPUS.md).

---

## Training Progress

| Run | Steps | Train tokens | Val Loss | Notes |
|---|---|---|---|---|
| ghost-tiny Phase 1 (pre-audit corpus) | 10,000 | 2.66M (leaky) | 2.74 | Superseded — leaky train/val split, archived under `archive/` |
| ghost-tiny Phase 2 (rebalanced corpus) | 10,000 | 2.66M | 3.7813 | Archived as `checkpoints/best_model_phase2.pt` |
| **ghost-tiny Phase 3 (post-NVD-pull corpus)** | **30,000** | **~30M** | **3.4458** | **Current canonical model.** Hardware-of-record: Mac Mini M4 (CPU), ~3h48m wall-clock |

> Phase 1's val_loss 2.74 was measured on a leaky split — not directly comparable to later phases. Phase 2 (3.78) and Phase 3 (3.45) are both on clean deterministic-hash splits, so the **0.34 nat drop is a real corpus-quality dividend** at fixed model size. Same recipe, ~12× the data, ~29% lower perplexity on val.

The Phase 3 checkpoint is `checkpoints/best_model.pt`. Phase 1 and Phase 2 checkpoints are preserved as `checkpoints/best_model_phase1.pt` and `checkpoints/best_model_phase2.pt` for archaeological reference.

The cross-phase perplexity benchmark (same hardcoded 10-sample cyber-text set across all phases — fair comparison):

| Model | Perplexity vs cyber-text benchmark |
|---|---|
| **ghost-tiny — Phase 3 (released)** | **142.09** |
| ghost-tiny — Phase 2 | 152.71 |
| ghost-tiny — Phase 1 | 2,183.94 |
| GPT-2 (124M baseline) | 26.76 |

Phase 2→3 moved this benchmark only ~7% — the 10-text set has a lot of generic security prose that already overlapped both corpora, so most of the perplexity dividend was earned at Phase 2 (corpus quality + clean split). The Phase 3 win shows up much more strongly on val_loss than on this benchmark. ghost-tiny is still 5.3× behind GPT-2 on raw perplexity, expected for a 14.7M model on ~30M tokens vs. a 124M model on ~40B tokens of WebText. See `logs/phase_comparison.png` for the 3-panel side-by-side and [MODEL_CARD.md](MODEL_CARD.md#evaluation-phase-3--30k-steps-post-nvd-pull-corpus) for the security-task eval (still mode-collapsed at this scale — expected).

## Sample Generations

These are real generations from `checkpoints/best_model.pt` (Phase 3, val_loss 3.45) at `temperature=0.8, top_k=50`. They reflect what a 14.7M-param model trained on ~30M tokens can actually do — and what it can't:

> **Prompt:** *A SQL injection attack works by*
> ...injecting a crafted payload in the "password" parameter. NOTE: some of these details are obtained from third party information. EINXCMS 3.4.7.2.0 contains a cross-site scripting vulnerability that allows users to embed arbitrary JavaScript code in the Web UI thus altering the intended functionality potentially leading to credentials disclosure within a trusted session. This issue is fixed in 1.0.10. Use after free in Google Chrome before 17.0.66.75 allowed a remote...

> **Prompt:** *To enumerate Active Directory users from a Linux box, an attacker can*
> ...exploit an attacker to inject malicious scripts as the target system and execute arbitrary code in the context of the current process. The vulnerability is due to the application... A successful exploit could allow the attacker to obtain sensitive information on the user. The BIO plugin for WordPress is vulnerable to Cross-Site Request Forgery in all versions up to, and including, 2.2. This is due to missing nonce validation on...

> **Prompt:** *Cross-site scripting (XSS) allows attackers to*
> ...inject arbitrary web script or HTML via unspecified vectors. The davcodec_server function in libav.c in the Linux kernel before 4.4.0.5, 4.x before 4.6.1, and 4.x before 4.3.0.x before 4.3.3, does not properly restrict access to (1) Ntcpd) PTP-based LDAP (2) X-3-NAP00-00N-N

**What this shows:** the model has learned the **CVE-database register** — phrases like "Cross-Site Request Forgery in all versions up to, and including, 2.2 — this is due to missing nonce validation," "use after free," "remote attacker," "submitting a crafted link" are real CVE language used in roughly the right context. Compare to Phase 2, which produced fragments like "the login page is used to the login page's name of the login page does not properly sanitization" — same architecture, same param count, just 12× more data. **Hallucinations are still rampant** (made-up products, scrambled version strings, mixed-up vendors) — the model has the *form* of CVE descriptions but not the *facts*. This is the expected outcome of corpus expansion at fixed model size: better surface fluency, no new factual capability. The fix is scale (more params), not more data at this param count — see the [Roadmap](#roadmap).

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

**Where we are (Phase 3, complete — v0.3.3):** ghost-tiny @ 30K steps on the post-NVD-pull ~30M-token corpus, val_loss 3.4458, perplexity 142.09 on the cyber-text benchmark. Same architecture as Phase 2, ~12× the data, **~29% lower perplexity on val**. The recipe scales with data — that's the result Phase 4 (ghost-small) gating was waiting on.

**Where we're going:**

1. **Corpus diversity** — break the NVD-87% lopsidedness. CTFtime archives, security research blogs (Project Zero, PortSwigger, Trail of Bits), MITRE ATT&CK, tool docs. This is the long-term moat and compounds even when compute is the bottleneck.
2. **ghost-small (~55M params)** — first scale-up rung. M4 GPU/MPS feasible. Phase 3 met the gating criterion (recipe-scales-with-data validated); the remaining gate is corpus diversity above.
3. **ghost-base (~350M params)** — first rung that needs rented GPU compute. Where domain-coherent generation should start to emerge.
4. **ghost-1B** — the long-term goal. The smallest scale at which a from-scratch cyber LM has a real shot at being genuinely useful. Will need either rented H100 hours or owned GPU.

**Realistic timeline:** 2–3 years of sustained work to a useful 1B from-scratch cyber LM. That is the actual shape of this work — there are no shortcuts for "from scratch" at scale. Detailed phase plan in [ROADMAP.md](ROADMAP.md).

For changelog history (v0.1.0 → v0.3.3), see [CHANGELOG.md](CHANGELOG.md).

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
