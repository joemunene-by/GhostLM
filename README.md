![CI](https://github.com/joemunene-by/GhostLM/actions/workflows/ci.yml/badge.svg) ![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg) ![Status](https://img.shields.io/badge/status-Phase%203.5%20Complete-green.svg)

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

The released v0.3.5 checkpoint was trained on the **rebalanced** Phase 3.5 corpus. NVD's full 333,540-record pull is on disk; its training contribution is capped at 6M tokens by deterministic content-hash subsample so the corpus isn't 90% CVE descriptions:

| Source | Records (raw → trained) | Trained tokens | Share | Type |
|---|---|---|---|---|
| NVD CVE Database | 333,540 → 71,828 | ~5.74M | **65.3%** | Real, capped via `--max-cve-tokens 6000000` |
| Synthetic CTF Writeups | 3,000 | ~1.51M | 17.2% | Synthetic, placeholder until real CTFtime grows |
| arXiv cs.CR Abstracts | 2,000 | ~0.74M | 8.4% | Real |
| CTFtime real writeups | 473 → 467 | ~0.47M | 5.3% | Real, inline-only, attributed |
| MITRE ATT&CK | 691 | ~0.26M | 2.9% | Real (Apache 2.0) |
| CAPEC | 609 | ~0.07M | 0.9% | Real (Apache 2.0) |
| **Total (post-dedup)** | **74,635** | **~8.79M** | | train: 70,965 / val: 3,670 |

Token share went from **NVD 87% in v0.3.3** → **NVD 65% in v0.3.5**. The pipeline produces a deterministic, leakage-proof split (content-hash bucketing, leakage check returns 0). The subsample is reproducible — `python3 scripts/rebuild_corpus.py --max-cve-tokens 6000000` always produces the same 71,828-record CVE prefix. `scripts/data_audit.py` runs the diagnostics and writes a 4-panel chart to `logs/data_audit.png`.

For where the corpus is heading — Phase 3.6 volume targets (CTFtime expansion, security research blogs, full-text papers, Exploit-DB) and licensing notes — see [CORPUS.md](CORPUS.md).

---

## Training Progress

| Run | Steps | Train tokens | Val Loss | Notes |
|---|---|---|---|---|
| ghost-tiny Phase 1 (pre-audit corpus) | 10,000 | 2.66M (leaky) | 2.74 | Superseded — leaky train/val split, archived under `archive/` |
| ghost-tiny Phase 2 (rebalanced corpus) | 10,000 | 2.66M | 3.7813 | Archived as `checkpoints/best_model_phase2.pt` |
| ghost-tiny Phase 3 (post-NVD-pull corpus) | 30,000 | ~30M | 3.4458 | NVD-dominated (87%); preserved as `checkpoints/phase3_refresh/best_model.pt` |
| **ghost-tiny Phase 3.5 (rebalanced corpus)** | **30,000** | **~8.8M** | **3.5518** | **Current canonical model.** NVD share 65%, six sources balanced. Hardware: Mac Mini M4 (CPU), ~3h13m wall-clock |

> Cross-phase val_loss is **not directly comparable** between v0.3.3 and v0.3.5: the validation distribution changed when we added MITRE/CAPEC/CTFtime to the corpus. v0.3.5 sits 0.10 nats higher on a different val set; that does not mean the model is worse. The eval-axis numbers below are the cleaner read.

The Phase 3.5 checkpoint is the current canonical model. Phase 3 is preserved as `checkpoints/phase3_refresh/best_model.pt` for cross-phase comparison; Phases 1–2 are preserved as `checkpoints/best_model_phase{1,2}.pt`.

### Cross-phase eval — fair comparison (fixed test set)

The cyber-text benchmark is 10 hand-picked external samples that overlap none of the training corpora. Directly comparable across phases:

| Model | Cyber-text perplexity (lower better) |
|---|---|
| **ghost-tiny — Phase 3.5 (released)** | **96.24** |
| ghost-tiny — Phase 3 | 142.09 |
| ghost-tiny — Phase 2 | 152.71 |
| ghost-tiny — Phase 1 | 2,183.94 |
| GPT-2 (124M baseline) | 26.76 |

Phase 3 → Phase 3.5 dropped this benchmark **32%** (142.09 → 96.24) at fixed parameter count and 1/3 the training tokens. ghost-tiny is now ~3.6× behind GPT-2 on raw cyber-text perplexity, with ~8× less capacity. The trajectory matters more than the absolute number; full breakdown in [MODEL_CARD.md](MODEL_CARD.md#evaluation).

### Per-source perplexity (val split)

The headline reason the rebalance worked — same model, same recipe, 1/3 the corpus, but with diversity sources actually represented:

| Source | v0.3.3 PPL | v0.3.5 PPL | Δ |
|---|---|---|---|
| MITRE ATT&CK | 615.43 | 55.14 | **−91%** |
| CTFtime real writeups | 184.24 | 60.71 | **−67%** |
| CAPEC | 326.11 | 133.81 | **−59%** |
| Synthetic CTF (same data) | 67.57 | 28.48 | **−58%** |
| arXiv (same data) | 671.09 | 354.95 | **−47%** |
| NVD CVE | 24.19 | 27.55 | +14% |
| **Overall** | **171.84** | **66.05** | **−62%** |

The first three sources were 0 records in v0.3.3's training; v0.3.5 modeled them as proper domains. The synthetic-CTF and arXiv 47–58% drops happened with **identical training data** — the gain is parameter capacity that v0.3.3 was burning on memorizing duplicate CVE descriptions being redirected onto already-present sources. NVD pays the small expected cost for less specialization.

### PMI-corrected security task accuracy

3 classification tasks × 10 samples. Random baseline 15%. Old length-normalized scoring was mode-collapsed at 4/30 = 13.3% across all phases (eval failure, not model failure); PMI scoring fixed it. Now: **6/30 (20%) → 12/30 (40%)** — doubled.

## Sample Generations

Real generations from the v0.3.5 best_model at `temperature=0.7, top_k=40`. The point of these isn't fluency — it's **register diversity**. v0.3.3 collapsed every prompt into CVE prose; v0.3.5 has multiple registers it can switch between depending on the prompt domain.

> **Prompt:** *MITRE ATT&CK technique T1003*
> T1003.011: defense-evasion
> Tactic: defense-evasion
> Adversaries may use-evasion, such as legitimate system-evasion, or malicious payloads.dll, execution, execution, and execution is used for creating a command execution of code is required by a...

This is **MITRE schema output** — the sub-technique ID format (`T1003.011`), the `Tactic:` header, the standard MITRE narrative opening (*"Adversaries may..."*). v0.3.3 had zero MITRE training data and would have produced a fake CVE for this prompt. v0.3.5 produces proper MITRE register because it now models that domain.

> **Prompt:** *The CTF challenge involved*
> ...leveraging a web application that was vulnerable to SQL injection attacks. The challenge required a specific flaw exists within the application. The issue results from the lack of proper validation of user-supplied data, which can result in a heap corruption condition. An attacker can leverage this vulnerability to execute code in the context...

CTF-writeup voice in the opening (*"leveraging a web application... The challenge required..."*) — narrative rather than the structured CVE template. Drifts back into CVE register mid-sentence (still ghost-tiny, still 14.7M params), but the opening voice is correctly CTF-flavored.

> **Prompt:** *A SQL injection attack works by*
> ...injecting malicious SQL payloads on a payload.php, allowing attackers to execute arbitrary SQL commands via the id parameter.The Web Client (Wise application in Google Chrome prior to 115.0.60.27.48 allowed a remote attacker to perform a heap corruption via a crafted HTML page...

Mixed: starts with the right framing (*"injecting malicious SQL payloads"*), then drifts into CVE-style with a fake Chrome version. CVE register is still default for security-prose prompts, but it's no longer the *only* register.

**What this shows:** the rebalance produced *behavioral* diversity, not just numerical eval improvements. The model can now switch into MITRE-format output, CTF-narrative voice, or CVE-template prose depending on what the prompt cues. Hallucinations remain rampant in absolute terms (fake Chrome version, scrambled MITRE narrative) — the model has *form* but not *facts*. The 14.7M-param scale is the binding constraint on factual capability; the corpus rebalance fixed what could be fixed at this param count. The fix for hallucinations is more parameters at the same training quality — see the [Roadmap](#roadmap).

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

**Where we are (Phase 3.5, complete — v0.3.5):** ghost-tiny @ 30K steps on the rebalanced ~8.8M-token corpus (NVD share 65%, six sources balanced). Cyber-text perplexity dropped 32% (142.09 → 96.24), per-source val PPL dropped 62% overall (172 → 66), PMI security task accuracy doubled (20% → 40%). The model now switches register between CVE / MITRE / CTF prompts where v0.3.3 collapsed everything into CVE prose. The recipe both scales with data (Phase 2→3) and benefits from source diversity (Phase 3→3.5) — both Phase 4 (ghost-small) gates met on the recipe side.

**Where we're going:**

1. **Corpus diversity** — break the NVD-87% lopsidedness. CTFtime archives, security research blogs (Project Zero, PortSwigger, Trail of Bits), MITRE ATT&CK, tool docs. This is the long-term moat and compounds even when compute is the bottleneck.
2. **ghost-small (~55M params)** — first scale-up rung. M4 GPU/MPS feasible. Phase 3 met the gating criterion (recipe-scales-with-data validated); the remaining gate is corpus diversity above.
3. **ghost-base (~350M params)** — first rung that needs rented GPU compute. Where domain-coherent generation should start to emerge.
4. **ghost-1B** — the long-term goal. The smallest scale at which a from-scratch cyber LM has a real shot at being genuinely useful. Will need either rented H100 hours or owned GPU.

**Realistic timeline:** 2–3 years of sustained work to a useful 1B from-scratch cyber LM. That is the actual shape of this work — there are no shortcuts for "from scratch" at scale. Detailed phase plan in [ROADMAP.md](ROADMAP.md).

For changelog history (v0.1.0 → v0.3.5), see [CHANGELOG.md](CHANGELOG.md).

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
