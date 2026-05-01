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

The configuration below is for **ghost-tiny**, the original architecture variant. As of v0.4.0, the canonical released checkpoint is **ghost-small** (~45M params, 6 layers / 512 d_model / 8 heads) trained on the Phase 3.6 corpus. Both variants share the same architecture with scaled layers / dim / heads — see the [Model Variants](#model-variants) table.

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
| ghost-tiny Phase 3.5 (rebalanced corpus) | 30,000 | ~8.8M | 3.5518 | Historical canonical for the existing PMI suite. NVD share 65%, six sources balanced. Hardware: Mac Mini M4 (CPU), ~3h13m wall-clock |
| ghost-tiny Phase 3.6 (+Exploit-DB) | 30,000 | ~12.56M | 3.8556 | Regressed on the eval suite (31.2% → 16.8%); ghost-tiny capacity ceiling found. Preserved at `checkpoints/phase3.6_exploitdb/best_model.pt` — see CHANGELOG v0.3.7 |
| **ghost-small Phase 4 (capacity reallocation)** | **30,000** | **~12.56M** | **2.3535** | **Current canonical model for density / generation.** ~45M params (6L / 512d / 8h) on the same Phase 3.6 corpus. **Per-source PPL 59–78% better than Phase 3.5 across every source**, overall PPL 66.05 → 11.12 (−83%). Hardware: Mac Mini M4 (MPS), ~15h wall-clock. See CHANGELOG v0.4.0 |

> Cross-phase val_loss is **not directly comparable** between phases when the corpus changes: each phase from 3.5 onward has a different validation distribution. The eval-axis numbers below are the cleaner read.

The Phase 4 ghost-small checkpoint at `checkpoints/phase4_ghost_small/best_model.pt` is the current canonical model for any density / completion / generation work — it dominates Phase 3.5 by 59–78% on per-source perplexity across every source. The Phase 3.5 ghost-tiny checkpoint at `checkpoints/phase3.5_balanced/best_model.pt` remains on disk as the historical canonical and is still the higher number on the existing **PMI** multiple-choice suite (a calibration artifact at small corpus size; see [CHANGELOG.md](CHANGELOG.md) v0.4.0 for the PMI vs logp scoring analysis). Both are kept; pick by use case.

### Chat tuning — v0.5 (current canonical chat model)

A supervised fine-tune on top of Phase 4 ghost-small turns the base completion model into a conversational cybersecurity assistant. The canonical chat model is **`checkpoints/phase5_chat_v3/best_model.pt`** — same 45M params as Phase 4, an additional ~1,800 step SFT run with three new role tokens (`<|ghost_user|>`, `<|ghost_assistant|>`, `<|ghost_end|>`) and 1,802 templated MCQ examples mixed into the chat training data.

| Checkpoint | CTIBench MCQ (n=2500) | Δ vs pretrain |
|---|---:|---:|
| `phase4_ghost_small` (pretrain only) | 17.8% (446) | — |
| `phase5_chat_v2` (free-form SFT) | 19.0% (475) | +1.2 pp |
| `phase5_chat_v2 + RAG(top4)` | 19.0% (476) | +1.2 pp |
| **`phase5_chat_v3` (MCQ-tuned)** | **36.9% (922)** | **+19.1 pp** |

Random baseline on 4-way MCQ is 25%. v3 is **1.48× random** — well above chance, well below frontier LLMs (85–95%). The next bump is a from-scratch retrain on a larger corpus (Phase 4.5, ~50M tokens) with the v0.5 architecture switches (RoPE + SwiGLU + RMSNorm) flipped on; that wiring is committed but gated on the corpus landing first. Full recipe in [`docs/chat_tuning.md`](docs/chat_tuning.md), benchmark in [`RESULTS.md`](RESULTS.md), CHANGELOG entry in v0.5.0.

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

The cleanest cross-phase read — does the model actually model each source it was trained on. The full trajectory across phases:

| Source | v0.3.3 (P3) | v0.3.5 (P3.5) | v0.3.7 (P3.6) | **v0.4.0 (P4)** | P4 vs P3.5 |
|---|---:|---:|---:|---:|---:|
| arXiv | 671.09 | 354.95 | 505.60 | **116.46** | **−67%** |
| CAPEC | 326.11 | 133.81 | 179.71 | **54.42** | **−59%** |
| CTFtime real writeups | 184.24 | 60.71 | 59.70 | **13.23** | **−78%** |
| Exploit-DB | — | — | 40.87 | **8.60** | new source |
| MITRE ATT&CK | 615.43 | 55.14 | 70.53 | **19.72** | **−64%** |
| NVD CVE | 24.19 | 27.55 | 35.44 | **11.29** | **−59%** |
| Synthetic CTF | 67.57 | 28.48 | 38.90 | **7.88** | **−72%** |
| **Overall** | **171.84** | **66.05** | **44.36** | **11.12** | **−83%** |

Three distinct phase-on-phase wins to read off this table:

- **v0.3.3 → v0.3.5 (corpus rebalance, fixed model):** the 47–91% drops on MITRE / CTFtime / CAPEC came from those sources being added to training, the synthetic-CTF / arXiv drops from same data with parameter capacity redirected away from memorizing duplicate CVEs.
- **v0.3.5 → v0.3.6 (corpus volume, fixed model):** every existing source got 28–42% worse — ghost-tiny ran out of capacity to hold seven sources at once. This is the result that diagnosed the ceiling.
- **v0.3.6 → v0.4.0 (model capacity, fixed corpus):** every single source improved 68–80% relative to v0.3.6, and 59–78% relative to v0.3.5. ghost-small at 45M params absorbs the corpus that broke ghost-tiny without the per-source tradeoff. **Capacity-reallocation hypothesis confirmed.**

### PMI-corrected security task accuracy

5 classification tasks × 25 samples = 125 evaluations (expanded from the 30-sample suite in v0.3.6). Old length-normalized scoring was mode-collapsed at 4/30 = 13.3% across all phases under logp scoring (eval failure, not model failure); PMI scoring fixed it.

| Task | Labels | Random | v0.3.5 | Most-common share |
|---|---|---|---|---|
| CVE Severity Classification | 4 | 25.0% | 8/25 (32.0%) | Critical 72% |
| Vulnerability Type Detection | 10 | 10.0% | 8/25 (32.0%) | IDOR 44% |
| Attack Technique Identification | 10 | 10.0% | 10/25 (40.0%) | LatMov 36% |
| CTF Challenge Categorization | 5 | 20.0% | 10/25 (40.0%) | Forensics 64% |
| MITRE ATT&CK Tactic Classification | 12 | 8.3% | 3/25 (12.0%) | LatMov 40% |
| **Overall** | — | ~14.5% | **39/125 (31.2%)** | — |

The 30-sample suite reported 12/30 = 40% on this same checkpoint. The drop to 31.2% is the eval getting more honest, not the model getting worse: with 25 balanced samples per task we now see CVE Severity is mode-collapsing toward "Critical" (72%) and MITRE Tactic is barely above random (12% vs 8.3% baseline). Vulnerability Type, Attack Technique, and CTF Categorization remain meaningfully above random (+22, +30, +20 pp) — those are the corpora that grew in the Phase 3.5 rebalance. See `CHANGELOG.md` v0.3.6 for the full discussion.

#### Phase 3.6 attempted next, regressed (v0.3.7)

The next training run added Exploit-DB (~3.77M tokens, 30% of the new corpus) and re-trained ghost-tiny at the same 30K-step recipe. The result was a 14.4 pp drop on the same eval suite:

| Task | Phase 3.5 | Phase 3.6 | Δ |
|---|---|---|---|
| CVE Severity Classification | 8/25 (32.0%) [72%] | 4/25 (16.0%) [60%] | −16 pp |
| Vulnerability Type Detection | 8/25 (32.0%) [44%] | 3/25 (12.0%) [**96%**] | −20 pp |
| Attack Technique Identification | 10/25 (40.0%) [36%] | 4/25 (16.0%) [60%] | −24 pp |
| CTF Challenge Categorization | 10/25 (40.0%) [64%] | 5/25 (20.0%) [48%] | −20 pp |
| MITRE ATT&CK Tactic Classification | 3/25 (12.0%) [40%] | 5/25 (20.0%) [76%] | +8 pp (mode-collapsed) |
| **Overall** | **31.2%** | **16.8%** | **−14.4 pp** |

Per-source perplexity confirmed the diagnosis: every existing source got 28–42% worse while Exploit-DB landed cleanly modeled (PPL 40.87). The "improved" overall PPL of −32.8% was misleading — Exploit-DB's heavy token share dragged the weighted average down regardless of how the existing sources fared.

**Conclusion:** ghost-tiny at 14.7M params is at capacity. More corpus at fixed model size has hit diminishing returns at this rung. The path forward is the model (ghost-small at 55M params), not more data. Phase 3.6 corpus + checkpoint preserved at `checkpoints/phase3.6_exploitdb/best_model.pt` as the ghost-small training target — if ghost-small absorbs the same corpus without per-source regression, the capacity-reallocation hypothesis is confirmed. See `CHANGELOG.md` v0.3.7 for the full per-source breakdown and reasoning.

#### Phase 4 ghost-small — capacity-reallocation hypothesis confirmed (v0.4.0)

ghost-small (~45M params, 6 layers / 512 d_model / 8 heads) trained on the same Phase 3.6 corpus that broke ghost-tiny. 30k steps, MPS, 15h wall-clock. Final val_loss **2.3535** — a 1.20-nat (~3.3× perplexity) drop relative to Phase 3.5 ghost-tiny (3.5518), and the loss curve was still descending at the final step.

The PMI security suite is more nuanced. Headline number drops vs Phase 3.5 (39/125 → 29/125, 31.2% → 23.2%), but with **logp scoring** (no PMI-correction) Phase 4 actually beats Phase 3.5 (24/125 vs 22/125, 19.2% vs 17.6%). The PMI advantage at Phase 3.5 is a calibration artifact — PMI subtracts the unconditional candidate log-prob to break ties, and a higher-capacity model with a tighter probability distribution gives PMI less separation to work with. On a 25-sample-per-task suite this can flip the headline.

| Task | P3.5 PMI | P3.5 logp | **P4 PMI** | **P4 logp** |
|---|---:|---:|---:|---:|
| CVE Severity | 32% | 24% | 24% | 24% |
| Vuln Type | 32% | 20% | **40%** | 16% |
| Attack Tech | **40%** | 8% | 16% | 12% |
| CTF Cat | **40%** | 28% | 28% | 28% |
| MITRE Tactic | 12% | 8% | 8% | **16%** |
| **Overall** | **31.2%** | 17.6% | 23.2% | **19.2%** |

Ranking by metric, honestly:
1. **Per-source PPL (density):** Phase 4 wins decisively (−83% overall vs Phase 3.5).
2. **Logp eval (conservative scoring):** Phase 4 wins narrowly (+1.6 pp).
3. **PMI eval (favors loose-distribution models):** Phase 3.5 wins (+8.0 pp).

Phase 4 is the new canonical for any density / generation use; Phase 3.5 stays on disk as the historical canonical and the higher PMI scorer. See `CHANGELOG.md` v0.4.0 for the full breakdown.

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
