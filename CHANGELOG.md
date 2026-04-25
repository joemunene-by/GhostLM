# Changelog

All notable changes to GhostLM will be documented in this file.

Format: [Version] — Date — Description

---

## [0.1.0] — 2026-04-06 — Initial Release

### Added
- Decoder-only transformer architecture built from scratch in PyTorch
- CausalSelfAttention with manual scaled dot-product attention and causal masking
- Pre-norm TransformerBlock with residual connections
- FeedForward network with GELU activation
- GhostLMConfig dataclass with three presets: ghost-tiny, ghost-small, ghost-medium
- Weight-tied output projection (lm_head shares weights with token_embedding)
- Scaled residual initialization for stable deep network training
- GhostTokenizer wrapping GPT-2 BPE with 4 custom cybersecurity special tokens
- GhostDataset and build_dataloaders for PyTorch DataLoader integration
- GhostTrainer with cosine LR schedule, linear warmup, gradient clipping
- Checkpoint saving and loading with best_model.pt tracking
- JSON training log persistence
- Data collection pipeline: NVD CVE API, synthetic security papers, CTF writeups
- 10,925 cybersecurity training records (10,378 train / 547 validation)
- scripts/train.py — CLI training entry point with preset and override support
- scripts/generate.py — inference from checkpoint with temperature and top-k sampling
- scripts/evaluate.py — perplexity and generation quality benchmarks
- scripts/benchmark.py — GhostLM vs GPT-2 perplexity comparison
- scripts/chat.py — interactive terminal chat interface
- scripts/plot_training.py — training loss curve visualization
- scripts/push_to_hub.py — HuggingFace Hub upload utility
- notebooks/exploration.ipynb — architecture walkthrough notebook
- GitHub Actions CI workflow (10/10 tests on every push)
- Apache 2.0 license
- MODEL_CARD.md — HuggingFace-style model card
- CONTRIBUTING.md — contributor guide
- Makefile — one-command workflow (make train-tiny, make chat, etc.)

### First Training Run
- ghost-tiny (14.5M params) trained for 500 steps on CPU
- Loss reduced from 10.04 → 6.27 (val_loss)
- CVE language patterns emerged after 500 steps
- Checkpoint saved to checkpoints/best_model.pt

### Known Limitations
- ghost-tiny only trained for 500 steps — not yet useful for real tasks
- Training on CPU is slow (~1.8s/step) — GPU or TPU needed for ghost-small
- Synthetic data used for papers and CTF writeups — real datasets planned

---

## [0.2.0] — 2026-04-09 — Phase 1 Training Complete (10K Steps)

### Training Milestone
- ghost-tiny (14.5M params) trained to 10,000 steps on CPU
- Final training loss: ~1.97
- Final validation loss: ~2.74
- No overfitting observed — stable loss curves throughout

### Evaluation Results
- Cybersecurity perplexity: 2,183.94 (vs GPT-2 baseline: 26.76)
- CVE Severity Classification: 20.0% accuracy
- Vulnerability Type Detection: 10.0% accuracy
- Attack Technique Identification: 10.0% accuracy
- Overall security eval score: 13.3%
- Model generates security domain vocabulary but lacks reasoning capability at this scale

### Architecture
- Simplified model: learned positional embeddings + GELU FFN
- 2 layers, 256 dim, 4 heads, 1024 context length

### Updated
- MODEL_CARD.md with full evaluation results and benchmark comparison
- Training curve plots and benchmark logs

---

## [0.2.1] — 2026-04-22 — Phase 2 Readiness

### Added
- **RoPE (Rotary Position Embeddings)** — config-toggled via `use_rope=True`; replaces learned positional embeddings with the relative-position encoding used by LLaMA / Mistral.
- **Flash Attention** path — config-toggled via `use_flash_attention=True`; routes through PyTorch 2.0+ `scaled_dot_product_attention` for `O(n)` memory.
- **Safetensors export** with `config.json` sidecar and SHA-256 checksum (see `scripts/export.py`). Pickle-free distribution path for HF Hub.

### Changed
- Pinned dependency versions; added PEP 639 license metadata.
- Test suite grown from 10 → 16 tests.

---

## [0.2.2] — 2026-04-23 — Data Audit + Corpus Rebalancing

### Added
- `scripts/data_audit.py` — length percentiles, dedup rate, CVE-year distribution, CTF category share, token share, train/val leakage check. Writes a 4-panel diagnostic chart to `logs/data_audit.png`.

### Changed
- **CVE collector** rewritten to 119-day NVD windows with append mode; coverage extended from 1999–2005 to 1999–2025 (27 years, ~19,925 records).
- **Paper collector** switched from hand-written synthetic `× 50` padding to the arXiv cs.CR Atom API — 1,000 real abstracts.
- **Synthetic CTF generator** emits unique templates only (fixed a rotation bug that limited output to 12 of ~22 templates).
- `merge_datasets` now uses a **deterministic MD5-bucket split** — identical or near-duplicate texts always land in the same split, eliminating the train/val leakage that affected v0.2.0.

### Note
- Previous Phase 1 evaluation numbers (val_loss 2.74, perplexity 2,183.94) were measured on the pre-audit corpus with ~9% train/val leakage. They are preserved in the v0.2.0 entry above for archaeological reference but should be treated as superseded.

---

## [0.3.0] — 2026-04-25 — Phase 2 Training Complete

### Training Milestone
- ghost-tiny (14.7M params) trained for 10,000 steps on the rebalanced corpus.
- Hardware-of-record: Mac Mini M4 (CPU). Training time: ~70 minutes wall-clock.
- Resumed from Phase 1 step-4000 checkpoint pulled from the corpus-prep box and continued on the leakage-free split.
- **Final validation loss: 3.7813** (perplexity ≈ 44) — the first trustworthy held-out measurement of GhostLM.
- Phase 1's lower val_loss (2.74) is preserved in v0.2.0 but not directly comparable: it was measured on a leaky split. Phase 2's number is the honest baseline going forward.

### Added
- Phase 2 corpus: 19,925 NVD CVE records + 1,000 arXiv cs.CR abstracts + 3,000 synthetic CTF writeups (Ollama-pipeline) → 23,049 records / ~2.66M tokens after dedup.
- `checkpoints/best_model.pt` — Phase 2 best (val_loss 3.7813). Phase 1's `best_model.pt` preserved as `checkpoints/best_model_phase1.pt`.
- `checkpoints/checkpoint_step_10000.pt` — final Phase 2 checkpoint.
- `logs/training_log.json` — periodic eval snapshots from the Phase 2 run.
- Sample generations added to MODEL_CARD with honest characterization (vocabulary acquired; semantics absent).
- New `ROADMAP.md` — multi-year scale ladder (ghost-tiny → ghost-small → ghost-base → ghost-1B), corpus targets per rung, compute estimates.
- New `CORPUS.md` — current sources, expansion targets (CTFtime, security blogs, MITRE ATT&CK, tool docs), licensing notes.

### Changed
- License: standardized on **MIT** (LICENSE was MIT, MODEL_CARD/CHANGELOG previously said Apache 2.0 — fixed).
- README & MODEL_CARD updated with grounded Phase 2 numbers, scale ladder, and honest framing.
- CONTRIBUTING.md adds corpus expansion as a first-class contribution track.

---

## [0.3.1] — 2026-04-25 — Phase 2 Evaluation Refresh

### Evaluation Results (Phase 2 checkpoint)
- **Cyber-text perplexity vs GPT-2:** 152.71 (Phase 2) vs 2,183.94 (Phase 1) vs 26.76 (GPT-2 124M baseline) — **14.3× improvement** over Phase 1 on the same hardcoded `BENCHMARK_TEXTS` set; still 5.7× off GPT-2, expected for the params/tokens budget.
- **Security-domain task eval:** 4/30 (13.3%) — same numerical score as Phase 1, but with a different mode-collapse pattern. Phase 2 predicts "High" for every CVE Severity, "Cross-Site Scripting" for every Vuln Type, and "Supply Chain Compromise" for every Attack Technique. The model has learned the most-frequent label per task, not the discriminative structure. Random baseline is ~33%; 13.3% is below random — confirms structured-task evaluation is not yet meaningful at this scale.

### Added
- `logs/benchmark_phase2.json` — Phase 2 GPT-2 perplexity benchmark output.
- `logs/eval_security_phase2.json` — Phase 2 security-task eval output with per-question detail.
- `scripts/plot_phase_comparison.py` — generates `logs/phase_comparison.png` (3-panel: final val_loss, perplexity, security accuracy for Phase 1 vs Phase 2 vs GPT-2).
- MODEL_CARD `Evaluation` section expanded with concrete Phase 2 perplexity and mode-collapse details.

### Note
- Per-step training logs (`logs/training_log.json` and `archive/logs_v1_pre_corpus_fix/training_log.json`) only flushed entries late in training (3 and 5 endpoint datapoints respectively), so a true side-by-side curve plot was not feasible. The phase-comparison plot uses the endpoint metrics that actually exist.

---

## [0.3.2] — 2026-04-25 — Phase 3 NVD-at-scale

Corpus milestone, not a model release. The released checkpoint is still v0.3.0's (val_loss 3.78). The next ghost-tiny refresh run will be the first to train on this corpus.

### Corpus
- **Full NVD pull complete:** 333,540 CVE records, 1999–2026 (28 years), via `scripts/collect_nvd_full.py` with proper `startIndex` pagination.
- **~12× corpus expansion** vs. v0.3.0 baseline (2.66M → ~30M tokens).
- After dedup + merge: ~309K unique records, ~293K train / ~15K val. Deterministic-hash split, leakage 0.
- NVD year skew (expected): 2020s 189,946 / 2010s 102,581 / 2000s 40,156 / 1990s 857. Weighted toward 2018+, reflecting actual CVE publication scaling.
- NVD intra-source duplication: 7.9% (4,635 dup groups, 26,316 extra records) caught by merge dedup.
- **Token-share now lopsided** — NVD 87%, CTF 5%, papers 2%. The next *corpus* track is diversity (CTFtime + MITRE ATT&CK + security research blogs), not deeper NVD.

### Build
- `Makefile` switched to `PYTHON ?= python3` so it works on Mac (no `python` symlink) and Linux without local hacks.

### Docs
- `CORPUS.md`: current corpus snapshot updated to post-pull numbers; v0.3.0 baseline preserved as the corpus the released checkpoint was trained on.
- `ROADMAP.md`: Phase 3 NVD-at-scale marked done; ghost-small gating note updated with current ~30M-token state vs. ~500M-1B target.

---

## [Unreleased] — Upcoming

### Planned for v0.3.x — ghost-tiny refresh on the new corpus
- Training run on the ~30M-token Phase 3 corpus to validate the recipe scales with data.
- Same architecture, more steps to match the larger corpus.
- Headline metric: does val_loss meaningfully drop below v0.3.0's 3.78 with 12× more data?

### Planned for v0.4.0 — Corpus expansion
- CTFtime archive ingestion.
- Curated security-research-blog corpus (Project Zero, PortSwigger, Trail of Bits, etc.).
- MITRE ATT&CK structured + unstructured.
- Target: 10–100× current corpus size as substrate for ghost-small.

### Planned for v1.0.0 — Release
- ghost-small fully trained weights released.
- Public REST API.
- HuggingFace Hub publication (safetensors).
