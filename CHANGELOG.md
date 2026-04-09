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

## [Unreleased] — Upcoming

### Planned for v0.3.0
- Phase 2 training: 100K steps on Mac Mini M4
- Real arXiv cs.CR paper dataset integration
- HuggingFace Hub model weights release
- Gradio web demo

### Planned for v1.0.0
- ghost-small fully trained weights released
- REST API for inference
- Fine-tuning scripts for domain adaptation
