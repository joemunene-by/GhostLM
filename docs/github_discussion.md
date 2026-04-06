# Announcing GhostLM v0.1.0 — Help Wanted!

Hello everyone 👋

I'm Joe Munene, and I've just released **GhostLM** — an open-source cybersecurity-focused language model built entirely from scratch in PyTorch.

## What is GhostLM?
A decoder-only transformer trained on CVE vulnerability descriptions, CTF writeups, and security research. No pretrained weights, no wrappers — built from the ground up.

**Current status:**
- ghost-tiny (14.5M params) trained to 5,000 steps, val_loss = 3.46
- Loss reduced from 10.04 → 3.46 (65% reduction)
- Generating CVE-style security language already

## I Need Your Help With

### 1. Data Sources
What cybersecurity datasets should I add? I currently have:
- NVD CVE descriptions (9,925 records)
- Synthetic security papers
- Synthetic CTF writeups

Would love suggestions for real CTF writeup datasets, exploit-db data, OWASP docs, etc.

### 2. Architecture Improvements
I'm considering:
- RoPE instead of learned positional embeddings
- Flash Attention for faster training
- SwiGLU activation instead of GELU

Has anyone implemented these from scratch? Would love a PR.

### 3. Compute
I'm training on a ThinkPad with 4GB RAM (yes, really). Applied for Google TPU credits. If anyone has GPU/TPU access and wants to contribute compute time to train ghost-small, reach out.

### 4. Evaluation
What should a cybersecurity LLM be benchmarked on? I currently use perplexity. Should I build a CTF-solving benchmark? A CVE classification task?

## How to Contribute
See CONTRIBUTING.md — all skill levels welcome.

## Links
- GitHub: https://github.com/joemunene-by/GhostLM
- Dev.to article: (link when published)
- Reddit thread: (link)

Looking forward to building this together 🔐
