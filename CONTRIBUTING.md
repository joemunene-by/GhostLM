# Contributing to GhostLM

Thank you for your interest in contributing to GhostLM — the open-source cybersecurity language model built from scratch.

## Ways to Contribute

### 1. Data
- Find and add new cybersecurity datasets to `data/collect.py`
- Improve data cleaning in the `clean_text()` function
- Add new data sources: exploit-db, security blogs, OWASP documentation

### 2. Model Architecture
- Experiment with attention mechanisms (grouped query attention, sliding window)
- Try SwiGLU activation instead of GELU in FeedForward (see issue #9)
- Add RMSNorm as an alternative to LayerNorm
- Experiment with MoE (Mixture of Experts) layers

> RoPE and Flash Attention already landed in PR #13 — both are config-toggled on
> `GhostConfig` via `use_rope=True` and `use_flash_attention=True`.

### 3. Training
- Improve the learning rate schedule
- Add gradient accumulation for larger effective batch sizes
- Add distributed training support (`torch.distributed`) — see issue #8
- Add seed handling for reproducible runs
- Add checkpoint resumption robustness (LR scheduler state, RNG state)

> Mixed precision training (`torch.autocast`) already landed in an earlier PR.

### 4. Evaluation
- Add cybersecurity-specific benchmarks
- Build a CTF challenge evaluation suite
- Compare against other security-focused models

### 5. Documentation
- Improve docstrings
- Write tutorials and usage examples
- Translate documentation

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/GhostLM.git
   cd GhostLM
   ```
3. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   make install
   ```
5. Run tests to confirm everything works:
   ```bash
   make test
   ```
6. Create a branch:
   ```bash
   git checkout -b feat/your-feature-name
   ```
7. Make your changes, add tests if applicable
8. Push and open a Pull Request

## Code Style
- Follow PEP 8
- Add docstrings to all functions and classes
- Keep functions focused and under 50 lines where possible
- Add type hints to all function signatures

## Commit Message Format
```
feat: add grouped query attention to CausalSelfAttention
fix: resolve weight decay bug in configure_optimizers
docs: update README with training instructions
data: add exploit-db scraper to collect.py
```

## Questions?
Open an issue on GitHub or reach out directly.

Built with ❤️ by Joe Munene — Complex Developers

GitHub: https://github.com/joemunene-by/GhostLM
