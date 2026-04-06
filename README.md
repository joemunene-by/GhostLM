# GhostLM

An open-source cybersecurity-focused language model built from scratch.

## About

GhostLM is a transformer-based language model designed specifically for cybersecurity reasoning tasks. It is built entirely from scratch in Python using PyTorch, with the goal of providing a transparent, auditable, and domain-specialized model for security research and analysis.

## Architecture

GhostLM uses a decoder-only transformer architecture with:

- **Tokenization**: GPT-2 BPE tokenizer (via tiktoken) with custom special tokens
- **Embeddings**: Learned token and positional embeddings
- **Transformer Blocks**: Multi-head causal self-attention + feed-forward networks with pre-norm and residual connections
- **Output Head**: Linear projection with weight tying to token embeddings
- **Optimizer**: AdamW with weight decay separation and cosine LR scheduling with warmup

### Model Presets

| Preset         | Layers | Dim  | Heads | Use Case              |
|----------------|--------|------|-------|-----------------------|
| `ghost-tiny`   | 2      | 256  | 4     | Development & testing |
| `ghost-small`  | 6      | 512  | 8     | Default training      |
| `ghost-medium` | 12     | 768  | 12    | Production            |

## Project Structure

```
GhostLM/
├── data/
│   ├── collect.py          # Data collection pipeline (CVE, papers, CTF writeups)
│   ├── raw/                # Raw downloaded data
│   └── processed/          # Processed train/val JSONL files
├── ghostlm/
│   ├── __init__.py         # Package exports
│   ├── config.py           # GhostLMConfig dataclass with presets
│   ├── model.py            # GhostLM transformer (attention, FFN, blocks)
│   ├── tokenizer.py        # GhostTokenizer (tiktoken wrapper)
│   ├── trainer.py          # GhostTrainer (training loop, checkpointing)
│   └── dataset.py          # GhostDataset and DataLoader utilities
├── scripts/
│   ├── train.py            # Training entry point with CLI args
│   ├── evaluate.py         # Perplexity + generation quality benchmarks
│   └── generate.py         # Inference with checkpoint loading
├── tests/
│   └── test_model.py       # Pytest test suite
└── notebooks/              # Experiment notebooks
```

## Installation

```bash
git clone https://github.com/joemunene-by/GhostLM.git
cd GhostLM
pip install -r requirements.txt
```

## Usage

### 1. Collect Data

```bash
python data/collect.py
```

Downloads CVE descriptions, security papers, and CTF writeups from HuggingFace datasets, then merges them into train/validation splits.

### 2. Train

```bash
# Train with default (ghost-small)
python scripts/train.py

# Train with a different preset
python scripts/train.py --preset ghost-tiny

# Resume from checkpoint
python scripts/train.py --checkpoint checkpoints/best_model.pt

# Override hyperparameters
python scripts/train.py --preset ghost-tiny --max-steps 10000 --batch-size 16 --lr 1e-3
```

### 3. Evaluate

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```

Computes perplexity on cybersecurity texts and generates responses to security prompts. Results saved to `logs/eval_results.json`.

### 4. Generate

```bash
python scripts/generate.py \
  --checkpoint checkpoints/best_model.pt \
  --prompt "A SQL injection attack works by" \
  --max-tokens 200 \
  --temperature 0.8
```

### 5. Run Tests

```bash
pytest tests/ -v
```

## Roadmap

- [ ] Flash attention integration
- [ ] Multi-GPU training with DDP
- [ ] Fine-tuning pipeline for specific security tasks (malware analysis, threat intel)
- [ ] Quantization and ONNX export
- [ ] Web-based inference demo
- [ ] Benchmark against existing security LLMs

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
