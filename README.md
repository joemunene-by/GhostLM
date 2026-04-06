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

| Preset         | Layers | Dim  | Heads | Params   | Use Case              |
|----------------|--------|------|-------|----------|-----------------------|
| `ghost-tiny`   | 2      | 256  | 4     | ~3M      | Development & testing |
| `ghost-small`  | 6      | 512  | 8     | ~55M     | Default training      |
| `ghost-medium` | 12     | 768  | 12    | ~160M    | Production            |

## Project Structure

```
GhostLM/
├── data/
│   ├── collect.py          # Data collection pipeline (NVD API, papers, CTF)
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
│   ├── generate.py         # Inference with checkpoint loading
│   ├── benchmark.py        # Compare GhostLM vs GPT-2 perplexity
│   ├── plot_training.py    # Plot training loss curves
│   ├── chat.py             # Interactive terminal chat interface
│   └── push_to_hub.py      # Upload to HuggingFace Hub
├── tests/
│   └── test_model.py       # Pytest test suite
├── notebooks/
│   └── exploration.ipynb   # Architecture exploration notebook
├── Makefile                # Common development commands
├── CONTRIBUTING.md         # Contribution guidelines
├── MODEL_CARD.md           # HuggingFace-style model card
└── LICENSE                 # Apache 2.0
```

## Quick Start

```bash
git clone https://github.com/joemunene-by/GhostLM.git
cd GhostLM
make install
make test
```

## Usage

### Makefile Commands

```bash
make help          # Show all available commands
make install       # Install all dependencies
make test          # Run unit tests
make data          # Download and prepare training data
make train-tiny    # Train ghost-tiny (CPU-friendly, ~3M params)
make train-small   # Train ghost-small (~55M params, GPU recommended)
make generate      # Generate text from trained checkpoint
make chat          # Interactive chat with trained model
make benchmark     # Compare GhostLM vs GPT-2 perplexity
make plot          # Plot training loss curve
make clean         # Remove cache files
```

### 1. Collect Data

```bash
python data/collect.py
```

Downloads CVE descriptions from the NVD REST API, curated cybersecurity paper abstracts, and CTF writeups. Merges into train/validation JSONL splits.

### 2. Train

```bash
# Train with default (ghost-small)
python scripts/train.py

# Train ghost-tiny on CPU
python scripts/train.py --preset ghost-tiny --max-steps 2000 --batch-size 2 --device cpu

# Resume from checkpoint
python scripts/train.py --checkpoint checkpoints/best_model.pt

# Override hyperparameters
python scripts/train.py --preset ghost-tiny --max-steps 10000 --batch-size 16 --lr 1e-3
```

### 3. Evaluate

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```

Computes perplexity on cybersecurity texts and generates responses to 8 security prompts. Results saved to `logs/eval_results.json`.

### 4. Generate

```bash
python scripts/generate.py \
  --checkpoint checkpoints/best_model.pt \
  --prompt "A SQL injection attack works by" \
  --max-tokens 200 \
  --temperature 0.8
```

### 5. Interactive Chat

```bash
python scripts/chat.py --checkpoint checkpoints/best_model.pt
```

### 6. Benchmark vs GPT-2

```bash
python scripts/benchmark.py --checkpoint checkpoints/best_model.pt
```

### 7. Plot Training Curve

```bash
python scripts/plot_training.py --output logs/training_curve.png
```

### 8. Upload to HuggingFace

```bash
python scripts/push_to_hub.py \
  --checkpoint checkpoints/best_model.pt \
  --repo-id joemunene/GhostLM-tiny
```

### 9. Run Tests

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
