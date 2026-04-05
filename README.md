# GhostLM

An open-source cybersecurity-focused language model built from scratch.

## About

GhostLM is a transformer-based language model designed specifically for cybersecurity reasoning tasks. It is built entirely from scratch in Python using PyTorch, with the goal of providing a transparent, auditable, and domain-specialized model for security research and analysis.

## Architecture

GhostLM uses a standard decoder-only transformer architecture:

- **Embeddings**: Learned token and positional embeddings
- **Transformer Blocks**: Multi-head self-attention + feed-forward networks with layer normalization
- **Output Head**: Linear projection over the vocabulary for next-token prediction

Configuration defaults and hyperparameters are defined in `ghostlm/config.py`.

## Installation

```bash
git clone https://github.com/joemunene-by/GhostLM.git
cd GhostLM
pip install -r requirements.txt
```

## Training

```bash
# Collect and prepare data
python data/collect.py

# Train the model
python scripts/train.py
```

Training configuration can be adjusted in `ghostlm/config.py`.

## Roadmap

- [ ] Implement tokenizer training on cybersecurity corpora
- [ ] Build transformer blocks (attention, FFN, layer norm)
- [ ] Pre-training pipeline with wandb logging
- [ ] Evaluation benchmarks for cybersecurity tasks
- [ ] Inference and text generation scripts
- [ ] Model checkpointing and loading

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
