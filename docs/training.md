# GhostLM Training Guide

## Prerequisites
- Python 3.10+
- 4GB+ RAM (for ghost-tiny on CPU)
- GPU recommended for ghost-small and ghost-medium

## Quick Start
```bash
git clone https://github.com/joemunene-by/GhostLM.git
cd GhostLM
make install
make data
make train-tiny
```

## Step by Step

### 1. Install Dependencies
```bash
make install
```

### 2. Collect Training Data
```bash
make data
```
This downloads ~10,000 CVE descriptions from the NVD API and generates synthetic cybersecurity training data. Output: data/processed/train.jsonl and data/processed/val.jsonl

### 3. Train ghost-tiny (CPU-friendly)
```bash
python scripts/train.py --preset ghost-tiny --max-steps 5000 --batch-size 2 --device cpu
```

### 4. Train ghost-small (GPU recommended)
```bash
python scripts/train.py --preset ghost-small --max-steps 100000 --batch-size 32 --device cuda
```

### 5. Resume from checkpoint
```bash
python scripts/train.py --preset ghost-tiny --max-steps 10000 --checkpoint checkpoints/best_model.pt
```

### 6. Monitor training
```bash
make plot
```

## Hyperparameter Guide

| Parameter | Default | Notes |
|---|---|---|
| batch_size | 32 | Reduce to 2-4 for CPU |
| learning_rate | 3e-4 | Good default for AdamW |
| warmup_steps | 2000 | ~1% of max_steps |
| grad_clip | 1.0 | Prevents exploding gradients |
| grad_accum_steps | 4 | Simulates larger batch size |
| context_length | 128 | Reduce for low RAM |
| dropout | 0.1 | Increase if overfitting |

## Hardware Requirements

| Variant | RAM | VRAM | Time (CPU) | Time (GPU) |
|---|---|---|---|---|
| ghost-tiny | 4GB | 2GB | ~2h/1000 steps | ~5min/1000 steps |
| ghost-small | 16GB | 8GB | Not recommended | ~30min/1000 steps |
| ghost-medium | 32GB | 16GB | Not recommended | ~2h/1000 steps |

## Tips
- Always start with ghost-tiny to validate your pipeline
- Use --grad-accum 8 on very low RAM systems
- Monitor val_loss — if it stops decreasing, training has converged
- Save checkpoints frequently with --save-interval 500
