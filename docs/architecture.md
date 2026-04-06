# GhostLM Architecture Deep-Dive

## Overview
GhostLM is a decoder-only transformer language model built entirely from scratch in PyTorch. This document explains every architectural decision in detail.

## Tokenizer
- Uses GPT-2 BPE (Byte Pair Encoding) via tiktoken
- Base vocabulary: 50,257 tokens
- 4 custom special tokens added: BOS, EOS, PAD, UNK
- Total vocabulary: 50,261 tokens
- Chunking support for documents longer than context window

## Embeddings
- Token embeddings: nn.Embedding(vocab_size, d_model)
- Positional embeddings: nn.Embedding(context_length, d_model) — learned, not sinusoidal
- Both summed and passed through dropout
- Weight tying: output projection shares weights with token embedding (reduces parameters, improves performance)

## Attention
- Multi-head causal self-attention
- Single combined QKV projection for efficiency
- Causal mask registered as buffer using torch.tril
- Scaled dot-product: softmax(QK^T / sqrt(head_dim)) * V
- Separate attention dropout and residual dropout
- Head dimension: d_model // n_heads

## Transformer Block
- Pre-norm architecture (LayerNorm before attention and FFN, not after)
- Pre-norm is more stable for deep networks than post-norm
- Residual connections around both attention and FFN sub-layers
- Two LayerNorms per block

## Feed-Forward Network
- Position-wise: applied independently to each token
- Architecture: Linear(d_model, d_ff) → GELU → Linear(d_ff, d_model) → Dropout
- d_ff = 4 * d_model by default
- GELU activation (smoother than ReLU, used in GPT-2 and BERT)

## Weight Initialization
- Linear layers: normal(mean=0, std=0.02)
- Embedding layers: normal(mean=0, std=0.02)
- Biases: zeros
- Scaled residual init: projection layers use std=0.02/sqrt(2*n_layers) for stability

## Output Head
- Linear projection from d_model to vocab_size
- No bias
- Weight tied to token embedding

## Model Variants
| Variant | Layers | d_model | n_heads | d_ff | Params |
|---|---|---|---|---|---|
| ghost-tiny | 2 | 256 | 4 | 1024 | ~14.5M |
| ghost-small | 6 | 512 | 8 | 2048 | ~55M |
| ghost-medium | 12 | 768 | 12 | 3072 | ~160M |

## Training
- Optimizer: AdamW (beta1=0.9, beta2=0.95, weight_decay=0.1)
- No weight decay on: biases, LayerNorm weights, embeddings
- LR schedule: linear warmup → cosine decay to 1e-5 minimum
- Gradient clipping: 1.0
- Gradient accumulation: 4 steps (effective batch size = batch_size * accum_steps)

## Design Decisions
- Why decoder-only? Autoregressive generation is natural for security text completion
- Why pre-norm? More stable training, easier to scale to more layers
- Why learned positional embeddings? Simpler than RoPE, sufficient for our context length
- Why weight tying? Reduces parameters by vocab_size * d_model (~25M params saved)
- Why GPT-2 tokenizer? Well-tested, handles code and technical text well
