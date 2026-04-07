# Building GhostLM: Architecture Decisions for a Cybersecurity Language Model

*Joe Munene, 2026*

---

## Why build from scratch

The obvious question: why not just fine-tune GPT-2? It's 124M parameters, well-documented, and there are a hundred tutorials on how to do it. I had three reasons.

First, I wanted to actually understand transformers. Not "I read the Attention Is All You Need paper" understand — I mean line-by-line, gradient-by-gradient understand. You don't get that by calling `model = GPT2LMHeadModel.from_pretrained("gpt2")` and swapping in a LoRA adapter. You get it by implementing causal masking by hand, debugging why your attention weights are NaN at step 300, and figuring out that your positional embeddings were initialized wrong.

Second, GPT-2's architecture carries baggage. Post-norm residual connections. A tokenizer trained on WebText. Hyperparameters tuned for general English text at 2019-era scale. If I'm building something for cybersecurity, I want to control every decision from the ground up — normalization placement, weight initialization scaling, dropout strategy, all of it.

Third, I wanted a model I could actually train on a ThinkPad with 4GB of RAM. Fine-tuning GPT-2 small (124M params) requires holding the full model in memory plus optimizer states. Ghost-tiny at 14.5M params fits comfortably, trains in reasonable time on CPU, and still teaches you everything about the architecture. You scale up when you have the compute, not before.

## Architecture overview

GhostLM is a standard decoder-only transformer. Nothing exotic. The value isn't in inventing new architecture — it's in getting the details right at small scale.

```
Input tokens
    → Token embedding + Learned positional embedding
    → Dropout
    → N x [LayerNorm → Causal Self-Attention → Residual
            → LayerNorm → FFN → Residual]
    → Final LayerNorm
    → Linear projection (weight-tied to token embedding)
    → Logits
```

Three model sizes:

| Variant | Layers | d_model | Heads | d_ff | Params |
|---|---|---|---|---|---|
| ghost-tiny | 2 | 256 | 4 | 1024 | 14.5M |
| ghost-small | 6 | 512 | 8 | 2048 | 55M |
| ghost-medium | 12 | 768 | 12 | 3072 | 160M |

Context length is 1024 tokens across all variants. Tokenizer is GPT-2 BPE via tiktoken with 50,261 tokens (50,257 base + BOS, EOS, PAD, UNK).

## Pre-norm vs post-norm

Original transformer (Vaswani et al.) uses post-norm: apply the sub-layer, add the residual, then normalize. GPT-2 uses this. It works, but it's known to be unstable during training, especially as you go deeper. The gradients through the residual stream get scaled by the LayerNorm at each layer, which compounds.

Pre-norm flips it: normalize first, then apply the sub-layer, then add the residual. The residual connection now carries un-normalized activations straight through the network. This makes the gradient flow much cleaner.

```python
# Post-norm (GPT-2 style) — not what I use
x = self.ln_1(x + self.attn(x))
x = self.ln_2(x + self.ffn(x))

# Pre-norm (GhostLM) — what I use
x = x + self.attn(self.ln_1(x))
x = x + self.ffn(self.ln_2(x))
```

The practical difference: pre-norm models are easier to train. I can use a higher learning rate, training is more stable across different model sizes, and I don't need to babysit the loss curve as much. For a project where I'm iterating quickly on a CPU, that stability matters more than any marginal quality difference post-norm might give at convergence.

One thing pre-norm requires: a final LayerNorm after the last transformer block, before the output projection. Without it, the un-normalized residual stream feeds directly into the logits, and you get garbage. This is easy to forget if you're following post-norm tutorials.

## Learned vs sinusoidal positional embeddings

Sinusoidal embeddings (the original transformer approach) are deterministic — you compute them from a formula involving sin/cos at different frequencies. They're elegant and they generalize to sequence lengths not seen during training.

Learned embeddings are just an `nn.Embedding(context_length, d_model)` that gets trained alongside everything else. Simpler to implement, one line of code, and the model learns whatever positional patterns work best for the data.

I went with learned embeddings for a few reasons:

1. At 1024 context length, the positional embedding table is small. For ghost-tiny it's 1024 * 256 = 262K parameters. Negligible.
2. The model never needs to generalize beyond its context length. I'm not doing extrapolation.
3. Learned embeddings consistently match or slightly beat sinusoidal in practice at these scales.

The tradeoff is that I can't process sequences longer than 1024 tokens without retraining. That's fine for now. If I needed length extrapolation, I'd switch to RoPE (Rotary Position Embeddings), not sinusoidal.

## Weight tying

The output projection (the linear layer that maps d_model back to vocab_size to produce logits) shares its weight matrix with the token embedding. This is a single line:

```python
self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
self.lm_head.weight = self.token_embedding.weight
```

For ghost-small, this saves vocab_size * d_model = 50,261 * 512 = ~25.7M parameters. That's almost half the model's total parameter count. The intuition is that tokens which are semantically similar should have similar embeddings, and the output distribution should respect that same similarity structure. Empirically, weight tying improves performance at small scale — you're effectively giving the model a stronger inductive bias with fewer parameters to learn.

The only subtlety: you have to be careful with weight decay. The tied weight serves double duty (embedding lookup and output projection), so I exclude it from weight decay to avoid conflicting regularization signals.

## Attention implementation

Nothing unusual here, but a few details worth noting.

I use a single combined QKV projection instead of three separate linear layers:

```python
self.c_qkv = nn.Linear(d_model, 3 * d_model, bias=True)
```

Then split the output into Q, K, V. This is slightly more efficient than three separate projections because it's a single matrix multiply. The compiler can also fuse operations better.

Causal masking is done with `torch.tril` registered as a persistent buffer. The mask is pre-allocated at the full context length and sliced to the actual sequence length during forward. No recomputation per step.

```python
self.register_buffer(
    "causal_mask",
    torch.tril(torch.ones(context_length, context_length))
    .view(1, 1, context_length, context_length),
    persistent=False,
)
```

Scaling is `1/sqrt(head_dim)` applied after the QK^T product, before masking and softmax. Standard.

Separate dropout layers for attention weights and residual connections. Both default to 0.1. At 14.5M parameters on ~10K documents, the model needs all the regularization it can get.

## Feed-forward network

Two linear layers with GELU activation in between:

```
d_model → d_ff → GELU → d_model → Dropout
```

The expansion ratio d_ff/d_model = 4 is standard (following GPT-2 and most subsequent work). GELU over ReLU because it's smoother at zero, which helps gradient flow. The practical difference at this scale is small, but GELU is the default for a reason.

I considered SwiGLU (used in Llama and most modern architectures) but decided against it. SwiGLU adds a gating mechanism that slightly increases the parameter count per layer and complicates the implementation. At 14.5M params, the simpler architecture is easier to debug and the performance difference is negligible.

## Weight initialization

Standard normal initialization with std=0.02 for all linear and embedding layers. Biases initialized to zero.

The important detail is scaled residual initialization. Output projections in the attention and FFN blocks (the layers right before the residual add) use a reduced standard deviation:

```python
std = 0.02 / math.sqrt(2 * n_layers)
```

This comes from the GPT-2 paper. Each transformer block has two residual additions (attention + FFN), so there are 2*n_layers additions total. Scaling the initialization of these projections by 1/sqrt(2*n_layers) keeps the variance of the residual stream roughly constant regardless of depth. Without this, deeper models have exploding activations in early training.

For ghost-tiny (2 layers), the scaling factor is 0.02/sqrt(4) = 0.01. For ghost-small (6 layers), it's 0.02/sqrt(12) ≈ 0.0058. Small numbers, big difference in training stability.

## Training pipeline

### Data collection

The backbone of the training data is CVE descriptions from NIST's National Vulnerability Database. I hit the NVD REST API v2.0, pull vulnerability descriptions in batches, clean them, and write to JSONL. About 9,925 real CVE records.

This is supplemented with 500 synthetic security research abstracts and 500 synthetic CTF writeups covering topics like SQL injection, buffer overflows, privilege escalation, reverse engineering, cryptography, and network forensics. "Synthetic" means I generated structured writeups that follow realistic patterns — they're not copy-pasted from real sources, but they're domain-accurate.

Total dataset: 10,925 documents, roughly 515K tokens after BPE encoding. Split 95/5 into train (10,378) and validation (547).

515K tokens is tiny by language model standards. GPT-2 was trained on 8 billion tokens. I'm working with 0.006% of that. This is the biggest constraint on model quality, and it's the thing I'd fix first with more resources.

### Tokenization

I use GPT-2's BPE tokenizer via tiktoken. Four special tokens added: BOS (beginning of sequence), EOS (end of sequence), PAD, and UNK. Total vocabulary: 50,261 tokens.

Why GPT-2's tokenizer instead of training my own? Two reasons. First, BPE trained on WebText handles code, technical terminology, and mixed-case text well — exactly what cybersecurity documents look like. CVE descriptions have identifiers like `CVE-2023-12345`, function names like `memcpy`, and mixed prose/code. GPT-2's tokenizer handles all of this without pathological splits.

Second, if I ever want to initialize from GPT-2 weights or compare directly against GPT-2, having the same tokenizer eliminates a confounding variable.

Documents longer than the 1024-token context window are chunked into overlapping segments during dataset preparation.

## Training loop

### AdamW with weight decay separation

The optimizer setup is more nuanced than just `torch.optim.AdamW(model.parameters(), lr=3e-4)`. I separate parameters into two groups:

**Weight decay applied (0.1):** All `nn.Linear` weight matrices.

**No weight decay:** All biases, all `nn.LayerNorm` parameters, all `nn.Embedding` parameters, and the tied lm_head weight.

The logic: weight decay is a regularizer that pushes weights toward zero. This makes sense for large weight matrices that might overfit. It doesn't make sense for biases (they're small and need to shift activations), LayerNorm parameters (they're scale/shift parameters, not feature extractors), or embeddings (they need to be free to represent token semantics without a zero-pulling prior).

The implementation walks the module tree, checks `isinstance` for each module type, and categorizes every parameter. An assertion verifies no parameters are missed. This is one of those things that's easy to get wrong silently — if you accidentally apply weight decay to LayerNorm, training still works, it's just slightly worse, and you'd never know.

### Cosine learning rate schedule

Linear warmup from 0 to 3e-4 over 2,000 steps, then cosine decay to a minimum of 1e-5.

```python
if step < warmup:
    lr = base_lr * (step + 1) / warmup
else:
    decay_ratio = (step - warmup) / (max_steps - warmup)
    lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(pi * decay_ratio))
```

The warmup is critical. Without it, the model sees a high learning rate on randomly initialized weights and the loss spikes or goes NaN in the first few hundred steps. 2,000 steps of warmup (roughly 1% of training for ghost-small) gives the model time to find a reasonable region of the loss landscape before the learning rate peaks.

Cosine decay is gentler than linear decay — it stays near the peak LR for longer during the middle of training, then drops smoothly toward the end. The 1e-5 minimum prevents the learning rate from hitting exactly zero, which would freeze training.

### Gradient accumulation

With a batch size of 32 and 4 accumulation steps, the effective batch size is 128. On CPU with ghost-tiny, I drop to batch_size=2 with 4 accumulation steps for an effective batch of 8.

The implementation splits each batch into micro-batches, does a forward/backward pass on each, accumulates the gradients, then does a single optimizer step with gradient clipping at 1.0.

```python
for mx, my in zip(micro_x, micro_y):
    _, loss = self.model(mx, targets=my)
    scaled_loss = loss / len(micro_x)
    scaled_loss.backward()

torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

The `set_to_none=True` in `zero_grad` is a minor optimization — it sets gradient tensors to None instead of filling them with zeros, which saves a memset operation and reduces memory fragmentation.

### Mixed precision

AMP (Automatic Mixed Precision) is enabled automatically on CUDA. The GradScaler handles loss scaling to prevent underflow in fp16 gradients. On CPU and MPS, training runs in fp32.

This isn't critical for ghost-tiny (it fits in memory regardless), but for ghost-small and ghost-medium on GPU, AMP roughly doubles throughput and halves memory usage.

## What the model can and can't do at 14.5M params

Let me be direct: 14.5M parameters trained on 515K tokens is not going to produce a useful cybersecurity assistant. That's not the point.

**What it can do:**
- Learn the statistical patterns of cybersecurity text
- Generate plausible-looking CVE-style descriptions
- Complete security-related prompts with domain-relevant vocabulary
- Demonstrate that the architecture and training pipeline work correctly
- Achieve decreasing perplexity on the validation set (training loss went from ~11 to ~6.7 in the first 500 steps)

**What it can't do:**
- Provide accurate security analysis
- Reason about vulnerabilities
- Generate correct exploit explanations
- Anything you'd actually want to use in a security workflow

The gap between "generates text that looks like security writing" and "understands security concepts" is enormous. Closing that gap requires orders of magnitude more data and parameters. Ghost-tiny is a proof of concept, not a product.

## Lessons learned

**Start small, validate everything.** I spent the first week just getting ghost-tiny to train without NaN losses on CPU. Every bug in the architecture, initialization, or training loop showed up at small scale. If I'd started with ghost-small on GPU, debugging would have been 10x slower.

**Data quality matters more than quantity at small scale.** The real CVE descriptions from NVD are higher quality than the synthetic data. The model learns more coherent patterns from 9,925 real documents than from mixing in synthetic ones. If I had more real data, I'd drop the synthetic entirely.

**Weight decay separation is not optional.** Early experiments without proper decay separation produced noticeably worse validation loss. It's one of those things where the code "works" either way, but the details compound.

**Checkpointing aggressively is cheap insurance.** Saving every 500-1000 steps costs almost nothing in disk space for a 14.5M parameter model. I've been glad to have old checkpoints multiple times when experimenting with hyperparameters.

## What I'd do differently with more compute

**More data, first and always.** The 515K token dataset is the bottleneck. I'd scrape exploit-db, pull real arXiv cs.CR papers in full text, grab OWASP documentation, index Metasploit module descriptions, and crawl HackTheBox/TryHackMe writeups. Target: 50M+ tokens of real cybersecurity text. This alone would probably improve quality more than any architecture change.

**Scale to ghost-medium (160M params).** The architecture supports it already. 160M params on 50M+ tokens would be in the range where language models start showing emergent capabilities. Whether that emerges for domain-specific cybersecurity tasks is an open question worth answering.

**Switch to RoPE.** Learned positional embeddings work fine at 1024 context, but they don't extrapolate. RoPE (Rotary Position Embeddings) encodes position through rotation of the query/key vectors, which generalizes better to unseen sequence lengths. If I wanted to extend context to 4096 or 8192, RoPE is the move.

**Flash Attention.** My current attention implementation is the naive O(n^2) version — compute the full attention matrix, mask it, softmax, multiply by values. Flash Attention fuses these operations into a single CUDA kernel with tiling, reducing memory from O(n^2) to O(n) and improving wall-clock speed 2-4x. At 1024 context length this isn't a bottleneck, but at 4096+ it's mandatory.

**SwiGLU activation.** Replace the GELU feed-forward with SwiGLU (the gated variant used in Llama). Slightly more parameters per layer, but consistently better performance at the same compute budget.

**Distributed training.** PyTorch DDP or FSDP to train across multiple GPUs. Ghost-medium at 160M params is still small enough for a single GPU, but if I push to 1B+ parameters, distributed training becomes necessary.

**Instruction tuning.** After pre-training, fine-tune on cybersecurity Q&A pairs, structured vulnerability reports, and CTF solution walkthroughs. This is what turns a text completion model into something that can actually follow instructions and provide useful answers.

**Evaluation beyond perplexity.** Build a proper benchmark: can the model correctly identify vulnerability types from descriptions? Can it classify CVE severity? Can it generate syntactically valid security recommendations? Perplexity tells you the model is learning; task-specific evaluation tells you if it's learning anything useful.

## Wrapping up

GhostLM is a from-scratch implementation of a decoder-only transformer for cybersecurity text. The architecture decisions are intentionally conservative — pre-norm, learned positional embeddings, GELU, weight tying — because the goal is a correct, well-understood foundation that can be scaled up, not a novel architecture.

The model at 14.5M parameters is a teaching tool and proof of concept. The path to something actually useful runs through more data, more parameters, and more training — all of which the current codebase is designed to support.

Code is at [github.com/joemunene-by/GhostLM](https://github.com/joemunene-by/GhostLM). Apache 2.0 licensed. Contributions welcome.
