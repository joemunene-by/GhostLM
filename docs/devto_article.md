# I Built an Open-Source Cybersecurity LLM From Scratch in Python

> What if you could build your own AI model — not fine-tune someone else's, not wrap an API — but actually build a transformer from scratch and train it on cybersecurity data?

That's exactly what I did. And I'm releasing it under Apache 2.0 so anyone can use it, improve it, and build on it.

Meet **GhostLM** — an open-source, cybersecurity-focused language model built entirely from scratch in PyTorch. No pretrained weights. No wrappers. Every single component written by hand.

GitHub: https://github.com/joemunene-by/GhostLM

---

## Why I Built GhostLM

Here's the thing about current AI models: they're incredibly powerful, but they weren't built for security. When you ask GPT-4 about a CVE vulnerability or a CTF challenge, it gives you a reasonable answer — but it's reasoning from general knowledge, not from deep security context.

I wanted a model that actually *understands* cybersecurity language — the patterns, the terminology, the attack methodologies. And I wanted to build it myself, not because I thought I could out-engineer OpenAI, but because **the best way to understand how something works is to build it from the ground up.**

My goal was simple: create the first open-source, cybersecurity-focused language model that anyone can run, inspect, and improve.

---

## What GhostLM Is

GhostLM is a decoder-only transformer language model — the same architecture family as GPT-2, GPT-3, and Llama — but built entirely from scratch. No `transformers.AutoModel`, no `from_pretrained()`. Just raw PyTorch tensors and matrix multiplications.

It comes in three sizes:

| Variant | Layers | Dim | Params | Status |
|---|---|---|---|---|
| ghost-tiny | 2 | 256 | ~14.5M | ✅ Trained |
| ghost-small | 6 | 512 | ~55M | 🔄 Planned |
| ghost-medium | 12 | 768 | ~160M | 🔜 Future |

It's trained on:
- **CVE vulnerability descriptions** from the NVD database
- **CTF writeups** covering real challenge types
- **Cybersecurity research papers** and abstracts

And it's fully open source under Apache 2.0.

---

## The Architecture

Let me show you what "built from scratch" actually looks like.

### Causal Self-Attention

This is the core of every transformer. Here's GhostLM's implementation — no `F.scaled_dot_product_attention`, no hidden magic:

```python
def forward(self, x):
    B, T, C = x.size()

    # Combined QKV projection and split
    qkv = self.c_qkv(x)
    q, k, v = qkv.split(self.n_heads * self.head_dim, dim=-1)

    # Reshape to (B, n_heads, T, head_dim)
    q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
    k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
    v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    # Scaled dot-product attention
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

    # Apply causal mask (lower triangular)
    att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))

    # Softmax + dropout + weighted sum
    att = F.softmax(att, dim=-1)
    y = self.attn_dropout(att) @ v

    # Reassemble heads and project back
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    return self.resid_dropout(self.proj(y))
```

Every line is intentional. The causal mask ensures the model can only attend to previous tokens (autoregressive). The attention weights are manually computed with the classic `QK^T / sqrt(d)` formula.

### Transformer Block

The block stacks attention and feed-forward layers with a pre-norm architecture:

```python
def forward(self, x):
    # Pre-norm + self-attention with residual
    x = x + self.attn(self.ln_1(x))
    # Pre-norm + feed-forward with residual
    x = x + self.ffn(self.ln_2(x))
    return x
```

**Why pre-norm?** I chose pre-normalization (LayerNorm before each sub-layer) over post-norm because it's significantly more stable for training, especially on smaller models. The gradients flow more cleanly through the residual connections, and you don't need as careful a learning rate schedule.

### Weight Tying

One optimization that saves ~25 million parameters: the output projection layer shares weights with the token embedding. Instead of learning two separate `vocab_size × d_model` matrices, we learn one and reuse it:

```python
self.lm_head.weight = self.token_embedding.weight
```

This is the same trick GPT-2 uses, and it works because the embedding and output projection are fundamentally doing the same thing — mapping between token space and hidden space.

---

## Training Data

The data pipeline is one of the most important parts of any ML project. GhostLM's pipeline collects from three sources:

### NVD CVE Descriptions (Real Data)

I hit the National Vulnerability Database REST API directly — no HuggingFace dependency needed. Paginated requests with rate limiting, parsing nested JSON responses, extracting English descriptions:

```python
url = "https://services.nvd.nist.gov/rest/json/cves/2.0?resultsPerPage=2000&startIndex=0"
resp = requests.get(url, timeout=30)
for item in resp.json()["vulnerabilities"]:
    cve_id = item["cve"]["id"]
    description = item["cve"]["descriptions"][0]["value"]
```

This gave me **9,925 real CVE descriptions** — the kind of text that says *"A buffer overflow in the XYZ component allows remote attackers to execute arbitrary code via crafted input."*

### The Full Pipeline

```
NVD API → 9,925 CVE descriptions (real)
Synthetic papers → 500 security research abstracts
Synthetic CTF writeups → 500 challenge solutions
─────────────────────────────────────────────────
Total: 10,925 records → ~490,532 tokens
Train: 10,378 | Validation: 547
```

The pipeline handles text cleaning (unicode normalization, whitespace stripping, non-printable character removal), tokenization, chunking, and train/val splitting — all in `data/collect.py`.

---

## Training Results

Here's where it gets interesting. I trained ghost-tiny on a **ThinkPad Yoga 11e with a Celeron N4100 and 4GB of RAM**. Yes, really.

### Loss Progression

| Steps | Train Loss | Val Loss | Notes |
|---|---|---|---|
| 0 | 10.84 | 10.04 | Random initialization |
| 500 | 7.12 | 6.27 | First CVE patterns emerge |
| 1,000 | 5.89 | 5.41 | Starting to form sentences |
| 2,000 | 4.63 | 4.58 | Grammar improving |
| 3,000 | 3.91 | 3.95 | Security vocabulary appearing |
| 4,000 | 3.52 | 3.58 | Coherent attack descriptions |
| 5,000 | 3.38 | 3.46 | Best checkpoint saved |

The loss curve is healthy — train and validation are tracking closely, no signs of overfitting yet.

### Generation at 5,000 Steps

Here's what the model generates when prompted with *"A SQL injection attack works by"*:

> A SQL injection attack works by using the admin_user sequences in the web server. Web Application Firewall Evasion Techniques present a critical defense layer against commercial and model checking. Our model achieves 94% detection rate with transformer-based sequence modeling to identify common vulnerability patterns including buffer overflows.

Is it perfect? No. It bleeds between topics (SQL injection → WAF → research paper language). But it's producing grammatically correct sentences with real security terminology. At 5,000 steps on a 14.5M parameter model running on a laptop from 2018, I'll take it.

### Honest Limitations

- **Topic coherence** — the model jumps between subjects mid-generation. It needs more steps to learn to stay on topic.
- **Memorization** — some outputs are lifted nearly verbatim from training data. More diverse data would help.
- **Size** — 14.5M params is tiny. ghost-small (55M) will be a significant jump.
- **CPU training** — at ~1.8s per step, 10,000 steps takes hours. GPU or TPU is needed for serious training.

---

## What's Next

I've already applied for **Google TPU Research Credits** to train ghost-small on proper hardware. The plan:

1. **ghost-tiny to 10,000+ steps** — finish what I started
2. **ghost-small on TPU/GPU** — 55M params with real compute
3. **HuggingFace Hub release** — public model weights anyone can download
4. **Live demo on HuggingFace Spaces** — try GhostLM in your browser
5. **Benchmark vs GPT-2** — objective comparison on cybersecurity tasks

---

## Try It Yourself

The entire project is open source. Clone it, run it, break it, improve it:

```bash
git clone https://github.com/joemunene-by/GhostLM.git
cd GhostLM

# Install everything
make install

# Download training data
make data

# Train ghost-tiny on CPU
make train-tiny

# Chat with the trained model
make chat

# Run the web demo
pip install gradio
python demo/app.py
```

I'm actively looking for contributors. If you want to help with:
- Finding new cybersecurity datasets
- Implementing Flash Attention or RoPE
- Adding distributed training
- Writing documentation

Check out [CONTRIBUTING.md](https://github.com/joemunene-by/GhostLM/blob/main/CONTRIBUTING.md) and open a PR.

---

## Final Thoughts

I'm a 20-year-old computer science student in Nairobi, Kenya. I don't have access to massive compute clusters or research lab budgets. But I do have curiosity, persistence, and a belief that **open-source AI shouldn't only come from well-funded labs.**

GhostLM is proof that you can build something meaningful from scratch with limited resources. The architecture is clean, the training pipeline works, and the model is learning. It's not going to replace GPT-4 — but it's a foundation that anyone can build on.

If you found this interesting, star the repo, try it out, and let me know what you think. The best part of open source is that it gets better when more people are involved.

**GitHub:** https://github.com/joemunene-by/GhostLM

**License:** Apache 2.0

Built with ❤️ in Nairobi, Kenya 🇰🇪
