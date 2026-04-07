---
language:
  - en
license: apache-2.0
library_name: pytorch
tags:
  - cybersecurity
  - transformer
  - language-model
  - decoder-only
  - from-scratch
  - cve
  - ctf
  - security
datasets:
  - custom
pipeline_tag: text-generation
model-index:
  - name: ghost-tiny
    results: []
  - name: ghost-small
    results: []
---

# GhostLM — Cybersecurity Language Model

## Model Details

| Field | Value |
|---|---|
| **Model Names** | `ghostlm/ghost-tiny` (14.5M params), `ghostlm/ghost-small` (55M params) |
| **Architecture** | Decoder-only transformer |
| **Author** | [Joe Munene](https://github.com/joemunene-by) |
| **License** | Apache 2.0 |
| **Language** | English |
| **Framework** | PyTorch (built from scratch, no pretrained weights) |
| **Version** | 0.1.0 (pre-training) |

## Model Description

GhostLM is a cybersecurity-focused decoder-only transformer language model built entirely from scratch in PyTorch. No pretrained weights, no wrappers — every component (attention, feed-forward, embeddings, training loop) is hand-implemented.

The model is trained on CVE vulnerability descriptions from the National Vulnerability Database, CTF writeups, and security research papers. It is designed for cybersecurity reasoning tasks: CVE analysis, exploit explanation, penetration testing assistance, and security concept generation.

## Model Variants

| Variant | Layers | d_model | Heads | d_ff | Context | Params | Status |
|---|---|---|---|---|---|---|---|
| `ghostlm/ghost-tiny` | 2 | 256 | 4 | 1024 | 1024 | ~14.5M | Training |
| `ghostlm/ghost-small` | 6 | 512 | 8 | 2048 | 1024 | ~55M | Pre-training |
| `ghostlm/ghost-medium` | 12 | 768 | 12 | 3072 | 1024 | ~160M | Planned |

## Architecture

- **Type:** Decoder-only transformer with causal self-attention
- **Normalization:** Pre-norm (LayerNorm before attention and FFN sub-layers)
- **Positional encoding:** Learned positional embeddings
- **Activation:** GELU
- **Tokenizer:** GPT-2 BPE via tiktoken (50,257 base tokens + 4 special tokens = 50,261 total)
- **Weight tying:** Output projection shares weights with token embedding
- **Attention:** Multi-head causal self-attention with combined QKV projection
- **Initialization:** Normal(0, 0.02) with scaled residual init (std=0.02/sqrt(2*n_layers)) for projection layers

## Training Data

| Source | Records | Type | Description |
|---|---|---|---|
| NVD CVE Database | 9,925 | Real | Vulnerability descriptions from the NVD REST API v2.0 |
| Security Research Papers | 500 | Synthetic | Abstracts covering 10 cybersecurity research areas |
| CTF Writeups | 500 | Synthetic | Challenge solutions across 10 CTF categories |
| **Total** | **10,925** | | **~515K tokens** |

**Data splits:** 10,378 train / 547 validation

**Topics covered:** vulnerability detection, adversarial ML, network intrusion, cryptographic protocols, fuzzing, side-channel attacks, ransomware detection, supply chain security, memory safety, WAF evasion, SQL injection, XSS, buffer overflow, privilege escalation, reverse engineering, binary exploitation, steganography, network forensics.

## Training Details

| Parameter | Value |
|---|---|
| Optimizer | AdamW (beta1=0.9, beta2=0.95, weight_decay=0.1) |
| Learning rate | 3e-4 |
| LR schedule | Cosine decay with linear warmup |
| Warmup steps | 2,000 |
| Min LR | 1e-5 |
| Gradient clipping | 1.0 |
| Gradient accumulation | 4 steps |
| Batch size | 32 (ghost-small), 2-4 (ghost-tiny on CPU) |
| Dropout | 0.1 |
| Mixed precision | AMP on CUDA, fp32 on CPU |

**Weight decay separation:** No weight decay applied to biases, LayerNorm parameters, or embedding weights. Only linear layer weights receive weight decay.

**Hardware:** ghost-tiny trained on a ThinkPad Yoga 11e (Celeron N4100, 4GB RAM). ghost-small targets GPU training.

## Intended Uses

### Primary use cases
- CVE analysis and vulnerability explanation
- CTF challenge reasoning and methodology
- Penetration testing report generation
- Security concept explanation and education
- Cybersecurity text completion and generation

### Out-of-scope uses
- **Production security decisions:** This is a small research model. Do not use it to make real security assessments.
- **Malware creation:** The model should not be used to develop malicious software or exploits for unauthorized use.
- **Attacking systems without authorization:** Any use for illegal cybersecurity activity is prohibited.

## Limitations

- **Small model size:** At 14.5M-55M parameters, GhostLM is significantly smaller than production LLMs. Output quality is limited accordingly.
- **Limited training data:** ~10K documents is small for language model pre-training. The model has narrow domain coverage even within cybersecurity.
- **May generate inaccurate information:** The model can produce plausible-sounding but factually incorrect security information. Always verify outputs against authoritative sources.
- **No instruction tuning:** The model is a base language model, not instruction-tuned. It generates text completions, not structured answers.
- **Synthetic data bias:** Approximately 9% of training data is synthetic, which may introduce patterns not found in real security writing.
- **English only:** The model was trained exclusively on English text.

## Ethical Considerations

GhostLM is trained on cybersecurity content that inherently includes offensive security knowledge — exploit techniques, attack methodologies, and vulnerability details. This is the same information freely available in CVE databases, security conferences, and published research.

**Responsible use:**
- This model is intended for defensive security, education, and research.
- Users should follow responsible disclosure practices when working with vulnerability information.
- The model's outputs should not be used to attack systems without explicit authorization.
- Security professionals should apply the same ethical standards they would to any security tool.

**Dual-use risk:** Like any cybersecurity knowledge base, the information the model generates could theoretically be misused. However, the model's small size and limited capabilities make it far less capable than freely available tools and resources already in the security community.

## How to Use

```python
import torch
from ghostlm import GhostLM, GhostLMConfig, GhostTokenizer

# Load ghost-tiny
config = GhostLMConfig.from_preset("ghost-tiny")
model = GhostLM(config)
tokenizer = GhostTokenizer()

# Load trained weights
checkpoint = torch.load("checkpoints/best_model.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Generate
prompt = "A SQL injection attack works by"
ids = tokenizer.encode(prompt)
input_tensor = torch.tensor(ids).unsqueeze(0)
output = model.generate(input_tensor, max_new_tokens=100, temperature=0.8, top_k=50)
print(tokenizer.decode(output[0].tolist()))
```

## Evaluation

- **Metric:** Perplexity on cybersecurity holdout set
- **Benchmark:** Generation quality on 8 cybersecurity prompts (SQL injection, XSS, buffer overflow, privilege escalation, etc.)
- **Results:** To be published after training completes

## Citation

```bibtex
@misc{ghostlm2026,
  author = {Joe Munene},
  title = {GhostLM: An Open-Source Cybersecurity-Focused Language Model},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/joemunene-by/GhostLM}
}
```

## Links

- **GitHub:** [github.com/joemunene-by/GhostLM](https://github.com/joemunene-by/GhostLM)
- **Author:** [Joe Munene](https://github.com/joemunene-by)
- **License:** [Apache 2.0](LICENSE)
