# GhostLM — Cybersecurity Language Model

## Model Description

GhostLM is an open-source, decoder-only transformer language model built entirely from scratch in PyTorch. It is purpose-built for cybersecurity reasoning tasks, including CVE analysis, CTF challenge solving, exploit explanation, and penetration testing assistance.

| Field       | Value          |
|-------------|----------------|
| **Author**  | Joe Munene     |
| **License** | Apache 2.0     |
| **Version** | 0.1.0 (pre-training) |

## Model Architecture

| Parameter            | Value                      |
|----------------------|----------------------------|
| Architecture         | Decoder-only Transformer   |
| Parameters (ghost-small) | ~55M                   |
| Context Length       | 1024 tokens                |
| Layers               | 6                          |
| Attention Heads      | 8                          |
| Embedding Dimension  | 512                        |
| Feed-forward Dimension | 2048                     |
| Tokenizer            | GPT-2 BPE (50,261 tokens)  |
| Positional Encoding  | Learned                    |

## Training Data

- **NVD CVE Database** — 10,000+ vulnerability descriptions fetched from the NVD REST API
- **arXiv cs.CR Papers** — Cybersecurity and cryptography research paper abstracts
- **CTF Writeups** — Capture The Flag challenge solutions and methodology descriptions
- **Total** — ~15,000+ cybersecurity documents merged into train/validation splits

## Intended Uses

**Allowed:**
- CVE analysis and vulnerability explanation
- CTF challenge reasoning assistance
- Security research and education
- Penetration testing report generation
- Cybersecurity concept explanation

**Not Allowed:**
- Creating malware or exploit code for malicious purposes
- Attacking systems without authorization
- Any illegal cybersecurity activity

## Model Variants

| Variant        | Layers | Dim  | Params   | Status              |
|----------------|--------|------|----------|---------------------|
| `ghost-tiny`   | 2      | 256  | ~3M      | Available for testing |
| `ghost-small`  | 6      | 512  | ~55M     | Pre-training        |
| `ghost-medium` | 12     | 768  | ~160M    | Planned             |

## Training Details

- **Framework:** PyTorch (built from scratch, no pretrained weights)
- **Optimizer:** AdamW with weight decay separation (no decay on biases, LayerNorm, embeddings)
- **LR Schedule:** Cosine decay with linear warmup (2,000 warmup steps)
- **Gradient Clipping:** 1.0
- **Hardware:** ThinkPad Yoga 11e (Celeron N4100, 4GB RAM) for ghost-tiny
- **Status:** Pre-training in progress

## Evaluation

- **Metric:** Perplexity on cybersecurity holdout set
- **Benchmark:** Generation quality on 8 cybersecurity prompts (SQL injection, XSS, buffer overflow, etc.)
- **Results:** To be updated after training completes

## Quick Start

```python
import torch
from ghostlm import GhostLM, GhostLMConfig, GhostTokenizer

config = GhostLMConfig.from_preset("ghost-small")
model = GhostLM(config)
tokenizer = GhostTokenizer()

prompt = "A SQL injection attack works by"
ids = tokenizer.encode(prompt)
input_tensor = torch.tensor(ids).unsqueeze(0)
output = model.generate(input_tensor, max_new_tokens=100, temperature=0.8, top_k=50)
print(tokenizer.decode(output[0].tolist()))
```

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

- **GitHub:** https://github.com/joemunene-by/GhostLM
- **Issues:** https://github.com/joemunene-by/GhostLM/issues
- **License:** Apache 2.0
