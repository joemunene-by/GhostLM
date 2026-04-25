---
language:
  - en
license: mit
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
| **Model Names** | `ghostlm/ghost-tiny` (14.7M params, current). Future: `ghost-small`, `ghost-base`, `ghost-1B` |
| **Architecture** | Decoder-only transformer |
| **Author** | [Joe Munene](https://github.com/joemunene-by) |
| **License** | MIT |
| **Language** | English |
| **Framework** | PyTorch (built from scratch, no pretrained weights) |
| **Version** | 0.3.0 (Phase 2 complete — 10K steps on rebalanced, leakage-free corpus) |

## Model Description

GhostLM is a cybersecurity-focused decoder-only transformer language model built entirely from scratch in PyTorch. No pretrained weights, no wrappers — every component (attention, feed-forward, embeddings, training loop) is hand-implemented.

The model is trained on CVE vulnerability descriptions from the National Vulnerability Database, CTF writeups, and security research papers. It is designed for cybersecurity reasoning tasks: CVE analysis, exploit explanation, penetration testing assistance, and security concept generation.

## Model Variants

| Variant | Layers | d_model | Heads | d_ff | Context | Params | Status |
|---|---|---|---|---|---|---|---|
| `ghostlm/ghost-tiny` | 2 | 256 | 4 | 1024 | 1024 | 14.7M | Phase 2 complete (10K steps, rebalanced corpus, val_loss 3.78) |
| `ghostlm/ghost-small` | 6 | 512 | 8 | 2048 | 1024 | ~55M | Planned (next scale rung) |
| `ghostlm/ghost-base` | 12 | 768 | 12 | 3072 | 1024 | ~350M | Planned (rented GPU) |
| `ghostlm/ghost-1B` | 24 | 1024 | 16 | 4096 | 1024 | ~1B | Long-term goal |

ghost-tiny is the iteration vehicle. The scale ladder above is the path to a genuinely useful from-scratch cyber LM. See [ROADMAP.md](ROADMAP.md) for phased milestones, compute requirements, and corpus targets.

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
| NVD CVE Database | 19,925 | Real | Vulnerability descriptions from NVD REST API v2.0, balanced per-year cap, 1999–2025 (27 years) |
| arXiv cs.CR Abstracts | 1,000 | Real | Recent-first via arXiv Atom API |
| Synthetic CTF Writeups | 3,000 | Synthetic | Generated via local LLM (Ollama-based pipeline), varied topic + template mix |
| **Total (raw)** | **23,925** | | |
| **Total (post-dedup)** | **23,049** | | **~2.66M tokens** (train: 2,525,245 / val: 136,869) |

**Data splits:** 21,872 train / 1,177 validation. Split is **deterministic by content hash** — identical or near-duplicate texts always land in the same split, eliminating the train/val leakage that affected the v0.2.0 corpus.

**Topics covered:** vulnerability detection, adversarial ML, network intrusion, cryptographic protocols, fuzzing, side-channel attacks, ransomware detection, supply chain security, memory safety, WAF evasion, SQL injection, XSS, buffer overflow, privilege escalation, reverse engineering, binary exploitation, steganography, network forensics.

For corpus expansion plans (CTFtime, security blogs, MITRE ATT&CK, tool docs) and licensing notes, see [CORPUS.md](CORPUS.md).

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

**Hardware (Phase 2):** ghost-tiny trained on Mac Mini M4 (CPU; MPS/GPU acceleration not used for this run). Cross-machine workflow: Linux box for data prep and corpus curation; Mac Mini M4 for the training loop. Phase 1 was run on a ThinkPad Yoga 11e (Celeron N4100) and is preserved for archaeological reference.

**Training time (Phase 2):** ~70 minutes total wall-clock for 10K steps on M4 CPU.

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

- **Small model size:** At 14.7M parameters, ghost-tiny is two-to-three orders of magnitude below production LLMs. Output quality reflects this.
- **Limited training data:** ~23K records / ~2.7M tokens is small for language-model pre-training (Chinchilla-optimal for 14.7M params would be ~300M tokens; for ghost-1B, ~20B tokens). The corpus needs to grow ~100× for the upper rungs of the scale ladder.
- **No semantic grounding at this scale:** the model has learned the *surface vocabulary* of cyber writing (CTF terminology, exploit techniques, CVE-string format, vuln types) but cannot bind topic, hold grammatical structure across clauses, or generate factually correct content. See [Sample Generations](#sample-generations) below.
- **Hallucinated CVE-shaped output:** prompting with `CVE-YYYY-NNNNN is a vulnerability in...` will produce plausible-looking but entirely fabricated CVE descriptions, including invented version chains. Do not use these as factual.
- **No instruction tuning:** ghost-tiny is a base language model. It generates text completions, not structured answers.
- **Synthetic CTF share:** ~13% of training records are synthetic CTF writeups generated by a local LLM. This is a known risk for distribution drift relative to real CTF writing and will be replaced with real CTFtime / GitHub writeup corpus in upcoming corpus expansion.
- **English only.**

## Sample Generations

Real outputs from `checkpoints/best_model.pt` (Phase 2, val_loss 3.78), at `temperature=0.7-0.8, top_k=50`:

> **Prompt:** *A SQL injection attack works by*
> ...the login page. The login page is used to the login page's name of the login page does not properly sanitization of the password, which allows attackers to cause a denial of service via a long GET request. In this CTF challenge, participants are tasked with a classic vulnerability in exploiting a vulnerable to exploit a vulnerable to SQL injection vulnerability...

> **Prompt:** *CVE-2021-44228 is a vulnerability in*
> ...the C privileged user account, which allows local users to gain root privileges. Buffer overflow in x86 program in Oracle MySQL 5.1.2.5.0.0.2.1.0.1 and 7.1.2.1.0.x before 7.2.2.2.x before 7.1.x before 7.1.1.2.

> **Prompt:** *To enumerate Active Directory users from a Linux box, an attacker can*
> ...obtain the binary using the binary. The binary. The binary using UPX, the binary exploitation, we were a stack with a ret2libc base address of the stack with Ghidra's memory with pwndbg and function...

> **Prompt:** *Cross-site scripting (XSS) allows attackers to*
> ...inject arbitrary web script that HTML via the (1) search parameter in the search parameter. The web server in OpenView Network Manager (aka TCCQ: How can an attacker exploit a web application that leverages a web applications using a reverse proxy service. The attacker can craft a malicious server.

These samples are honest evidence of model capability at this scale — useful for understanding what 14.7M params trained on ~2.7M tokens looks like. Genuine task usefulness is not expected until the upper rungs of the scale ladder.

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

## Evaluation (Phase 2 — 10K Steps, rebalanced corpus)

### Validation Loss

- **Final training loss (step 10000):** 4.59 (single-batch noisy estimate, see training_log.json)
- **Final validation loss (step 10000):** **3.7813** (perplexity ≈ 44)
- **Validation loss at step 9000:** 3.7972 — flat plateau over the last 1000 steps suggests the model has extracted what it can from this corpus at this scale.

### Note on Phase 1 vs Phase 2 numbers

The previous version of this card reported Phase 1 numbers (val_loss 2.74, perplexity 2,183.94 vs GPT-2). Those are **not directly comparable** to Phase 2:

- Phase 1 was trained on a corpus with ~98% duplication in papers/CTF and ~9% train/val leakage. Its val_loss 2.74 reflected memorization of held-in samples, not held-out generalization.
- Phase 2 was trained on a deterministic-hash split with leakage eliminated. val_loss 3.78 is the first trustworthy measurement.
- Treat Phase 1 numbers as superseded. Phase 2's higher val_loss is not a regression — it is the first honest read.

### Perplexity vs GPT-2 (cyber-text benchmark)

Re-run on the Phase 2 checkpoint against the same hardcoded `BENCHMARK_TEXTS` set used for Phase 1 (10 cyber-text samples, fair comparison):

| Model | Perplexity (lower is better) |
|---|---|
| GPT-2 (124M baseline) | **26.76** |
| ghost-tiny — Phase 2 | **152.71** |
| ghost-tiny — Phase 1 | 2,183.94 |

Phase 2 is **14.3× better** than Phase 1 on this benchmark while still 5.7× worse than GPT-2 — expected for a 14.7M model trained on ~2.7M tokens against a 124M model trained on ~40B tokens of WebText. The Phase 1→Phase 2 gain is the corpus-quality dividend (clean split + deduplication + balanced CTF), not extra training. Raw output: `logs/benchmark_phase2.json`.

### Security-domain task evaluation

Re-run on the Phase 2 checkpoint via `scripts/eval_security.py` (3 tasks, 30 questions total: CVE Severity Classification, Vulnerability Type Detection, Attack Technique Identification):

| Phase | Score | Failure mode |
|---|---|---|
| Phase 1 | 4/30 (13.3%) | Mode-collapsed predictions |
| Phase 2 | 4/30 (13.3%) | Mode-collapsed predictions: predicts "High" for every CVE Severity, "Cross-Site Scripting" for every Vuln Type, "Supply Chain Compromise" for every Attack Technique |

Same numerical score as Phase 1, but with a different mode-collapse pattern — the model has learned the *most frequent label per task* rather than the discriminative structure. This is exactly what a 14.7M-param classifier trained on a corpus where these labels appear unevenly would predict. **Random-guess baseline is ~33%** (each task is 4-way multiple choice), so 13.3% is below random, confirming the model is not yet doing real classification. Raw output: `logs/eval_security_phase2.json`.

**What this means:** the perplexity dividend is real (corpus quality matters) but the model is still well below the threshold where structured-task evaluation is meaningful. Both numbers are baselines for the next scale rung.

### Phase comparison plot

`logs/phase_comparison.png` shows final val_loss, perplexity (vs GPT-2 baseline), and security-task accuracy for both phases side by side. Generated by `scripts/plot_phase_comparison.py`.

### Generation Quality

See [Sample Generations](#sample-generations) above. In summary: vocabulary acquisition is real and on-domain; semantic grounding is absent; topic-binding fails; CVE strings are fluent-sounding hallucinations. Expected behavior at this scale.

### Training Curves

- ghost-tiny shows healthy loss decrease through Phase 1 (10K steps on pre-audit corpus) and stable training through Phase 2 (10K steps resumed on rebalanced corpus).
- No catastrophic forgetting or divergence observed in Phase 2.
- Per-step training loss has high single-batch variance (batch_size=2, grad_accum=4, effective batch=8) — see `logs/training_log.json` for periodic eval snapshots.

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
- **License:** [MIT](LICENSE)
