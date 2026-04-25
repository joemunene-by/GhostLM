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
| **Version** | 0.3.3 (Phase 3 ghost-tiny refresh — 30K steps on the post-NVD-pull ~30M-token corpus) |

## Model Description

GhostLM is a cybersecurity-focused decoder-only transformer language model built entirely from scratch in PyTorch. No pretrained weights, no wrappers — every component (attention, feed-forward, embeddings, training loop) is hand-implemented.

The model is trained on CVE vulnerability descriptions from the National Vulnerability Database, CTF writeups, and security research papers. It is designed for cybersecurity reasoning tasks: CVE analysis, exploit explanation, penetration testing assistance, and security concept generation.

## Model Variants

| Variant | Layers | d_model | Heads | d_ff | Context | Params | Status |
|---|---|---|---|---|---|---|---|
| `ghostlm/ghost-tiny` | 2 | 256 | 4 | 1024 | 1024 | 14.7M | **Phase 3 complete (30K steps, post-NVD-pull corpus, val_loss 3.45)** |
| `ghostlm/ghost-small` | 6 | 512 | 8 | 2048 | 1024 | ~55M | Planned (next scale rung; corpus diversity track gates the run) |
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

The released v0.3.3 checkpoint was trained on the post-NVD-pull corpus:

| Source | Records | Type | Description |
|---|---|---|---|
| NVD CVE Database | 333,540 | Real | Full pull, 1999–2026 (28 years), via `scripts/collect_nvd_full.py` with proper `startIndex` pagination |
| arXiv cs.CR Abstracts | 2,000 | Real | Recent-first via arXiv Atom API |
| Synthetic CTF Writeups | 3,000 | Synthetic | Generated via local LLM (Ollama-based pipeline), varied topic + template mix |
| **Total (post-dedup)** | **~309K** | | **~30M tokens** (train: ~293K / val: ~15K) |

**Data splits:** deterministic by content hash — identical or near-duplicate texts always land in the same split. Train/val leakage check returns 0.

**Token share (what the model sees):** NVD ~87%, CTF ~5%, papers ~2%. The lopsidedness motivates the next *corpus* track being diversity (CTFtime, MITRE ATT&CK), not deeper NVD.

**Topics covered:** vulnerability detection, adversarial ML, network intrusion, cryptographic protocols, fuzzing, side-channel attacks, ransomware detection, supply chain security, memory safety, WAF evasion, SQL injection, XSS, buffer overflow, privilege escalation, reverse engineering, binary exploitation, steganography, network forensics.

For corpus expansion plans (CTFtime, security blogs, MITRE ATT&CK, tool docs) and licensing notes, see [CORPUS.md](CORPUS.md).

## Training Details

| Parameter | Value |
|---|---|
| Optimizer | AdamW (beta1=0.9, beta2=0.95, weight_decay=0.1) |
| Learning rate | 3e-4 (with cosine decay to 1e-5) |
| Warmup steps | 2,000 |
| Gradient clipping | 1.0 |
| Gradient accumulation | 4 steps |
| Batch size (Phase 3) | 2 (effective batch = 8 with grad_accum) |
| Max steps (Phase 3) | 30,000 |
| Dropout | 0.1 |
| Mixed precision | AMP on CUDA, fp32 on CPU |

**Weight decay separation:** No weight decay applied to biases, LayerNorm parameters, or embedding weights. Only linear layer weights receive weight decay.

**Hardware (Phase 3):** Mac Mini M4 (CPU). ~3h48m wall-clock for 30K steps at ~2.4 it/s. Cross-machine workflow: Linux box for data prep and corpus curation; Mac Mini M4 for the training loop.

**Phase 1** was run on a ThinkPad Yoga 11e (Celeron N4100) and is preserved as `checkpoints/best_model_phase1.pt` for archaeological reference. **Phase 2** is preserved as `checkpoints/best_model_phase2.pt` (val_loss 3.78 on the 2.66M-token corpus).

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
- **Limited training data:** ~30M tokens is still small for language-model pre-training (Chinchilla-optimal for 14.7M params would be ~300M tokens; for ghost-1B, ~20B tokens). The corpus needs to grow another ~30× for the upper rungs of the scale ladder.
- **Surface-level fluency, weak grounding:** the model has learned the CVE-database register and surface vocabulary of cyber writing — it produces structurally correct CVE descriptions and security-prose grammar — but will hallucinate version chains, mix product names, and bind topic only loosely. See [Sample Generations](#sample-generations) below.
- **Hallucinated CVE-shaped output:** prompting with `CVE-YYYY-NNNNN is a vulnerability in...` will produce plausible-looking but entirely fabricated CVE descriptions, including invented version chains. **Do not use these as factual.**
- **No instruction tuning:** ghost-tiny is a base language model. It generates text completions, not structured answers.
- **Below random-guess on structured-task eval:** on 4-way multiple choice security tasks the model mode-collapses to the most-frequent label per task (4/30 = 13.3% vs ~33% random baseline). Confirms the model is not yet doing real classification at this scale.
- **Synthetic CTF share:** ~5% of training tokens are synthetic CTF writeups. Replacement with real CTFtime / GitHub writeup corpus is the priority next-track work.
- **NVD corpus skew:** 87% of training tokens are CVE descriptions. The model will tend toward CVE-style prose even when the prompt isn't CVE-shaped.
- **English only.**

## Sample Generations

Real outputs from `checkpoints/best_model.pt` (Phase 3, val_loss 3.45), at `temperature=0.8, top_k=50`:

> **Prompt:** *A SQL injection attack works by*
> ...injecting a crafted payload in the "password" parameter. NOTE: some of these details are obtained from third party information. EINXCMS 3.4.7.2.0 contains a cross-site scripting vulnerability that allows users to embed arbitrary JavaScript code in the Web UI thus altering the intended functionality potentially leading to credentials disclosure within a trusted session. This issue is fixed in 1.0.10. Use after free in Google Chrome before 17.0.66.75 allowed a remote...

> **Prompt:** *CVE-2021-44228 is a vulnerability in*
> ...the same product which allows local users to overwrite the filesystem via a large value (1) id parameter to the filename parameter to wp-admin/admin/mod_api/admin/admin/admin/login.php in QNAP Business Application Platform, iMC Net-based, iQAP_NAP-V, and MQ-IP-P, MQ-V-M DB2G, MQ-AN-W, MQ-2 A300-

> **Prompt:** *To enumerate Active Directory users from a Linux box, an attacker can*
> ...exploit an attacker to inject malicious scripts as the target system and execute arbitrary code in the context of the current process. The vulnerability is due to the application. An attacker could exploit this vulnerability by submitting a crafted link that executes in the application. A successful exploit could allow the attacker to obtain sensitive information on the user. The BIO plugin for WordPress is vulnerable to Cross-Site Request Forgery in all versions up to, and including, 2.2. This is due to missing nonce validation on...

> **Prompt:** *Cross-site scripting (XSS) allows attackers to*
> ...inject arbitrary web script or HTML via unspecified vectors. The davcodec_server function in libav.c in the Linux kernel before 4.4.0.5, 4.x before 4.6.1, and 4.x before 4.3.0.x before 4.3.3, does not properly restrict access to (1) Ntcpd) PTP-based LDAP (2) X-3-NAP00-00N-N

**What this shows compared to Phase 2:** the *register* is correct now — phrases like "Cross-Site Request Forgery in all versions up to, and including, 2.2 — this is due to missing nonce validation," "use after free," "remote attacker," "submitting a crafted link," "in the context of the current process" are all real CVE-database language used in roughly the right context. Phase 2 produced fragments like "the login page is used to the login page's name of the login page does not properly sanitization" — the same model class can't produce that anymore. **The hallucinations are still rampant** (made-up products, scrambled version strings, mixed-up vendor names) — the model has the *form* of CVE descriptions but not the *facts*. This is the expected outcome of corpus expansion at fixed model size: better surface fluency, no new factual capability.

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

# Load trained weights (v0.3.3 — Phase 3 ghost-tiny refresh)
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

## Evaluation (Phase 3 — 30K Steps, post-NVD-pull corpus)

### Validation loss

- **Final validation loss (step 30000):** **3.4458** (perplexity ≈ 31)
- **Curve shape:** monotonic decrease over 60 eval points; no instability, still slightly descending at step 30K (diminishing returns rather than plateau).
- Comparison: Phase 2 val_loss 3.7813 on the 2.66M-token corpus. Both runs use the deterministic-hash split, so the **0.34 nat drop is a real corpus-quality dividend at fixed model size**.

### Perplexity vs GPT-2 (cyber-text benchmark)

Same hardcoded `BENCHMARK_TEXTS` set used for every prior phase (10 cyber-text samples, fair comparison):

| Model | Perplexity (lower is better) |
|---|---|
| GPT-2 (124M baseline) | **26.76** |
| **ghost-tiny — Phase 3 (released)** | **142.09** |
| ghost-tiny — Phase 2 | 152.71 |
| ghost-tiny — Phase 1 | 2,183.94 |

Phase 3 is **7% better** than Phase 2 on this benchmark and **15.4× better** than Phase 1. Still 5.3× behind GPT-2, expected for a 14.7M-param model on ~30M tokens vs. a 124M-param model on ~40B tokens of WebText. The Phase 2→3 gain is modest because the 10-text benchmark contains generic security prose that already overlapped both corpora — most of the perplexity dividend was earned at Phase 2 (corpus quality + clean split), and the residual gain at Phase 3 is from the larger volume. Raw output: `logs/benchmark_phase3.json`.

### Security-domain task evaluation

Re-run on the Phase 3 checkpoint via `scripts/eval_security.py` (3 tasks, 30 questions: CVE Severity Classification, Vulnerability Type Detection, Attack Technique Identification):

| Phase | Score | Failure mode |
|---|---|---|
| Phase 1 | 4/30 (13.3%) | Mode-collapsed |
| Phase 2 | 4/30 (13.3%) | Mode-collapsed: predicts "High" / "Cross-Site Scripting" / "Supply Chain Compromise" |
| **Phase 3** | **4/30 (13.3%)** | Mode-collapsed: predicts "Medium-or-High" / "Cross-Site Scripting" / "DLL Search Order Hijacking" |

Same numerical score as prior phases, **but with a different mode-collapse pattern** — the model has learned the *most frequent label per task* rather than the discriminative structure, and at Phase 3 the most-frequent attack technique label has shifted (from Supply Chain Compromise to DLL Search Order Hijacking) reflecting the corpus shift. CVE-severity picks up some genuine discrimination (gets 2 right by mixing in Mediums). **Random-guess baseline is ~33%** (4-way multiple choice), so 13.3% is below random — confirming the model is not yet doing real classification at this scale. Raw output: `logs/eval_security_phase3.json`.

**What this means:** the corpus-expansion dividend is real on language modeling (val_loss + perplexity) but invisible on structured-task eval. Both numbers are baselines for the next scale rung — ghost-small at ~55M params is where structured-task eval should start to reward better corpus.

### Phase comparison plot

`logs/phase_comparison.png` shows final val_loss, perplexity (vs GPT-2 baseline), and security-task accuracy across all three phases side by side. Generated by `scripts/plot_phase_comparison.py`.

### Training curve

`logs/phase3_refresh/training_curve.png` shows the 30K-step Phase 3 curve. Phase 1 and Phase 2 logs were too sparse for real curves (3–5 endpoint datapoints); Phase 3 has 60 eval points, the first dense ghost-tiny training curve we've ever produced.

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
