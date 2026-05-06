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
| **Model Names** | `ghostlm/ghost-small-v0.7` (81M wide, top of the debiased CTIBench ceiling at 32.2% per-perm avg). `ghostlm/ghost-small` (~45M, v0.4.0 canonical for density / generation, val PPL 11.12). `ghostlm/ghost-tiny` (14.7M, historical PMI-suite scorer). In progress: `ghost-small-v0.9` on the 273M-token expansion corpus. Planned: `ghost-base`, `ghost-1B`. |
| **Architecture** | Decoder-only transformer (RoPE + SwiGLU + RMSNorm from v0.5 onward; learned PE + GELU + LayerNorm for v0.4 and earlier) |
| **Author** | [Joe Munene](https://github.com/joemunene-by) |
| **License** | MIT |
| **Language** | English |
| **Framework** | PyTorch (built from scratch, no pretrained weights) |
| **Version** | v0.9.2 + v1.0 corpus ready. v0.9 chat is the ghost-small bench winner (CTIBench full 28.9%, CTF eval 59.2%, SecQA 39.3%, fact recall 1/50). The 81M ghost-small line saturates as a register-matching parrot; free-form fact recall is at floor. The v1.0 corpus has been built (516,736 train / 27,049 val records, ~363M tokens, six domains: cybersec writeup / code / general language / math / authoritative reference / research-blog register) and the ghost-base launcher (`scripts/train_ghost_base.py`, 30L × 960d × 15h × 2560 d_ff, SmolLM2-360M shape, ~360M params) is shipped. v1.0 training is gated on rented-GPU access. |

## Model Description

GhostLM is a cybersecurity-focused decoder-only transformer language model built entirely from scratch in PyTorch. No pretrained weights, no wrappers — every component (attention, feed-forward, embeddings, training loop) is hand-implemented.

The model is trained on CVE vulnerability descriptions from the National Vulnerability Database, CTF writeups, and security research papers. It is designed for cybersecurity reasoning tasks: CVE analysis, exploit explanation, penetration testing assistance, and security concept generation.

## Model Variants

| Variant | Layers | d_model | Heads | d_ff | Ctx | Params | Status |
|---|---|---|---|---|---|---|---|
| `ghostlm/ghost-tiny` | 2 | 256 | 4 | 1024 | 1024 | 14.7M | Phase 3.5 (historical PMI canonical). 30K steps on ~8.8M tokens, overall PPL 66 |
| `ghostlm/ghost-small` (v0.4) | 6 | 512 | 8 | 2048 | 1024 | ~45M | Phase 4 base. 30K steps on ~12.56M tokens, val_loss 2.3535, overall val PPL 11.12 (−83%). Chat-v3 at **30.5%** debiased CTIBench |
| `ghost-small-v0.5` | 6 | 512 | 8 | 2048 | 512 | ~36M | RoPE + SwiGLU + RMSNorm + custom 32K BPE retrain. Chat at 29.7% debiased CTIBench |
| `ghost-small-v0.6` | 6 | 512 | 8 | 2048 | 512 | ~45M | v0.5 architecture + GPT-2 50K BPE on the v0.4.2 expanded corpus. Chat at 31.2% debiased (BPE-swap ablation) |
| `ghost-small-v0.7` | 6 | 768 | 12 | 3072 | 512 | ~81M | Wider variant of v0.6. Chat at **32.2%** debiased: single best in repo, but inside the noise band |
| `ghost-small-v0.8` | 6 | 768 | 12 | 3072 | 512 | ~81M | v0.7 arch + Qwen-14B-distilled fact-QA in pretrain. Chat at 31.2% (fact-density alone doesn't lift) |
| `ghost-small-v0.9` | 6 | 768 | 12 | 3072 | 512 | ~81M | v0.7 arch on 273M-token corpus (PRIMUS + CWE + OWASP + RFCs + fact-QA). Pretrain val 3.638. Chat is the canonical ghost-small bench winner: **28.9% on debiased CTIBench full (n=2500), 59.2% on in-repo CTF MCQ (n=30), 39.3% on SecQA (n=210, external)**. Free-form fact recall still at floor (1/50 on 50 hand-written prompts); MCQ wins measure register, not facts. |
| `ghostlm/ghost-base` | 30 | 960 | 15 | 2560 | 1024 | ~360M | Planned (rented GPU). SmolLM2-360M shape; the parameter rung where literature reports factual recall on cybersec MCQ starting to emerge |
| `ghostlm/ghost-1B` | 24 | 1024 | 16 | 4096 | 1024 | ~1B | Long-term goal |

ghost-tiny is the iteration vehicle. The v0.4 / v0.5 / v0.6 / v0.7 / v0.8 sequence is the same parameter rung (45-81M) under different recipes; the bench is consistent across all of them at 29-32% per-perm avg on debiased CTIBench, which is the diagnosis (param-count is the bottleneck at this rung). See [ROADMAP.md](ROADMAP.md) for phased milestones and [`docs/ctibench_bias_finding.md`](docs/ctibench_bias_finding.md) for the eval-methodology investigation.

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

The released v0.3.5 checkpoint was trained on the rebalanced Phase 3.5 corpus. NVD's full 333,540-record pull is on disk, but its training contribution is capped at 6M tokens by content-hash subsample so the corpus isn't 90% CVE descriptions:

| Source | Records (raw → trained) | Trained tokens | Share | Type |
|---|---|---|---|---|
| NVD CVE Database | 333,540 → 71,828 | ~5.74M | **65.3%** | Real, capped via `--max-cve-tokens 6000000` |
| Synthetic CTF Writeups | 3,000 | ~1.51M | 17.2% | Synthetic, placeholder until real CTFtime grows |
| arXiv cs.CR Abstracts | 2,000 | ~0.74M | 8.4% | Real |
| CTFtime real writeups | 473 → 467 | ~0.47M | 5.3% | Real, inline-only, per-record attribution |
| MITRE ATT&CK | 691 | ~0.26M | 2.9% | Real (Apache 2.0) |
| CAPEC | 609 | ~0.07M | 0.9% | Real (Apache 2.0) |
| **Total (post-dedup)** | **74,635** | **~8.79M** | | train: 70,965 / val: 3,670 |

**Data splits:** deterministic by content hash — identical or near-duplicate texts always land in the same split. Train/val leakage check returns 0.

**Token share comparison (what the model sees):**

| Phase | NVD share | Top non-NVD source | Overall |
|---|---|---|---|
| v0.3.3 (Phase 3) | 87% | CTF synthetic 5% | NVD-dominated |
| **v0.3.5 (Phase 3.5)** | **65.3%** | **synthetic 17.2%** | **balanced across 6 sources** |

The rebalance is reproducible: `python3 scripts/rebuild_corpus.py --max-cve-tokens 6000000` always produces the same 71,828-record CVE prefix.

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
| Batch size (Phase 3.5) | 2 (effective batch = 8 with grad_accum) |
| Max steps (Phase 3.5) | 30,000 |
| Dropout | 0.1 |
| Mixed precision | AMP on CUDA, fp32 on CPU |

**Weight decay separation:** No weight decay applied to biases, LayerNorm parameters, or embedding weights. Only linear layer weights receive weight decay.

**Hardware (Phase 3.5):** Mac Mini M4 (CPU). ~3h13m wall-clock for 30K steps at ~2.4 it/s. Cross-machine workflow: Linux box for data prep, corpus curation, and SSH-driven Mac orchestration; Mac Mini M4 for the training loop. The previous Nemotron-on-Mac harness was replaced this phase by direct `ssh ghostlm-mac` from Linux — drops the email-relay friction and lets the dev box drive the workhorse cleanly.

**Phase 1** was run on a ThinkPad Yoga 11e (Celeron N4100) and is preserved as `checkpoints/best_model_phase1.pt`. **Phase 2** is preserved as `checkpoints/best_model_phase2.pt` (val_loss 3.78 on the 2.66M-token corpus). **Phase 3 (v0.3.3)** is preserved as `checkpoints/phase3_refresh/best_model.pt` (val_loss 3.45 on the post-NVD-pull corpus, overall PPL 172).

## Evaluation

The v0.3.5 model is evaluated on two complementary axes — domain-modeling quality (per-source perplexity) and downstream reasoning (PMI-corrected security task accuracy).

### Per-source perplexity on the validation split

100 records sampled per source (deterministic seed). Lower is better.

| Source | v0.3.3 PPL | v0.3.5 PPL | Δ% | Reading |
|---|---|---|---|---|
| MITRE ATT&CK | 615.43 | 55.14 | **−91%** | Was OOD for v0.3.3; now in training |
| CTFtime real writeups | 184.24 | 60.71 | **−67%** | Was OOD for v0.3.3; now in training |
| CAPEC | 326.11 | 133.81 | **−59%** | Was OOD for v0.3.3; now in training |
| Synthetic CTF | 67.57 | 28.48 | **−58%** | Same data both phases — capacity reallocation |
| arXiv cs.CR | 671.09 | 354.95 | **−47%** | Same data both phases — capacity reallocation |
| NVD CVE | 24.19 | 27.55 | +14% | The expected, modest cost |
| **Overall** | **171.84** | **66.05** | **−62%** | |

The rebalance shifted the model from "knows NVD register, treats everything else as generic English" to "models each domain in proportion to its training share." The 47–58% improvements on synthetic CTF and arXiv are particularly notable because **the training data for those sources didn't change** — the gain comes from parameter capacity that v0.3.3 was burning on memorizing duplicate CVE descriptions being redirected onto already-present sources.

### PMI-corrected security task accuracy

Five classification tasks × 25 hand-crafted samples each (125 total). PMI scoring (commit `aee8008`) replaces the previous mode-collapsed length-normalized scoring that reported 4/30 = 13.3% on every phase under logp scoring. Per-task random baseline depends on the number of candidate labels.

The eval was expanded from 30 → 125 samples in v0.3.6 — the v0.3.5 model below was re-scored on the larger suite, so the numbers in this table are not directly comparable to the 30-sample numbers in older releases. The expanded suite is the new canonical measurement; future phases will be reported on it. The smaller suite is preserved at `logs/eval_security_phase3.5_pmi.json` for archaeology.

| Task | Labels | Random | v0.3.5 (125-sample) | Most-common share |
|---|---|---|---|---|
| CVE Severity Classification | 4 | 25.0% | 8/25 (32.0%) | Critical 72% |
| Vulnerability Type Detection | 10 | 10.0% | 8/25 (32.0%) | IDOR 44% |
| Attack Technique Identification | 10 | 10.0% | 10/25 (40.0%) | LatMov 36% |
| CTF Challenge Categorization | 5 | 20.0% | 10/25 (40.0%) | Forensics 64% |
| MITRE ATT&CK Tactic Classification | 12 | 8.3% | 3/25 (12.0%) | LatMov 40% |
| **Overall** | — | ~14.5% (avg) | **39/125 (31.2%)** | — |

Reading the table:

- **Vulnerability Type Detection (+22 pp), Attack Technique Identification (+30 pp), CTF Challenge Categorization (+20 pp)** are the three tasks where v0.3.5 is meaningfully above random. These map onto the corpora that grew during the Phase 3.5 rebalance (CWE-tagged CVE bodies, MITRE technique pages, CTFtime real writeups) and the eval picks up that the model has internalized those domains.
- **CVE Severity Classification (+7 pp above random with 72% prediction collapse onto Critical).** The model has learned that NVD descriptions usually accompany severe CVEs and bets that way regardless of input. The previous 10-sample suite happened to over-weight Critical/High labels in a way that masked this; the 25-sample suite with balanced severity distribution exposes it. This is the canary metric for whether subsequent training rungs learn calibrated severity reasoning.
- **MITRE ATT&CK Tactic Classification (+3.7 pp above random).** Tactic-level classification is the model's weakest task — distinguishing Persistence from Privilege Escalation from Defense Evasion is hard from a single description even for humans, and ghost-tiny at 14.7M params on 8.8M tokens has not built that abstraction. This is the metric to watch when ghost-small is trained: if scaling the model doesn't move tactic accuracy above ~25%, the architectural jump didn't produce reasoning gains.

#### Cross-phase trajectory on the expanded suite

Every preserved ghost-tiny checkpoint was re-scored on the new 125-sample suite so the trajectory is end-to-end comparable. Cells are `correct/total (accuracy) [most-common-share]`:

| Task | Phase 1 (2K) | Phase 2 (v0.3.0) | Phase 3 (v0.3.3) | Phase 3.5 (v0.3.5) | Phase 3.6 (v0.3.7) | **Phase 4 (v0.4.0)** |
|---|---|---|---|---|---|---|
| CVE Severity Classification | 7/25 (28.0%) [100%] | 5/25 (20.0%) [96%] | 4/25 (16.0%) [48%] | **8/25 (32.0%) [72%]** | 4/25 (16.0%) [60%] | 6/25 (24.0%) [72%] |
| Vulnerability Type Detection | 3/25 (12.0%) [48%] | 6/25 (24.0%) [76%] | 7/25 (28.0%) [48%] | 8/25 (32.0%) [44%] | 3/25 (12.0%) [96%] | **10/25 (40.0%) [44%]** |
| Attack Technique Identification | 2/25 (8.0%) [24%] | 3/25 (12.0%) [88%] | 5/25 (20.0%) [72%] | **10/25 (40.0%) [36%]** | 4/25 (16.0%) [60%] | 4/25 (16.0%) [52%] |
| CTF Challenge Categorization | 2/25 (8.0%) [84%] | 7/25 (28.0%) [76%] | 6/25 (24.0%) [88%] | **10/25 (40.0%) [64%]** | 5/25 (20.0%) [48%] | 7/25 (28.0%) [72%] |
| MITRE ATT&CK Tactic Classification | 1/25 (4.0%) [72%] | 2/25 (8.0%) [76%] | 3/25 (12.0%) [64%] | 3/25 (12.0%) [40%] | 5/25 (20.0%) [76%] | 2/25 (8.0%) [44%] |
| **Overall (PMI)** | **15/125 (12.0%)** | **23/125 (18.4%)** | **25/125 (20.0%)** | **39/125 (31.2%)** | **21/125 (16.8%)** | **29/125 (23.2%)** |

Phase 4 ghost-small (v0.4.0) is the new canonical model for density / generation work but lands lower than Phase 3.5 on the PMI scoring above. The honest read requires the second column type — **logp scoring** — which the suite also supports via `--scoring logp`:

| Phase | PMI | logp | Δ (PMI − logp) |
|---|---:|---:|---:|
| Phase 3.5 (ghost-tiny) | **31.2%** | 17.6% | +13.6 pp |
| Phase 4 (ghost-small) | 23.2% | **19.2%** | +4.0 pp |

Two things to note:

1. **PMI flatters Phase 3.5 by 13.6 pp.** PMI subtracts unconditional candidate log-prob to break ties — useful when the model is mode-collapsing because it normalizes for "this candidate is just inherently high-probability". A loose-distribution model with weakly differentiated logits gives PMI more separation to extract; a tight-distribution model gives less. Phase 3.5 (low capacity) gets the bigger PMI uplift; Phase 4 (higher capacity, sharper distribution) gets a smaller one.
2. **Logp — the more conservative scorer — picks Phase 4.** With logp scoring, Phase 4 narrowly beats Phase 3.5 (19.2% vs 17.6%) on this same 125-sample suite. The PMI vs logp gap diagnoses an eval-methodology limitation rather than a model regression.

The cleanest model metric remains per-source val PPL (no scoring rule, just density), where Phase 4 dominates Phase 3.5 by 59–78% across every source. See README's "Per-source perplexity" section for the full table.

The clean head-to-head between deliberate moves (PMI suite):
- **Phase 2→3 (3× training volume, fixed corpus): +1.6 pp**
- **Phase 3→3.5 (corpus rebalance, fixed model+steps): +11.2 pp**
- **Phase 3.5→3.6 (corpus volume, fixed model+steps): −14.4 pp** (ghost-tiny capacity ceiling)
- **Phase 3.6→4 (model capacity, fixed corpus+steps): +6.4 pp PMI / +1.6 pp logp / −75% per-source PPL** (capacity-reallocation hypothesis confirmed)

Use `make eval-security-all-phases` to re-run end-to-end, or `make eval-compare-phases` to regenerate the PMI table from saved JSONs. Run with `--scoring logp` to reproduce the logp column.

### Cyber-text perplexity vs GPT-2 (fixed external test set, ten samples)

The benchmark sample is held out from training and unchanged across phases — it's directly comparable.

| Phase | Perplexity | vs prior |
|---|---|---|
| Phase 1 | 2,183.94 | — |
| Phase 2 | 152.71 | −93% |
| Phase 3 (v0.3.3) | 142.09 | −7% |
| **Phase 3.5 (v0.3.5)** | **96.24** | **−32%** |
| GPT-2 small (117M) | 26.76 | (frozen baseline) |

ghost-tiny is 14.7M params vs GPT-2 small's 117M — so we're closing the cyber-text gap with ~8× less capacity. Still far behind GPT-2 in absolute terms, which is correct: a 14.7M-param ghost-tiny is a learning artifact, not a competitor. The trajectory is what matters.

### Note on val_loss

Final v0.3.5 val_loss is 3.5518 vs v0.3.3's 3.4458. **Do not read this as v0.3.3 being a better model.** The val sets are different — v0.3.5's val covers six sources (NVD, arxiv, ctftime, mitre, capec, synthetic) while v0.3.3's was NVD-dominated. A more diverse val set is harder to predict per-token regardless of model quality. The per-source perplexity table above is the cleaner read.

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
- **Mode-collapse on severity classification:** the model predicts "Critical" on 72% of CVE Severity samples regardless of input. Above random (32% vs 25% baseline) but the prior is doing most of the work. Calibrated severity reasoning is not present at this scale.
- **No tactic-level abstraction:** MITRE ATT&CK Tactic Classification accuracy is 12% (vs 8.3% random baseline) — the model can identify concrete techniques but not the higher-level adversary goal that groups them. Expected to be the slowest reasoning capability to emerge with scale.
- **Web/binary CTF confusion:** CTF Categorization scores 100% on Forensics and Cryptography but 0% on Web Exploitation — the model conflates web-exploit writeups with adjacent categories. Targeted corpus expansion in v0.4.0 should focus there.
- **Synthetic CTF share:** 17.2% of training tokens are synthetic CTF writeups. Replacement with real CTFtime / GitHub writeup corpus is the priority next-track work; the synthetic source will be dropped once real-writeup volume exceeds it.
- **NVD corpus weight:** 65.3% of training tokens are CVE descriptions (down from 87% pre-rebalance). The model still tilts toward CVE-style prose on neutral prompts.
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
