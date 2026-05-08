![CI](https://github.com/joemunene-by/GhostLM/actions/workflows/ci.yml/badge.svg) ![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg) ![Status](https://img.shields.io/badge/status-v0.9.8%20%2812%20bets%2C%201.7K%20records%2C%20full%20analyst%20workflow%29%2C%20ghost--base%20pending%20GPU-blue.svg)

# GhostLM

> An open-source language model built entirely from scratch in PyTorch. Purpose-built for cybersecurity, with code, general language, and math/reasoning folded into the v1.0 corpus.

> **2026-05-06 update.** The ghost-small (45-81M) line saturated at ~28% on debiased CTIBench and 0-2% on free-form fact recall: register-matching parrot, not a fact-knower. v0.9 chat is the bench winner across CTIBench full / in-repo CTF eval / external SecQA, but the truth metric (free-form fact recall) is at floor for the whole 81M parameter rung. The next move is parameter count, not corpus polish. **The v1.0 corpus is built**: 516,736 train / 27,049 val records / ~363M tokens spanning six domains (cybersec writeup, code, FineWeb-Edu general language, math/reasoning, NIST authoritative reference, security research blogs). Ghost-base (~360M params, 30L × 960d × 15h, SmolLM2-360M shape) is the v1.0 target. Launcher at [`scripts/train_ghost_base.py`](scripts/train_ghost_base.py), spec at [`docs/ghost_base_spec.md`](docs/ghost_base_spec.md), gated on rented GPU compute.

> **2026-05-08 update (v0.9.8): three new bets close the analyst-workflow loop.** Bet 10 ([docs/log_analysis bank](data/raw/log_analysis_patterns.jsonl)) trains on SOC-analyst log review across Windows Sysmon / Security, Linux auditbeat, network proxy/webserver/DNS, and email gateway logs across 30 ATT&CK techniques (120 records). Bet 11 ([cloud IaC security](data/raw/iac_security_patterns.jsonl)) trains on DevSecOps PR-review of Terraform / Kubernetes misconfigs across 15 patterns covering S3, IAM, security groups, RDS, Pod securityContext, NetworkPolicy, RBAC, secrets management (60 records). Bet 12 ([protocol field reading](data/raw/protocol_field_patterns.jsonl)) trains on network-forensics wire-format decoding across 20 patterns spanning datalink / network / transport / application layers including TLS 1.3, DNS, HTTP/2, BGP, IP, TCP, Ethernet, ARP, SMB2, Kerberos, QUIC, MQTT, RDP, plus JA3 fingerprinting (60 records). **Combined templated-synth corpus is now 1,745 records across 12 bets**, all parser-validated, reproducible from one command. Ghost-base SFT-data is fully prepared; the only remaining gate is training compute.

> **2026-05-08 update (v0.9.7): GhostBench v0.3 ships the behavioural tier.** The reserved `behavioral` slot is now fully implemented in [`ghostbench/behavioral.py`](ghostbench/behavioral.py). Two-path validators per format: lazy-import the canonical reference parser (`stix2`, `yara-python`, `pysigma`, `jsonschema`) when available for full validation, fall back to enhanced-structural checks otherwise. Catches edge cases parse misses: invalid UUID4 in STIX `id`, malformed YARA condition trees, Sigma logsource taxonomy violations, MISP attribute types outside the controlled vocab. Common `--behavioral` flag on every CLI subcommand opts every record into the tier at score time. **205 tests passing** (94 GhostLM + 111 ghostbench, +31 behavioural). End-to-end dogfooded on Mac: behavioural row appears alongside parse / fields / substrings in the per-tier breakdown.

> **2026-05-08 update (v0.9.6): GhostBench is now a real packaged eval suite.** [`ghostbench/`](ghostbench/) is a properly-packaged, statistically-rigorous, model-agnostic benchmark library built around the nine bets. Wilson 95% CIs for small-n binomial proportions, exact McNemar's test for paired comparisons, Cohen's h effect-size labelling, Newcombe-style paired-difference CIs. Multi-tier scoring (parse / fields / substrings / reserved semantic + behavioural). Matplotlib visualisations (forest plots, paired-comparison bar charts, projection charts), all designed to drop into a paper or blog post. CLI: `python -m ghostbench [score | summary | compare | suite-compare]`. **80 ghostbench tests + 94 GhostLM tests = 174 green.** Plus scaling-law projections at [`docs/ghost_base_projections.md`](docs/ghost_base_projections.md): given v0.9.5's templated-synth corpus, ghost-base should land ~61% on bet 6, ~75% on bet 9, with bets 7/8 under-resourced at ~3-6% pending bank growth or LLM-distillation. The same `Suite` machinery that scores GhostLM can be pointed at any small open LM (SmolLM2, Qwen2.5-0.5B, Llama-3.2-1B) for a head-to-head comparison.

> **2026-05-08 update (v0.9.5): nine differentiation bets, multi-modal-in-security.** Strategic frame at [docs/differentiation.md](docs/differentiation.md): the first six bets (tool grounding, freshness, tokenizer, context, MoE, structured-format literacy) made GhostLM narrowly competent. Bets 7-9 added in response to "what would make GhostLM exceptional, beyond what general-purpose small LMs offer?": **code-for-security** ([docs/code_security_synth.md](docs/code_security_synth.md), 12-pattern bank covering OWASP-Top-10-shaped CWE classes, 48 templated training records), **binary / hex literacy** ([docs/binary_literacy_synth.md](docs/binary_literacy_synth.md), 15-pattern bank covering PE / ELF / Mach-O / packers / shellcode / pe_field / disassembly, 44 records, **the most novel bet, no other small cybersec LM does this natively**), and **provenance with `<|cite|>...<|/cite|>` tags** ([docs/provenance_synth.md](docs/provenance_synth.md), cite-augmented tool-use traces, 429 records). Combined templated-synth corpus across all five shipping bets: **1,505 records, 99.4% parser-pass**, all reproducible from a public hand-curated bank. Bet 6's eval set grew to n=32 with Wilson 95% CIs; v0.9 baseline now locked at 0/32 = [0.0-10.7%].

> **2026-05-08 update (v0.9.4).** Six differentiation bets shipped. Tool-grounded SFT ([scripts/distill_tool_use.py](scripts/distill_tool_use.py)), nightly LoRA over fresh threat-intel ([scripts/daily_finetune.py](scripts/daily_finetune.py)), custom 32K BPE ([scripts/train_v1_bpe.py](scripts/train_v1_bpe.py), measured +4.0% on cyber / -2.5% on general vs GPT-2 BPE per [docs/bpe_corpus_ablation.md](docs/bpe_corpus_ablation.md), recommendation: ghost-base default to GPT-2 BPE), RoPE NTK 16K context ([scripts/extend_context_ntk.py](scripts/extend_context_ntk.py)), MoE for ghost-1B+ (`ghostlm/model.py:SparseMoE` with aux loss in `GhostLM.forward()`, presets `ghost-1b` 2.1B/1.2B-active and `ghost-3b` 6.0B/3.3B-active, 100-step training smoke PASS per [docs/moe_training_smoke.md](docs/moe_training_smoke.md)), and structured-format pretrain ([scripts/distill_format_aware.py](scripts/distill_format_aware.py)). v0.9 chat structural-compliance baseline locked at 0/8 = 0% per [docs/format_baseline_v09.md](docs/format_baseline_v09.md).

> **2026-05-07 update.** Eight pre-GPU artifacts shipped to make the v1.0 run more meaningful and unblock follow-on capability without spending GPU money. **RAG layer** wired into the demo Space (BGE-small embeddings over a 83K-chunk index of the cybersec corpus, top-4 passages prepended to each chat turn, gracefully falls back to bare chat if the index isn't loaded). **Fact-recall v2 bench** at [`data/raw/fact_recall_bench_v2.jsonl`](data/raw/fact_recall_bench_v2.jsonl) (n=100, growing to 200) with a smarter grader that adds boundary matching, disqualifier phrases, and composite-fact `must_appear` requirements; this is the truth metric for the ghost-base acceptance gate, doc at [`docs/fact_recall_v2.md`](docs/fact_recall_v2.md). **Corpus contamination audit** ([`scripts/audit_corpus_contamination.py`](scripts/audit_corpus_contamination.py), result in [`docs/contamination_audit.md`](docs/contamination_audit.md)) confirms the v1.0 corpus is essentially clean of CTIBench leakage so the GPU spend is safe. **Distillation pipeline** for ghost-3B+ corpus expansion is end-to-end complete: shared lib at [`scripts/distill_common.py`](scripts/distill_common.py) plus four per-type scripts (CTF walkthroughs, STRIDE threat models, deobfuscation, malware analysis), doc at [`docs/distillation.md`](docs/distillation.md). **Streaming chat** in the Space, **fp16 + int8 quantization** at [`scripts/quantize_v09.py`](scripts/quantize_v09.py), **MCP tool harness** expanded with NVD lookup + local MITRE + RAG query, **vendor research collector** for 11 missing TI feeds (Talos, Unit 42, Mandiant, Crowdstrike, etc), and the **multi-year hardware pathway** at [`docs/hardware_pathway.md`](docs/hardware_pathway.md) (RTX 6000 Pro Blackwell 96GB recommendation through ghost-7B; corpus is the harder ceiling past ghost-3B). All landed without spending a dollar on compute.

GhostLM is a decoder-only transformer language model. Pretrained from scratch on CVE descriptions, CTF writeups, MITRE/CWE/OWASP/RFC reference material, NIST SP 800 publications, security research blogs, security tool source code, FineWeb-Edu educational web text, and open-web-math reasoning. No pretrained weights, no wrappers, every component written by hand.

---

## Why GhostLM?

Security researchers currently rely on generic models (GPT-4, Llama) that weren't trained with security context. GhostLM is purpose-built for:

- CVE analysis and vulnerability explanation
- CTF challenge reasoning
- Penetration testing assistance
- Exploit and attack pattern understanding
- Security concept explanation

### Why from scratch and not a fine-tune?

Two reasons. **First**, most offensive-security content that the best general models have seen was filtered or RLHF-nudged away during alignment, so a fine-tune on top fights that prior. Training the tokenizer and weights from zero with security text in the mix lets the model treat CVE IDs, shell one-liners, and exploit technique names as first-class tokens rather than something to refuse. **Second**, GhostLM is also a study project. Every layer (attention, positional encoding, LR schedule, BPE) is hand-written so the codebase doubles as a readable reference for how a transformer is actually put together. A fine-tune hides that behind `AutoModel.from_pretrained`.

It is explicitly *not* trying to beat Llama on general benchmarks. It's trying to be the right tool for one narrow job, and a transparent one.

---

## Architecture

GhostLM is a multi-rung scale ladder. The smallest rung (**ghost-tiny**, 14.7M params) is the educational reference; the largest currently shipped is **ghost-small-v0.9** (81M params, RoPE + SwiGLU + RMSNorm); the v1.0 target is **ghost-base** (~360M, 12L × 768d × 12h, launcher at `scripts/train_ghost_base.py`).

| Variant | Layers | Heads | d_model | d_ff | Params | Tokenizer | Context |
|---|---:|---:|---:|---:|---:|---|---:|
| ghost-tiny | 2 | 4 | 256 | 1024 | 14.7M | GPT-2 BPE + 7 special | 1024 |
| ghost-small (v0.4) | 6 | 8 | 512 | 2048 | ~45M | GPT-2 BPE + 7 special | 1024 |
| ghost-small-v0.5 | 6 | 8 | 512 | 2048 | ~36M | custom 32K BPE + 7 special | 512 |
| ghost-small-v0.6 / v0.7 / v0.8 / v0.9 | 6 | 8 / 12 | 512 / 768 | 2048 / 3072 | 45M / 81M | GPT-2 50K BPE + 7 special | 512 |
| **ghost-base (v1.0 target)** | **12** | **12** | **768** | **3072** | **~360M** | **GPT-2 50K BPE + 7 special** | **1024 train / 2048 inference** |
| ghost-1b (preset, MoE) | 24 | 24 | 1536 | 6144 | ~2.1B total / 1.2B active | v1 BPE 32K + 11 special | 2048 |
| ghost-3b (preset, MoE) | 32 | 32 | 2048 | 10240 | ~6.0B total / 3.3B active | v1 BPE 32K + 11 special | 2048 |

Built with:
- Multi-head causal self-attention (manual implementation)
- **RoPE** (Rotary Position Embeddings), default-on for v0.5+, the relative-position encoding used by LLaMA / Mistral
- **SwiGLU** FFN, default-on for v0.5+, gated FFN with three projections (LLaMA-style)
- **RMSNorm**, default-on for v0.5+, half the params of LayerNorm with no quality loss at this scale
- **Flash Attention**, opt-in via `use_flash_attention=True`, routes through PyTorch 2.0+ `scaled_dot_product_attention` for `O(n)` memory
- Pre-norm transformer blocks with residual connections
- Cosine LR schedule with linear warmup
- Weight-tied output projection
- AdamW with weight decay separation
- **Safetensors** export for safe, arbitrary-code-free weight distribution (see `scripts/export.py`)

## Model Variants

GhostLM is a multi-year scale ladder. Each rung validates the recipe before climbing to the next:

| Variant | Layers | Dim | Params | Hardware target | Status |
|---|---|---|---|---|---|
| ghost-tiny | 2 | 256 | 14.7M | CPU | Historical, Phase 3.5 canonical on the PMI suite, superseded by ghost-small |
| ghost-small (v0.4) | 6 | 512 | ~45M | M4 GPU/MPS | Phase 4 base, learned PE / GELU / LayerNorm. Chat at 27.6% on debiased CTIBench full bench (n=2500), 50.0% CTF eval, 35.0% SecQA, 0/50 free-form fact recall |
| ghost-small-v0.5 | 6 | 512 | ~36M | M4 GPU/MPS | RoPE / SwiGLU / RMSNorm + custom 32K BPE. Chat clusters with the rest of the ghost-small line on debiased eval |
| ghost-small-v0.6 | 6 | 512 | ~45M | M4 GPU/MPS | v0.5 arch + GPT-2 50K BPE on the v0.4.2 expanded corpus. Chat at 28.2% debiased CTIBench (BPE swap ablation) |
| ghost-small-v0.7 | 6 | 768 | ~81M | M4 GPU/MPS | Wider variant of v0.6 (d_model 768, d_ff 3072). Chat at 27.2% / 50.0% / 37.6% / 1/50 across CTIBench full / CTF eval / SecQA / fact recall (param-count ablation; was the bench leader in n=500 sample) |
| ghost-small-v0.8 | 6 | 768 | ~81M | M4 GPU/MPS | v0.7 arch + Qwen-14B-distilled fact-QA in pretrain. Chat at 27.4% debiased CTIBench full; distilled Q&A alone doesn't lift |
| **ghost-small-v0.9** | 6 | 768 | ~81M | M4 GPU/MPS | **Bench winner of the ghost-small line**: 273M-token PRIMUS + CWE + OWASP + RFCs + fact-QA pretrain. Chat at **28.9% / 59.2% / 39.3% / 1/50** on CTIBench full / CTF eval / SecQA / fact recall. Wins every MCQ bench by 0.7-9.2 pp; free-form fact recall still at floor |
| **ghost-base** | **12** | **768** | **~360M** | **Rented GPU (A/H100)** | **v1.0 target. Corpus ready (516,736 train / ~363M tokens, six domains). Launcher at `scripts/train_ghost_base.py`, spec at `docs/ghost_base_spec.md`. Acceptance gate: ≥40% CTIBench OR ≥65% CTF eval OR ≥30% on the 50-question fact-recall set. Pending GPU access.** |
| ghost-1b | 24 | 1536 | ~2.1B total / 1.2B active | Rented or owned GPU (Blackwell 96GB) | Preset shipped (`from_preset("ghost-1b")` in `ghostlm/config.py`). MoE 4 experts top-2 (bet 5), v1 BPE 32K, RoPE+SwiGLU+RMSNorm+flash. Untrained, awaits compute |
| ghost-3b | 32 | 2048 | ~6.0B total / 3.3B active | Rented owned GPU | Preset shipped. MoE 4 experts top-2. Untrained |

ghost-tiny is the iteration vehicle and educational artifact. It is not, and at this scale will not become, a useful cyber-task model. The scale ladder above is the path to "useful." See [ROADMAP.md](ROADMAP.md) for phased milestones, corpus targets per rung, and honest compute estimates.

---

## Quick Start

### Installation
```bash
git clone https://github.com/joemunene-by/GhostLM.git
cd GhostLM
make install
```

### Prepare Training Data
```bash
make data
```

### Train
```bash
# CPU-friendly (ghost-tiny)
make train-tiny

# GPU (ghost-small)
make train-small
```

### Generate Text
```bash
make generate
```

### Interactive Chat
```bash
make chat
```

### Run as a Tool-Using Agent
```bash
# Smoke test with random ghost-tiny weights
python -m ghostlm.agent --query "What is CVE-2017-0144?" --offline

# Real checkpoint
python -m ghostlm.agent --query "..." --checkpoint runs/v09chat/best.pt
```
GhostAgent wraps any GhostLM checkpoint in a tool-using loop with
the four canonical tools (CVE / MITRE / CWE / RAG retrieval),
parses `<|tool_call|>` and `<|cite|>` tags, and emits a JSON-
serialisable trace. See `ghostlm/agent/` for the runtime and
`tests/test_agent.py` for the 31-case test suite.

### Serve as an HTTP API (OpenAI / Anthropic / Gemini / Ollama compatible)
```bash
python -m ghostlm.agent.server --checkpoint runs/v09chat/best.pt --port 8000
```
Exposes the agent loop over five vendor-compatible endpoint
families plus a native `/v1/agent/run`. Any client that already
targets OpenAI, Anthropic, Google Gemini, or Ollama can point at
the server unchanged. Tool calls happen server-side; the final
cite-tagged answer comes back in whatever shape the SDK expects.
Test suite at `tests/test_agent_server.py` (22 cases).

### Run Web Demo
```bash
pip install gradio
python demo/app.py
```

### Benchmark vs GPT-2
```bash
make benchmark
```

### Export Weights (safetensors or PyTorch)
```bash
# Safe, pickle-free weights for HuggingFace Hub distribution
python scripts/export.py --format safetensors

# Classic PyTorch checkpoint
python scripts/export.py --format pt
```

### Plot Training Curves
```bash
make plot
```

---

## Training Data

The v1.0 corpus has 516,736 train records / 27,049 val / ~363M tokens, six domains:

| Domain | Tokens (M) | Share | Sources |
|---|---:|---:|---|
| Cybersec writeup-style | ~265 | 73% | PRIMUS-Seed/FineWeb (Trend Micro, ODC-BY), NVD CVE (capped 6M tokens via deterministic-hash subsample), Exploit-DB (GPL-2.0), MITRE ATT&CK / CAPEC / CWE, OWASP family (cheatsheets / WSTG / ASVS / Top 10), CTFtime real writeups, arXiv cs.CR abstracts + full-text, fact-QA (Qwen-14B distilled), CISA KEV, IETF security RFCs |
| General language | ~46 | 13% | `HuggingFaceFW/fineweb-edu` (ODC-BY, classifier-filtered educational web) |
| Math / reasoning | ~21 | 6% | `open-web-math/open-web-math` (ODC-BY, math-filtered web) |
| Code (cybersec tools) | ~9 | 2.4% | 30 curated security tool repos (pwntools, impacket, scapy, sqlmap, volatility3, capa, plaso, AFL++, nuclei, trivy, prowler, paramiko, pyca/cryptography, etc.) |
| Authoritative reference | ~3 | 0.7% | 26 NIST SP 800 publications (RMF, controls, identity, IDS, zero trust, secure SDF, etc.); pymupdf-extracted, 12K-char chunks |
| Research-blog register | ~0.6 | 0.2% | 11 RSS/Atom feeds (Project Zero, PortSwigger Research, Trail of Bits, Google Security, GitHub SecurityLab, NCC Group, Doyensec, Krebs, DFIR Report, Ret2 Systems, MSRC) |
| **Total** | **~363** | **100%** | 26 distinct sources |

The pipeline produces a deterministic, leakage-proof split (content-hash bucketing, leakage check returns 0). NVD subsample is reproducible: `python3 scripts/rebuild_corpus.py --max-cve-tokens 6000000` always produces the same 71,828-record CVE prefix from the 333,540-record raw dump. Each new collector is a standalone CLI under `scripts/` (`collect_security_code.py`, `collect_fineweb_edu.py`, `collect_nist_sp800.py`, `collect_security_blogs.py`, `collect_math_reasoning.py`, plus the existing collectors); rebuild auto-globs every `data/raw/*.jsonl`.

For per-source record counts, license posture, and reproducibility commands, see [`CORPUS.md`](CORPUS.md).

---

## Training Progress

| Run | Steps | Train tokens | Val Loss | Notes |
|---|---|---|---|---|
| ghost-tiny Phase 1 (pre-audit corpus) | 10,000 | 2.66M (leaky) | 2.74 | Superseded, leaky train/val split, archived under `archive/` |
| ghost-tiny Phase 2 (rebalanced corpus) | 10,000 | 2.66M | 3.7813 | Archived as `checkpoints/best_model_phase2.pt` |
| ghost-tiny Phase 3 (post-NVD-pull corpus) | 30,000 | ~30M | 3.4458 | NVD-dominated (87%); preserved as `checkpoints/phase3_refresh/best_model.pt` |
| ghost-tiny Phase 3.5 (rebalanced corpus) | 30,000 | ~8.8M | 3.5518 | Historical canonical for the existing PMI suite. NVD share 65%, six sources balanced. Hardware: Mac Mini M4 (CPU), ~3h13m wall-clock |
| ghost-tiny Phase 3.6 (+Exploit-DB) | 30,000 | ~12.56M | 3.8556 | Regressed on the eval suite (31.2% → 16.8%); ghost-tiny capacity ceiling found. Preserved at `checkpoints/phase3.6_exploitdb/best_model.pt`, see CHANGELOG v0.3.7 |
| **ghost-small Phase 4 (capacity reallocation)** | **30,000** | **~12.56M** | **2.3535** | **Current canonical model for density / generation.** ~45M params (6L / 512d / 8h) on the same Phase 3.6 corpus. **Per-source PPL 59-78% better than Phase 3.5 across every source**, overall PPL 66.05 → 11.12 (−83%). Hardware: Mac Mini M4 (MPS), ~15h wall-clock. See CHANGELOG v0.4.0 |

> Cross-phase val_loss is **not directly comparable** between phases when the corpus changes: each phase from 3.5 onward has a different validation distribution. The eval-axis numbers below are the cleaner read.

The Phase 4 ghost-small checkpoint at `checkpoints/phase4_ghost_small/best_model.pt` is the current canonical model for any density / completion / generation work, it dominates Phase 3.5 by 59-78% on per-source perplexity across every source. The Phase 3.5 ghost-tiny checkpoint at `checkpoints/phase3.5_balanced/best_model.pt` remains on disk as the historical canonical and is still the higher number on the existing **PMI** multiple-choice suite (a calibration artifact at small corpus size; see [CHANGELOG.md](CHANGELOG.md) v0.4.0 for the PMI vs logp scoring analysis). Both are kept; pick by use case.

### Chat tuning, debiased real capability (v0.9.2)

A supervised fine-tune on top of the base ghost-small turns the completion model into a conversational cybersecurity assistant. As of v0.9.2 the canonical chat model is **`checkpoints/phase19_chat_v09/best_model.pt`** (81M params, v0.7 wide architecture, pretrained on the 273M-token PRIMUS + CWE + OWASP + RFC + fact-QA corpus, fine-tuned with the canonical chat-v3 SFT recipe).

Each chat-tune is evaluated on three independent MCQ sources plus one free-form fact-recall set:

- **CTIBench MCQ** (full test split, n=2500, 2 perms) — the AI4Sec/cti-bench benchmark.
- **In-repo CTF eval** (n=30, 4 perms) — hand-written cybersec MCQ at `data/raw/ctf_eval_bench.jsonl`.
- **SecQA** (n=210, 4 perms) — external benchmark, pulled via `scripts/fetch_secqa.py` from `zefang-liu/secqa` on HuggingFace.
- **Free-form fact recall** (n=50) — single-line factual prompts at `data/raw/fact_recall_bench.jsonl`, substring-graded.

All MCQ rows below use multi-permutation text-scoring: log P(option_text | prompt) per option under N option-letter orderings, no letter-token bias. Random baseline on 4-way MCQ is 25%. Fact-recall is free-form completion with substring grading; random baseline is ~0%.

| Checkpoint | CTIBench (n=2500) | CTF eval (n=30) | SecQA (n=210) | Fact recall (n=50) |
|---|---:|---:|---:|---:|
| `phase5_chat_v3` (v0.4 base, canonical from v0.5.0) | 27.6% | 50.0% | 35.0% | 0/50 (0.0%) |
| `phase10_chat_v06` (v0.6, BPE swap) | 28.2% | — | — | — |
| `phase15_chat_v07` (v0.7, 81M wide) | 27.2% | 50.0% | 37.6% | 1/50 (2.0%) |
| `phase20_chat_v07_ctx1024` (v0.7 ctx-1024 extension) | 26.7% | 45.8% | — | — |
| `phase17_chat_v08` (v0.8, 81M + fact-QA) | 27.4% | — | — | — |
| **`phase19_chat_v09` (canonical, 273M-token corpus)** | **28.9%** | **59.2%** | **39.3%** | **1/50 (2.0%)** |

**v0.9 wins every MCQ bench by 0.7-9.2 pp.** The corpus-density swing produced a real, consistent capability lift across CTIBench (+1.3-1.7 pp over v0.4/v0.7), the in-repo CTF eval (+9.2 pp), and the external SecQA bench (+1.7-4.3 pp). The ranking holds across all three independent sources.

**But fact-recall is at floor.** v0.4 / v0.7 / v0.9 all score 0-2% on 50 hand-written single-line factual prompts, and the two "hits" v0.7 and v0.9 each registered are arguably spurious (v0.7's "Injection" appears in unrelated tangent prose; v0.9's "256" comes from echoing "SHA-256" in the question itself). **The MCQ wins reflect register matching and topic distinctness, not factual recall.** The "cybersec parrot" diagnosis from v0.6.0 stands: at 81M parameters, the model has the *register* of cyber writing but not the *facts* in any retrievable form.

**Methodology correction (apples-to-apples re-bench, v0.9.2):** earlier README versions reported v0.4 at 30.5%, v0.5 at 29.7%, v0.6 at 31.2%, v0.7 at 32.2%, v0.8 at 31.2% on debiased CTIBench. All of those were on a 500-record subset; only v0.9 was scored on the full 2500. The apparent "v0.9 regressed against v0.7" was a sampling artifact. Re-benching every chat-tune on the full n=2500 set produces the table above, where v0.9 leads. The v0.9.0 / v0.9.1 release notes preserve the older numbers for historical record. Full investigation in [`docs/ctibench_bias_finding.md`](docs/ctibench_bias_finding.md), recipe in [`docs/chat_tuning.md`](docs/chat_tuning.md), raw data in [`RESULTS.md`](RESULTS.md), per-checkpoint JSONs in `logs/text_scoring/`.

The **next rung is ghost-base (~360M, rented GPU)** at [`docs/ghost_base_spec.md`](docs/ghost_base_spec.md). The v0.9 corpus-density gain on MCQ benches plus the floor result on free-form fact recall together make the case clearly: parameter count is what's missing for fact binding, and the v0.9 corpus is the right substrate to scale into. Acceptance criteria for ghost-base now include the free-form fact-recall benchmark explicitly: ≥40% per-perm avg on debiased CTIBench OR ≥65% on the CTF eval OR ≥30% on the 50-question fact-recall set; passing any one validates the rung.

### Cross-phase eval, fair comparison (fixed test set)

The cyber-text benchmark is 10 hand-picked external samples that overlap none of the training corpora. Directly comparable across phases:

| Model | Cyber-text perplexity (lower better) |
|---|---|
| **ghost-tiny, Phase 3.5 (released)** | **96.24** |
| ghost-tiny, Phase 3 | 142.09 |
| ghost-tiny, Phase 2 | 152.71 |
| ghost-tiny, Phase 1 | 2,183.94 |
| GPT-2 (124M baseline) | 26.76 |

Phase 3 → Phase 3.5 dropped this benchmark **32%** (142.09 → 96.24) at fixed parameter count and 1/3 the training tokens. ghost-tiny is now ~3.6× behind GPT-2 on raw cyber-text perplexity, with ~8× less capacity. The trajectory matters more than the absolute number; full breakdown in [MODEL_CARD.md](MODEL_CARD.md#evaluation).

### Per-source perplexity (val split)

The cleanest cross-phase read: does the model actually model each source it was trained on. The full trajectory across phases:

| Source | v0.3.3 (P3) | v0.3.5 (P3.5) | v0.3.7 (P3.6) | **v0.4.0 (P4)** | P4 vs P3.5 |
|---|---:|---:|---:|---:|---:|
| arXiv | 671.09 | 354.95 | 505.60 | **116.46** | **−67%** |
| CAPEC | 326.11 | 133.81 | 179.71 | **54.42** | **−59%** |
| CTFtime real writeups | 184.24 | 60.71 | 59.70 | **13.23** | **−78%** |
| Exploit-DB | - | - | 40.87 | **8.60** | new source |
| MITRE ATT&CK | 615.43 | 55.14 | 70.53 | **19.72** | **−64%** |
| NVD CVE | 24.19 | 27.55 | 35.44 | **11.29** | **−59%** |
| Synthetic CTF | 67.57 | 28.48 | 38.90 | **7.88** | **−72%** |
| **Overall** | **171.84** | **66.05** | **44.36** | **11.12** | **−83%** |

Three distinct phase-on-phase wins to read off this table:

- **v0.3.3 → v0.3.5 (corpus rebalance, fixed model):** the 47-91% drops on MITRE / CTFtime / CAPEC came from those sources being added to training, the synthetic-CTF / arXiv drops from same data with parameter capacity redirected away from memorizing duplicate CVEs.
- **v0.3.5 → v0.3.6 (corpus volume, fixed model):** every existing source got 28-42% worse, ghost-tiny ran out of capacity to hold seven sources at once. This is the result that diagnosed the ceiling.
- **v0.3.6 → v0.4.0 (model capacity, fixed corpus):** every single source improved 68-80% relative to v0.3.6, and 59-78% relative to v0.3.5. ghost-small at 45M params absorbs the corpus that broke ghost-tiny without the per-source tradeoff. **Capacity-reallocation hypothesis confirmed.**

### PMI-corrected security task accuracy

5 classification tasks × 25 samples = 125 evaluations (expanded from the 30-sample suite in v0.3.6). Old length-normalized scoring was mode-collapsed at 4/30 = 13.3% across all phases under logp scoring (eval failure, not model failure); PMI scoring fixed it.

| Task | Labels | Random | v0.3.5 | Most-common share |
|---|---|---|---|---|
| CVE Severity Classification | 4 | 25.0% | 8/25 (32.0%) | Critical 72% |
| Vulnerability Type Detection | 10 | 10.0% | 8/25 (32.0%) | IDOR 44% |
| Attack Technique Identification | 10 | 10.0% | 10/25 (40.0%) | LatMov 36% |
| CTF Challenge Categorization | 5 | 20.0% | 10/25 (40.0%) | Forensics 64% |
| MITRE ATT&CK Tactic Classification | 12 | 8.3% | 3/25 (12.0%) | LatMov 40% |
| **Overall** | - | ~14.5% | **39/125 (31.2%)** | - |

The 30-sample suite reported 12/30 = 40% on this same checkpoint. The drop to 31.2% is the eval getting more honest, not the model getting worse: with 25 balanced samples per task we now see CVE Severity is mode-collapsing toward "Critical" (72%) and MITRE Tactic is barely above random (12% vs 8.3% baseline). Vulnerability Type, Attack Technique, and CTF Categorization remain meaningfully above random (+22, +30, +20 pp), those are the corpora that grew in the Phase 3.5 rebalance. See `CHANGELOG.md` v0.3.6 for the full discussion.

#### Phase 3.6 attempted next, regressed (v0.3.7)

The next training run added Exploit-DB (~3.77M tokens, 30% of the new corpus) and re-trained ghost-tiny at the same 30K-step recipe. The result was a 14.4 pp drop on the same eval suite:

| Task | Phase 3.5 | Phase 3.6 | Δ |
|---|---|---|---|
| CVE Severity Classification | 8/25 (32.0%) [72%] | 4/25 (16.0%) [60%] | −16 pp |
| Vulnerability Type Detection | 8/25 (32.0%) [44%] | 3/25 (12.0%) [**96%**] | −20 pp |
| Attack Technique Identification | 10/25 (40.0%) [36%] | 4/25 (16.0%) [60%] | −24 pp |
| CTF Challenge Categorization | 10/25 (40.0%) [64%] | 5/25 (20.0%) [48%] | −20 pp |
| MITRE ATT&CK Tactic Classification | 3/25 (12.0%) [40%] | 5/25 (20.0%) [76%] | +8 pp (mode-collapsed) |
| **Overall** | **31.2%** | **16.8%** | **−14.4 pp** |

Per-source perplexity confirmed the diagnosis: every existing source got 28-42% worse while Exploit-DB landed cleanly modeled (PPL 40.87). The "improved" overall PPL of −32.8% was misleading: Exploit-DB's heavy token share dragged the weighted average down regardless of how the existing sources fared.

**Conclusion:** ghost-tiny at 14.7M params is at capacity. More corpus at fixed model size has hit diminishing returns at this rung. The path forward is the model (ghost-small at 55M params), not more data. Phase 3.6 corpus + checkpoint preserved at `checkpoints/phase3.6_exploitdb/best_model.pt` as the ghost-small training target, if ghost-small absorbs the same corpus without per-source regression, the capacity-reallocation hypothesis is confirmed. See `CHANGELOG.md` v0.3.7 for the full per-source breakdown and reasoning.

#### Phase 4 ghost-small, capacity-reallocation hypothesis confirmed (v0.4.0)

ghost-small (~45M params, 6 layers / 512 d_model / 8 heads) trained on the same Phase 3.6 corpus that broke ghost-tiny. 30k steps, MPS, 15h wall-clock. Final val_loss **2.3535**, a 1.20-nat (~3.3× perplexity) drop relative to Phase 3.5 ghost-tiny (3.5518), and the loss curve was still descending at the final step.

The PMI security suite is more nuanced. Headline number drops vs Phase 3.5 (39/125 → 29/125, 31.2% → 23.2%), but with **logp scoring** (no PMI-correction) Phase 4 actually beats Phase 3.5 (24/125 vs 22/125, 19.2% vs 17.6%). The PMI advantage at Phase 3.5 is a calibration artifact, PMI subtracts the unconditional candidate log-prob to break ties, and a higher-capacity model with a tighter probability distribution gives PMI less separation to work with. On a 25-sample-per-task suite this can flip the headline.

| Task | P3.5 PMI | P3.5 logp | **P4 PMI** | **P4 logp** |
|---|---:|---:|---:|---:|
| CVE Severity | 32% | 24% | 24% | 24% |
| Vuln Type | 32% | 20% | **40%** | 16% |
| Attack Tech | **40%** | 8% | 16% | 12% |
| CTF Cat | **40%** | 28% | 28% | 28% |
| MITRE Tactic | 12% | 8% | 8% | **16%** |
| **Overall** | **31.2%** | 17.6% | 23.2% | **19.2%** |

Ranking by metric, honestly:
1. **Per-source PPL (density):** Phase 4 wins decisively (−83% overall vs Phase 3.5).
2. **Logp eval (conservative scoring):** Phase 4 wins narrowly (+1.6 pp).
3. **PMI eval (favors loose-distribution models):** Phase 3.5 wins (+8.0 pp).

Phase 4 is the new canonical for any density / generation use; Phase 3.5 stays on disk as the historical canonical and the higher PMI scorer. See `CHANGELOG.md` v0.4.0 for the full breakdown.

## Sample Generations

Real generations from the v0.3.5 best_model at `temperature=0.7, top_k=40`. The point of these isn't fluency, it's **register diversity**. v0.3.3 collapsed every prompt into CVE prose; v0.3.5 has multiple registers it can switch between depending on the prompt domain.

> **Prompt:** *MITRE ATT&CK technique T1003*
> T1003.011: defense-evasion
> Tactic: defense-evasion
> Adversaries may use-evasion, such as legitimate system-evasion, or malicious payloads.dll, execution, execution, and execution is used for creating a command execution of code is required by a...

This is **MITRE schema output**, the sub-technique ID format (`T1003.011`), the `Tactic:` header, the standard MITRE narrative opening (*"Adversaries may..."*). v0.3.3 had zero MITRE training data and would have produced a fake CVE for this prompt. v0.3.5 produces proper MITRE register because it now models that domain.

> **Prompt:** *The CTF challenge involved*
> ...leveraging a web application that was vulnerable to SQL injection attacks. The challenge required a specific flaw exists within the application. The issue results from the lack of proper validation of user-supplied data, which can result in a heap corruption condition. An attacker can leverage this vulnerability to execute code in the context...

CTF-writeup voice in the opening (*"leveraging a web application... The challenge required..."*), narrative rather than the structured CVE template. Drifts back into CVE register mid-sentence (still ghost-tiny, still 14.7M params), but the opening voice is correctly CTF-flavored.

> **Prompt:** *A SQL injection attack works by*
> ...injecting malicious SQL payloads on a payload.php, allowing attackers to execute arbitrary SQL commands via the id parameter.The Web Client (Wise application in Google Chrome prior to 115.0.60.27.48 allowed a remote attacker to perform a heap corruption via a crafted HTML page...

Mixed: starts with the right framing (*"injecting malicious SQL payloads"*), then drifts into CVE-style with a fake Chrome version. CVE register is still default for security-prose prompts, but it's no longer the *only* register.

**What this shows:** the rebalance produced *behavioral* diversity, not just numerical eval improvements. The model can now switch into MITRE-format output, CTF-narrative voice, or CVE-template prose depending on what the prompt cues. Hallucinations remain rampant in absolute terms (fake Chrome version, scrambled MITRE narrative), the model has *form* but not *facts*. The 14.7M-param scale is the binding constraint on factual capability; the corpus rebalance fixed what could be fixed at this param count. The fix for hallucinations is more parameters at the same training quality, see the [Roadmap](#roadmap).

### Live test from chat-v3 canonical (v0.6.0 update)

The "model has form but not facts" diagnosis from v0.3.5 holds at v0.4.0 / v0.5.0 / v0.6.0 chat-tunes too. Direct free-form generation from `phase5_chat_v3` (the 36.9% single-order canonical) on five cybersec questions:

> **Q:** What is phishing?
> **A:** CAPEC-5, phishing attacks.

Knows the vocabulary association (phishing maps to CAPEC-5 in our training corpus). Answer truncates fast.

> **Q:** What does CVE-2017-0144 (EternalBlue) exploit?
> **A:** In the Linux kernel, the following vulnerability has been resolved: net/mlx5e: Fix a memory leak in mlx5e_core_add_dev()...

**Wrong.** EternalBlue is Windows SMB, not a Linux mlx5e leak. The model pattern-matched the CVE prefix to NVD-style descriptions and hallucinated a plausible-looking but factually wrong response.

> **Q:** How does a SQL injection attack work?
> **A:** The challenge presented a login form where user input was directly embedded in the database. Upon examining the login form, I noticed that the login form did not properly sanitize the input, allowing for SQL injection. This was a classic SQL injection vulnerability where...

Coherent on the mechanism (unsanitized input), framed as a CTF writeup because that dominates the training corpus.

> **Q:** Explain MITRE ATT&CK technique T1059 (Command and Scripting Interpreter).
> **A:** T1059, RDP is a technique used in conjunction with Command and Scripting Interpreter to execute arbitrary commands on the target system.

Knows the URL format and the phrase "Command and Scripting Interpreter," confuses T1059 with RDP (which is T1021).

**Diagnosis:** the model is a "cybersec parrot." It has learned vocabulary patterns, URL formats, and writing styles (heavily CTF-writeup-flavored). It has not learned cybersec facts. The 30% real ceiling on CTIBench MCQ is exactly what you would expect from a model that can sometimes pattern-match the right option text but cannot reliably distinguish factually correct from factually wrong cybersec sentences. Full investigation in [`docs/ctibench_bias_finding.md`](docs/ctibench_bias_finding.md). Fix is data-density (Phase 1 of v0.8: Qwen-distilled fact-dense Q&A; Phase 2: PRIMUS corpus) plus eventual scale.

---

## Project Structure

```
GhostLM/
├── ghostlm/ # Core library
│ ├── model.py # Transformer architecture (RoPE + Flash Attention toggles)
│ ├── config.py # Hyperparameters + ghost-tiny/small/medium presets
│ ├── tokenizer.py # GPT-2 BPE wrapper
│ ├── dataset.py # PyTorch dataset
│ ├── trainer.py # Training loop
│ └── agent/ # GhostAgent: tool-using runtime over a checkpoint
│   ├── runtime.py # GhostAgent loop + RuntimeConfig
│   ├── parser.py # bet 1 tool-call + bet 9 cite-tag parser
│   ├── tools.py # CVE / MITRE / CWE / RAG tool registry
│   ├── messages.py # AgentMessage + AgentTrace primitives
│   ├── runner.py # CLI: python -m ghostlm.agent --query ...
│   └── server.py # HTTP API: OpenAI / Anthropic / Gemini / Ollama
├── scripts/ # CLI tools
│ ├── train.py # Training entry point
│ ├── generate.py # Text generation
│ ├── chat.py # Interactive chat
│ ├── evaluate.py # Evaluation
│ ├── eval_security.py # Security-specific evaluation
│ ├── benchmark.py # GPT-2 comparison
│ ├── export.py # Weights export (safetensors / pt) + SHA-256 + config.json
│ ├── api.py # REST API server
│ ├── data_stats.py # Training-data statistics
│ ├── plot_training.py # Loss-curve plotter
│ ├── push_to_hub.py # HuggingFace Hub publisher
│ └── resume_train.sh # Resume an interrupted training run
├── data/ # Data pipeline
├── demo/ # Gradio web demo (demo/app.py)
├── tests/ # 181 unit tests (incl. 31 agent runtime + 24 SFT pipeline + 10 GhostBench agent + 22 HTTP server + bet 1-12 differentiation)
└── Makefile # One-command workflow
```

---

## Roadmap

GhostLM is a multi-year effort. The honest framing is that ghost-tiny is a learning artifact and a working pipeline, *not* a useful cyber-task model. The path to "useful" is the scale ladder below, paired with a corpus that grows by ~100× from where it is today. See [ROADMAP.md](ROADMAP.md) for full milestones, compute estimates, and corpus targets.

**Where we are (v0.9.3 + six differentiation bets, 2026-05-08):** the ghost-small line saturated at ~28% on debiased CTIBench and 0-2% on free-form fact recall, register-matching parrot, not a fact-knower. v0.9 chat is the bench winner across CTIBench full / in-repo CTF eval / external SecQA but the truth metric is at floor for the whole 81M parameter rung. Diagnostic: retriever surfaces the right passage 41% of the time; the 81M model extracts the fact 1% of the time. The bottleneck is generation capacity, not retrieval, and parameter scaling is the answer. **The v1.0 corpus is built**: 516,736 train / 27,049 val / ~363M tokens across six domains. **Ghost-base (~360M)** is the v1.0 target, launcher and spec ready, gated on rented GPU. **Six differentiation bets** are scaffolded in the repo: tool-grounded SFT, daily LoRA over fresh threat-intel, custom 32K BPE (measured +1.6% vs GPT-2 BPE, opt-in), RoPE NTK 16K context extension, MoE for ghost-1B+ with `ghost-1b` (2.1B total / 1.2B active) and `ghost-3b` (6.0B total / 3.3B active) presets shipped, and structured-format pretrain (STIX / YARA / Sigma / MISP). Strategic frame at [docs/differentiation.md](docs/differentiation.md).

**Where we're going:**

1. **Ghost-base v1.0 GPU run:** rented H100 hours, 360M params on the 363M-token corpus. The acceptance gate is ≥40% CTIBench OR ≥65% CTF eval OR ≥30% on the 50-question fact-recall set. This is the spend that converts the saturated ghost-small line into a model with measurable fact-recall capability. Spec at [docs/ghost_base_spec.md](docs/ghost_base_spec.md).
2. **Bet 1 (tool-use SFT) on top of ghost-base:** ~$200 distillation budget for 10K traces, 1-2 GPU hours to fine-tune. The point of the GPU spend, the meta-skill of "lookup before answering" beats memorization at small scale.
3. **Bet 4 (long context to 16K):** RoPE NTK rebase + 3-5 GPU hours of long-form fine-tune. Unlocks IR triage workflows where a 50K-token threat report goes in the prompt.
4. **Bet 2 (daily LoRA cron):** practical once owned hardware lands (Blackwell 96GB recommendation); rented-GPU expense before that.
5. **Ghost-1b with native MoE from step 0:** 24-layer / 1536-d / 4-expert top-2. Bet 5's preset already in `ghostlm/config.py` so the architecture is settled; the remaining work is the actual pretrain run on owned compute.

**Realistic timeline:** 2-3 years of sustained work to a useful 1B from-scratch cyber LM. The shape of the curve from here is "park at the small-cybersec-LM benchmark plateau OR climb to ghost-base on rented H100s and re-bench." The differentiation bets are the strategic answer to "park is a crowded place." Detailed phase plan in [ROADMAP.md](ROADMAP.md), full multi-year hardware pathway in [docs/hardware_pathway.md](docs/hardware_pathway.md).

For changelog history (v0.1.0 onward), see [CHANGELOG.md](CHANGELOG.md).

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get involved.

---

## License

MIT. See [LICENSE](LICENSE).

---

## Author

**Joe Munene**, [Complex Developers](https://github.com/joemunene-by)

Built in Nairobi, Kenya.
