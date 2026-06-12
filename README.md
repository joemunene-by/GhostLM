<p align="center">
  <img src="assets/ghostlm_wordmark.png" alt="GhostLM" width="560">
</p>

<p align="center">
  <img src="https://github.com/joemunene-by/GhostLM/actions/workflows/ci.yml/badge.svg" alt="CI">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/version-0.9.35-blue.svg" alt="Version">
</p>

# GhostLM

> An open-source language model built entirely from scratch in PyTorch. Purpose-built for cybersecurity, with code, general language, and math/reasoning folded into the v1.0 corpus.

> **Sister project**: [`ghostloop`](https://github.com/joemunene-by/ghostloop) is the embodied-AI sibling — same `GhostAgent`-shaped tool-using runtime + fail-closed safety pipeline + GhostBench-shaped paired-comparison eval, applied to **robot motion primitives** instead of CVE / MITRE / CWE lookups. Shipped v0.3.0 on 2026-05-10 with PyBullet + MuJoCo backends, MuJoCo Menagerie loader, episode catalogue, trace replay, five policy gates (DenyList / RateLimit / Geofence / ForceCap / HITL), and a `python -m ghostloop` CLI. The thesis: as VLA models become the policy substrate, the runtime around them needs the same rigor we already apply to LLM tool use.

> **Status (v0.9.34 — 2026-06-10):** training/inference stack hardened ahead of the ghost-base GPU run — KV-cached generation (5.4× faster decoding), memory-mapped pretokenized corpus, real DDP data sharding, LR-schedule and SwiGLU-init fixes, plus live wandb metrics, `--compile`, and `--grad-checkpoint` flags, all dress-rehearsed end-to-end on Mac. The v1.0 pretrain corpus stands at 768,741 train / 40,429 val records (~422M tokens), code share 11.6%, cybersec sources ~65% of text. The GhostAgent tool-using runtime, multi-vendor HTTP server (OpenAI / Anthropic / Gemini / Ollama wire formats), MCP integration, and the GhostBench statistical eval suite are all shipped. **ghost-base (~360M params) is the v1.0 training target, gated on rented GPU compute.** Dated, per-version detail lives in [CHANGELOG.md](CHANGELOG.md).

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

# Optional but recommended for large corpora: pretokenize once into
# memory-mapped .bin files (instant startup, near-zero RAM at train time).
make pretokenize
```

### Train
```bash
# CPU-friendly (ghost-tiny)
make train-tiny

# GPU (ghost-small)
make train-small

# Multi-GPU (DDP): data is sharded per rank via DistributedSampler
torchrun --nproc_per_node=4 scripts/train.py --preset ghost-small ...

# GPU-run extras: live wandb metrics, torch.compile, gradient checkpointing
python scripts/train_ghost_base.py --wandb --compile --grad-checkpoint ...
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
nine cybersec tools (CVE / MITRE / CWE / RAG / CISA KEV /
GreyNoise / VirusTotal / Shodan / OTX), parses `<|tool_call|>` and
`<|cite|>` tags, and emits a JSON-serialisable trace. Each tool
tries its real upstream API when keys are set
(GREYNOISE_API_KEY, VIRUSTOTAL_API_KEY, SHODAN_API_KEY, OTX_API_KEY)
and falls back to an in-package offline cache otherwise. See
`ghostlm/agent/` for the runtime and `tests/test_agent.py` for the
47-case test suite.

### Serve as an HTTP API (OpenAI / Anthropic / Gemini / Ollama compatible)
```bash
python -m ghostlm.agent.server --checkpoint runs/v09chat/best.pt --port 8000
```
Exposes the agent loop over five vendor-compatible endpoint
families plus a native `/v1/agent/run`. Any client that already
targets OpenAI, Anthropic, Google Gemini, or Ollama can point at
the server unchanged. Tool calls happen server-side; the final
cite-tagged answer comes back in whatever shape the SDK expects.
Open `http://localhost:8000/` in a browser for the built-in chat
demo UI (single-page, no JS framework, hits `/v1/agent/run` and
renders the trace inline). Test suite at
`tests/test_agent_server.py` (24 cases).

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

The v1.0 corpus has **768,741 train records / 40,429 val / ~422M tokens** (post-v0.9.32 rebuild, was 516K / 27K / 363M before the open-source code pull):

| Domain | Tokens (M) | Share | Sources |
|---|---:|---:|---|
| Cybersec writeup-style | ~280 | 65% | PRIMUS-Seed/FineWeb (Trend Micro, ODC-BY) — 46.5% of train chars alone, NVD CVE (capped 6M tokens via deterministic-hash subsample), Exploit-DB (GPL-2.0), MITRE ATT&CK / CAPEC / CWE, OWASP family (cheatsheets / WSTG / ASVS / Top 10), CTFtime real writeups, arXiv cs.CR abstracts + full-text, fact-QA (Qwen-14B distilled), CISA KEV, IETF security RFCs |
| Code (open-source, v0.9.31 pull) | ~40 | 9.5% | **105 repos / 26,012 files**: cpython stdlib + numpy/scipy/pandas + sklearn/transformers + Flask/FastAPI/Django (Python, 7,469 files); golang stdlib + gin/cobra/k8s/terraform/docker/caddy (Go, 4,351); rustlang std + tokio/serde/clap/ripgrep/uv (Rust, 4,029); vue/svelte/next/typescript/vite/nestjs (TS, 2,318); express/koa/lodash/react/preact (JS, 1,507); redis/sqlite/curl/openssl/postgres (C, 2,299); protobuf/leveldb/grpc/folly (C++, 1,840); spring/commons-lang/guava (Java, 1,436); rails/sinatra/rspec (Ruby, 461); plus swift/elixir/phoenix. 100% permissive licenses. Per-source totals at [`data/code_corpus_manifest.json`](data/code_corpus_manifest.json). |
| General language | ~46 | 11.0% | `HuggingFaceFW/fineweb-edu` (ODC-BY, classifier-filtered educational web) |
| Math / reasoning | ~21 | 5.0% | `open-web-math/open-web-math` (ODC-BY, math-filtered web) |
| Code (cybersec tools) | ~9 | 2.1% | 30 curated security tool repos (pwntools, impacket, scapy, sqlmap, volatility3, capa, plaso, AFL++, nuclei, trivy, prowler, paramiko, pyca/cryptography, etc.) |
| Authoritative reference | ~3 | 0.6% | 26 NIST SP 800 publications (RMF, controls, identity, IDS, zero trust, secure SDF, etc.); pymupdf-extracted, 12K-char chunks |
| Research-blog register | ~0.6 | 0.1% | 11 RSS/Atom feeds (Project Zero, PortSwigger Research, Trail of Bits, Google Security, GitHub SecurityLab, NCC Group, Doyensec, Krebs, DFIR Report, Ret2 Systems, MSRC) |
| **Total** | **~422** | **100%** | 27 distinct sources (+ 1 new code corpus). **Combined code share: 11.6%** (was 2.4%). |

The pipeline produces a deterministic, leakage-proof split (content-hash bucketing, leakage check returns 0). NVD subsample is reproducible: `python3 scripts/rebuild_corpus.py --max-cve-tokens 6000000` always produces the same 71,828-record CVE prefix from the 333,540-record raw dump. Each new collector is a standalone CLI under `scripts/` (`collect_security_code.py`, `collect_fineweb_edu.py`, `collect_nist_sp800.py`, `collect_security_blogs.py`, `collect_math_reasoning.py`, plus the existing collectors); rebuild auto-globs every `data/raw/*.jsonl`.

For per-source record counts, license posture, and reproducibility commands, see [`CORPUS.md`](CORPUS.md).

### SFT Corpus (chat-tuning)

On top of the pretrain corpus above, GhostLM ships a separate SFT (supervised fine-tune) corpus that teaches the model conversational patterns, tool use, cite-tagged answers, and broad coding chat. Built from hand-curated patterns + deterministic templated synth, every record is parser-clean and reproducible from a single CLI line.

**Cybersec SFT (12 differentiation bets, ~1,940 records):**

| Bet | Records | Eval | Description |
|---|---:|---:|---|
| Bet 1 (tool use) | 424 | n=15 | `<|tool_call|>{json}<|/tool_call|>` 4-message traces (search_cve_nvd / lookup_mitre_technique / lookup_cwe / rag_retrieve) |
| Bet 6 (format-aware) | 560 | n=32 | STIX 2.1 / YARA / Sigma / MISP structured artifacts with two-path validators (real-library + structural fallback) |
| Bet 7 (code-security) | 243 | n=50 | 62 patterns across 11 languages (Python, JavaScript, Java, Go, C, Ruby, PHP, Rust, C#, Swift, Kotlin) with vuln + patch + CWE mapping |
| Bet 8 (binary literacy) | 109 | n=35 | 40 patterns covering file magics (PE/ELF/Mach-O/ZIP/PDF/PNG/JPEG/GIF/MP4/SQLite/DEX/WASM), shellcode, ROP gadgets, hash format recognition, encoding (base64/hex/UTF-8 BOM) |
| Bet 9 (provenance) | 429 | n=15 | `<|cite|>type:id#field<|/cite|>` cite-tagged tool-use traces |
| Bet 10 (log analysis) | 120 | n=25 | 30 patterns across Windows Sysmon / Linux auditbeat / network proxy / DNS / email-gateway logs mapped to 30 ATT&CK techniques |
| Bet 11 (cloud IaC) | 60 | n=15 | 15 patterns across Terraform/AWS + Kubernetes (S3 ACL, IAM trust, security groups, RDS encryption, Pod privileged, NetworkPolicy, RBAC) |
| Bet 12 (protocol fields) | 60 | n=20 | 15 patterns across datalink / network / transport / application layers (TLS 1.3, DNS, HTTP/2, BGP, IP, TCP, Ethernet, ARP, SMB2, Kerberos, QUIC, MQTT, RDP, JA3) |
| Cybersec subtotal | **2,005** | | bets 1+6+7+8+9+10+11+12 |

**Code SFT (broader-than-security, ~1,981 records):**

| Bank | Records | Description |
|---|---:|---|
| Bet 7 code-security (above) | 243 | (also counted in cybersec) |
| Binary literacy (above) | 109 | (also counted in cybersec) |
| `code_explain` templated synth | 975 | 195 patterns × 5 variants (pretrain prose / identify language / explain purpose / walkthrough / concepts) covering algorithms, data structures, idioms, design patterns, web frameworks, databases, testing, concurrency primitives, build systems |
| `code_write` templated synth | 588 | 195 patterns × 3-4 variants (pretrain prose / write function / write idiomatic / compare alternatives) covering everyday coding tasks across Python / JavaScript / Go / Rust / Java |
| Programming Q&A (hand-written) | 66 | 12 topics: Python basics, concepts, JavaScript basics, tooling, Rust basics, code explain, Go basics, debug help, refactor, testing, performance, Java basics |

**Cross-domain chat seeds (~375 records):**

| Bank | Records | Description |
|---|---:|---|
| `small_talk.jsonl` | 153 | Identity, greetings, persona, project context |
| `general_knowledge.jsonl` | 98 | 15 topics: programming, math, science, geography, etymology, uncertainty/refusal, how-to, identity, comparison, definitions, reasoning, history, cross-domain, philosophy, conversation |
| `programming_qa.jsonl` | 66 | (also counted in code SFT) |
| `math_reasoning.jsonl` | 58 | 10 topics: arithmetic, algebra, geometry, word problems, probability, statistics, logic, proofs, combinatorics, concepts |

**Trace distillation:** [`ghostlm/agent/teacher.py`](ghostlm/agent/teacher.py) + [`scripts/distill_agent_traces.py`](scripts/distill_agent_traces.py) generate fresh bet-1+9 traces by driving any OpenAI-compatible teacher (Ollama running Qwen-14B locally, real OpenAI / Anthropic API, vLLM, etc.) through the GhostAgent runtime. Output drops directly into the SFT pipeline.

**Combined-corpus build:** `python3 scripts/build_v15_combined_synth.py` streams every individual synth output, tags each record with its bet number and training-time use (pretrain prose vs SFT Q&A), and writes one unified file. Mix tag is what lets ghost-base's SFT recipe weight bets selectively.

**M4-runnable SFT pipeline (no GPU):** [`scripts/prep_tool_use_sft.py`](scripts/prep_tool_use_sft.py) + [`scripts/finetune_chat.py`](scripts/finetune_chat.py) + [`scripts/eval_agent.py`](scripts/eval_agent.py) form an end-to-end loop: convert templated synth into chat-format records, fine-tune v0.9 chat on top, score against the held-out provenance eval. Wall time: a few hours per pass. Documented in CHANGELOG v0.9.10.

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

The samples below are historical, captured during v0.3.5 / chat-v3 (v0.6.0). They illustrate the "form but not facts" diagnosis that drove the project from a small-corpus parrot toward the v0.9.x bench-winning chat-tunes and the v0.9.11+ agent runtime that pushes factual answers through tool dispatch instead of memorisation. Newer chat-tunes win every MCQ benchmark by 0.7-9.2 pp but the 81M parameter rung still floors on free-form fact recall — the agent loop and ghost-base GPU run are the two complementary fixes for that. These older snapshots stay here as the cleanest demonstration of what the corpus-only path can and cannot produce.

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
│   ├── server.py # HTTP API: OpenAI / Anthropic / Gemini / Ollama
│   ├── teacher.py # OpenAI-compat client: any teacher as a Generator
│   └── web_ui.py # Static HTML demo UI served at GET /
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
├── tests/ # 276 unit tests covering 12 differentiation bets +
│           # GhostAgent runtime (47) + SFT prep (24) + GhostBench
│           # agent runner (10) + HTTP server (24) + distillation (13) +
│           # MCP agent (5) + bet 7 code-security expansion (9) +
│           # bet 8 binary-literacy expansion (11) + general-knowledge
│           # bank (11) + programming-Q&A bank (8) + math-reasoning
│           # bank (9) + code-explain templated synth at 195 patterns
│           # (8) + code-write templated synth at 195 patterns (8)
└── Makefile # One-command workflow
```

---

## Roadmap

GhostLM is a multi-year effort. The honest framing is that ghost-tiny is a learning artifact and a working pipeline, *not* a useful cyber-task model. The path to "useful" is the scale ladder below, paired with a corpus that grows by ~100× from where it is today. See [ROADMAP.md](ROADMAP.md) for full milestones, compute estimates, and corpus targets.

**Where we are (v0.9.32, 2026-05-09):** the ghost-small line saturated at ~28% on debiased CTIBench and 0-2% on free-form fact recall, register-matching parrot, not a fact-knower. v0.9 chat is the bench winner across CTIBench full / in-repo CTF eval / external SecQA but the truth metric is at floor for the whole 81M parameter rung. The bottleneck is generation capacity, not retrieval, and parameter scaling is the answer. **The v1.0 pretrain corpus is built**: 516,736 train / 27,049 val / ~363M tokens across six domains. **The SFT corpus is now ghost-base ready**: ~1,940 records of cybersec SFT across 12 differentiation bets, plus ~1,981 records of code SFT across two new templated-synth banks (code-explain + code-write) that surpass cybersec scale, plus ~375 records of cross-domain chat seeds. **Pretrain code expansion landed**: 120-repo collector pulled 105/120 successfully (4h11m on Mac), then `rebuild_corpus.py` re-merged train/val. Pretrain corpus now **422M tokens, 768K train records, code share 11.6%** (was 2.4%, 4.8x growth) — into the SmolLM2 / Phi training-mix band without losing the cybersec edge (~65% of corpus is still cybersec text). **Ghost-base (~360M)** is the v1.0 target, launcher and spec ready, gated on rented GPU. Strategic frame at [docs/differentiation.md](docs/differentiation.md).

**Infrastructure shipped this push session (v0.9.11 → v0.9.32):**

1. **GhostAgent runtime** ([`ghostlm/agent/`](ghostlm/agent/)). Tool-using loop wrapping any GhostLM checkpoint. Bet-1 tool-call parser, bet-9 cite-tag emission, JSON-serialisable trace, three-state termination (answer_emitted / max_iterations / model_error). 9 cybersec tools (CVE / MITRE / CWE / RAG / CISA KEV / GreyNoise / VirusTotal / Shodan / OTX) with try-real-then-cache backends.
2. **Multi-vendor HTTP server** ([`ghostlm/agent/server.py`](ghostlm/agent/server.py)). Speaks OpenAI Chat Completions, Anthropic Messages, Google Gemini, and Ollama wire formats. Any client SDK targeting one of those drops in unchanged. Static demo UI served at `GET /` so visitors can chat in a browser.
3. **MCP server retrofit** ([`scripts/mcp_server.py`](scripts/mcp_server.py)). New `ghostlm_agent` tool exposes the full agent loop to Claude Desktop / Cursor / any MCP-compatible client.
4. **Trace distillation** ([`ghostlm/agent/teacher.py`](ghostlm/agent/teacher.py)). `OpenAICompatGenerator` lets any OpenAI-compatible teacher (Ollama + Qwen-14B local, real OpenAI / Anthropic / vLLM / etc.) generate fresh bet-1+9 SFT records that drop into the SFT pipeline.
5. **GhostBench agent runner** ([`scripts/ghostbench_agent_run.py`](scripts/ghostbench_agent_run.py)). Scores the agent loop end-to-end across all 7 bet evals with paired-comparison vs no-tools baseline. Wilson CIs, McNemar p-values via existing `python -m ghostbench compare`.
6. **SFT pipeline** ([`scripts/prep_tool_use_sft.py`](scripts/prep_tool_use_sft.py) + [`eval_agent.py`](scripts/eval_agent.py)). M4-runnable end-to-end (synth → prep → fine-tune → eval) without GPU; closes the loop between corpus and trained model.
7. **Code SFT expansion**. Bet 7 grew from 12 patterns (48 records) to 62 patterns / 11 languages (243 records). Bet 8 from 15 to 40 patterns. Two new templated-synth banks (code-explain at 195 patterns / 975 records, code-write at 195 patterns / 588 records) surpass cybersec SFT scale.
8. **Cross-domain chat seeds**. Three new banks: `general_knowledge.jsonl` (98 records, 15 topics), `programming_qa.jsonl` (66 records, 12 topics), `math_reasoning.jsonl` (58 records, 10 topics). Cross-domain SFT floor moved from 0% to ~16% of unique records.
9. **Open-source code corpus collector + landed pull** ([`scripts/collect_code_corpus.py`](scripts/collect_code_corpus.py) + [`data/code_corpus_repos.json`](data/code_corpus_repos.json) + [`data/code_corpus_manifest.json`](data/code_corpus_manifest.json)). 120-repo / 15-language config, permissive-license allowlist, per-repo + per-language caps, sha256 dedup, sidecar manifest, `--append` resume. **Pull executed on Mac (105/120 OK, 26K files, 168M chars). Rebuild folded into train/val: code share 2.4% → 11.6%, train 516K → 768K records / ~422M tokens.**

**What's next (gated on rented GPU compute):**

0. **(Optional) Re-pull failed mega-monorepos via `python3 scripts/collect_code_corpus.py --append`** to recover ~10-15M more tokens from pytorch / nodejs / kafka / etc. Not blocking — the bulk of the value is already on disk.
1. **Ghost-base v1.0 GPU run:** rented H100 hours, 360M params on the 363M-token pretrain corpus + the now-balanced SFT corpus. Acceptance gate: ≥40% CTIBench OR ≥65% CTF eval OR ≥30% on the 50-question fact-recall set. Spec at [docs/ghost_base_spec.md](docs/ghost_base_spec.md).
2. **Run the SFT pipeline on v0.9 chat (M4, no GPU needed):** prep the bet-1+9 traces into chat-SFT shape, fine-tune v0.9 chat on top, score against the provenance eval. Tests whether the agent runtime can be fed a checkpoint that uses it correctly *before* ghost-base lands. Documented in CHANGELOG v0.9.10.
3. **Bet 4 (long context to 16K):** RoPE NTK rebase + 3-5 GPU hours of long-form fine-tune. Unlocks IR triage where a 50K-token threat report goes in the prompt.
4. **Ghost-1b with native MoE from step 0:** 24-layer / 1536-d / 4-expert top-2. Bet 5's preset already in `ghostlm/config.py` so the architecture is settled; the remaining work is the actual pretrain run on owned compute.

**Realistic timeline:** 2-3 years of sustained work to a useful 1B from-scratch cyber LM. The shape of the curve from here is "park at the small-cybersec-LM benchmark plateau OR climb to ghost-base on rented H100s and re-bench." The 12 differentiation bets + agent runtime + code SFT push are the strategic answer to "park is a crowded place." Detailed phase plan in [ROADMAP.md](ROADMAP.md), full multi-year hardware pathway in [docs/hardware_pathway.md](docs/hardware_pathway.md).

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
