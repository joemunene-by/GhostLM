# GhostLM Corpus

The training corpus is the long-term moat for this project. Its size and quality bound everything downstream — no architecture trick will rescue a 14.7M-param model from a 2.7M-token corpus, and no architecture trick will hold back a 1B-param model trained on a properly-curated 20B-token cyber corpus.

This document is the working record of what's currently in the corpus, what's known to be missing, and the licensing constraints that govern what can be added.

---

## Generalist corpus profile (de-specialization)

**Goal (2026-06):** broaden GhostLM from a cybersecurity-only model into a small generalist that keeps cybersecurity as its deepest specialty. Historically the corpus was ~65-73% cybersec text; that is what makes the model a "cybersec parrot" and caps its general usefulness. The fix is a corpus rebalance, not a rewrite: cybersec stays the largest single specialty, but it stops owning the token budget.

**Mechanism.** `data/collect.py` maps every record `source` to a coarse training **domain** (`domain_of`, table `SOURCE_DOMAINS`): `cybersec`, `general_web`, `code`, `math`, `knowledge`, `instruction`, `other`. `rebalance_by_domain(records, domain_token_budgets)` then deterministically hash-subsamples each budgeted domain down to a token cap, the same content-hash subsample used for `--max-cve-tokens`, generalized from one source to a whole domain. Domains without a budget pass through whole. The merge prints a **Domain mix** report (tokens + share per domain) so the realized balance is visible.

**Profiles** (`scripts/rebuild_corpus.py`, `CORPUS_PROFILES`):

| Profile | Effect |
|---|---|
| `cybersec` (default) | No domain caps; legacy behaviour. Use `--max-cve-tokens` alone. |
| `generalist` | Caps `cybersec` below `general_web`; brings code/math/knowledge to real share. The de-specialization lever. |
| `balanced` | Every domain capped to a similar budget for an even mix. |

```bash
# Rebalance an existing raw/ pull into a generalist mix and read the Domain mix report:
python3 scripts/rebuild_corpus.py --profile generalist
# Override a single domain's cap (repeatable, accepts k/m/b suffixes):
python3 scripts/rebuild_corpus.py --profile generalist --domain-budget cybersec=100m
```

Budgets are token *caps*, not exact shares: a domain contributes `min(collected, cap)` tokens, so the achieved mix depends on how much each collector pulled. Run a collection pass first, then rebuild and read the report.

**First realized mix (2026-06-16, preview).** A `--profile generalist` rebuild over the first general-domain pull (FineWeb-Edu 78K, open-web-math 60K, broad Wikipedia 50K) against the on-disk cybersec sources produced 195,850 train / 10,248 val records (leakage check 0):

| Domain | ~Tokens | Share |
|---|---:|---:|
| general_web | 75.7M | 40.0% |
| knowledge | 45.0M | 23.8% |
| math | 45.0M | 23.8% |
| **cybersec** | 23.3M | **12.3%** |
| code | pending pull | — |

The headline: cybersec dropped from ~65-73% of the corpus to a minority domain. This snapshot under-represents cybersec (the large PRIMUS/NVD-full sources weren't on disk; with them present the `cybersec` cap of 120M binds) and excludes code (the `collect_code_corpus.py` pull was still running). The final v1.0 generalist corpus re-runs this once FineWeb-Edu reaches target and the code pull lands.

### Decontamination

Adding general web and Wikipedia raises a real risk: ARC / OpenBookQA questions may appear verbatim in FineWeb-Edu or Wikipedia, which would inflate the new general benchmarks. `scripts/decontaminate.py` fingerprints every benchmark (cybersec MCQ sets + the general rulers) with exact-question and 12-word-shingle detectors, scans the corpus, and can remove offending records (`--write-clean`).

First run over the preview generalist corpus (195,850 records) against all benchmarks (CTIBench-adjacent eval sets, SecQA, the CTF set, fact-recall v1/v2, and the 4,030-question ARC/OpenBookQA general set): **7 contaminated records (0.004%)**, 5 exact + 2 shingle. The corpus is effectively uncontaminated, so the eval numbers reflect capability, not memorization. The final rebuild runs `decontaminate.py --write-clean` to strip those few records.

```bash
python3 scripts/decontaminate.py --include-answers \
  --write-clean data/processed/train_clean.jsonl
```

**New general-domain collectors** feeding the non-cybersec domains:

| Domain | Collector | Source / license |
|---|---|---|
| `general_web` | `scripts/collect_fineweb_edu.py` (target raised to 150K records) | HuggingFaceFW/fineweb-edu, ODC-BY |
| `math` | `scripts/collect_math_reasoning.py` (target 60K records) | open-web-math/open-web-math, ODC-BY |
| `code` | `scripts/collect_code_corpus.py` (120 permissive repos / 15 languages) | per-repo SPDX permissive allowlist |
| `knowledge` | **NEW** `scripts/collect_wikipedia_general.py` | wikimedia/wikipedia (broad, not the cyber BFS slice), CC BY-SA 4.0 |

**Measuring generalist capability.** The eval suite was 100% cybersec (CTIBench, SecQA, in-repo CTF, cybersec fact-recall). `scripts/fetch_general_mcq.py` adds non-cybersec rulers (ARC-Easy, ARC-Challenge, OpenBookQA), and `scripts/eval_text_scoring.py --prompt-style general` scores them with the same debiased multi-permutation text-scoring methodology (records tagged `"domain": "general"` auto-drop the cybersec framing). This is how the pivot is verified rather than assumed.

The SFT persona was reframed in the same pass: `data/raw/chat/small_talk.jsonl` no longer introduces GhostLM as a cybersec-only specialist ("not a general assistant"), and `data/raw/chat/general_knowledge.jsonl` was broadened across history, reasoning, philosophy, science, and cross-domain identity.

---

## v1.0 corpus, post-code-pull rebuild (2026-05-09, v0.9.32: 768,741 train / 40,429 val records, ~422M tokens)

Per-source breakdown of `data/processed/train.jsonl` after the v0.9.31 code corpus was folded in via `scripts/rebuild_corpus.py`:

| Source | Records | Chars (M) | Share | Notes |
|---|---:|---:|---:|---|
| primus_fineweb | 284,874 | 786 | 46.5% | Trend Micro PRIMUS-FineWeb (cybersec writeup-style, ODC-BY) |
| primus_seed | 65,160 | 192 | 11.3% | Trend Micro PRIMUS-Seed |
| fineweb_edu | 47,510 | 185 | 11.0% | HuggingFaceFW/fineweb-edu (educational web, ODC-BY) |
| **code_corpus** | **24,692** | **160** | **9.5%** | **NEW v0.9.31 — 105 permissively-licensed repos / 15 languages / `data/code_corpus_manifest.json`** |
| nvd | 291,788 | 98 | 5.8% | CVE descriptions (deterministic-hash subsample, capped 6M tokens) |
| arxiv_full | 1,880 | 96 | 5.7% | cs.CR full-text |
| math_reasoning | 18,991 | 84 | 5.0% | open-web-math/open-web-math (math-filtered web, ODC-BY) |
| security_code | 6,235 | 35 | 2.1% | 30 cybersec-tool repos (pwntools, impacket, scapy, etc.) |
| exploitdb | 4,711 | 14 | 0.9% | Exploit-DB exploit collection |
| nist_sp800 | 1,001 | 10 | 0.6% | 26 NIST SP 800 publications, pymupdf-extracted |
| Others (synthetic/fact_qa/wikipedia_cyber/arxiv/security_blogs/cwe/owasp_*/mitre_*/rfcs/cisa_kev/capec/ctftime/vendor_research) | ~80K | ~30 | ~1.7% | Long-tail cybersec authoritative + reference + research-blog sources |
| **Total** | **768,741** | **1,689** | **100%** | (+ 40,429 val records, leakage check 0) |

**Code share: 9.5% (code_corpus) + 2.1% (security_code) = 11.6%**, up from 2.4% pre-pull. Lands in the SmolLM2 / Phi / TinyLlama training-mix band. Cybersec text share remains ~65% of total (primus_fineweb + primus_seed + nvd + arxiv_full + security_code + exploitdb + nist_sp800 + small sources).

Dedup during rebuild removed 45,027 cross-source duplicates from the 854K input pool, yielding 809,170 unique records → 95% / 5% deterministic-hash split.

---

## v1.0 corpus (built 2026-05-06, 516,736 train / 27,049 val records, ~363M tokens)

The v1.0 corpus expansion landed: cybersec writeup-style content (the v0.9 substrate) plus three new domains the ghost-small line never saw, code (cybersec tool source), general language (FineWeb-Edu), and math/reasoning (open-web-math). Single rebuild via `scripts/rebuild_corpus.py --max-cve-tokens 6000000`, leakage check returns 0.

| Source | Records | Tokens (M) | Notes |
|---|---:|---:|---|
| primus_fineweb | 284,905 | 196 | TinyBERT-filtered cybersec subset of CommonCrawl |
| primus_seed | 65,160 | 48 | Trend Micro hand-curated cybersec text |
| nvd | 64,559 | 5.5 | NVD CVE descriptions, capped at 6M tokens via deterministic-hash subsample |
| **fineweb_edu** | **47,510** | **46** | **NEW: HuggingFaceFW/fineweb-edu, classifier-filtered educational web** |
| **math_reasoning** | **18,991** | **21** | **NEW: open-web-math/open-web-math, mathematical reasoning** |
| fact_qa | 10,561 | 0.5 | Qwen-14B-distilled cybersec Q&A |
| **security_code** | **6,235** | **8.8** | **NEW: source code from 30 curated cybersec tool repos (pwntools, impacket, scapy, sqlmap, volatility3, capa, AFL++, nuclei, etc.)** |
| exploitdb | 4,711 | 3.6 | Exploit-DB GPL-2.0 PoCs |
| synthetic | 2,847 | 1.4 | Phase-2 synthetic CTF placeholder |
| arxiv | 1,890 | 0.7 | arXiv cs.CR abstracts |
| arxiv_full | 1,880 | 24 | arXiv cs.CR full-text PDFs |
| cisa_kev | 1,526 | 0.2 | CISA Known Exploited Vulns catalog |
| mitre_full | 1,064 | 0.2 | MITRE ATT&CK full bundle |
| **nist_sp800** | **1,001** | **2.6** | **NEW: 26 NIST SP 800 publications, pymupdf-extracted, 12K-char chunks** |
| cwe | 927 | 0.3 | MITRE CWE entries |
| **wikipedia_cyber** | **730** | **1.1** | **NEW (resumed): Wikipedia BFS over cybersec categories** |
| mitre_attack | 655 | 0.2 | MITRE ATT&CK enterprise techniques |
| capec | 563 | 0.1 | MITRE CAPEC attack patterns |
| ctftime | 451 | 0.5 | CTFtime inline writeups |
| **security_blogs** | **199** | **0.6** | **NEW: 11 RSS feeds (Project Zero, PortSwigger, Trail of Bits, etc.)** |
| owasp_wstg | 126 | 0.3 | OWASP Web Security Testing Guide |
| owasp_cheatsheets | 106 | 0.3 | OWASP Cheat Sheet Series |
| owasp_asvs | 75 | 0.04 | OWASP Application Security Verification Standard 5.0 |
| rfcs | 48 | 0.2 | Curated security IETF RFCs |
| owasp_top10 | 15 | 0.04 | OWASP Top 10 (2021) per-category markdown |
| **Total (post-dedup)** | **516,736 train / 27,049 val** | **~363M** | **Six domains: cybersec writeup, code, general language, math, authoritative reference, research-blog register** |

Token share by domain:

- **Cybersec writeup-style** (PRIMUS-Seed/FineWeb, NVD, ExploitDB, MITRE family, OWASP family, RFCs, CTFtime, blogs, NIST, arXiv, fact-QA, etc.): ~265M tokens / 73%. The v0.9 substrate plus the v1.0 reference / blog additions.
- **General language** (FineWeb-Edu): ~46M / 13%. Textbook-style educational web.
- **Math / reasoning** (open-web-math): ~21M / 6%. For chain-of-thought on numeric / logical prompts.
- **Code** (security tool source): ~9M / 2.4%. Cybersec-relevant Python / C / Go / JS.
- **Mixed crawl** (PRIMUS-FineWeb subset is broadly cybersec + general web): the dominant pre-existing block.

This is the corpus ghost-base v1.0 will train on. The diagnostic from v0.9.2 was that 81M params can't bind facts retrievably regardless of how dense the corpus is on a single domain; ghost-base lifts to 360M and the corpus lifts to multi-domain in the same step. Whichever lever was missing should reveal itself in the v1.0 fact-recall numbers.

---

## v1.0 collectors (the run that built the table above)

For ghost-base v1.0 the corpus needs to grow beyond pure cybersec writeups to support coding ability, general language, and authoritative reference recall. Five collectors are pulling concurrently as of 2026-05-06; outputs land at `data/raw/{security_code, fineweb_edu, nist_sp800, security_blogs, wikipedia_cyber}.jsonl`. None are folded into `data/processed/train.jsonl` yet; that happens via the next `rebuild_corpus.py` run.

| Source | Collector | Target | License |
|---|---|---|---|
| Security tool source code | `scripts/collect_security_code.py` + `data/security_code_repos.json` | 30 curated repos (pwntools, impacket, scapy, sqlmap, volatility3, capa, plaso, AFL++, nuclei, trivy, prowler, paramiko, pyca/cryptography, etc.) walked at .py / .c / .h / .cpp / .js / .ts / .go / .rs / .sh, capped at 2000 files per repo | per-repo SPDX (MIT, Apache-2.0, BSD-2/3, GPL-2.0, GPL-3.0, LGPL) |
| FineWeb-Edu | `scripts/collect_fineweb_edu.py` | 50K records of `HuggingFaceFW/fineweb-edu` (sample-10BT split) at edu-score >= 3.0; classifier-filtered educational subset of CommonCrawl | ODC-BY |
| NIST SP 800 | `scripts/collect_nist_sp800.py` | 26 curated SP 800 publications (RMF, controls, incident handling, identity, IDS, pen-test guide, zero trust, secure SDF, etc.); pymupdf text extraction; chunked at 12K chars | US gov public domain |
| Security research blogs | `scripts/collect_security_blogs.py` | 11 RSS/Atom feeds (Project Zero, PortSwigger Research, Trail of Bits, Google Security, GitHub SecurityLab, NCC Group, Doyensec, Krebs, DFIR Report, Ret2 Systems, MSRC); stdlib HTML body extractor strips chrome | per-feed (research / non-commercial use, attributed) |
| Wikipedia cybersec | `scripts/collect_wikipedia_cyber.py` | BFS to depth 2 over `Computer security / Cryptography / Cyberattacks / Network security / Computer security exploits` categories, capped at 2500 articles | CC BY-SA 3.0 |
| **Open-source code corpus (v0.9.31, pulled — 105 / 120 OK)** | `scripts/collect_code_corpus.py` + `data/code_corpus_repos.json` + manifest at `data/code_corpus_manifest.json` (4h11m wall on Mac, 26,012 files / 168M chars / ~42M tokens; per-language: python 7,469 / go 4,351 / rust 4,029 / typescript 2,318 / c 2,299 / cpp 1,840 / javascript 1,507 / java 1,436 / ruby 461 / swift 228 / elixir 74; 15 monorepo timeouts recoverable via `--append`) | **120 curated permissively-licensed repos across 15 languages**: cpython stdlib + numpy/scipy/pandas + sklearn/pytorch/transformers + Flask/FastAPI/Django + requests/httpx/aiohttp + pydantic/sqlalchemy/pytest + rich/textual + Pillow (Python, 35 repos); golang stdlib + gin/echo/cobra/viper/prometheus/k8s/terraform/vault/docker/containerd/caddy/bubbletea (Go, 20); rustlang std + cargo/tokio/axum/serde/clap/hyper/actix-web/rustls/ripgrep/bat/alacritty/deno/wasmtime/ruff/uv (Rust, 21); express/node/koa/lodash/axios/react/preact/svelte/next/typescript/vite/prettier/eslint/jest/tailwind/nestjs (JS+TS, 17); redis/sqlite/curl/openssl/git/postgres/protobuf/leveldb/abseil/grpc/folly/httpd (C/C++, 11); spring/commons-lang/guava/kafka/kotlin/scala (5); rails/sinatra/rspec (Ruby, 3); plus elixir/phoenix/erlang/zig/swift/vapor. Per-repo cap 600 files, content-hash dedup, sidecar manifest with per-source totals. | SPDX allowlist filter: MIT / MIT-0 / MIT-CMU / Apache-2.0 / BSD-2-Clause / BSD-3-Clause / ISC / MPL-2.0 / PSF-2.0 / Unlicense / CC0-1.0 / Zlib / blessing (sqlite) / PostgreSQL. GPL/LGPL/AGPL repos in the config are skipped at run time (override with `--license-allowlist all`). |

Expected size at completion (legacy collectors): ~150-200M additional tokens beyond v0.9. Plus the v0.9.30 code corpus pull adds an estimated **50-150M tokens** depending on per-repo cap, bringing total post-merge corpus into the **460-650M token** range — pretrain code share moves from 2.4% to 12-25%, the rough distribution that small open-source LMs train on (SmolLM2 / TinyLlama / Phi family).

The v1.0 framing is "code + language + cybersec depth" not "more of the same." Ghost-small saturates on register matching at 81M params; ghost-base at 360M is the parameter rung where factual binding and instruction-following are expected to emerge, and that needs corpus diversity not just volume.

---

## Current corpus (v0.9, 273M train tokens — what phase18_v09_pretrain trained on)

The corpus has grown ~30× since the v0.5.0 release. Driven by the diagnosis that 60M tokens of CTF-writeup-heavy text is below the threshold for emergent factual recall on cybersec MCQ benchmarks, v0.9 mixes in open-license cybersec text from PRIMUS, MITRE CWE, OWASP, IETF RFCs, and a Qwen-14B-distilled fact-QA pipeline.

| Source | Records | Type | License | Notes |
|---|---:|---|---|---|
| Primus-FineWeb (Trend Micro AI Lab) | ~300K | Real | ODC-BY | TinyBERT-filtered cybersec subset of CommonCrawl FineWeb (EMNLP 2025) |
| Primus-Seed (Trend Micro AI Lab) | ~85K | Real | MIT-style | Hand-curated cybersec text (security company sites, wikis, MITRE) |
| NVD CVE Database | 71,828 (capped) | Real | US gov, public domain | Capped via `--max-cve-tokens 6000000` |
| Exploit-DB | 5,000 | Real | GPL-2.0 | PHP webapps + Linux locals + Python PoCs, 2019-2025 |
| Synthetic CTF | 3,000 | Synthetic | Self-generated (Ollama) | Phase 2 placeholder |
| arXiv cs.CR Abstracts | 2,000 | Real | arXiv terms | |
| Fact-QA (Qwen-14B distilled) | 11,234 | Real (distilled) | MIT (model output) | Q&A pairs from MITRE / CWE / NVD via Qwen-14B Ollama, v0.8 pipeline |
| MITRE CWE | 969 | Real | MITRE free redistribution | Title + description + extended desc + consequences + mitigations |
| MITRE ATT&CK | 691 | Real | Apache 2.0 | Enterprise techniques |
| CAPEC | 609 | Real | Apache 2.0 | Attack patterns |
| CTFtime Writeups | 467 | Real | User-submitted | Inline body only, per-record attribution |
| OWASP WSTG | 133 | Real | CC BY-SA 4.0 | Web Security Testing Guide markdown |
| OWASP Cheat Sheets | 110 | Real | CC BY-SA 4.0 | Per-topic security guidance |
| OWASP ASVS | 80 | Real | CC BY-SA 4.0 | Application Security Verification Standard 5.0, grouped by section |
| IETF Security RFCs | 48 | Real | IETF (free redistribution) | Curated security RFCs (TLS 1.3, OAuth, JWT, DNSSEC, X.509, IPsec, SSH, ChaCha20, DKIM, etc.) |
| OWASP Top 10 (2021) | 18 | Real | CC BY-SA 4.0 | One record per category + ancillary docs |
| **Total (post-dedup)** | **train 669,085 / val 35,189** | | | **~273M train tokens / ~14.5M val tokens** |

**Rebuild commands:**
```bash
python3 scripts/rebuild_corpus.py
python3 scripts/build_chat_dataset.py
```

The v0.9 corpus is what `checkpoints/phase18_v09_pretrain` is training on. v0.6 / v0.7 / v0.8 trained on smaller subsets of the same source list (everything above except PRIMUS-FineWeb, the OWASP family, and the RFCs).

NVD CVE distribution (full file): 2025: 43,381 · 2024: 38,840 · 2023: 25,198 · 2022: 24,279 · 2021: 22,729. By decade: 1990s: 857 · 2000s: 40,156 · 2010s: 102,581 · 2020s: 189,946. The subsample preserves this year skew because hash-based selection is uniform across the input.

### Collectors

Each new source has a dedicated CLI under `scripts/`. They emit standard `{"id", "source", "text"}` JSONL into `data/raw/`, are resume-safe where the upstream supports it, and are polite about request rate.

| Source | Collector | Notes |
|---|---|---|
| Primus-Seed / Primus-FineWeb | `scripts/collect_primus.py` | Streams from HuggingFace, gated (forms required) |
| Fact-QA | `scripts/build_fact_qa_data.py` | Local Qwen-14B via Ollama `/api/generate`, ~14h on M4 |
| MITRE CWE | `scripts/collect_cwe.py` | Parses the official XML zip dump |
| OWASP Cheat Sheets | `scripts/collect_owasp_cheatsheets.py` | Walks the CheatSheetSeries repo |
| OWASP WSTG | `scripts/collect_owasp_wstg.py` | Walks the wstg/document/ subtree |
| OWASP ASVS | `scripts/collect_owasp_asvs.py` | Pulls the project's signed flat.json release |
| OWASP Top 10 | `scripts/collect_owasp_top10.py` | Per-file via raw.githubusercontent.com (avoids flaky git-clone) |
| IETF RFCs | `scripts/collect_rfcs.py` | Curated list of ~50 security RFCs from rfc-editor.org |
| Wikipedia cybersec | `scripts/collect_wikipedia_cyber.py` | BFS Wikipedia category tree, polite 0.6s delay |

---

## Historical: v0.5.0 corpus (Phase 3.5 endpoint, NVD-subsampled rebalance)

What `checkpoints/phase4_ghost_small/` and the v0.5.0 chat checkpoints were trained on. Preserved here for archaeology and so the v0.4.0/v0.5.0 numbers in CHANGELOG.md remain reproducible.

| Source | Records on disk | After rebuild | Tokens (post-subsample) | Share | Type |
|---|---|---|---|---|---|
| NVD CVE Database | 333,540 | 71,828 | ~5.74M | **65.3%** | Real — capped via `--max-cve-tokens 6000000` |
| Synthetic CTF Writeups | 3,000 | 3,000 | ~1.51M | **17.2%** | Synthetic — placeholder until CTFtime grows |
| arXiv cs.CR Abstracts | 2,000 | 2,000 | ~0.74M | **8.4%** | Real |
| CTFtime Writeups | 473 | 467 | ~0.47M | **5.3%** | Real, inline-only, attributed |
| MITRE ATT&CK | 691 | 691 | ~0.26M | **2.9%** | Real (Apache 2.0) |
| CAPEC | 609 | 609 | ~0.07M | **0.9%** | Real (Apache 2.0) |
| **Total (post-dedup)** | **340,313** | **74,635** | **~8.79M** | | train: 70,965 / val: 3,670 |

This was the structural rebalance corpus: NVD share dropped from 90% (Phase 3) to 65% (Phase 3.5) via deterministic content-hash subsample. Adding Exploit-DB at Phase 3.6 brought the corpus to ~12.56M tokens, which is what `checkpoints/phase4_ghost_small` and every v0.4-base chat-tune trained on.

---

## Phase 2 baseline (v0.3.0, what the released checkpoint was trained on)

| Source | Records | Tokens (approx) | Type | Notes |
|---|---|---|---|---|
| NVD CVE Database | 19,925 | ~1.6M | Real | NVD REST API v2.0, 119-day windows, **per-year cap 500**, 1999–2025 |
| arXiv cs.CR Abstracts | 1,000 | ~0.5M | Real | arXiv Atom API, recent-first by submittedDate descending |
| Synthetic CTF Writeups | 3,000 | ~0.6M | Synthetic | Generated via local LLM (Ollama-based pipeline), varied template + topic mix |
| **Total (post-dedup)** | **23,049** | **~2.66M** | | train: 21,872 / val: 1,177 |

Preserved verbatim because `checkpoints/best_model.pt` was trained on this exact corpus. The per-year cap of 500 was a stopgap — and was masking the fact that `collect_cve_descriptions` only fetched `startIndex=0` of each window. The new `collect_cve_full` paginates properly; both are kept in `data/collect.py`.

---

## Expansion targets

Roughly ordered by leverage (records-per-effort × content-quality × license-friendliness).

### 1. Full NVD dump
- **What:** every CVE record from 1999 to present, properly chunked.
- **Source:** NVD REST API v2.0 with `startIndex` pagination, 119-day windows.
- **License:** US government work, public domain. Free to redistribute.
- **Status:** **done** (2026-04-25). 333,540 records pulled into `data/raw/cve_full.jsonl` via `scripts/collect_nvd_full.py`. Resume-safe; can be re-run to top up with newly published CVEs.

### 2. CTFtime archive
- **What:** real CTF writeups across years and categories. Replaces the current 3,000 synthetic CTF set.
- **Source:** CTFtime task pages + linked writeups via the on-site `id_description` body container. The collector walks `/event/<id>/tasks/` → `/task/<id>` → `/writeup/<id>` for an explicit, config-driven list of events.
- **License:** user-submitted to CTFtime under site terms; this corpus treats them as research-archivable with full per-record attribution (`ctftime_url`, `original_url`, `team`, `event_name`). Off-site links (gitbook / personal blogs) are **not** followed because their licensing posture isn't auditable from a config. Each record carries `license: "ctftime-user-submitted"` so downstream consumers can filter.
- **Status:** **collector ready** — `scripts/collect_ctftime.py` reads a JSON config (see `data/ctftime_events.example.json`) of `{id, name}` events, walks tasks → writeups, parses the inline body, and emits attributed JSONL. Resume-safe (skips writeup IDs already on disk) and polite (default 1 req/sec, configurable). Skips writeups without an inline body. To deploy: edit a config, then `make data-ctftime` (or `python scripts/collect_ctftime.py --config <path>`).

### 3. GitHub CTF writeup repos
- **What:** community-maintained writeup collections (e.g. `ctfs/write-ups-*`, `p4-team/ctf`, etc.).
- **License:** repo-by-repo (mostly MIT or CC-BY). Need to honor per-repo license and attribution.
- **Status:** **collector ready** — `scripts/collect_ctf_repos.py` shallow-clones a JSON-config'd list of repos, walks `*.md` files, and emits JSONL records each tagged with the source repo URL, file path, and SPDX license. The "which repos" decision lives in the config (see `data/ctf_repos.example.json`) so license choices are auditable rather than baked into code. To deploy: edit a config, then `python scripts/collect_ctf_repos.py --config <path>`.

### 4. Security research blogs (curated)
- **What:** primary-source technical blogs from established security research groups.
- **Targets:** Project Zero, PortSwigger Research, Trail of Bits, Google Security, Microsoft Security Response Center, GitHub Security Lab, NCC Group, Doyensec.
- **License:** terms vary by site. Many allow non-commercial redistribution with attribution; some require explicit permission. **Each source needs an individual license check before ingestion.**
- **Status:** wanted; license-gated per source.

### 5. MITRE ATT&CK
- **What:** structured technique data + unstructured procedure / detection text.
- **Source:** MITRE ATT&CK STIX 2.1 enterprise bundle (`raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json`).
- **License:** Apache 2.0 / public — explicitly intended for redistribution.
- **Status:** **collector ready and verified** (2026-04-26). `collect_mitre_attack()` in `data/collect.py` produces ~691 records, ~1 MB JSONL, avg 1,492 chars per record. Skips revoked / deprecated techniques. Run via `make data-mitre`.

### 5b. CAPEC (Common Attack Pattern Enumeration and Classification)
- **What:** structured attack-pattern descriptions complementing MITRE ATT&CK.
- **Source:** MITRE CAPEC STIX bundle (`raw.githubusercontent.com/mitre/cti/master/capec/2.1/stix-capec.json`).
- **License:** Apache 2.0 / public — same redistribution terms as ATT&CK.
- **Status:** **collector ready and verified** (2026-04-26). `collect_capec()` produces ~610 records, ~330 KB JSONL, avg 496 chars per record. Run via `make data-capec`.

### 6. Tool documentation
- **What:** primary docs for security tools — nmap, metasploit, burp, ghidra, pwntools, sqlmap, etc.
- **Source:** official documentation pages and man pages.
- **License:** typically the upstream tool's license (GPL, BSD, etc.); docs usually inherit. Per-source check.
- **Status:** wanted.

### 7. Full-text security research papers
- **What:** beyond abstracts — full-text papers from arXiv cs.CR (already collecting abstracts) and selected open-access venues.
- **Source:** arXiv full-text API for cs.CR; USENIX Security / IEEE S&P / NDSS open-access archives.
- **License:** arXiv allows full-text download; conference papers vary.
- **Status:** wanted at later phases (helpful for ghost-base+).

### 8. Real exploit-DB entries
- **What:** Exploit-DB contains structured exploit metadata + PoC code. Useful for binding CVE descriptions to actual exploitation context.
- **Source:** Exploit-DB CSV + per-entry pages.
- **License:** terms allow research use; redistribution within a training corpus needs explicit attribution.
- **Status:** in-progress — see `scripts/` (Exploit-DB scraper landed in PR #19, pre-rebalance).

---

## Licensing principles

For a published training corpus and downstream model weights, the safe baseline is:

- **Public-domain or permissively-licensed** sources (NVD, MITRE ATT&CK, arXiv) can go in without restriction.
- **CC-BY** sources can go in with proper attribution preserved in dataset metadata.
- **All-rights-reserved** sources are out unless we have explicit permission or rely on fair-use research carve-outs (which we will not lean on for redistributable training data).
- **Synthetic data** generated by another LLM inherits whatever obligations attach to that LLM's terms — for the current 3,000 synthetic CTF records, the local-LLM-generated nature avoids commercial-LLM ToS concerns.

When in doubt, document the source URL, license, and attribution requirement in the per-source ingestion script. Do not ingest "found on the internet" content without checking.

---

## Data quality notes

- **Synthetic CTF share:** dropped from ~13% (v0.3.0 baseline) to ~5% (post-NVD pull) just because the denominator grew. Replacement with real CTFtime / GitHub writeups is still the priority — the absolute count of synthetic records hasn't changed, and they still introduce distribution drift relative to real CTF writing.
- **NVD token-share lopsidedness:** post-pull, NVD is ~90% of training tokens. The diversity collectors (MITRE ATT&CK, CAPEC, CTFtime) move the share by single percentage points each because they're 1–10× smaller than NVD's 27M tokens — diversity sources can't compete on share without subsampling NVD itself. The fix is `scripts/rebuild_corpus.py --max-cve-tokens N` (added 2026-04-26), which deterministically samples NVD records by content-hash prefix until the token budget is reached. Setting `--max-cve-tokens 6000000` brings NVD share to ~67% with the existing diversity sources, which is the actual rebalance the project has been chasing. Subsampling is reproducible (same input → same prefix) so train/val splits stay stable across rebuilds.
- **CVE distribution skew:** strongly weighted toward 2018+. 2020s alone is 189,946 of 333,540 records. This reflects how CVE publication has actually scaled (more software, more disclosure programs) — not a sampling artifact. Not a fix priority.
- **Length skew:** most CVE records are short (p50 ~250 chars / ~62 tokens), most CTF writeups are medium (~2,000 chars / ~500 tokens), arXiv abstracts cluster at ~1,500 chars. CVE p99 is 1,645 chars, max 3,998 — short, factual descriptions dominate.
- **Tokenization:** GPT-2 BPE (50,257 base + 4 special tokens). No domain-adapted tokenizer yet; cyber-specific tokens (CVE-IDs, hex addresses, hashes) get split into multiple sub-tokens, costing context-length efficiency. Not fixing pre-ghost-base.

---

## How to contribute corpus

See [CONTRIBUTING.md](CONTRIBUTING.md). The lowest-friction contributions are:

1. A scraper for one of the wanted sources above, written as a script under `scripts/` that drops `data/raw/<source>.jsonl`.
2. A license-and-licensing audit for a source we haven't formally checked.
3. A deduplication / quality-scoring pass over `data/raw/` that proposes records to drop.

For larger corpus contributions, open an issue first to coordinate scope and licensing.
