# GhostLM Corpus

The training corpus is the long-term moat for this project. Its size and quality bound everything downstream — no architecture trick will rescue a 14.7M-param model from a 2.7M-token corpus, and no architecture trick will hold back a 1B-param model trained on a properly-curated 20B-token cyber corpus.

This document is the working record of what's currently in the corpus, what's known to be missing, and the licensing constraints that govern what can be added.

---

## Current corpus (Phase 3 in progress, post-NVD pull)

After the full NVD pull on 2026-04-25 (`scripts/collect_nvd_full.py`):

| Source | Records | Tokens (approx) | Type | Notes |
|---|---|---|---|---|
| NVD CVE Database | 333,540 | ~27.4M | Real | Full pull, paginated, 1999–2026 (28 years). `data/raw/cve_full.jsonl` |
| arXiv cs.CR Abstracts | 2,000 | ~0.7M | Real | arXiv Atom API, recent-first |
| Synthetic CTF Writeups | 3,000 | ~1.5M | Synthetic | Local-LLM generated; will be replaced by real CTFtime when the scraper lands |
| **Total (post-dedup)** | **~309,000** | **~30M** | | train: ~293,500 / val: ~15,500 |

NVD CVE distribution: 2025: 43,381 · 2024: 38,840 · 2023: 25,198 · 2022: 24,279 · 2021: 22,729. By decade: 1990s: 857 · 2000s: 40,156 · 2010s: 102,581 · 2020s: 189,946. The corpus is heavily weighted toward 2018+ — a real reflection of how CVE publication has scaled, not a sampling artifact.

NVD has 7.9% intra-source duplication (4,635 dup groups, 26,316 extra records) caught by the merge dedup. Remaining cross-source duplication is negligible.

**Status:** raw files collected and merged into `data/processed/{train,val}.jsonl`. **No model has been trained on this corpus yet** — the v0.3.0 ghost-tiny checkpoint was trained on the v0.3.0 baseline below. The next training run (ghost-tiny refresh on the new corpus) will be the first to use this. Split is deterministic by content hash — `scripts/data_audit.py` runs the diagnostics.

This is a **~12× corpus expansion** vs. the v0.3.0 baseline. Token share is now 87% NVD / 5% CTF / 2% papers — the lopsidedness is the case for prioritizing CTFtime + MITRE ATT&CK next so the next-but-one expansion improves diversity, not just CVE volume.

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
- **Source:** CTFtime task pages + linked writeups (rate-limit-aware scraper needed).
- **License:** writeups are typically CC-BY or unspecified — needs per-writeup attribution; redistribution within a research/training corpus is generally accepted but should be documented.
- **Status:** wanted; highest priority on the CTF side.

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
- **Source:** MITRE ATT&CK STIX bundle (`https://attack.mitre.org/`).
- **License:** Apache 2.0 / public — explicitly intended for redistribution.
- **Status:** wanted.

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
- **NVD token-share lopsidedness:** post-pull, NVD is 87% of training tokens. This is the *case* for prioritizing CTFtime + MITRE ATT&CK next — more CVEs alone won't help diversity. ghost-tiny will re-train on this corpus to validate the recipe scales with data, but the next *corpus* track is non-NVD breadth, not deeper NVD.
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
