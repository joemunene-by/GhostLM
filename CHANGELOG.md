# Changelog

All notable changes to GhostLM will be documented in this file.

Format: [Version] — Date — Description

---

## [0.1.0] — 2026-04-06 — Initial Release

### Added
- Decoder-only transformer architecture built from scratch in PyTorch
- CausalSelfAttention with manual scaled dot-product attention and causal masking
- Pre-norm TransformerBlock with residual connections
- FeedForward network with GELU activation
- GhostLMConfig dataclass with three presets: ghost-tiny, ghost-small, ghost-medium
- Weight-tied output projection (lm_head shares weights with token_embedding)
- Scaled residual initialization for stable deep network training
- GhostTokenizer wrapping GPT-2 BPE with 4 custom cybersecurity special tokens
- GhostDataset and build_dataloaders for PyTorch DataLoader integration
- GhostTrainer with cosine LR schedule, linear warmup, gradient clipping
- Checkpoint saving and loading with best_model.pt tracking
- JSON training log persistence
- Data collection pipeline: NVD CVE API, synthetic security papers, CTF writeups
- 10,925 cybersecurity training records (10,378 train / 547 validation)
- scripts/train.py — CLI training entry point with preset and override support
- scripts/generate.py — inference from checkpoint with temperature and top-k sampling
- scripts/evaluate.py — perplexity and generation quality benchmarks
- scripts/benchmark.py — GhostLM vs GPT-2 perplexity comparison
- scripts/chat.py — interactive terminal chat interface
- scripts/plot_training.py — training loss curve visualization
- scripts/push_to_hub.py — HuggingFace Hub upload utility
- notebooks/exploration.ipynb — architecture walkthrough notebook
- GitHub Actions CI workflow (10/10 tests on every push)
- Apache 2.0 license
- MODEL_CARD.md — HuggingFace-style model card
- CONTRIBUTING.md — contributor guide
- Makefile — one-command workflow (make train-tiny, make chat, etc.)

### First Training Run
- ghost-tiny (14.5M params) trained for 500 steps on CPU
- Loss reduced from 10.04 → 6.27 (val_loss)
- CVE language patterns emerged after 500 steps
- Checkpoint saved to checkpoints/best_model.pt

### Known Limitations
- ghost-tiny only trained for 500 steps — not yet useful for real tasks
- Training on CPU is slow (~1.8s/step) — GPU or TPU needed for ghost-small
- Synthetic data used for papers and CTF writeups — real datasets planned

---

## [0.2.0] — 2026-04-09 — Phase 1 Training Complete (10K Steps)

### Training Milestone
- ghost-tiny (14.5M params) trained to 10,000 steps on CPU
- Final training loss: ~1.97
- Final validation loss: ~2.74
- No overfitting observed — stable loss curves throughout

### Evaluation Results
- Cybersecurity perplexity: 2,183.94 (vs GPT-2 baseline: 26.76)
- CVE Severity Classification: 20.0% accuracy
- Vulnerability Type Detection: 10.0% accuracy
- Attack Technique Identification: 10.0% accuracy
- Overall security eval score: 13.3%
- Model generates security domain vocabulary but lacks reasoning capability at this scale

### Architecture
- Simplified model: learned positional embeddings + GELU FFN
- 2 layers, 256 dim, 4 heads, 1024 context length

### Updated
- MODEL_CARD.md with full evaluation results and benchmark comparison
- Training curve plots and benchmark logs

---

## [0.2.1] — 2026-04-22 — Phase 2 Readiness

### Added
- **RoPE (Rotary Position Embeddings)** — config-toggled via `use_rope=True`; replaces learned positional embeddings with the relative-position encoding used by LLaMA / Mistral.
- **Flash Attention** path — config-toggled via `use_flash_attention=True`; routes through PyTorch 2.0+ `scaled_dot_product_attention` for `O(n)` memory.
- **Safetensors export** with `config.json` sidecar and SHA-256 checksum (see `scripts/export.py`). Pickle-free distribution path for HF Hub.

### Changed
- Pinned dependency versions; added PEP 639 license metadata.
- Test suite grown from 10 → 16 tests.

---

## [0.2.2] — 2026-04-23 — Data Audit + Corpus Rebalancing

### Added
- `scripts/data_audit.py` — length percentiles, dedup rate, CVE-year distribution, CTF category share, token share, train/val leakage check. Writes a 4-panel diagnostic chart to `logs/data_audit.png`.

### Changed
- **CVE collector** rewritten to 119-day NVD windows with append mode; coverage extended from 1999–2005 to 1999–2025 (27 years, ~19,925 records).
- **Paper collector** switched from hand-written synthetic `× 50` padding to the arXiv cs.CR Atom API — 1,000 real abstracts.
- **Synthetic CTF generator** emits unique templates only (fixed a rotation bug that limited output to 12 of ~22 templates).
- `merge_datasets` now uses a **deterministic MD5-bucket split** — identical or near-duplicate texts always land in the same split, eliminating the train/val leakage that affected v0.2.0.

### Note
- Previous Phase 1 evaluation numbers (val_loss 2.74, perplexity 2,183.94) were measured on the pre-audit corpus with ~9% train/val leakage. They are preserved in the v0.2.0 entry above for archaeological reference but should be treated as superseded.

---

## [0.3.0] — 2026-04-25 — Phase 2 Training Complete

### Training Milestone
- ghost-tiny (14.7M params) trained for 10,000 steps on the rebalanced corpus.
- Hardware-of-record: Mac Mini M4 (CPU). Training time: ~70 minutes wall-clock.
- Resumed from Phase 1 step-4000 checkpoint pulled from the corpus-prep box and continued on the leakage-free split.
- **Final validation loss: 3.7813** (perplexity ≈ 44) — the first trustworthy held-out measurement of GhostLM.
- Phase 1's lower val_loss (2.74) is preserved in v0.2.0 but not directly comparable: it was measured on a leaky split. Phase 2's number is the honest baseline going forward.

### Added
- Phase 2 corpus: 19,925 NVD CVE records + 1,000 arXiv cs.CR abstracts + 3,000 synthetic CTF writeups (Ollama-pipeline) → 23,049 records / ~2.66M tokens after dedup.
- `checkpoints/best_model.pt` — Phase 2 best (val_loss 3.7813). Phase 1's `best_model.pt` preserved as `checkpoints/best_model_phase1.pt`.
- `checkpoints/checkpoint_step_10000.pt` — final Phase 2 checkpoint.
- `logs/training_log.json` — periodic eval snapshots from the Phase 2 run.
- Sample generations added to MODEL_CARD with honest characterization (vocabulary acquired; semantics absent).
- New `ROADMAP.md` — multi-year scale ladder (ghost-tiny → ghost-small → ghost-base → ghost-1B), corpus targets per rung, compute estimates.
- New `CORPUS.md` — current sources, expansion targets (CTFtime, security blogs, MITRE ATT&CK, tool docs), licensing notes.

### Changed
- License: standardized on **MIT** (LICENSE was MIT, MODEL_CARD/CHANGELOG previously said Apache 2.0 — fixed).
- README & MODEL_CARD updated with grounded Phase 2 numbers, scale ladder, and honest framing.
- CONTRIBUTING.md adds corpus expansion as a first-class contribution track.

---

## [0.3.1] — 2026-04-25 — Phase 2 Evaluation Refresh

### Evaluation Results (Phase 2 checkpoint)
- **Cyber-text perplexity vs GPT-2:** 152.71 (Phase 2) vs 2,183.94 (Phase 1) vs 26.76 (GPT-2 124M baseline) — **14.3× improvement** over Phase 1 on the same hardcoded `BENCHMARK_TEXTS` set; still 5.7× off GPT-2, expected for the params/tokens budget.
- **Security-domain task eval:** 4/30 (13.3%) — same numerical score as Phase 1, but with a different mode-collapse pattern. Phase 2 predicts "High" for every CVE Severity, "Cross-Site Scripting" for every Vuln Type, and "Supply Chain Compromise" for every Attack Technique. The model has learned the most-frequent label per task, not the discriminative structure. Random baseline is ~33%; 13.3% is below random — confirms structured-task evaluation is not yet meaningful at this scale.

### Added
- `logs/benchmark_phase2.json` — Phase 2 GPT-2 perplexity benchmark output.
- `logs/eval_security_phase2.json` — Phase 2 security-task eval output with per-question detail.
- `scripts/plot_phase_comparison.py` — generates `logs/phase_comparison.png` (3-panel: final val_loss, perplexity, security accuracy for Phase 1 vs Phase 2 vs GPT-2).
- MODEL_CARD `Evaluation` section expanded with concrete Phase 2 perplexity and mode-collapse details.

### Note
- Per-step training logs (`logs/training_log.json` and `archive/logs_v1_pre_corpus_fix/training_log.json`) only flushed entries late in training (3 and 5 endpoint datapoints respectively), so a true side-by-side curve plot was not feasible. The phase-comparison plot uses the endpoint metrics that actually exist.

---

## [0.3.2] — 2026-04-25 — Phase 3 NVD-at-scale

Corpus milestone, not a model release. The released checkpoint is still v0.3.0's (val_loss 3.78). The next ghost-tiny refresh run will be the first to train on this corpus.

### Corpus
- **Full NVD pull complete:** 333,540 CVE records, 1999–2026 (28 years), via `scripts/collect_nvd_full.py` with proper `startIndex` pagination.
- **~12× corpus expansion** vs. v0.3.0 baseline (2.66M → ~30M tokens).
- After dedup + merge: ~309K unique records, ~293K train / ~15K val. Deterministic-hash split, leakage 0.
- NVD year skew (expected): 2020s 189,946 / 2010s 102,581 / 2000s 40,156 / 1990s 857. Weighted toward 2018+, reflecting actual CVE publication scaling.
- NVD intra-source duplication: 7.9% (4,635 dup groups, 26,316 extra records) caught by merge dedup.
- **Token-share now lopsided** — NVD 87%, CTF 5%, papers 2%. The next *corpus* track is diversity (CTFtime + MITRE ATT&CK + security research blogs), not deeper NVD.

### Build
- `Makefile` switched to `PYTHON ?= python3` so it works on Mac (no `python` symlink) and Linux without local hacks.

### Docs
- `CORPUS.md`: current corpus snapshot updated to post-pull numbers; v0.3.0 baseline preserved as the corpus the released checkpoint was trained on.
- `ROADMAP.md`: Phase 3 NVD-at-scale marked done; ghost-small gating note updated with current ~30M-token state vs. ~500M-1B target.

---

## [0.3.3] — 2026-04-25 — Phase 3 ghost-tiny Refresh

**This is a model release.** The Phase 3 checkpoint is now canonical at `checkpoints/best_model.pt`. Phase 2 archived as `checkpoints/best_model_phase2.pt`.

### Training Milestone
- ghost-tiny (14.7M params, unchanged) trained from scratch for 30,000 steps on the ~30M-token post-NVD-pull corpus.
- Hardware-of-record: Mac Mini M4 (CPU). Training time: ~3h48m wall-clock at ~2.4 it/s.
- **Final val_loss: 3.4458** (perplexity ≈ 31). 0.34 nat lower than Phase 2's 3.78 — **the recipe scales with data at fixed model size.**
- Curve is monotonic and clean over 60 eval points; first dense ghost-tiny training curve in project history (`logs/phase3_refresh/training_curve.png`).

### Evaluation
- **Cyber-text perplexity vs GPT-2:** 142.09 (Phase 3) vs 152.71 (Phase 2) vs 2,183.94 (Phase 1) vs 26.76 (GPT-2 124M). Modest 7% gain over Phase 2 on this benchmark — most of the perplexity dividend was earned at Phase 2 (corpus quality + clean split); Phase 3's win is more visible on val_loss than on the 10-text benchmark, which already overlaps both corpora. Still 5.3× behind GPT-2.
- **Security-domain task eval:** 4/30 (13.3%) — same numerical score as Phase 2 but with a *different mode-collapse pattern* (Phase 3 predicts "Medium-or-High" / "Cross-Site Scripting" / "DLL Search Order Hijacking" instead of Phase 2's "High" / "XSS" / "Supply Chain Compromise"). CVE-severity task picks up partial discrimination (got 2 right by mixing Mediums). Confirms model is still too small for structured-task eval — the corpus dividend is invisible there until ghost-small.

### Generation Quality
- Phase 3 sample generations now produce **CVE-database register**: phrases like "Cross-Site Request Forgery in all versions up to, and including, 2.2 — this is due to missing nonce validation," "use after free," "remote attacker," "submitting a crafted link" are real CVE language used in roughly the right context. Phase 2's broken-grammar fragments ("the login page is used to the login page's name of the login page") are gone.
- Hallucinations are still rampant — made-up products, scrambled version strings, mixed-up vendors. Form is right; facts are not. Expected outcome of corpus-expansion at fixed model size.

### Added
- `checkpoints/best_model.pt` — Phase 3 best (val_loss 3.4458). Phase 2 preserved as `checkpoints/best_model_phase2.pt`.
- `checkpoints/phase3_refresh/best_model.pt`, `checkpoints/phase3_refresh/checkpoint_step_30000.pt` — same artifact in the run-name'd dir.
- `logs/phase3_refresh/training_log.json` — 60-eval Phase 3 training log.
- `logs/phase3_refresh/training_curve.png` — first real training curve (Phase 1 + Phase 2 logs were too sparse).
- `logs/benchmark_phase3.json` — Phase 3 GPT-2 perplexity benchmark output.
- `logs/eval_security_phase3.json` — Phase 3 security-task eval output.
- `logs/phase_comparison.png` — re-rendered with all three phases populated across val_loss / perplexity / security panels.

### Changed
- README badge: "Phase 2 Complete" → "Phase 3 Complete".
- README + MODEL_CARD: training-data table updated to the post-NVD-pull corpus (~309K records / ~30M tokens). Sample Generations replaced with Phase 3 outputs. Evaluation section reflects Phase 3 numbers.
- ROADMAP: Phase 3 ghost-tiny refresh marked done; Phase 4 (ghost-small) gating updated — recipe-scales-with-data validated, remaining gate is corpus diversity.

### Note
- The Phase 2→3 perplexity gain on the cyber-text benchmark (152.71 → 142.09, 7%) is smaller than the val_loss gain (3.78 → 3.45, 9% / 29% perplexity). This is consistent: the benchmark is 10 hand-picked cyber-text samples that overlapped both corpora, so the residual gain is from volume rather than quality. The val_loss measurement is the cleaner read on whether the recipe scales.

---

## [0.3.5] — 2026-04-26 — Phase 3.5 Corpus Rebalance

### Diversity collectors
- MITRE ATT&CK collector — 691 enterprise techniques pulled from the Apache 2.0 STIX bundle, ~258K tokens. `make data-mitre`.
- CAPEC collector — 609 attack patterns from the CAPEC STIX bundle, ~75K tokens. `make data-capec`.
- CTFtime real-writeup collector — `collect_ctftime_writeups()` walks event → tasks → writeups, extracts the inline body from the `id_description` container, skips off-site redirects (their licensing is not auditable per-event). Resume-safe by writeup_id. Polite default 1 req/sec, configurable. Each record carries full attribution: ctftime_url, original_url, event_id, event_name, task_id, task_name, team, rating, license. `make data-ctftime`.
- Curated CTFtime starter list (`data/ctftime_events.json`) — 28 events 2020–2024, weight ≥ 50, ≥ 50 participants. PlaidCTF, Google CTF, HITCON, ASIS, 0CTF/TCTF, Hack.lu, DEF CON Quals, Real World CTF, plus regional events.
- GitHub-CTF-repos collector (parametric, JSON-config'd) — `collect_ctf_repos()` shallow-clones a list of repos, walks `*.md` files, tags each with SPDX license. Per-repo licensing is auditable rather than baked into code.

### NVD subsampling
- New `scripts/rebuild_corpus.py --max-cve-tokens N` flag — caps the NVD CVE contribution to N tokens via deterministic content-hash prefix (sort by md5(text), take prefix until cumulative chars/4 reaches N). Without the cap, NVD's 27.4M tokens dominate at ~90% share; with `--max-cve-tokens 6000000` NVD share drops to ~65% and real diversity sources hold ~35%.
- Reproducible: same input + same target → byte-identical output. Train/val splits stay stable across rebuilds.
- 6 new tests for the subsample logic in `tests/test_data.py` (cap enforcement, no-op when under budget or no CVEs, determinism, order-independence, end-to-end via merge_datasets).

### Audit fixes
- `scripts/data_audit.py` token-share now computed from processed splits, not raw files. The previous version reported raw chars regardless of subsampling — with the cap applied, the report claimed "NVD 89.9%" while the actual training corpus had NVD at 65.3%. Now shows both columns side-by-side ("raw" / "kept %") when subsampling has materially shrunk a source.
- `scripts/data_audit.py` source-selection now mirrors `rebuild_corpus.py`'s `select_corpus_sources()` so `cve.jsonl` isn't double-counted when `cve_full.jsonl` is present.
- 9 new tests for the CTFtime parser quirks hit on real HTML (entity-encoded titles, team links shadowed by user links, empty rating spans).

### Bumped
- CTFtime collector default `--max-chars` raised from 12000 → 30000. The first scrape audit showed length p90/p95/p99/max all hitting the 12K cap — most real writeups with full exploit transcripts run 15–25K chars and were truncated mid-narrative.

### Corpus state
- Train: 70,965 records · val: 3,670 records · 8.79M tokens total
- Token share: nvd 65.3% · synthetic-ctf 17.2% · arxiv 8.4% · ctftime-real 5.3% · mitre 2.9% · capec 0.9%
- Leakage: 0
- 39/39 data tests pass on Linux; 49/49 (data + model) on Mac

### Workflow
- Direct SSH from Linux dev box to Mac M4 workhorse via `ssh ghostlm-mac` alias (mDNS-resolved, key-authed). Replaces the prior Nemotron-relay loop, which couldn't compose long jobs against a 120s tool timeout. Cross-machine rule still applies — Mac owns long jobs, Linux owns code edits — SSH just removes the email-relay friction.

### Status
- ghost-tiny refresh on the rebalanced corpus: **complete** (run-name `phase3.5_balanced`, 30,000 steps, CPU, ~3h13m wall-clock on M4). The v0.3.3 release artifacts are preserved in `checkpoints/phase3_refresh/`.

### Training results (v0.3.5 ghost-tiny)
- Final val_loss: 3.5518 (vs v0.3.3's 3.4458, +0.106). Note: val sets are not directly comparable across phases because the v0.3.5 val distribution covers six sources vs v0.3.3's NVD-dominated val. The cleaner read is per-source perplexity below.
- Same model, same recipe, same step count as v0.3.3. Smaller training corpus (8.8M vs 26.4M tokens) and more diverse mix.

### Eval results — the rebalance bought what we projected

**Per-source perplexity (val split, 100 records per source, lower is better):**

| Source | v0.3.3 | v0.3.5 | Δ% |
|---|---|---|---|
| mitre_attack | 615.43 | 55.14 | **−91%** |
| ctftime | 184.24 | 60.71 | **−67%** |
| capec | 326.11 | 133.81 | **−59%** |
| synthetic CTF | 67.57 | 28.48 | **−58%** |
| arxiv | 671.09 | 354.95 | **−47%** |
| nvd | 24.19 | 27.55 | +14% |
| **overall** | **171.84** | **66.05** | **−62%** |

The directional shift is exactly what was hypothesized: every diversity source dropped 47–91%, NVD paid a small expected cost (+14%), overall PPL fell 62%. The model went from "knows NVD register, treats everything else as generic English" to "models each domain in proportion to its training share."

The synthetic CTF improvement (−58%) is a free win the rebalance produced without any new data — the same 3K records were used in both phases. Subsampling NVD freed up parameter capacity that v0.3.3 was spending on memorizing duplicate CVE descriptions, and v0.3.5 redirected that capacity onto sources it hadn't fully modeled.

**PMI-corrected security task accuracy (3 tasks, 30 questions, random baseline 15%):**

| Task | v0.3.3 | v0.3.5 |
|---|---|---|
| CVE Severity Classification | 1/10 (10%) | 4/10 (40%) |
| Vulnerability Type Detection | 3/10 (30%) | 4/10 (40%) |
| Attack Technique Identification | 2/10 (20%) | 4/10 (40%) |
| **Overall** | **6/30 (20%)** | **12/30 (40%)** |

Doubled accuracy at fixed model size and (slightly smaller) training data. The previous logp-based scoring reported every phase at 4/30 = 13.3% (below random) — that was the eval being mode-collapsed, not the model failing. PMI scoring (commit aee8008) fixed the eval; this is the first phase where the eval can actually discriminate.

**Cyber-text perplexity vs GPT-2 baseline (fixed external test set, directly comparable across phases):**

| Phase | Perplexity | Δ vs prior |
|---|---|---|
| Phase 1 | 2,183.94 | (baseline) |
| Phase 2 | 152.71 | −93% |
| Phase 3 (v0.3.3) | 142.09 | −7% |
| **Phase 3.5 (v0.3.5)** | **96.24** | **−32%** |
| GPT-2 (117M, frozen baseline) | 26.76 | — |

Phase 3.5 is the largest single-phase perplexity improvement since Phase 1→2. The cyber-text benchmark is 10 hand-picked external samples that overlap none of our training corpus, so the improvement is genuinely from better domain modeling, not from corpus-level memorization shifts.

---

## [0.3.6] — 2026-04-27 — Eval Harness Expansion

Eval-only release. No new training, no new data, no checkpoint changes — just a more honest measurement of the v0.3.5 model so future phases have signal worth trusting.

### Why expand the eval before the next training phase

The v0.3.5 release reported 12/30 (40%) on the security suite. Three tasks × 10 samples is small enough that a swing of three correct answers is worth ten percentage points, which makes it impossible to tell whether a v0.3.6 of 50% is real progress or coin-flip noise. The lesson from Phase 3→3.5 was that the eval matters as much as the corpus — PMI scoring (commit aee8008) was the only reason the rebalance gain was visible at all, since the previous logp scoring sat at the random-baseline floor across every phase. Going into 3-to-6 months of corpus volume work for v0.4.0, the eval needs to be precise enough to register sub-3pp moves and surface mode collapse without relying on lucky sample picks.

### What changed

- Existing three classification tasks expanded from 10 → 25 samples each (75 total). New samples cover label classes the original 10-sample set under-represented: more authenticated/local-vector CVEs, more vuln-type variants on each CWE, more attack-technique scenarios per ATT&CK technique.
- New task: **CTF Challenge Categorization** (25 samples, 5-way: Web Exploitation / Cryptography / Binary Exploitation / Reverse Engineering / Forensics). Maps directly onto the CTFtime corpus that grew in Phase 3.5 and tests whether the model has internalized that taxonomy.
- New task: **MITRE ATT&CK Tactic Classification** (25 samples, 12-way: Initial Access through Impact). Distinct from the existing technique-level task — tactics are the higher-level *why* whereas techniques are the *how*. The MITRE corpus is the source for this concept and the eval should reflect it.
- Total: 5 tasks × 25 samples = 125 evaluations, up from 30. Same PMI scoring engine, no new code paths.
- New `make eval-security` target points at `checkpoints/phase3.5_balanced/best_model.pt` and writes to `logs/eval_security_phase3.5_expanded.json`. Old `*_pmi.json` log files preserved for comparison; the new filename signals that these numbers are not directly comparable to the 30-sample suite.

### What the larger eval reveals about v0.3.5

Run on the same v0.3.5 checkpoint that scored 12/30 (40%) on the small suite:

| Task | Acc | Random | Above-random | Most-common share |
|---|---|---|---|---|
| CVE Severity Classification | 8/25 (32.0%) | 25.0% | +7.0 pp | Critical 72% |
| Vulnerability Type Detection | 8/25 (32.0%) | 10.0% | +22.0 pp | IDOR 44% |
| Attack Technique Identification | 10/25 (40.0%) | 10.0% | +30.0 pp | LatMov 36% |
| CTF Challenge Categorization | 10/25 (40.0%) | 20.0% | +20.0 pp | Forensics 64% |
| MITRE ATT&CK Tactic Classification | 3/25 (12.0%) | 8.3% | +3.7 pp | LatMov 40% |
| **Overall** | **39/125 (31.2%)** | ~14.5% (avg) | **+16.7 pp** | — |

The headline number drops from 40% to 31.2% — that is the eval getting more honest, not the model getting worse. Three things the small suite was hiding:

1. **CVE Severity is mode-collapsing toward "Critical" (72% of predictions).** The 10-sample suite happened to have 4 Critical/High labels matching that bias and scored as if the model had learned severity reasoning. With 25 samples spanning Critical/High/Medium/Low more evenly, the prior is exposed: the model has learned that NVD descriptions usually accompany severe CVEs and bets that way regardless of input.
2. **MITRE Tactics is barely above random (12% vs 8.3% baseline).** Technique identification works (+30 pp above random) because techniques map onto recognizable concrete actions. Tactics are abstract goals (Persistence vs Privilege Escalation vs Defense Evasion can be hard to disambiguate even for humans on a single description) and the model hasn't built that abstraction at 14.7M params on 8.8M tokens. This is the right negative result — it tells us where corpus volume should go (more MITRE tactic-explicit text in v0.4.0) and what to expect to improve at scale.
3. **CTF Categorization scores 100% on Forensics and Cryptography but 0% on Web Exploitation.** Pwn/Reverse split too. The Phase 3.5 corpus has enough crypto challenge writeups for the model to recognize them, but web/binary exploits get conflated with the categories that *visually share their vocabulary* (forensics shares "memory dump", "binary", "extract"; web exploitation shares less unique vocabulary with the others). Useful and actionable signal for v0.4.0 corpus targeting.

The three tasks where the model is meaningfully above random — Vuln Type (+22 pp), Attack Technique (+30 pp), CTF Categorization (+20 pp) — are exactly the tasks where Phase 3.5's corpus rebalance added real domain text. The story holds; the measurement just got finer.

### Cross-phase trajectory on the expanded suite

Every preserved ghost-tiny checkpoint was re-scored on the new 125-sample suite so the trajectory is comparable end-to-end. Cells show `correct/total (accuracy) [most-common-share]`:

| Task | Phase 1 (2K, val 5.19) | Phase 2 (10K, v0.3.0) | Phase 3 (30K, v0.3.3) | Phase 3.5 (30K, v0.3.5) |
|---|---|---|---|---|
| CVE Severity Classification | 7/25 (28.0%) [100%] | 5/25 (20.0%) [96%] | 4/25 (16.0%) [48%] | 8/25 (32.0%) [72%] |
| Vulnerability Type Detection | 3/25 (12.0%) [48%] | 6/25 (24.0%) [76%] | 7/25 (28.0%) [48%] | 8/25 (32.0%) [44%] |
| Attack Technique Identification | 2/25 (8.0%) [24%] | 3/25 (12.0%) [88%] | 5/25 (20.0%) [72%] | 10/25 (40.0%) [36%] |
| CTF Challenge Categorization | 2/25 (8.0%) [84%] | 7/25 (28.0%) [76%] | 6/25 (24.0%) [88%] | 10/25 (40.0%) [64%] |
| MITRE ATT&CK Tactic Classification | 1/25 (4.0%) [72%] | 2/25 (8.0%) [76%] | 3/25 (12.0%) [64%] | 3/25 (12.0%) [40%] |
| **Overall** | **15/125 (12.0%)** | **23/125 (18.4%)** | **25/125 (20.0%)** | **39/125 (31.2%)** |

Reading the trajectory:

- **Phase 2→3 (training volume) bought +1.6 pp overall. Phase 3→3.5 (corpus rebalance, same 30K steps) bought +11.2 pp.** The central thesis of the project — corpus quality outweighs training volume at this scale — now has a clean head-to-head measurement. Tripling training steps from 10K to 30K against an NVD-dominant corpus (Phase 2→3) was nearly free of downstream-task gains; restructuring the corpus at fixed step count (Phase 3→3.5) produced 7× more capability lift.
- **Mode-collapse share declines monotonically across phases on every task except CVE Severity.** Phase 1 picks one label for 100% of severity samples; Phase 3.5 picks "Critical" 72% of the time. The rebalance gave back some severity discrimination — Phase 3's mode-collapse share was 48%, the lowest of any phase — by training on so much NVD that severity signal was rich. Phase 3.5 reduced NVD share 87%→65% and the model lost some of that signal. This is the trade we deliberately accepted: it cost CVE Severity calibration to buy the per-source perplexity drops on MITRE/CTFtime/CAPEC.
- **Attack Technique Identification is the cleanest improvement story:** 8% → 12% → 20% → 40% accuracy with mode-collapse falling 24% → 88% → 72% → 36%. (Phase 1's low collapse share is because the model was barely trained and predictions were near-uniform; from Phase 2 onward each phase reduces collapse.) This is the task most aligned with the Attack Technique-rich corpora that grew across phases.
- **MITRE Tactic Classification is the slowest mover:** 4% → 8% → 12% → 12%. Stuck near the 8.3% random baseline. Tactic-level abstraction does not appear to emerge from corpus changes alone at 14.7M params — this is the v0.4.0 ghost-small canary metric.

The new `make eval-security-all-phases` target re-runs the suite on every preserved checkpoint and prints this table; `make eval-compare-phases` regenerates the table from saved JSONs without re-running. Future ghost-small / ghost-base evals will appear as new columns automatically.

### Practical implications for v0.4.0

- A 3pp move on the new suite represents ~4 correct/incorrect samples, comfortably above noise floor. The previous suite couldn't reliably distinguish two models within 10pp of each other.
- Most-common-share is now reported per task and exposes mode collapse the small suite would have masked. Future runs that score well on accuracy but show >60% most-common-share on any task should be treated as suspect.
- MITRE Tactic accuracy is the canary metric for whether ghost-small actually learns abstraction or just memorizes more text. It is currently 12% vs 8.3% random; if the next training rung does not move this above ~25%, the architecture/scale jump didn't produce reasoning gains and should be diagnosed before further compute.
- **CVE Severity calibration is a Phase 4 acceptance criterion.** Phase 3.5 traded some of it away for diversity — that was the right trade at this rung — but ghost-small needs to recover it. Watch the CVE Severity most-common-share: if it stays above ~60% at the next rung, the model has memorized "Critical is common in NVD" rather than learned to read severity from context.

### Files touched

- `scripts/eval_security.py` — sample lists expanded, new tasks added, `main()` rewired for 5 tasks. Same scoring engine, no behavior change for callers using `--scoring logp` or older checkpoints.
- `scripts/compare_phase_evals.py` — new helper that loads `logs/eval_security_*_expanded.json` and prints the cross-phase comparison table. Used by `make eval-compare-phases` and `make eval-security-all-phases`.
- `Makefile` — eval targets added to `.PHONY` and `help`, including `eval-security-all-phases` (runs on every preserved checkpoint and prints the comparison) and per-phase targets `eval-security-phase{1,2,3}`.
- `logs/eval_security_phase{1,2,3,3.5}_expanded.json` — full per-phase outputs preserved for archaeology and to support the comparison script.

---

## [0.3.7] — 2026-04-28 — Phase 3.6 attempted; ghost-tiny capacity ceiling found

Honest-result release. Two units: collector infrastructure (Exploit-DB,
arXiv full-text scaffolding, CTFtime event discovery, audit + tracker
tooling) plus the Phase 3.6 ghost-tiny training run those collectors
enabled. The training run regressed on the eval suite (31.2% → 16.8%).
Documented end-to-end so the negative result is the artifact: future
phases inherit the lesson that more corpus at fixed model size has hit
diminishing returns at this rung.

The released canonical model stays at v0.3.5
(`checkpoints/phase3.5_balanced/best_model.pt`). Phase 3.6 weights are
preserved at `checkpoints/phase3.6_exploitdb/best_model.pt` for
archaeology and as the cleanest training target for ghost-small —
if/when ghost-small absorbs the same corpus without per-source
regression, the capacity-reallocation hypothesis is confirmed.

### Phase 3.6 corpus-volume work — Exploit-DB landed

**Exploit-DB collector hardened and pulled.** The collector existed but had several problems that made it unsuitable for routine v0.4.0 corpus pulls. Reworked end-to-end and ran a real pull; corpus jumped from 8.79M to ~12.56M tokens (+43%) and NVD share dropped from 65.3% to 45.7% — first time below 50%.

#### Collector changes

- **Persistent local mirror at `data/raw/_exploitdb_mirror`.** The previous version cloned the ~1.5 GB Exploit-DB repository into a tempdir on every run, threw it away, and re-cloned next time. The new version clones once and `git pull --ff-only`s on subsequent runs. A pull failure on an existing mirror is non-fatal — the on-disk snapshot is used and the warning is logged.
- **Resume support.** Re-running against an existing `data/raw/exploitdb.jsonl` loads the existing record ids and only appends new records. Lets long pulls survive interruption.
- **Metasploit-module filter (default on).** Metasploit framework modules carry boilerplate (`include Msf::Exploit::Remote`, `class Metasploit < Msf::Exploit::Local`, etc.) that is repetitive enough to dilute the corpus signal. Path-based detection (`metasploit/` in the file path) plus content-based detection (Msf:: needles in the first ~600 chars) identifies them. 198 modules filtered on the real pull. Pass `--keep-metasploit` to override.
- **Date-descending CSV sort (default on).** The CSV's natural order is by Exploit-DB id (oldest first); without sorting, the first 5K records skewed 73% to legacy ASP / hardware advisories from 2003–2010. Default sort by `date_published` descending pulls the most recent exploits first. The first audit caught this directly — discovered by running `scripts/audit_exploitdb.py` on the unsorted output and seeing `hardware 41.9% / asp 31.0%` and CVE-prefix top of `CVE-2006 12.1%`. Pass `--no-sort` to preserve historical order.
- **Structured metadata per record.** Each record now carries `platform`, `type`, `codes` (CVEs), `language` (file extension), `date`, `license` (`GPL-2.0`) as top-level fields, not just inline header text. Downstream filtering can act on these without re-parsing the body.
- **Truncate vs. drop.** Records longer than `max_chars` (default 12000) are truncated rather than dropped. The header (Exploit-DB id, platform, CVE, date, author) is at the start so it survives the cut.
- **CLI wrapper at `scripts/collect_exploitdb.py`** with `--max-records`, `--min-chars`, `--max-chars`, `--mirror`, `--keep-metasploit`, `--no-sort`. Parity with the other collectors.
- **`make data-exploitdb`** runs the wrapper with defaults.

#### Audit script

- **`scripts/audit_exploitdb.py`** — structural audit complementary to `scripts/data_audit.py`. The latter audits the merged train/val corpus; this one audits the raw `exploitdb.jsonl` so bad-distribution problems can be caught before the merge bakes them in. Reports record count, length percentiles, distribution by platform / type / language / year, CVE coverage with year-prefix breakdown, and license sanity check. Wired up as `make data-exploitdb-audit`.
- This script earned its keep on the very first run: caught the legacy-skew problem in 30 seconds of staring at the output. The collector update was a direct consequence.

#### What the pull contains (5,000 records, post-sort)

| Dimension | Distribution |
|---|---|
| Total | 5,000 records · 15.2 MB chars · ~3.80M tokens |
| Length (chars) | p10=920 · p50=2108 · p90=6538 · p99=12000 |
| By platform (top 5) | php 43.8% · windows 19.4% · multiple 14.5% · hardware 11.1% · linux 3.1% |
| By type | webapps 67.7% · local 15.2% · remote 9.3% · dos 7.3% · hardware 0.5% |
| By language | txt 65.1% · py 27.5% · sh 1.5% · c 1.4% · html 0.9% |
| By date year | 2020 26.3% · 2021 22.6% · 2019 14.9% · 2023 13.7% · 2022 7.9% · 2025 7.4% · 2024 6.3% |
| CVE coverage | 36.4% (1,822 of 5,000 records carry a CVE id) · top years CVE-2019 21.7% · CVE-2020 17.5% · CVE-2021 14.8% · CVE-2023 11.6% |
| License | 100% GPL-2.0 |

Of note:
- **Python share jumped 8.6% → 27.5%** versus the unsorted pull. Modern Exploit-DB submissions are predominantly Python PoCs (vs the old Perl/C era), which is exactly the format the eval revealed the model is weakest at (CTF Web Exploitation 0%, Binary Exploitation poor).
- **PHP webapps 43.8%** dominates by platform. Reasonable read of the recent CVE landscape — PHP web stacks (WordPress plugins, Joomla, etc.) generate continuous CVE flow.
- **CVE coverage is 36.4%, lower than the unsorted sample's 69.8%** because recent EDB entries are often submitted before a CVE is assigned. Older advisories had years to accumulate CVE-id back-references.

#### Corpus state after `make data-rebuild`

| Source | Phase 3.5 share | Phase 3.6 share | Phase 3.6 tokens |
|---|---|---|---|
| nvd | 65.3% | **45.7%** | ~5.74M (cap held at 6M) |
| exploitdb | — | **30.0%** | ~3.77M (new) |
| synthetic CTF | 17.2% | 12.0% | ~1.51M |
| arxiv | 8.4% | 5.9% | ~0.74M |
| ctftime real | 5.3% | 3.7% | ~0.47M |
| mitre_attack | 2.9% | 2.1% | ~0.26M |
| capec | 0.9% | 0.6% | ~0.07M |
| **Total** | **8.79M** | **~12.56M** | **+43%** |

Records: 79,601 total (up from 74,635) · train 75,676 / val 3,925 · 26,356 cross-source duplicates collapsed during merge · leakage check 0.

NVD share crossed below 50% for the first time. The structural rebalance (Phase 3.5 NVD subsample) plus the volume add (Phase 3.6 Exploit-DB) move us materially closer to the 50–100M token v0.4.0 target — still ~37M+ tokens away, but on the right trajectory and entirely from non-NVD sources.

#### Tests

- **Eight new unit tests** (`tests/test_data.py`) covering: `_is_metasploit_module` path-and-content detection, metadata extraction, Metasploit filtering with the keep-flag override, resume across runs, max-chars truncation with header preservation, min-chars dropping, missing-CSV handled as a no-op, and date-desc CSV sort behavior with and without the `sort_by_date_desc` flag. All use a `_stub_exploitdb_mirror` helper that lays out a fake mirror in `tmp_path`, so no network access during the test run. 63/63 tests pass.

#### Training run — ghost-tiny on the Phase 3.6 corpus

Ghost-tiny refresh on the new mix: same recipe as Phase 3.5 (30K steps, batch 2 + grad-accum 4, AdamW with cosine LR), MPS device on the M4 (faster than the 3h13m CPU run). Final val_loss 3.8556, best at step 29000 (3.8555) — converged. Run name `phase3.6_exploitdb`, weights at `checkpoints/phase3.6_exploitdb/best_model.pt`.

#### Eval results — the v0.3.6 expanded suite paid for itself

The headline number, on the same 5×25=125-sample suite that evaluated every prior phase:

| Phase | Overall accuracy | Δ vs prior |
|---|---|---|
| Phase 1 (early, 2K steps) | 12.0% | (baseline) |
| Phase 2 (10K, v0.3.0) | 18.4% | +6.4 pp |
| Phase 3 (30K, v0.3.3) | 20.0% | +1.6 pp |
| Phase 3.5 (30K, rebalanced corpus) | **31.2%** | **+11.2 pp** |
| **Phase 3.6 (30K, +Exploit-DB at fixed model size)** | **16.8%** | **−14.4 pp** |

Per-task breakdown:

| Task | Phase 3.5 | Phase 3.6 | Δ |
|---|---|---|---|
| CVE Severity Classification | 8/25 (32.0%) [72%] | 4/25 (16.0%) [60%] | −16 pp |
| Vulnerability Type Detection | 8/25 (32.0%) [44%] | 3/25 (12.0%) [**96%**] | −20 pp |
| Attack Technique Identification | 10/25 (40.0%) [36%] | 4/25 (16.0%) [60%] | −24 pp |
| CTF Challenge Categorization | 10/25 (40.0%) [64%] | 5/25 (20.0%) [48%] | −20 pp |
| MITRE ATT&CK Tactic Classification | 3/25 (12.0%) [40%] | 5/25 (20.0%) [76%] | +8 pp (mode-collapsed) |

The MITRE Tactic +8 pp is not a real win — most-common-share went from 40% to 76%, meaning the model is now predicting one tactic for 19 of 25 samples and getting some of them right by accident. Vuln Type collapsed even harder: 96% of predictions on a single label, meaning the model picks the same answer for 24/25 samples.

#### Per-source perplexity — the capacity reallocation story

Same val split (100 records per source, deterministic seed) on both checkpoints:

| Source | Phase 3.5 PPL | Phase 3.6 PPL | Δ% |
|---|---|---|---|
| nvd | 27.55 | 35.44 | +28.6% |
| synthetic CTF | 28.48 | 38.90 | +36.6% |
| ctftime | 60.71 | 59.70 | −1.7% |
| mitre_attack | 55.14 | 70.53 | +27.9% |
| capec | 133.81 | 179.71 | +34.3% |
| arxiv | 354.95 | 505.60 | +42.4% |
| **exploitdb** | — (not in train) | **40.87** | **(new, modeled)** |
| **Overall (token-weighted)** | **66.05** | **44.36** | **−32.8%** |

The overall PPL number looks like an improvement (−32.8%) and is **misleading**. Exploit-DB landed in the corpus at 30% token share with low intrinsic perplexity (40.87, second-lowest of any source after NVD), which drags the weighted average down regardless of what happens to the other sources. Per-source the picture is unambiguous: every existing source got 28–42% worse. CTFtime is the lone exception, and it survives flat rather than improves.

#### What this means

Ghost-tiny at 14.7M parameters is at capacity for the diversity the corpus is now asking it to model. Adding Exploit-DB without scaling the model up made the model split its parameters less efficiently across the existing seven sources. The eval suite — built in v0.3.6 specifically because earlier suites couldn't reliably distinguish 10pp-noise from real signal — caught the regression cleanly with mode-collapse share and per-task accuracy moving in the same direction.

This is exactly the kind of finding the project's "trajectory matters more than absolute numbers" framing was set up to surface. Three concrete takeaways:

1. **More corpus at fixed model size has hit diminishing returns at this rung.** Phase 2→3 (3× training volume) bought +1.6 pp; Phase 3→3.5 (corpus rebalance at fixed model+steps) bought +11.2 pp; Phase 3.5→3.6 (corpus volume at fixed model+steps) cost 14.4 pp. The corpus-first thesis from the rebalance does not extend indefinitely — it ran out of headroom inside ghost-tiny.
2. **The path forward is the model, not the data.** ghost-small at 55M params (~3.7× capacity) is the next training rung, with the existing Phase 3.5 corpus or a subsampled Phase 3.6. Adding more corpus at ghost-tiny would be wasted compute.
3. **Token-weighted overall PPL hides per-source regressions.** Every future ROADMAP report needs the per-source breakdown next to the overall — the headline number can move the wrong way for the right reasons (capacity reallocation onto a new heavy source) and look like a win on paper.

#### Decision: roll back the canonical model to v0.3.5; keep Phase 3.6 corpus + checkpoint as a learning artifact

The released "current canonical model" reference in MODEL_CARD.md and README.md stays at v0.3.5 (`checkpoints/phase3.5_balanced/best_model.pt`). Phase 3.6's checkpoint is preserved at `checkpoints/phase3.6_exploitdb/best_model.pt` for archaeology and for future comparisons (if/when ghost-small is trained on the same corpus, we'll see whether the regression goes away with more parameters — that's the cleanest test of the "ghost-tiny is at capacity" hypothesis).

The Phase 3.6 corpus (12.56M tokens with Exploit-DB) stays in `data/processed/` ready for the ghost-small training run. Re-running `make data-rebuild --max-cve-tokens 6000000` reproduces the same 79,601-record corpus from `data/raw/`.

#### `make eval-security-phase3.6` target

Added to `Makefile` so the Phase 3.6 numbers can be reproduced: `make eval-security-phase3.6` runs the 5×25 suite against the preserved checkpoint and writes to `logs/eval_security_phase3.6_expanded.json`. `make eval-security-all-phases` and `make eval-compare-phases` now include Phase 3.6 in the cross-phase comparison.

### Planned for v0.4.0 — ghost-small training rung (now the immediate next move)
- ghost-small (55M params) training on the Phase 3.6 corpus, GPU-required (Mac MPS or rented GPU). The same corpus that broke ghost-tiny is the test case: if ghost-small absorbs it without per-source regression, the capacity-reallocation hypothesis is confirmed and ghost-base/ghost-1B planning is unblocked.
- Continuing corpus expansion toward 50–100M tokens via full-text security papers (arXiv cs.CR PDFs, currently abstract-only — collector landed in v0.3.6 but data not yet pulled), additional CTFtime events (discovery script `scripts/discover_ctftime_events.py` landed but has not been run), and tool docs (pwntools, scapy, impacket via a generic Sphinx scraper, not yet built).
- Drop the synthetic 3K CTF set once real CTFtime + GitHub-CTF-repos + Exploit-DB corpus exceeds it in token volume. Already true under Phase 3.6 (Exploit-DB 30% > synthetic 12%).

### Planned for v1.0.0 — Release
- ghost-small fully trained weights released.
- Public REST API.
- HuggingFace Hub publication (safetensors).

---

## [0.9.2] — 2026-05-06 — Apples-to-apples re-bench; v0.9 wins three MCQ surfaces, free-form fact recall is at floor for everyone

The third revision of the v0.9 story in 24 hours. v0.9.0 said the
ghost-small line was capped at ~30% on CTIBench. v0.9.1 said the
cap was CTIBench-specific because v0.9 led on the in-repo CTF eval.
v0.9.2 closes the loop with three corrections:

1. **The "v0.9 regressed on CTIBench" framing was a sampling
   artifact.** Every prior CTIBench number in this repo (v0.4 30.5%,
   v0.5 29.7%, v0.6 31.2%, v0.7 32.2%, v0.8 31.2%) was scored on a
   500-record subset; only v0.9 (28.9%) was on the full 2500. After
   re-benching every chat-tune on the full set, every other variant
   landed 4-5 pp lower than its n=500 number, and v0.9 leads.

2. **External MCQ confirms the inversion.** SecQA (210q from
   `zefang-liu/secqa` on HF, pulled via `scripts/fetch_secqa.py`)
   reproduces the same v0.9 > v0.7 > v0.4 ordering: 39.3% > 37.6% >
   35.0%. Same direction as the in-repo CTF eval, same direction as
   the corrected CTIBench full bench. Three independent surfaces.

3. **Free-form fact recall is at floor for the whole ghost-small
   line.** A new 50-question hand-written fact-recall set
   (`data/raw/fact_recall_bench.jsonl`, scored by
   `scripts/eval_fact_recall.py` with substring grading) gets 0/50
   on v0.4, 1/50 on v0.7, 1/50 on v0.9 — and both "hits" are
   spurious (v0.7's "Injection" appears in unrelated tangent prose;
   v0.9's "256" comes from echoing "SHA-256" in the question
   itself). The MCQ wins reflect register matching and topic
   distinctness, not facts.

### What landed (scripts + data)

- **`scripts/eval_text_scoring.py`** now records
  `per_perm_per_question` per-question correctness so the
  contamination subset split (`scripts/analyze_contamination_split.py`)
  can run.
- **`scripts/fetch_secqa.py`** — HF puller for SecQA v1 + v2,
  converts to the project's `{question, choices, answer}` JSONL
  schema. Fetcher is the source of truth, the cached JSONL is
  gitignored.
- **`scripts/eval_fact_recall.py`** + **`data/raw/fact_recall_bench.jsonl`**
  — 50 hand-written fact-recall prompts (CVE / CWE / MITRE /
  OWASP / crypto / protocol / misc), free-form chat completion,
  permissive substring grading. JSONL bench is whitelisted in
  `.gitignore` so it travels with the repo.

### Apples-to-apples bench table (v0.9.2 corrected)

| Variant | CTIBench (n=2500) | CTF eval (n=30) | SecQA (n=210) | Fact recall (n=50) |
|---|---:|---:|---:|---:|
| v0.4 chat-v3 (canonical) | 27.6% | 50.0% | 35.0% | 0/50 |
| v0.6 chat | 28.2% | — | — | — |
| v0.7 chat (81M wide) | 27.2% | 50.0% | 37.6% | 1/50 |
| v0.7 chat-ctx1024 | 26.7% | 45.8% | — | — |
| v0.8 chat (fact-dense) | 27.4% | — | — | — |
| **v0.9 chat (273M corpus)** | **28.9%** | **59.2%** | **39.3%** | **1/50** |

v0.9 wins every MCQ bench by 0.7-9.2 pp. Fact recall is at floor
(~0%) for every checkpoint.

### Contamination split landed too

The contamination subset analysis (`scripts/analyze_contamination_split.py`)
on both v0.7 and v0.9 shows -3.0 pp and -2.2 pp deltas respectively
on contaminated questions. Both models regress equally. Those
questions are intrinsically harder for everyone, contamination
isn't the lever.

### What this means for ghost-base

The spec at `docs/ghost_base_spec.md` stands. Acceptance criteria
gain a fact-recall threshold: **≥40% per-perm avg on CTIBench OR
≥65% on the CTF eval OR ≥30% on the 50-question fact-recall set**;
passing any one validates the rung. The fact-recall threshold is
the most important: that's the metric that actually distinguishes
"register-matching parrot" from "model that knows the facts."

### What's superseded

- v0.9.0's "30% real-capability ceiling at the ghost-small rung"
  framing. The apparent ceiling was the n=500 subsetting; the
  real spread on full bench is 26.7-28.9%, and v0.9 is the top.
- v0.9.1's "CTIBench-specific regression" framing — v0.9 didn't
  regress on CTIBench either; it leads.
- The "v0.7 chat is the bench winner" claim everywhere in earlier
  README / MODEL_CARD / ROADMAP. Replaced with v0.9.

The v0.9.0 / v0.9.1 release notes preserve the older framings as
historical record. The README, RESULTS, MODEL_CARD, ROADMAP all
now point at v0.9.2 numbers.

---

## [0.9.1] — 2026-05-06 — Cross-bench validation; the CTIBench ceiling was CTIBench-specific

The hours-after correction to the v0.9.0 release. v0.9.0 read the
six-attempt 28-32% band on debiased CTIBench as a firm ~30%
real-capability ceiling at the ghost-small rung, with v0.9 slightly
regressing from v0.7's 32.2%. The cross-bench validation work
queued in v0.9.0's [Unreleased] section landed the same day and
flipped that diagnosis.

### What landed

- **`scripts/eval_text_scoring.py --bench-jsonl <path>`** — same
  multi-permutation text-scoring methodology, but now accepts any
  MCQ JSONL with `{question, choices: {A,B,C,D}, answer}` records,
  not just CTIBench. CTIBench path unchanged when the flag is unset.
- **`checkpoints/phase20_chat_v07_ctx1024`** — context-extension
  fine-tune of the v0.7 best chat (the prior bench winner), 500
  steps at lr 1e-5 from `phase15_chat_v07/best_model.pt` with
  `--context-length 1024`. Recovers the long-form CTI input range
  the ctx-512 base couldn't handle. Final val_loss 2.6236.

### Result: v0.9 leads on the in-repo CTF MCQ eval (+9 pp vs v0.7)

Ran the same four chat-tunes on the in-repo CTF eval set
(`data/raw/ctf_eval_bench.jsonl`, 30 hand-written cybersec MCQ
questions, debiased text-scoring, 4 permutations):

| Variant | CTIBench (n=2500) | CTF eval (n=30) |
|---|---:|---:|
| v0.4 chat-v3 (canonical) | 30.5% | 50.0% |
| v0.7 chat (81M wide) | **32.2%** | 50.0% |
| v0.7 chat ctx-1024 (extension) | (not benched) | 45.8% |
| **v0.9 chat (273M-token corpus)** | 28.9% | **59.2%** |

The v0.9 → v0.7 ranking flips between the two benchmarks. v0.9 is
the new bench-winner among ghost-small variants when scored on a
non-CTIBench MCQ set, by 9 percentage points.

### What this overturns from v0.9.0

The "30% real-capability ceiling at the ghost-small rung,
regardless of corpus density" framing in v0.9.0 was CTIBench-
specific, not a model property. PRIMUS-FineWeb's TinyBERT-filtered
crawl text appears to shift the model's prior away from CTIBench's
particular threat-intel register (the v0.9 regression on CTIBench
is real and reproducible) but improves performance on a broader
cybersec-MCQ test set covering practical exploitation, web /
crypto / forensics CTF categories, and CWE-style fact recall.

The corpus-density swing worked. CTIBench wasn't the right
yardstick for it.

### Caveats

- 30 questions is a small bench. A 4-point swing is ~5 questions,
  well within noise. Treat the *absolute* CTF eval numbers as
  indicative, not authoritative. The *ranking* (v0.9 > v0.7 > v0.4)
  is consistent and informative.
- The CTF eval was hand-written by the project maintainer, with
  no external validation. Its question topics overlap with the
  v0.9 corpus expansion (CWE / OWASP / RFC content), so part of
  the v0.9 lead may be in-distribution test-set recovery rather
  than capability gain. A larger external bench (CySecBench,
  SecQA, or a CTF MCQ set someone else wrote) is the right next
  move to confirm the inversion.
- The v0.7 ctx-1024 extension (45.8%) is slightly below v0.7
  base (50.0%) on the same bench. Context-extension cost ~4 pp
  on the 30-question set, probably because the fine-tune adapted
  weights toward longer-context patterns at the cost of MCQ-format
  sharpness. Below the noise floor for n=30.

### Implications for ghost-base

The v1.0 spec at `docs/ghost_base_spec.md` still stands, but the
framing shifts. Ghost-base goes from "needed because the ceiling
is real" to "needed to validate that corpus density and parameter
count compound rather than substitute." The acceptance criteria
get a third bench: ≥40% per-perm avg on debiased CTIBench OR
≥65% on the CTF eval OR ≥50% on a hand-written fact-recall
benchmark; passing any one is enough to validate the rung.

### Files touched

- `scripts/eval_text_scoring.py` (--bench-jsonl flag)
- `data/raw/ctf_eval_bench.jsonl` (already in repo from issue #6)
- `checkpoints/phase20_chat_v07_ctx1024/best_model.pt` (new)
- `logs/text_scoring/{v04_chat_v3_ctf, v07_chat_ctf, v09_chat_ctf, v07_ctx1024_chat_ctf}.json`
- README + RESULTS + MODEL_CARD + ROADMAP updated with the dual-bench picture

### Qualitative comparison (follow-up investigation)

`scripts/compare_chat_completions.py` runs a fixed set of five
fact-recall prompts (CVE for EternalBlue, CWE for SQLi, MITRE
technique for LSASS dumping, ChaCha20 vs AES-GCM, SameSite=Strict
vs CSRF) through v0.4 / v0.7 / v0.9 chat with the same chat-format
wrapper. Full transcripts and honest interpretation in
[`docs/v0.9_qualitative_compare.md`](docs/v0.9_qualitative_compare.md).

The two-line read:

1. **v0.9 IS qualitatively the most-fact-aware chat-tune in the
   ghost-small line.** On near-greedy decode it lands on the right
   magic numbers ("CAPEC-89 — SQL injection", wrong framework but
   89 is exactly the correct CWE number for SQLi) and the right
   format ("T1559.001 — credential extraction. Tactic: ..." with
   sub-technique notation and citation structure) where v0.4 and
   v0.7 drift into CTF-writeup tangents without surfacing
   identifiers at all.
2. **None of the three reliably answers a fact-recall question.**
   The 50%+ on the in-repo CTF MCQ bench is real signal from
   text-scoring's preference for option-strings that match topic +
   register, not from the model knowing the answer in any
   actionable sense. v0.9 also shows new repetition pathology
   ("Online Online Online…", "SSL-enabled SSL-enabled…" loops)
   under near-greedy decode that v0.4 and v0.7 don't, plausibly
   from PRIMUS-FineWeb's web-crawl repetition density.

The cross-bench finding stands but is more nuanced than "v0.9 is
smarter." Implications for ghost-base spec: the acceptance gate
needs to measure factual binding directly (free-form fact-recall
with rubric grading), not just MCQ accuracy; PRIMUS-FineWeb shards
should be filtered for repetition before the v1.0 pretrain.

### Contamination audit (follow-up investigation)

`scripts/audit_ctibench_contamination.py` — 8-word-shingle overlap
check between every CTIBench MCQ question (with options) and the v0.9
pretrain corpus (PRIMUS-Seed + PRIMUS-FineWeb, 149M unique shingles).

**Result: 275 / 2500 (11.0%) of CTIBench questions have at least one
shingle overlap with PRIMUS.** Avg per-question overlap is 0.92% of
shingles. Worst offender hits 40/50 shingles (80%) on a question
about phishing-bot camouflage; the top-10 list is dominated by
MITRE ATT&CK technique descriptions (T1037.004 Login Hook, T1087.003,
T1555.004), NTFS Alternate Data Streams, OPC server descriptions,
and a GDPR personal-data-processing question.

**Interpretation:** the 11% rate is real but mostly reflects
*shared source material*, not test-set leakage. Both CTIBench and
PRIMUS-Seed draw from the same public corpora (MITRE / OWASP / NIST
docs), so shared phrases are expected and don't constitute v0.9
"reading the answer" during pretrain. If contamination actually
*helped* v0.9 on CTIBench, we'd expect a gain over v0.7; instead
v0.9 regressed. The honest read is one of:

1. Contamination is a wash on CTIBench (the model sees the source
   material but still has to do the question→answer mapping, which
   it does about as well as any chat-tune).
2. Contamination actively *confuses* the model (half-remembered
   phrasings from training are nearby in option-space, biasing
   the score-by-text rule toward the wrong option in CTIBench's
   particular framing).
3. The CTIBench regression is genuinely about register-shift from
   PRIMUS-FineWeb (the unique-to-v0.9 source), not contamination
   from PRIMUS-Seed (which v0.7's corpus didn't share but v0.9's
   FineWeb didn't either).

**Subset split landed (`scripts/analyze_contamination_split.py`):**
re-ran the CTIBench bench on v0.9 chat with per-question correctness
saved (`--out-json` now records `per_perm_per_question`), then
cross-referenced with the audit's `per_question` overlap data.

| subset | n     | per-perm avg | delta vs clean |
|--------|------:|-------------:|---------------:|
| clean (no shingle overlap) | 2225 | 0.2915 | (baseline)    |
| contaminated (>=1 overlap) |  275 | 0.2691 | **-2.2 pp**    |

v0.9 does *worse* on the questions where it saw the source material
during pretrain. If contamination were helping, we would see a
positive delta; instead it is meaningfully negative. Combined with
the cross-bench CTF result (where v0.9 leads), this rules out the
contamination-helps hypothesis and keeps the most likely explanation
the *register-shift* story: PRIMUS-FineWeb's TinyBERT-filtered
crawl text shifts v0.9's prior away from CTIBench's particular
threat-intel framing on the overlapping questions, plausibly because
half-remembered training-corpus phrasings bias the score-by-text
rule toward the wrong distractor when the right answer's phrasing is
slightly different from the one v0.9 absorbed during pretrain. The
clean-subset 29.2% is the cleaner read of v0.9's CTIBench capability
on questions with no PRIMUS overlap.

The ghost-base story is unchanged: parameter count remains the next
lever to pull.

`logs/ctibench_contamination.json` has the full per-question
overlap counts and the top-N list; `scripts/audit_ctibench_contamination.py`
reproduces from any corpus JSONL.

---

## [0.9.0] — 2026-05-06 — Corpus-density attempt; the 30% ceiling is firm at the ghost-small rung

The end of the ghost-small (81M) line. v0.7 ruled out parameter count
as the bottleneck below 81M; v0.8 ruled out fact-density via Qwen-14B
distillation; v0.9 was the corpus-density swing, a 4× expansion of
the pretrain corpus. It also failed to break the ceiling, and
slightly regressed.

### What landed

- **273M-token corpus** (vs ~60M for v0.6/v0.7/v0.8). Sources mixed
  in: Trend Micro PRIMUS-Seed (~85K hand-curated cybersec records,
  EMNLP 2025), Primus-FineWeb (~300K TinyBERT-filtered cybersec
  CommonCrawl pages), MITRE CWE (969 weakness records with
  consequences and mitigations), the OWASP family (cheatsheets 110,
  WSTG 133, ASVS 80, Top 10 18), 48 IETF security RFCs (TLS 1.3 RFC
  8446, OAuth 2.0 RFC 6749, JWT RFC 7519, DNSSEC RFC 4033, IKEv2 RFC
  7296, X.509 RFC 5280, ChaCha20+Poly1305 RFC 8439, EdDSA, DKIM,
  SPF, DMARC, etc.), plus the v0.8 fact-QA. Train: 669,085 records
  / ~273M tokens. Val: 35,189 records.
- **Phase 18 pretrain (`phase18_v09_pretrain`)** — same v0.7
  architecture (81M wide, RoPE + SwiGLU + RMSNorm), from-scratch on
  the 273M-token corpus. 15K steps, ~12h on M4 across two crashed
  resumes (disk-full, ModuleNotFoundError) before completing
  cleanly. Final val_loss 3.638. Note: not directly comparable to
  v0.7 (3.17) or v0.8 (3.56) since v0.9's val set is drawn from the
  same expanded corpus and is much more diverse per token.
- **Phase 19 chat (`phase19_chat_v09`)** — canonical chat-v3 SFT
  recipe (lr 3e-5, 1800 steps, batch 8 × accum 4, ctx 512). Final
  val_loss 2.802.

### Result: 28.9% per-perm avg on the full 2500-q debiased CTIBench

`logs/text_scoring/chat-v09.json`: 28.7% / 29.1% on the two
permutations, 28.9% averaged. **Below v0.7's 32.2% and the prior
29-32% band.** Prediction distributions are clean (no letter
collapse), so the regression is genuine, not an artifact.

The most likely explanation: PRIMUS-FineWeb's TinyBERT-filtered
crawl text dilutes the cyber-text register that the smaller, more
focused corpus of v0.6/v0.7 had concentrated. The model is sharper
on general cybersec prose but loses the MCQ-format completion
sharpness that scored well on CTIBench.

### Six attempts at the ghost-small rung, all in 28-32%

| Variant | Pretrain | Recipe | Debiased per-perm avg |
|---|---|---|---:|
| v0.4 chat-v3 | ~12.6M tokens, learned PE / GELU / LayerNorm | MCQ-tuned | 30.5% |
| v0.5 chat-v5 | ~60M, custom 32K BPE, RoPE/SwiGLU/RMSNorm | hybrid raw + CoT | 29.7% |
| v0.6 chat | ~60M, GPT-2 50K BPE, RoPE/SwiGLU/RMSNorm | canonical chat-v3 | 31.2% |
| v0.7 chat (best) | ~60M, 81M wide | canonical chat-v3 | **32.2%** |
| v0.8 chat | ~60M + 11K Qwen-14B fact-QA | canonical chat-v3 | 31.2% |
| **v0.9 chat** | **~273M PRIMUS + CWE + OWASP + RFC + fact-QA** | canonical chat-v3 | **28.9%** |

Three architectural axes (BPE, positional encoding + FFN +
normalization, parameter count up to 81M), one SFT-objective axis
(letter-loss vs text-loss), and two corpus-density axes (60M with
fact-QA, 273M with PRIMUS+OWASP+RFC) have all been ablated to within
this band. The ~30% real-capability ceiling at this rung is firm.

### Diagnosis: 81M is below the threshold for emergent factual recall

Live testing on every variant exhibits the same pattern:
register-correct prose, factually wrong content. EternalBlue gets a
wrong CVE; MITRE technique IDs get conflated; CVE-to-CWE mappings
hallucinate. The model is a "cybersec parrot" at this scale,
regardless of how much factual content it sees during pretrain.

The pattern matches the literature: SmolLM2-360M and Phi-3.5-mini
both report factual-recall capability emerging in the 300M-400M
parameter range. **Ghost-base (~350M, 12L × 768d) is the next
rung**, gated on rented GPU compute.

### What ships at v0.9.0

- All v0.6-v0.9 checkpoints (best_model.pt only) preserved on disk.
  The canonical chat model for the ghost-small rung remains v0.7
  (`checkpoints/phase15_chat_v07/best_model.pt`), the bench winner.
- Ten new corpus collectors (`scripts/collect_primus.py`,
  `collect_cwe.py`, `collect_owasp_*.py`, `collect_rfcs.py`,
  `collect_wikipedia_cyber.py`).
- Updated docs across README / CHANGELOG / RESULTS / CORPUS /
  MODEL_CARD / ROADMAP reflecting the ceiling diagnosis.

### What v1.0 (next major) needs

- Ghost-base (~350M) trained on rented GPU compute. Same v0.7 arch
  scaled up. Same v0.9 corpus or its successor.
- The ghost-base eval needs to clear ~40% per-perm avg on debiased
  CTIBench to validate that the bottleneck was indeed parameter
  count and not something else (eval methodology, recipe, etc.).
- If ghost-base also stalls at 28-32%, the diagnosis flips again
  and we need to look at the eval itself rather than the model.

---

## [0.8.0] — 2026-05-05 — Fact-dense pretrain via Qwen-14B distillation; ceiling holds

The fact-density attempt at the 30% CTIBench ceiling. v0.7 had ruled out
"the model is too small": 81M params at v0.7 wide hit 32.2% on debiased
text-scoring, identical to every smaller variant. The remaining
hypothesis was data density: a corpus dominated by CTF writeups doesn't
contain the structured fact lookups CTIBench tests. v0.8 attacks that
directly.

### What landed

- **`scripts/build_fact_qa_data.py`** — overnight pipeline that pulls
  MITRE ATT&CK / CISA KEV / CWE / NVD descriptions and prompts a local
  Qwen-14B (via Ollama `/api/generate`) to extract concrete factual Q&A
  pairs (e.g. `Q: What CWE category is CVE-2021-44228? A: CWE-502`).
  Resume-safe (id-keyed), 14h on M4. Produced 11,234 records in
  `data/raw/fact_qa.jsonl`.
- **Phase 16 pretrain (`phase16_v08_pretrain`)** — same v0.7 architecture
  (81M wide, RoPE + SwiGLU + RMSNorm), pretrained from scratch on the
  v0.7 corpus + the new fact-QA records. 15K steps, ~6h on M4.
- **Phase 17 chat (`phase17_chat_v08`)** — canonical chat-v3 recipe SFT
  on the v0.8 base. Best checkpoint at step 1800, val_loss 2.60.

### Result: 31.2% per-perm avg on debiased text-scoring

`logs/text_scoring/chat-v08.json`: 31.0% / 31.4% on the two
permutations, 31.2% averaged. **0 pp improvement over v0.7.** Adding
~11K Qwen-distilled Q&A pairs to a 60M-token CTF-writeup-heavy corpus
moved the bench by less than the noise floor.

The ceiling holds. Five independent attempts now sit between 29-32% on
debiased CTIBench: v0.4 base (30.5%), v0.5 base (29.7%), v0.5 chat-text
text-loss (30.1%), v0.6 BPE-swap (31.2%), v0.7 wide (32.2%),
**v0.8 fact-dense (31.2%)**. The diagnosis is firm: the model is
interpolating between memorized writeup patterns, not doing structured
factual recall. Distilled Q&A doesn't fix the underlying corpus
density problem because 11K records in a ~60M-token corpus is a 0.2%
share, not enough to shift the model's prior toward fact lookup.

### What v0.9 will test (in progress)

Drop the distillation crutch, expand the corpus 4× by mixing in real
open-license cybersec text: Trend Micro's PRIMUS dataset (EMNLP 2025,
~85K Seed + ~300K FineWeb records), MITRE CWE (969), OWASP cheatsheets
(110), OWASP WSTG (133), OWASP ASVS (80), OWASP Top 10 (18), 48 IETF
security RFCs, plus the v0.8 fact-QA. New corpus is 273M train tokens
(4× v0.6/v0.7). If the ceiling still holds, the diagnosis is firm at
"81M params is below the threshold for emergent factual recall, and
the next move is the ghost-base (~350M) rung."

### Files touched

- `scripts/build_fact_qa_data.py` (new)
- `scripts/collect_primus.py`, `scripts/collect_cwe.py`,
  `scripts/collect_owasp_*.py`, `scripts/collect_rfcs.py`,
  `scripts/collect_wikipedia_cyber.py` (new corpus collectors)
- `checkpoints/phase16_v08_pretrain/`, `checkpoints/phase17_chat_v08/`
  (training artifacts; intermediate step ckpts cleaned, final + best
  retained)
- `logs/text_scoring/chat-v08.json` (debiased eval result)

---

## [0.7.0] — 2026-05-04 — 81M wide variant; param-count ablation against the ceiling

The capacity ablation. v0.4 / v0.5 / v0.6 all sat at 29-32% on debiased
CTIBench despite different architectures, BPEs, and corpora. v0.7 keeps
v0.6's recipe and corpus but widens the model to 81M params (6L × 768d,
d_ff 3072, 12 heads). If the ceiling is a parameter-count limit at 45M,
nearly doubling capacity should move the bench.

### What landed

- **`scripts/train_v07.py`** — wider variant launcher. Same v0.5
  architecture (RoPE + SwiGLU + RMSNorm) and GPT-2 50K BPE as v0.6, but
  `d_model=768`, `d_ff=3072`, `n_layers=6`, `n_heads=12` → 81.1M params.
  Resume-safe via `--resume <ckpt>`.
- **Phase 14 pretrain (`phase14_v07_pretrain_v3`)** — from-scratch on
  the v0.6 expanded corpus, 15K steps, ~7h on M4. Two earlier attempts
  (v0.7_pretrain, v0.7_pretrain_v2) crashed mid-run from MPS contention
  with concurrent fact-QA generation; v3 is the clean run.
- **Phase 15 chat (`phase15_chat_v07`)** — canonical chat-v3 SFT recipe.
  OOM-killed mid-training at step 700; step 600 checkpoint loaded as
  best_model.pt.

### Result: 32.2% per-perm avg, +1 pp over v0.6

`logs/text_scoring/chat-v07.json`: 31.2% / 33.2% on the two
permutations, 32.2% averaged. The single best debiased CTIBench score
in the project, but inside the existing 29-32% noise band, not a real
break of the ceiling.

The ablation confirms what the bias finding implied: param-count alone
isn't the bottleneck at this rung. From v0.4 (45M) to v0.7 (81M, 1.8×
params) the bench moved 1.7 pp. Live testing on v0.7 still shows the
same factual gaps (wrong CVE bindings, conflated MITRE technique IDs)
as v0.4. The model is a smarter cybersec parrot, not a cybersec
expert.

### What v0.8 will test (next)

Targeted fact-density injection via a Qwen-14B-distilled Q&A pipeline,
keeping the v0.7 81M architecture fixed. If facts injected as direct
Q&A pairs move the bench, the bottleneck is data type (we need
fact-lookup format, not just writeup register). If they don't, the
bottleneck is data volume.

### Files touched

- `scripts/train_v07.py` (new)
- `checkpoints/phase14_v07_pretrain_v3/`, `checkpoints/phase15_chat_v07/`
- `logs/text_scoring/chat-v07.json`

---

## [0.6.0] — 2026-05-03 — CTIBench bias artifact discovered; debiased eval ships

The methodology release. Live testing of the v0.5.0 canonical chat-v3
exposed that the model "knows" CTIBench answers as a position bias
rather than as content reasoning: it emits "C" on 98.6% of questions,
and CTIBench's gold-letter distribution is 15/32/37/15 (A/B/C/D), so a
model that always emits "C" scores 37.1% on the v0.5.0 single-order
metric. That's what 36.9% chat-v3 was actually doing.

### Bias-finding investigation

`docs/ctibench_bias_finding.md` documents the full diagnosis: per-letter
emission distribution per checkpoint, gold-letter distribution check
on CTIBench, and what numerical headlines this overturns. The bias
artifact is intrinsic to the v0.5.0 eval, not specific to chat-v3.

### Two debiased eval scripts

- **`scripts/eval_debiased.py`** — multi-permutation letter scoring.
  Scores log-prob of each option letter under N option-letter
  orderings (default A,B,C,D + C,B,D,A) and reports the mean per-perm
  accuracy plus per-letter prediction distributions.
- **`scripts/eval_text_scoring.py`** — skips the letter token entirely.
  Scores log P(option_text | prompt) per option, length-normalized,
  under the same multi-permutation scheme. The cleanest read of real
  capability: a single-letter emitter collapses to 25% (random).

Both write JSON outputs to `logs/debiased/` and `logs/text_scoring/`.

### Re-scored every checkpoint in the project

| Checkpoint | Single-order (biased) | Text per-perm avg (real) | Latched letter |
|---|---:|---:|---|
| `phase5_chat_v3` (v0.5.0 canonical) | 36.9% | **30.5%** | C (98.6%) |
| `phase5_chat_v3_repro2` | 31.2% | 31.7% | B/C dual |
| `phase8_chat_v05_v5` (v0.5 base) | 34.8% | 29.7% | C (79.6%) |
| `phase10_chat_v06` (v0.6 BPE-swap) | 29.8% | 31.2% | B (86.2%) |
| `phase13_chat_text` (text-loss SFT) | 19.6% | 30.1% | mixed |

Every chat-tune sits in a 29-32% per-perm-avg band. ~5-7 points of
real signal above 25% random, not the 12+ that single-order suggested.

### v0.6 base + chat: BPE-swap ablation

The v0.5.0 canonical was on the v0.4 base (custom 32K BPE). v0.6 is the
v0.5 architecture (RoPE + SwiGLU + RMSNorm) plus the GPT-2 50K BPE,
trained from scratch on the v0.4.2-expanded corpus.

- **Phase 9 pretrain (`phase9_v06_pretrain`)** — 15K steps from-scratch
  on the v0.4.2 corpus (~60M tokens including +MITRE-full and +CISA-KEV).
- **Phase 10 chat (`phase10_chat_v06`)** — canonical chat-v3 recipe SFT.
  31.2% per-perm avg, on par with the band.

The BPE swap doesn't move the ceiling. Combined with the v0.4-vs-v0.5
arch comparison from v0.5.0, three architectural axes (BPE size,
positional encoding, FFN, normalization) have all been ablated to
within the 29-32% band.

### Live testing reveals the "cybersec parrot" diagnosis

Free-form generation on v0.5.0 chat-v3, v0.5 chat, v0.6 chat all
exhibit the same pattern: register-correct prose, factually wrong
content. EternalBlue gets a wrong CVE; MITRE technique IDs get
conflated; CVE-to-CWE mappings hallucinate. The model has internalized
the *shape* of CTI writing, not the *facts*.

Five independent AI sources (ChatGPT, Gemini, separate Claude
sessions, local Qwen reasoning chain, internal benchmarks against
SmolLM2) converged on the same diagnosis: at 60M tokens of
CTF-writeup-heavy text, 45-81M params has enough capacity to model
the language but not enough density to hold the facts.

### Files touched

- `scripts/eval_debiased.py`, `scripts/eval_text_scoring.py` (new)
- `docs/ctibench_bias_finding.md` (new)
- `README.md` — debiased numbers added to chat-tuning section, single-
  order numbers preserved, em-dash separators stripped (70→0)
- `RESULTS.md` — debiased text-scoring table added below single-order
- `checkpoints/phase9_v06_pretrain/`, `checkpoints/phase10_chat_v06/`,
  `checkpoints/phase11_chat_v06_hybrid/`, `checkpoints/phase13_chat_text/`
  (training artifacts)
- `logs/debiased/*.json`, `logs/text_scoring/*.json` (per-checkpoint
  debiased outputs preserved)

---

## [0.5.0] — 2026-05-01 — Chat tuning, MCQ, RAG, and v0.5 architecture wiring

The first chat-tunable, benchmark-scoreable rung of the project. Phase 4
ghost-small (the v0.4.0 base model below) is now the substrate for a
proper supervised fine-tune that turns it into a conversational
cybersecurity assistant — and the first GhostLM model with a credible
public benchmark number.

### chat-v3 — 36.9% on CTIBench MCQ (2500 questions)

The headline result of v0.5. Three iterations of chat tuning landed on
top of the Phase 4 base, scored on the full 2500-question CTIBench
multiple-choice benchmark:

| Checkpoint | n | Correct | Accuracy |
|---|---:|---:|---:|
| `phase4_ghost_small` (pretrain only, no chat) | 2500 | 446 | **17.8%** |
| `phase5_chat_v2` (free-form SFT, small-talk-balanced) | 2500 | 475 | **19.0%** |
| `phase5_chat_v2 + RAG(top4)` | 2500 | 476 | **19.0%** |
| **`phase5_chat_v3` (MCQ-format SFT)** | **2500** | **922** | **36.9%** |

v3 lifts +17.9 pp over v2 — **+447 questions correct** — and lands at
**1.48× random** (random baseline 25.0% on 4-way MCQ). The model is
still 45M params trained on 12.56M cybersec tokens; only the
fine-tuning data distribution changed for v3. RESULTS.md tracks the
table going forward; `scripts/run_bench.py` regenerates rows.

Honest framing: 36.9% is well above random and a respectable result
for a 45M from-scratch model, but well below frontier-LLM scores
(85-95%). The gap to "useful" is the v0.4.2 corpus expansion + the
v0.5 from-scratch retrain (architecture switches below) + a more
demanding chat dataset, not a clever inference-time trick.

### Chat-tune pipeline (commits `4219637`, `b67987d`)

Built from scratch, no `transformers.Trainer` dependency:

- **3 new role tokens** appended to the GhostTokenizer (vocab
  50,261 → 50,264): `<|ghost_user|>`, `<|ghost_assistant|>`,
  `<|ghost_end|>`. Phase 4 weights load by expanding the token
  embedding three rows and re-tying `lm_head`.
- **Assistant-only loss masking** — `GhostTokenizer.encode_chat()`
  emits both token ids and a per-token mask; the trainer fills
  non-assistant target tokens with `-1` so the existing
  `cross_entropy(..., ignore_index=-1)` does the rest.
- **`scripts/build_chat_dataset.py`** — walks the pretrain corpus,
  applies per-source templates (NVD, MITRE, CAPEC, Exploit-DB,
  CTFtime, synthetic CTF), and merges in `data/raw/chat/small_talk.jsonl`
  with `--small-talk-multiplier` (v1 used 1× and produced a model that
  ignored conversational structure; v2/v3 use 30× for ~30% small-talk
  share in the training mix).
- **`scripts/finetune_chat.py`** — loads Phase 4, expands the embedding,
  runs the standard `GhostTrainer` with SFT-appropriate hyperparameters
  (lr 3e-5, 1800 steps, 120 warmup, batch 8 × grad_accum 4).
- **`scripts/eval_chat.py`** — held-out chat eval (identity, refusals,
  small-talk, free-form cybersec). Confirms v2 → v3 preserved identity
  and refusal behavior while improving MCQ-format compliance.

Full recipe in `docs/chat_tuning.md`, including the v1 → v2 → v3
progression and what each iteration actually fixed.

### MCQ training data (commit `879219d`)

`scripts/build_mcq_data.py` templates 1,802 MCQ examples into
`data/raw/chat/mcq.jsonl`:

- **1,000 NVD CWE-class MCQs** — vulnerability type from description, 4
  candidates from a 20-class taxonomy.
- **655 MITRE tactic MCQs** — "which tactic does T1234 belong to?"
- **147 acronym MCQs** — XSS / SSRF / RCE / etc. → expansion.

Answer-letter distribution balanced (A=455 / B=404 / C=497 / D=446) so
there's no positional bias to memorize. The assistant turn is the
bare letter `A`/`B`/`C`/`D`, with a 30% subset followed by a one-line
justification — teaches the model to actually emit a single letter
after the `Answer:` cue rather than continuing into prose.

`build_chat_dataset.py` grew `--mcq-jsonl` + `--mcq-multiplier`; v3
uses 2× oversampling (~20% of the training mix is MCQ-format examples).

### RAG retrieval scaffolding (commits `b67987d`, `80b0cea`, `bd95ada`)

- **`scripts/build_rag_index.py`** — embeds the corpus chunks; index
  lands at `data/rag/{chunks.jsonl, index.npy, meta.json}` (~177 MB,
  not committed; rebuild deterministic from the corpus).
- **`scripts/rag_chat.py`** — retrieval-augmented chat: top-k chunks
  prepended to the prompt as `[CONTEXT-i]` blocks.
- **`scripts/run_bench.py --rag-dir`** — same eval harness, RAG-aware.

**Result on the full 2500-q bench: +1 question over no-RAG (476 vs
475 at chat-v2).** A 100-q smoke had suggested RAG hurt by ~3 pp; the
smoke variance was the artifact, not RAG. Honest read: at 45M params
the model can't actually use the retrieved context — it doesn't have
the in-context-reading capability that RAG benefits depend on. RAG is
preserved as infrastructure for the v0.5 retrain when the bigger
model can use it.

### v0.5 architecture switches (commit `879219d`)

Three new flags in `GhostLMConfig`, all defaulting to `False` so every
existing Phase 1-4 checkpoint loads unchanged:

- `use_rope: bool = False` — already wired in attention.
- `use_swiglu: bool = False` — gated FFN with three projections, hidden
  shrunk to ⅔ d_ff to match GELU FeedForward parameter count.
- `use_rmsnorm: bool = False` — half the params of LayerNorm, no bias;
  matches LayerNorm quality at this scale per LLaMA / Mistral / Gemma.

A new `ghost-small-v0.5` preset flips all three on. `ghostlm/model.py`
gains `RMSNorm`, `SwiGLU`, and `make_norm()` / `make_ffn()` dispatch
helpers that keep the existing `TransformerBlock` and `GhostLM` init
paths one line.

Verified: `ghost-small-v0.5` runs forward+loss end-to-end (45.0M
params, matched parameter budget vs v0.4's 45.2M). Phase 4 checkpoint
still loads into the new model class with zero missing/unexpected
keys. **The retrain that actually uses these switches is gated on
the v0.4.2 corpus expansion** — there's no point retraining on the
same 12.56M tokens when the architecture can absorb meaningfully more.

### Other infra

- **MLX export** — `phase5_chat_v2_mlx_q4/` has the q4-quantized MLX
  weights for fast inference on Apple silicon (`scripts/mlx_chat.py`).
- **MCP server** — chat over MCP for IDE integrations (`docs/mcp.md`).
- **`run_bench.py`** — first benchmark harness in the project. Auto-
  appends rows to `RESULTS.md`. Currently wired for CTIBench MCQ;
  schema is benchmark-agnostic so adding e.g. SecQA or CySecBench is
  a drop-in.

### What's canonical now

- **Base completion model:** ghost-small Phase 4 (v0.4.0) at
  `checkpoints/phase4_ghost_small/best_model.pt`. Unchanged.
- **Canonical chat model:** **`checkpoints/phase5_chat_v3/best_model.pt`**.
  This is the model to use for any chat / MCQ / instruction-following
  task. v1 and v2 are preserved as `phase5_chat/` and `phase5_chat_v2/`
  for ablation reference.
- **v0.5 architecture pretrain:** does not exist yet; the retrain is
  v0.4.2's job once the corpus lands.

---

## [0.4.0] — 2026-04-30 — Phase 4 ghost-small; capacity-reallocation hypothesis confirmed

The headline training result the project has been working toward since
Phase 3.6 told us ghost-tiny had hit its capacity ceiling. ghost-small
(~45M params, 6 layers / 512 d_model / 8 heads) trained for 30,000
steps on the same 12.56M-token Phase 3.6 corpus that broke ghost-tiny —
local Mac M4 MPS, ~15 hours wall-clock, batch_size=8 × grad_accum=4
(effective 32). Full training-log JSON is committed at
`logs/phase4_ghost_small/training_log.json` and the canonical checkpoint
is at `checkpoints/phase4_ghost_small/best_model.pt`.

**Loss trajectory (lower is better):**
- Step 1,000 (mid-warmup): val_loss 5.0758
- Step 10,000: val_loss 2.6548
- Step 20,000: val_loss 2.4031
- Step 30,000 (final): **val_loss 2.3535** — a **1.20-nat improvement**
  over the Phase 3.5 ghost-tiny canonical (3.5518), equivalent to ~3.3×
  lower perplexity. Loss was still descending at the final step (no
  overfitting plateau visible across 30k steps), so further training on
  the same corpus would likely keep paying.

### Per-source perplexity — the cleanest test

The Phase 3.6 regression diagnosis was per-source: every existing source
got 28–42% worse on ghost-tiny when Exploit-DB content was added.
ghost-small absorbs the same corpus and dominates every source — by
**59–78% relative to Phase 3.5**, by 68–80% relative to Phase 3.6:

| Source | Phase 3.5 | Phase 3.6 | **Phase 4** | vs 3.5 | vs 3.6 |
|---|---:|---:|---:|---:|---:|
| arxiv | 354.95 | 505.60 | **116.46** | **−67%** | −77% |
| capec | 133.81 | 179.71 | **54.42** | **−59%** | −70% |
| ctftime | 60.71 | 59.70 | **13.23** | **−78%** | −78% |
| exploitdb | — | 40.87 | **8.60** | — | −79% |
| mitre_attack | 55.14 | 70.53 | **19.72** | **−64%** | −72% |
| nvd | 27.55 | 35.44 | **11.29** | **−59%** | −68% |
| synthetic | 28.48 | 38.90 | **7.88** | **−72%** | −80% |
| **overall** | **66.05** | **44.36** | **11.12** | **−83%** | **−75%** |

This is the empirical confirmation of the capacity-reallocation
hypothesis. Phase 3.6 didn't fail because the corpus was bad — it
failed because 14.7M params couldn't hold seven sources at once. 45M
params hold all seven without the tradeoff. **The path forward is the
model, not the data — confirmed.**

### Security task suite — mixed, with a methodology finding

The 5×25 = 125-sample multiple-choice suite (CVE Severity / Vuln Type /
Attack Technique / CTF Categorization / MITRE Tactic) gives a more
nuanced read, and reveals an eval-methodology issue worth documenting
before users misread the numbers:

| Task | Phase 3.5 (PMI) | **Phase 4 (PMI)** | Phase 3.5 (logp) | **Phase 4 (logp)** |
|---|---:|---:|---:|---:|
| CVE Severity | 8/25 (32%) | 6/25 (24%) | 6/25 (24%) | 6/25 (24%) |
| Vuln Type | 8/25 (32%) | **10/25 (40%)** | 5/25 (20%) | 4/25 (16%) |
| Attack Tech | 10/25 (40%) | 4/25 (16%) | 2/25 (8%) | 3/25 (12%) |
| CTF Cat | 10/25 (40%) | 7/25 (28%) | 7/25 (28%) | 7/25 (28%) |
| MITRE Tactic | 3/25 (12%) | 2/25 (8%) | 2/25 (8%) | **4/25 (16%)** |
| **Overall** | **39/125 (31.2%)** | 29/125 (23.2%) | 22/125 (17.6%) | **24/125 (19.2%)** |

Read the columns top-to-bottom rather than the rows: **with logp
scoring (the more conservative scorer that does not subtract the
unconditional log-prob), Phase 4 beats Phase 3.5.** The PMI advantage
that flatters Phase 3.5 (+13.6 pp PMI vs logp) shrinks dramatically on
Phase 4 (+4.0 pp PMI vs logp). The mechanism is calibration: PMI
subtracts the unconditional candidate log-prob to break ties, and a
higher-capacity model with a tighter probability distribution gives PMI
less separation to extract. The 25-sample-per-task suite is small
enough that this calibration asymmetry can flip the headline.

The honest ranking by metric:
1. **Per-source PPL (density):** Phase 4 wins decisively (−83% overall vs Phase 3.5).
2. **Logp eval (conservative scoring):** Phase 4 wins narrowly (+1.6 pp).
3. **PMI eval (favors loose-distribution models):** Phase 3.5 wins (+8.0 pp).

For any user-facing generation work — completion, rewriting, register
matching — Phase 4 is strictly better. The PMI eval result is preserved
honestly in the comparison table; ghost-small is promoted to canonical
for everything except backwards-compatibility with the existing PMI
suite.

### What changed code-side

- `scripts/train.py` gains a `--warmup-steps` flag so future short smoke
  runs aren't dominated by the default 2000-step warmup. Used to land
  the batch=4 and batch=8 smoke runs that informed the full Phase 4
  recipe (final config: batch=8, grad_accum=4, MPS, 30k steps).
- `checkpoints/phase4_smoke/`, `checkpoints/phase4_b8_smoke/`, and
  `checkpoints/phase4_ghost_small/` are all on disk; only the last is
  the canonical artifact. The smoke checkpoints are kept as the
  "what would 300 / 100 steps of ghost-small look like" reference.

### What didn't change

- Phase 3.5 ghost-tiny (`checkpoints/phase3.5_balanced/best_model.pt`)
  remains on disk as the historical canonical and continues to be the
  better answer on the existing PMI suite. It is **not** removed.
- The corpus is unchanged (12.56M tokens, Phase 3.6 mix). This release
  is purely a model-capacity scale-up at fixed corpus.

### Next rung

Phase 4 leaves two open questions the current setup can't answer:
1. Does the loss curve keep descending past 30k steps on Phase 3.6
   corpus, or does it overfit? (Worth running — cheap.)
2. Does ghost-base (~350M) absorb a 50–100M-token corpus the same way
   ghost-small absorbed 12.56M? (External GPU territory; gated on
   either compute budget or a meaningful corpus expansion first.)

The Unreleased section below tracks both.

---

## [Unreleased]

The next release will land whatever follow-ups arrive before the
ghost-base v1.0 GPU run. Currently empty.

---

## [0.9.15] — 2026-05-09 — five new real-world cybersec tools (CISA KEV + GreyNoise + VirusTotal + Shodan + OTX)

The agent went from 4 demo-grade tools (CVE / MITRE / CWE / RAG) to
9 tools that correspond to the actual lookups a SOC analyst does
during an investigation. Three of them have live API paths
(GREYNOISE_API_KEY / VIRUSTOTAL_API_KEY / SHODAN_API_KEY / OTX_API_KEY
trigger the real upstream); CISA KEV is keyless and tries the public
CISA feed by default. All five ship with hand-curated offline caches
so tests are deterministic and the agent works without network egress.

### New tools

  lookup_cisa_kev(cve_id)
       Is this CVE on CISA's Known Exploited Vulnerabilities list?
       Returns the KEV entry with vendor, product,
       vulnerability name, required-action text, due date, and
       known-ransomware-use status. Tries the public CISA JSON feed
       at https://www.cisa.gov/sites/default/files/feeds/...; falls
       back to the offline cache (6 well-known KEV entries: Log4Shell,
       BlueKeep, Zerologon, EternalBlue, xz-utils backdoor, HTTP/2
       Rapid Reset). No API key required.

  lookup_greynoise(ip)
       Classify an IP as internet-noise, benign infrastructure (Google
       DNS, Cloudflare), targeted malicious, or unknown. Reads
       GREYNOISE_API_KEY for live community-API lookups; falls back
       to the offline cache (RFC 5737 documentation prefixes + known
       benign DNS resolvers).

  lookup_virustotal_hash(hash)
       File-hash reputation (MD5 / SHA1 / SHA256). Returns
       malicious / suspicious / harmless detection counts plus a
       threat label. Reads VIRUSTOTAL_API_KEY; falls back to the
       offline cache (EICAR test file + WannaCry public hash). The
       backend rejects malformed hex with an error blob.

  lookup_shodan(ip)
       Service profile for an IP: hostnames, country, org, open
       ports, banners. Reads SHODAN_API_KEY; falls back to the
       offline cache (Google + Cloudflare DNS resolvers as
       reference shape).

  lookup_alienvault_otx(indicator)
       OTX pulse search for IOCs (IP, domain, hash, APT name).
       Returns pulse count and short summaries with tags + TLP.
       Reads OTX_API_KEY; falls back to the offline cache (Lazarus
       Group, APT28 reference summaries).

### Architecture

Each tool follows the canonical try-real-then-cache pattern that
v0.9.9 established:

  1. If API key is set AND GHOST_AGENT_OFFLINE != 1, attempt the
     live upstream HTTP call with a short timeout.
  2. On any URL/HTTP/OS error, fall through silently.
  3. Look up the offline cache by the same key.
  4. If neither matches, return a structured `{found: false}`
     response so the model can recover via the bet-1 not-found
     pattern.

Tool errors during dispatch (unknown name, missing required arg,
backend exception) still get captured into ToolResult.error rather
than raising, so the agent loop continues and the model can recover.

### Updated default system prompt

`RuntimeConfig.system_prompt` now lists all 9 tools by name so the
model sees the full catalog when deciding which to call. Previous
4-tool prompt is replaced; existing checkpoints that were SFT'd on
the 4-tool prompt continue to work because the new prompt is a
strict superset.

### Tests

[`tests/test_agent.py`](tests/test_agent.py) gains 15 new cases
covering the new tools:

  - **Registry** (1): all five new tools registered.
  - **CISA KEV** (3): offline hit, not-found, missing-arg.
  - **GreyNoise** (2): known benign IP, unknown IP returns unknown.
  - **VirusTotal** (3): EICAR cache hit, invalid hex format, case-
    insensitive lookup (uppercase hash matches lowercase cache).
  - **Shodan** (2): known IP, unknown IP returns not-found.
  - **OTX** (3): known APT, case-insensitive, unknown indicator.
  - **Offline env var** (1): GHOST_AGENT_OFFLINE=1 keeps every
    request in-memory (sub-100ms latency).

The pre-existing `test_registry_has_four_canonical_tools` is
relaxed to assert the four canonical tools are a SUBSET (the 9-tool
registry no longer equals the 4-tool set).

Total tests now 210, all green.

### Why this matters

A SOC analyst's day is dominated by lookups against exactly these
five services (plus the four bet-1 originals). Before today the
agent could only do CVE / MITRE / CWE / RAG, which is roughly the
"intel triage" surface. With CISA KEV + GreyNoise + VirusTotal +
Shodan + OTX, the agent covers the full investigative loop:

  - "Is this CVE actively exploited?"          (CISA KEV)
  - "Is this scanning IP background noise?"     (GreyNoise)
  - "Is this dropped file known-bad?"           (VirusTotal)
  - "What services does this IP expose?"        (Shodan)
  - "Has anyone seen this indicator before?"    (OTX)

Each tool has a real public-data path that GhostAgent uses
automatically when the relevant API key is set. The offline caches
mean the agent demo runs end-to-end without any creds or network
egress, which matters for CI, air-gapped environments, and the
"clone the repo, see it work" first-time experience.

---

## [0.9.14] — 2026-05-09 — MCP server retrofit: ghostlm_agent tool exposes the full agent loop

The existing MCP server (`scripts/mcp_server.py`) shipped before
the agent runtime existed; its six tools were direct-model
invocations or deterministic lookups. v0.9.14 retrofits the server
with a seventh tool, `ghostlm_agent`, that runs the full GhostAgent
loop and returns the cite-tagged final answer. Claude Desktop /
Claude Code / Cursor / any MCP-compatible client can now invoke
the cybersec agent loop the same way they invoke any other tool.

### scripts/mcp_server.py

New tool:

```
ghostlm_agent(query, max_iters=6, include_trace=False) -> str
```

The tool wires GhostAgent around the same model the MCP server
already loaded for the older direct-chat tools. No second
checkpoint load: the new helper `make_generator_from_loaded` in
`ghostlm/agent/runner.py` builds the Generator from an already-
loaded model + tokenizer, so the MCP server's GhostLMRuntime
shares one set of weights between the direct-chat and agent-loop
code paths. `runtime.agent(max_iters=N)` is a lazy factory that
caches the GhostAgent instance and rebuilds when the iteration
cap changes.

The `include_trace=True` flag prepends a JSON-serialised trace
block (every message, every tool call, every cite tag) before the
final answer, which lets a Claude session inspect the loop's
reasoning step-by-step. Useful for debugging an answer that looks
wrong: the trace shows whether the model emitted a tool call,
whether the tool succeeded, and what the model did with the
response.

### ghostlm/agent/runner.py

Refactored `make_generator(checkpoint_path, ...)` to delegate the
Generator-building part to a new helper:

```
make_generator_from_loaded(model, config, tokenizer, device, ...)
                                  -> Generator
```

`make_generator(checkpoint_path)` now does
`load_model -> make_generator_from_loaded`, which preserves the
existing CLI behaviour while exposing the underlying builder for
callers that already have the model in memory. This is what the
MCP server uses; it is also what tests and any future shared-
runtime caller (a multi-tenant server, a notebook context) will
use to avoid duplicate model loading.

### Tests

[`tests/test_mcp_agent.py`](tests/test_mcp_agent.py) covers the
MCP-server-specific wiring (3 cases: agent lazy-built, same
max_iters returns cached, different max_iters rebuilds, ghostlm
_agent tool returns final answer, include_trace emits JSON block).
The whole file skips cleanly via `pytest.importorskip("mcp")` on
machines without the `mcp` package installed; it runs end-to-end
on machines with `pip install mcp`.

[`tests/test_agent.py`](tests/test_agent.py) gains one case
(`TestMakeGeneratorFromLoaded`) that drives the agent loop against
random ghost-tiny weights via the new builder. This runs
everywhere (no MCP dependency) and verifies the refactor preserves
the Generator contract.

Total tests now 195 (one new in test_agent + 5 new in test_mcp_agent
that are guarded by importorskip), all green where runnable.

### Why this matters

The MCP server is the contact surface between GhostLM and the rest
of the AI tooling ecosystem. Before today it was useful only for
direct-chat invocation; now any MCP-aware client can call into the
full agent loop with one tool name. Combined with v0.9.12's HTTP
server, GhostLM is now reachable from:

  - Any OpenAI-SDK client            (HTTP /v1/chat/completions)
  - Any Anthropic-SDK client         (HTTP /v1/messages)
  - Any Gemini-SDK client            (HTTP /v1beta/models/...)
  - Any Ollama-compatible client     (HTTP /api/chat)
  - Any MCP-compatible client        (stdio MCP, ghostlm_agent)
  - Direct CLI                       (python -m ghostlm.agent)

That is a deliberately oversized client surface for an 81M-param
model. The point is to make the model's *availability* irrelevant
to its quality: when ghost-base trains and the quality jumps, every
existing integration just gets better without any glue code.

---

## [0.9.13] — 2026-05-09 — agent-trace distillation: bet 1 + bet 9 traces from any OpenAI-compatible teacher

The 850 templated bet 1 + bet 9 traces produced by
`scripts/synth_tool_use*.py` are structurally correct but come from
a fixed template bank: every `search_cve_nvd` trace asks "What is
{cve} about?" and every `lookup_cwe` trace asks "Explain CWE-{id}".
A model trained only on those will overfit the templates. v0.9.13
adds the pipeline that generates fresh, varied, real-teacher traces
by driving any OpenAI-compatible teacher (Ollama, vLLM, OpenAI,
Anthropic via a translator, anything that speaks the OpenAI wire
format) through the GhostAgent runtime.

### ghostlm/agent/teacher.py

`OpenAICompatGenerator` is a Generator (a callable
`(history) -> str`) that proxies to any OpenAI-compatible chat-
completions endpoint. The constructor takes a base_url, api_key,
and model identifier; the call wraps the agent's message history
into the OpenAI request shape (TOOL role maps to user, with the
existing `<|tool_response|>...<|/tool_response|>` wrapping inline)
and returns the assistant's content verbatim. The runtime parses
the tool-call blocks the same way it does for any other generator.

The teacher is constructed once per distillation run and reused
across all prompts. An `httpx.Client` is used for the actual HTTP
calls so the same generator is testable via `httpx.MockTransport`
without ever touching the network.

### scripts/distill_agent_traces.py

CLI that drives a teacher through GhostAgent across a prompts
JSONL and writes the resulting traces in the bet-1 4-message text
format (USER / ASSISTANT / TOOL / ASSISTANT) that
`scripts/prep_tool_use_sft.py` already consumes. The output is
drop-in compatible with the SFT pipeline:

```
PYTHONPATH=. python3 scripts/distill_agent_traces.py \
  --teacher-base-url http://localhost:11434/v1 \
  --teacher-model qwen2.5:14b \
  --teacher-api-key ollama \
  --prompts data/raw/curated_prompts.jsonl \
  --out data/processed/distilled_tool_use.jsonl \
  --require-cite

PYTHONPATH=. python3 scripts/prep_tool_use_sft.py \
  --in-tool-use data/processed/distilled_tool_use.jsonl \
  --in-provenance data/processed/synth_tool_use_provenance.jsonl \
  --base-train data/processed/chat_train.jsonl \
  --out-train data/processed/chat_train_distilled.jsonl \
  --out-val data/processed/chat_val_distilled.jsonl
```

Quality gates:
  - `trace_to_bet1_text` only writes traces that fit the canonical
    USER / ASSISTANT / TOOL / ASSISTANT shape; multi-tool-call
    traces and traces missing a final answer are skipped.
  - `--require-cite` skips traces without parseable cite tags
    (the bet 9 quality bar). Use this when distilling for SFT.
  - Stats line at end-of-run reports total / kept / skipped-shape
    / skipped-no-cite / errors.

### Why this matters

Until today, the bet 1 + bet 9 SFT corpus was bounded by what the
synth scripts could template. The templates produce 850 records;
beyond that you get diminishing returns from variation. With this
release, anyone can:

  - Run Ollama with Qwen-14B locally and generate thousands more
    high-quality bet-1 + bet-9 traces overnight, on M4 hardware.
  - Mix templated + distilled corpora to get both structural
    correctness AND natural variation.
  - Distill from genuinely strong teachers (frontier APIs) when
    creds are available, raising the upper bound on what GhostLM
    can learn from SFT.

Combined with v0.9.10's prep pipeline and v0.9.9's runtime, this
closes the data-quality loop: stronger teachers -> more varied
traces -> better SFT -> measurably better agent (now scorable
through v0.9.11's GhostBench agent runner).

### Tests

[`tests/test_agent_distill.py`](tests/test_agent_distill.py)
covers 13 cases:

  - **OpenAICompatGenerator** (5): request body shape, role mapping
    (TOOL -> user), non-200 raises, malformed response raises,
    no-api-key omits Authorization header. All using
    `httpx.MockTransport` so no network is needed.
  - **trace_to_bet1_text** (3): valid 4-message trace produces
    correct format, no-tool-call returns None, no-final-answer
    returns None.
  - **trace_has_cite_tag** (3): present in assistant counts,
    absent does not, present only in user does not count.
  - **End-to-end** (1): a stub-handler teacher emitting a perfect
    bet-1 + bet-9 trace through the agent loop produces a valid
    distilled record.
  - **CLI subprocess** (1): `--teacher-base-url` pointed at an
    unreachable port logs errors gracefully and exits 0.

Total tests now 194, all green.

---

## [0.9.12] — 2026-05-09 — multi-vendor HTTP server: OpenAI + Anthropic + Gemini + Ollama

GhostAgent now speaks the request/response shapes of every major
LLM provider API, so any client that already targets OpenAI,
Anthropic, Google, or Ollama can point at the GhostLM URL unchanged
and get a compatible response back. The agent loop runs server-side:
tool calls happen behind the API and the model's final cite-tagged
answer comes back in whatever shape the caller's SDK expects.

### Endpoint matrix

  POST /v1/chat/completions
       OpenAI Chat Completions API. Returns `chat.completion`
       objects with `tool_calls` populated when the loop dispatched
       any tool. `stream=true` produces SSE chunks (one delta per
       agent iteration plus a closing chunk with the final answer
       and `[DONE]` sentinel). Many other providers (Mistral, xAI,
       vLLM, TGI, Together, Groq) re-implement this same shape, so
       the OpenAI endpoint is also a Mistral / xAI / vLLM endpoint.

  POST /v1/messages
       Anthropic Messages API. Accepts both string and content-
       block content (`{"type": "text", "text": "..."}` and
       `{"type": "tool_result", ...}`), surfaces tool calls as
       `tool_use` blocks in the response content array, and maps
       termination reasons (`answer_emitted` -> `end_turn`,
       `max_iterations` -> `max_tokens`).

  POST /v1beta/models/{model}:generateContent
       Google Gemini API. Accepts the `contents/parts/role` shape
       and returns `candidates[].content.parts` plus `usageMetadata`.

  POST /api/chat
  POST /api/generate
  GET  /api/tags
       Ollama API. Local-first clients (Open WebUI, LobeChat,
       continue.dev, llama.cpp's web UI) typically default to this.

  POST /v1/agent/run    Native: full AgentTrace as JSON for clients
                         that want loop visibility.
  GET  /v1/models       OpenAI-compat model list.
  GET  /healthz         Readiness probe with registered tools.

### Architecture

The server is a factory: `create_app(generator, config, model_name,
tools)` returns a FastAPI app wired around a supplied generator,
which makes test injection trivial. Pydantic request models live
at module level (Pydantic v2 forward-ref resolution requires this)
so the same shape definitions feed both runtime validation and
OpenAPI schema generation. Vendor-specific shape conversion lives
in module-level helpers (`_anthropic_extract_query`,
`_trace_to_anthropic_content`, `_gemini_extract_query`, etc.) so
the per-endpoint handlers stay terse and the conversion logic is
unit-testable.

The streaming OpenAI endpoint emits one SSE delta per assistant
iteration plus a closing chunk with the final answer. Token-level
streaming would require generator-callback hooks the runtime does
not yet expose; iteration-level is what the current generator
abstraction allows without re-architecture.

### Tests

[`tests/test_agent_server.py`](tests/test_agent_server.py) covers
22 cases:

  - **Introspection** (3): `/healthz`, `/v1/models`, `/api/tags`.
  - **Native /v1/agent/run** (3): trace + metadata, max_iters
    override, include_trace=False.
  - **OpenAI** (5): basic completion, tool_calls surfaced with
    JSON arguments, no tool_calls when plain answer, 400 on
    missing user message, streaming yields `[DONE]` sentinel.
  - **Anthropic** (5): string content, content-block format,
    tool_use blocks in content array, text block, 400 on missing
    user message.
  - **Gemini** (3): basic generateContent, 400 on missing user
    content, usageMetadata shape.
  - **Ollama** (3): `/api/chat`, `/api/generate`, 400 on missing
    user message.

Total tests now 181, all green.

### CLI

```
python -m ghostlm.agent.server \\
    --checkpoint runs/v09chat/best.pt --port 8000 --offline
```

Without `--checkpoint`, spins up random ghost-tiny so the wiring
can be smoke-tested without a model. Once a real checkpoint is
loaded, ANY of these clients work against the same URL:

```python
# OpenAI SDK
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="anything")
client.chat.completions.create(model="ghostlm",
    messages=[{"role": "user", "content": "What is CVE-2017-0144?"}])

# Anthropic SDK
from anthropic import Anthropic
client = Anthropic(base_url="http://localhost:8000", api_key="anything")
client.messages.create(model="ghostlm", max_tokens=512,
    messages=[{"role": "user", "content": "..."}])

# Ollama Python client
import ollama
ollama.Client(host="http://localhost:8000").chat(model="ghostlm",
    messages=[{"role": "user", "content": "..."}])
```

Plus curl, LangChain, LlamaIndex, Open WebUI, continue.dev, anything
that talks one of the four shapes.

### Why this matters

Before today, GhostLM was a CLI tool. After today, it is a service
with the broadest possible client surface. Any team that already has
an OpenAI / Anthropic / Gemini / Ollama integration in their stack
can point it at GhostLM by changing a base_url. This collapses the
"how do I use this model in my product" friction to zero, which is
exactly what a small-cybersec-LM project needs to be useful in real
SOC environments.

The server also unblocks evaluation against external benchmark
harnesses that target these APIs (LMSYS Chat Arena, OpenAI evals,
Anthropic's stress tests). When ghost-base lands, those harnesses
become available without any glue code.

---

## [0.9.11] — 2026-05-09 — GhostBench agent runner: every bet now scores end-to-end through the agent loop

The piece that turns the agent runtime into a real research artifact.
v0.9.9 shipped GhostAgent. v0.9.10 shipped the SFT pipeline that
trains a checkpoint to use it. v0.9.11 makes the agent loop scorable
across every bet GhostBench knows about: bet 6 format-aware, bet 7
code-security, bet 8 binary-literacy, bet 9 provenance, bet 10
log-analysis, bet 11 IaC-security, bet 12 protocol-fields. Seven held-
out evals, one CLI line, full statistical machinery (Wilson CIs,
McNemar, Cohen's h, paired-difference confidence intervals).

### scripts/ghostbench_agent_run.py

Composes GhostAgent with GhostBench. For every Bench in
`Suite.from_dir(eval_dir)`, runs the agent loop on every prompt,
serialises each trace into a `Prediction` record (using the new
`AgentTrace.to_scored_text` helper), and writes one JSONL per bench
to `--predictions-dir/<bench>.jsonl`. The output drops cleanly into
the existing `python -m ghostbench summary` and `python -m
ghostbench compare` commands; no changes to GhostBench core required.

A `--baseline` flag forces `max_iters=1` so the model emits one
message and the loop terminates without dispatching tools, the
no-tools control for paired comparison. Same checkpoint, same
prompt, same generation params, same agent runtime, same system
prompt, but the model never sees a tool response. Compare via:

```
python -m ghostbench compare \
  --eval data/raw/<eval>.jsonl \
  --a-predictions logs/<run>/<bench>.jsonl --a-name agent \
  --b-predictions logs/<run>_baseline/<bench>.jsonl --b-name baseline \
  --bench-name <bench>
```

A `--write-traces` flag dumps the full trace structure to a
sidecar `<bench>.traces.jsonl` for audit and replay, so a
disagreement between agent and baseline is forensic-recoverable.
A `--only` flag accepts a comma-separated bench list when you only
want to re-run a subset.

### AgentTrace.to_scored_text refactor

The `trace_to_full_text` helper that lived in `scripts/eval_agent.py`
moves onto `AgentTrace` itself as `to_scored_text(include_user=False,
include_system=False)`. The default kept content is ASSISTANT
messages plus TOOL responses, the same convention v0.9.10 introduced
to avoid crediting substrings present in the eval prompt rather than
substrings the model produced or grounded through tool dispatch. The
old helper is now a thin shim around the method, so existing call
sites continue to work.

Opt-in flags expose USER and SYSTEM content for cases where you
*do* want the full conversation surface (e.g. logging, debugging,
or evals where the system prompt must show up in the trace text).

### Tests

[`tests/test_ghostbench_agent.py`](tests/test_ghostbench_agent.py)
covers 10 cases:

  - **AgentTrace.to_scored_text** (4): default excludes user +
    system, `include_user` opts user back in, `include_system`
    opts system back in, both flags.
  - **trace_to_prediction** (3): propagates eval tags
    (format/prompt/required_substrings/required_fields/seed_id),
    predicted_artifact correctly excludes user content, the dict
    is loadable into `Prediction.from_dict`.
  - **end-to-end** (1): a stub generator that emits a perfect bet-1
    + bet-9 trace produces a Prediction that `Bench.score` for
    `bet9_provenance` recognises as passing the substrings tier.
  - **CLI subprocess** (2): runs against the real
    `data/raw/provenance_eval.jsonl` (n=15) with random ghost-tiny
    weights, asserts well-formed Prediction JSONL output, and
    confirms `--baseline` flag triggers `max_iters=1`.

Total tests now 159, all green.

### M4 invocation: every bet scored through the agent

Once the v0.9.10 SFT lands a checkpoint, this is the one-shot path
to the headline artifact, a per-bet table of agent vs baseline:

```bash
PYTHONPATH=. python3 scripts/ghostbench_agent_run.py \
  --checkpoint checkpoints/phase20_chat_v09_tools/best_model.pt \
  --eval-dir data/raw \
  --predictions-dir logs/v09tools_agent \
  --run-name v09tools_agent \
  --offline

PYTHONPATH=. python3 scripts/ghostbench_agent_run.py \
  --checkpoint checkpoints/phase20_chat_v09_tools/best_model.pt \
  --eval-dir data/raw \
  --predictions-dir logs/v09tools_baseline \
  --run-name v09tools_baseline \
  --baseline --offline

python -m ghostbench summary \
  --eval-dir data/raw \
  --predictions-dir logs/v09tools_agent \
  --run-name v09tools_agent \
  --out logs/v09tools_agent/suite_summary.md

python -m ghostbench summary \
  --eval-dir data/raw \
  --predictions-dir logs/v09tools_baseline \
  --run-name v09tools_baseline \
  --out logs/v09tools_baseline/suite_summary.md

# Per-bench paired comparison with McNemar p-values:
for b in bet9_provenance bet6_format_aware bet7_code_security \
         bet8_binary_literacy bet10_log_analysis bet11_iac_security \
         bet12_protocol_fields; do
  python -m ghostbench compare \
    --eval data/raw/${b}*.jsonl \
    --a-predictions logs/v09tools_agent/${b}.jsonl --a-name agent \
    --b-predictions logs/v09tools_baseline/${b}.jsonl --b-name baseline \
    --bench-name ${b} \
    --out logs/comparisons/${b}_agent_vs_baseline.md
done
```

The aggregate output is the answer to the falsifiability question:
on which bets does the agent runtime measurably help, and at what
significance level? Prior to v0.9.11, the runtime was unfalsifiable
infrastructure; now it is a measurable component with a real eval
behind it.

### Why this matters

The 12-bet differentiation work (v0.9.4 through v0.9.8) produced
seven held-out eval sets. Until today they only measured the model
directly. The agent runtime (v0.9.9) wrapped the model in a tool-
using loop. The SFT pipeline (v0.9.10) bridged the runtime back to
v0.9 chat. This release closes the loop: every bet now scores
through the agent, with paired comparison against a no-tools control,
with statistical significance reported.

When ghost-base lands, the same one-line invocation produces a
publishable-shape table comparing ghost-base-with-tools vs ghost-
base-baseline vs v0.9-chat-with-tools, with McNemar p-values on
each bet. That is the kind of result that distinguishes a research
project from a demo.

---

## [0.9.10] — 2026-05-08 — tool-use SFT pipeline + agent eval harness

The bridge between v0.9.9's runtime and a v0.9 chat checkpoint that
can actually use it. Three pieces: a prep script that converts the
existing bet 1 + bet 9 synth traces into chat-format SFT records, a
held-out-eval runner that scores agent traces with strict + soft
pass rates and Wilson CI, and 24 unit tests covering the pipeline.

### scripts/prep_tool_use_sft.py

Converts the bet-1 four-message trace string (USER / ASSISTANT /
TOOL / ASSISTANT) into the two-role chat shape the existing
`ChatDataset` expects. The TOOL response maps to the next USER turn
(carrying its `<|tool_response|>...` wrapping verbatim), so the
model sees the bet-1 wire format inline in conversation. Loss is
auto-masked to assistant tokens by `ChatDataset`, which means the
model learns BOTH "when to emit a tool call" (assistant turn 2) and
"how to synthesize cite-tagged answers from tool responses"
(assistant turn 4). The script also takes optional `--base-train` /
`--base-val` flags so the converted records mix into the existing
chat SFT corpus, preserving v0.9's small-talk + identity behaviour
instead of overwriting it.

A deterministic 95/5 train/val split keyed on a stable hash of
each record (`source` + `seed_id` + first-100-chars of user content)
guarantees the same record always lands in the same split across
runs, which matters when comparing multiple SFT runs.

### scripts/eval_agent.py

Runs the agent loop against a held-out eval set
([`data/raw/provenance_eval.jsonl`](data/raw/provenance_eval.jsonl),
n=15) and scores each trace by `required_substrings` presence:
strict pass-rate (all substrings present, with Wilson 95% CI) and
soft pass-rate (mean fraction present). Crucially the scorer
**excludes USER and SYSTEM messages**: many provenance eval prompts
mention the entity ("What is CVE-2017-0144?"), so naive
concatenation would credit substrings already in the question. The
scored content is ASSISTANT messages plus TOOL responses, which
honestly measures what the model produced or grounded through tool
dispatch.

A `--baseline` flag forces `max_iters=1` so the model emits one
message and the loop terminates without dispatching tools. This is
the no-tools control for paired comparison: same prompt, same
generation params, same agent runtime, but no chance for the model
to see tool responses. Comparing tools-on vs tools-off scores tells
you whether the SFT actually changed tool-use behaviour.

### Tests

[`tests/test_agent_sft.py`](tests/test_agent_sft.py) covers 24
cases:

  - **parse_trace** (6): happy path, missing roles, wrong first
    role, empty input, non-string input, outer whitespace.
  - **trace_to_chat_record** (5): four turns alternating roles,
    tool call in assistant 1, tool response in user 2, answer in
    assistant 2, metadata preserved.
  - **hash_for_split** (2): deterministic, distinguishes records.
  - **prep CLI** (1): subprocess-invokes the script and asserts
    train/val files are written.
  - **eval scoring** (4): full text concats only assistant + tool,
    all-required present, partial match, user substring excluded.
  - **wilson_ci** (4): n=0 edge, full-pass upper bound, zero-pass
    lower bound, half-centred symmetry.
  - **stub-generator end-to-end** (1): a perfect bet-1+9 trace
    scores 100% strict pass on a provenance-style eval prompt.

Total tests are now 149, all green.

### M4 invocation (no GPU dependency)

```bash
# 1. Convert synth traces into chat-SFT records, mixed with v0.9's
#    existing chat data so small-talk + identity SFT survives.
PYTHONPATH=. python3 scripts/prep_tool_use_sft.py \
  --in-tool-use data/processed/synth_tool_use.jsonl \
  --in-provenance data/processed/synth_tool_use_provenance.jsonl \
  --base-train data/processed/chat_train.jsonl \
  --base-val data/processed/chat_val.jsonl \
  --out-train data/processed/chat_train_with_tools.jsonl \
  --out-val data/processed/chat_val_with_tools.jsonl

# 2. Fine-tune on top of v0.9 chat. Smaller LR than pretrain,
#    fewer steps because the SFT data is narrow.
PYTHONPATH=. python3 scripts/finetune_chat.py \
  --checkpoint checkpoints/phase19_chat_v09/best_model.pt \
  --train-data data/processed/chat_train_with_tools.jsonl \
  --val-data data/processed/chat_val_with_tools.jsonl \
  --run-name phase20_chat_v09_tools \
  --learning-rate 1e-5 \
  --max-steps 2000 \
  --warmup-steps 100

# 3. Eval the new checkpoint on the provenance held-out set.
PYTHONPATH=. python3 scripts/eval_agent.py \
  --checkpoint checkpoints/phase20_chat_v09_tools/best_model.pt \
  --eval data/raw/provenance_eval.jsonl

# 4. Paired comparison: same checkpoint with tools off.
PYTHONPATH=. python3 scripts/eval_agent.py \
  --checkpoint checkpoints/phase20_chat_v09_tools/best_model.pt \
  --eval data/raw/provenance_eval.jsonl --baseline

# 5. Reference: pre-SFT v0.9 chat with the agent runtime.
PYTHONPATH=. python3 scripts/eval_agent.py \
  --checkpoint checkpoints/phase19_chat_v09/best_model.pt \
  --eval data/raw/provenance_eval.jsonl
```

The expected result is steps 3 > 4 (tools help if SFT worked) and
3 > 5 (SFT improves over the pre-SFT v0.9 chat). M4 wall time for
the full pipeline (n~850 records, 2000 steps): a few hours.

### Why this matters

Up to v0.9.9 the runtime existed but had no checkpoint that could
exercise it. After this change the path is:

  synth_v1.jsonl  ->  prep  ->  SFT on v0.9 chat  ->  agent eval

with every step reproducible from a single CLI line. Format
compliance is the kind of narrow signal small models *can* learn
at 81M params even when fact recall floors at this scale, so v0.9-
chat-with-tools could plausibly produce an actually-working agent
demo before GPU compute lands. The held-out eval will quantify
whether that's true.

---

## [0.9.9] — 2026-05-08 — GhostAgent: the runtime that turns the checkpoint into a deployed assistant

A production-shaped agent runtime that exercises bets 1 (tool-use
trace format) and 9 (cite tags) end-to-end. This is the missing
architectural layer between "we trained a model" and "an analyst
queried the model and got back a cite-grounded answer." It is
generator-agnostic, model-agnostic, and offline-testable; it runs
against any GhostLM checkpoint today and will run unchanged against
ghost-base when the synth-tool-use SFT data lands.

### Module layout: `ghostlm/agent/`

[`ghostlm/agent/messages.py`](ghostlm/agent/messages.py) defines the
conversation primitives: `MessageRole` (USER / ASSISTANT / TOOL /
SYSTEM), `AgentMessage` with class methods that build correctly
shaped messages including the bet-1 `<|tool_response|>...
<|/tool_response|>` wrapping for TOOL replies, and `AgentTrace`
which captures the full back-and-forth plus termination metadata
(reason, iteration count, latency, token count). Every dataclass is
JSON-serialisable so traces can be logged and replayed by the
GhostBench paired-comparison machinery.

[`ghostlm/agent/parser.py`](ghostlm/agent/parser.py) parses raw
assistant outputs. Three regex passes pull out
`<|tool_call|>{json}<|/tool_call|>` blocks, `<|cite|>type:id#field
<|/cite|>` tags, and the surrounding plain text. Lenient tag
normalisation tolerates the noisy patterns a not-yet-trained
checkpoint produces (`<|tool call|>` with a space, `<TOOL_CALL>`
casing variants, fenced ```json bodies). Strict-mode parsing for
training-data validation stays in `scripts/distill_tool_use.py`; the
runtime parser is permissive so the loop never crashes on partial
output.

[`ghostlm/agent/tools.py`](ghostlm/agent/tools.py) is the registry
plus the four canonical tools the bet 1 SFT data trained on:
`search_cve_nvd`, `lookup_mitre_technique`, `lookup_cwe`,
`rag_retrieve`. Every backend follows a graceful-degradation
pattern: try the real upstream (NVD JSON API, MITRE Workbench),
fall back to a hand-curated offline cache that ships with the
package, fall back to a structured `not_found` response so the
model recovers using the failure-mode shape it was trained on.
`GHOST_AGENT_OFFLINE=1` forces offline mode for deterministic CI.
`dispatch(call_name, args, registry)` validates required args,
times the backend, captures all exceptions into `ToolResult.error`,
and returns the result instead of raising; tool errors get fed back
to the model on the next loop iteration as recoverable failures.

[`ghostlm/agent/runtime.py`](ghostlm/agent/runtime.py) is the loop.
`RuntimeConfig` exposes the knobs (max_iters, generation params,
system prompt, stop sequences). `GhostAgent(generator, config,
tools)` accepts any callable `(history) -> str` as the model
abstraction, which decouples the loop from any specific HF /
transformers wiring; you can drop in a remote API, a local
llama.cpp server, or a unit-test stub equally. `agent.run(query)`
builds a history, generates, parses, dispatches tool calls, feeds
results back, repeats. Three terminal states are exhaustive:
`answer_emitted` (model produced a no-tool-call message),
`max_iterations` (cap hit), `model_error` (generator raised). A
small repair pass re-attaches `<|/tool_call|>` if the generator's
stop sequence ate it, so the parser always sees well-formed blocks.

[`ghostlm/agent/runner.py`](ghostlm/agent/runner.py) +
[`ghostlm/agent/__main__.py`](ghostlm/agent/__main__.py) wire the
runtime to a real GhostLM checkpoint. CLI:

```
python -m ghostlm.agent --query "What is CVE-2017-0144?"
python -m ghostlm.agent --query "..." --checkpoint runs/v09chat/best.pt
python -m ghostlm.agent --query "..." --offline --json --max-iters 4
```

Without `--checkpoint`, the runner spins up random ghost-tiny
weights so the loop can be smoke-tested without a trained model
(output is noise, but every mechanic exercises). With v0.9 chat,
the model emits poor tool calls because it wasn't trained on the
bet 1 format; the loop terminates safely via `answer_emitted` or
the max-iterations safety. When ghost-base trains on
`synth_v1.jsonl`, the runtime is already wrapped around it.

### Why this matters

Up to v0.9.8 the project shipped data and models. Bet 1 produced
288 templated tool-use traces, bet 9 produced 252 cite-tagged
traces, but nothing actually executed those formats end-to-end.
v0.9.9 makes the formats live: the parser reads them, the
dispatcher executes the tools, the loop feeds responses back, and
the trace captures every step in JSON for audit and replay. The
moment a checkpoint emits a structurally valid `<|tool_call|>`,
GhostAgent will execute it against the real NVD API, get a real
CVE record, feed it back, and produce a cite-tagged final answer.
That makes the project a deployable artifact, not just a research
prototype.

### Tests

[`tests/test_agent.py`](tests/test_agent.py) covers 31 cases:

  - **Parser (10 tests):** single + multiple tool calls, cite tags
    with and without field, normalised spaced tags, code-fence
    stripping, malformed JSON warnings, missing-name warnings,
    block stripping for the spoken-answer path, non-string input.
  - **Tools (9 tests):** registry shape, offline CVE hit, unknown-
    CVE not_found, MITRE lookup, CWE prefix normalisation, RAG
    retrieval, unknown-tool error, missing-required-arg error,
    backend-exception capture.
  - **Messages (3 tests):** TOOL message wraps in `<|tool_response|>`
    tags, error shape, trace round-trips through `to_dict`.
  - **Runtime (9 tests):** terminates on no tool call, dispatches
    then emits answer, max-iterations safety, model-error capture,
    tool-error recoverability, stop-sequence repair, system-prompt
    disable, cite metadata stash, JSON-serialisable trace.

All 31 pass; full suite is now 156 tests, all green.

### Strategic frame: from corpus to product

The 12-bet differentiation work answered "what does GhostLM know?";
GhostAgent answers "what does GhostLM do?" The project now has:

  - A trained 81M base + chat checkpoint (v0.9 series)
  - 1,745 templated SFT records spanning 12 cybersec bets
  - A packaged eval suite (GhostBench v0.3) with statistical rigor
  - **A production-shape agent runtime that wraps the checkpoint**
  - An MCP server, RAG layer, GGUF + MLX exports

When GPU compute lands and ghost-base trains on the combined synth
corpus, the deployment story is already wired: same `python -m
ghostlm.agent --query "..."` line, same trace shape, same eval
harness. The runtime's defensive design (lenient parsing, graceful
tool fallback, three-way terminal states, offline-testable
backends) means the loop survives whatever the trained model
produces.

---

## [0.9.8] — 2026-05-08 — three new bets: log analysis + cloud IaC security + protocol field reading

The release that takes GhostLM from "9 bets covering cybersec prose
+ code + binary + structured CTI + tool use + provenance" to **"12
bets covering the full security analyst workflow surface"**: the
SOC analyst's daily log review (bet 10), the DevSecOps engineer's
PR review of cloud IaC (bet 11), and the network forensicist's
protocol field decoding (bet 12).

### Bet 10: log analysis & event reasoning

[`scripts/synth_log_analysis.py`](scripts/synth_log_analysis.py)
templates 4 record variants per pattern (pretrain prose +
identify-technique Q&A + explain-detection Q&A + field-citation
Q&A) from a 30-pattern bank covering Windows Sysmon / Windows
Security / Linux auditbeat / network proxy / network webserver /
network DNS / email gateway logs across 30 unique ATT&CK technique
ids. **120 records** at 100% parser-pass.

Held-out eval at [`data/raw/log_analysis_eval.jsonl`](data/raw/log_analysis_eval.jsonl)
covers 25 prompts spanning T1078 (Valid Accounts), T1071 (Web
Protocols C2), T1565 (hosts file modification), T1053 (scheduled
tasks), T1003 variants (LSASS / SAM / NTDS dump), T1204 (User
Execution from temp), T1197 (BITS abuse), T1566.002 (link-based
phishing), T1098.001 (cloud IAM access-key creation), T1556
(Modify Authentication Process / DNS hijack), T1003.008 (/etc/
shadow read), T1567 (cloud bucket exfiltration), T1562.004
(firewall disable), T1490 (VSS shadow delete), TA0010 (DNS
exfiltration tunneling), T1059.001 (PowerShell Mimikatz),
T1053.003 (cron persistence), T1218 / T1218.007 (InstallUtil /
msiexec proxy execution), T1047 (wmic remote exec).

### Bet 11: cloud IaC security

[`scripts/synth_iac_security.py`](scripts/synth_iac_security.py)
templates 4 record variants per pattern (pretrain prose +
identify-and-fix Q&A + explain-the-diff Q&A + severity-mapping
Q&A with CWE) from a 15-pattern bank covering Terraform/AWS (S3
ACL, IAM trust, security groups, RDS encryption, IAM wildcard
actions, S3 logging, CloudFront HTTPS, EBS encryption, IAM MFA,
ALB+WAF) and Kubernetes (Pod privileged, NetworkPolicy default-
deny, Secret stored plaintext, RBAC cluster-admin, hostNetwork).
**60 records** at 100% parser-pass.

Held-out eval covers 15 prompts on Lambda secrets in env vars,
K8s dangerous capabilities, S3 bucket-policy wildcard, RDS
publicly_accessible + plaintext password, K8s automount + latest
tag, public LoadBalancer to DB port, IAM s3:* wildcard,
CloudTrail disabled, EKS public endpoint with 0.0.0.0/0,
hostPath mount of /, port range 0-65535, RBAC system:authenticated
secret reader, Lambda Function URL with no auth, Ingress without
TLS, KMS key with rotation off and wildcard policy.

### Bet 12: network protocol field reading

[`scripts/synth_protocol_fields.py`](scripts/synth_protocol_fields.py)
templates 3 record variants per pattern (pretrain prose +
identify-protocol Q&A + read-field Q&A walkthrough) from a 20-
pattern bank spanning every layer: datalink (Ethernet, ARP),
network (IPv4, ICMP), transport (TCP, QUIC), application (TLS
1.3 ClientHello, TLS Application Data, TLS SNI, TLS Certificate,
DNS query, DNS response, HTTP/2 frame header, BGP UPDATE, DHCP
Discover, SMB2 Negotiate, Kerberos AS-REQ, MQTT CONNECT, RDP
X.224 Connection Request), plus the JA3 fingerprint derivation.
**60 records** at 100% parser-pass.

Held-out eval covers 20 prompts including TLS ServerHello
identification, IPv4+UDP decoding, TCP destination port reading,
HTTP/2 SETTINGS frame, DNS response flags, ARP reply, ICMP Time
Exceeded (traceroute), TCP flag 0x14 (RST+ACK), IPv6 EtherType,
SMB2 TREE_CONNECT, TLS 1.3 version detection via supported_versions
extension, DNS QType 0x000F (MX), IPv4 protocol field 0x32 (ESP),
DHCP OFFER message type, BGP marker semantics, MQTT PUBLISH
opcode, QUIC long-header Initial type, RDP cleartext mstshash
cookie, Kerberos AP-REQ message-type 14.

### Total templated-synth corpus

**1,745 records** ready for ghost-base SFT mixing once GPU lands:

| Bet | Records | Acceptance |
|---|---:|---:|
| 1 (tool-use, plain) | 424 | 98.6% |
| 6 (STIX / YARA / Sigma / MISP) | 560 | 99.8% |
| 7 (code-for-security) | 48 | 100.0% |
| 8 (binary / hex literacy) | 44 | 100.0% |
| 9 (cite-augmented tool-use) | 429 | 99.8% |
| **10 (log analysis)** | **120** | **100.0%** |
| **11 (cloud IaC security)** | **60** | **100.0%** |
| **12 (protocol field reading)** | **60** | **100.0%** |
| **TOTAL** | **1,745** | **99.5%** |

`scripts/build_v15_combined_synth.py` extended with the three new
synth pipelines and updated CATEGORY_RULES so the unified
combined-corpus output now includes all eight pipelines, tagged by
training-time use (pretrain vs SFT).

### GhostBench Suite auto-discovery

`ghostbench.bench.Suite.from_dir` extended to discover the three
new eval files (bet10_log_analysis, bet11_iac_security,
bet12_protocol_fields) by their canonical filenames, with proper
descriptions. `python -m ghostbench summary --eval-dir data/raw`
will now produce a 7-row table covering all measurable bets the
moment ghost-base trains.

### Strategic frame: full analyst-workflow coverage

The bet sequence is now complete across the security-analyst-
workflow envelope:

  - **Threat-intel analyst** (bet 6): STIX / YARA / Sigma / MISP
  - **SOC analyst** (bet 10 + bet 9): logs + cite-tagged answers
  - **DevSecOps engineer** (bet 11 + bet 7): IaC + code review
  - **Reverse engineer / forensicist** (bet 8 + bet 12): binary +
    protocol fields
  - **Operator / auditor** (bet 1 + bet 9): tool-grounded + cite-
    backed answers
  - **Plus**: bet 3 tokenizer, bet 4 long context, bet 5 MoE
    architecture for ghost-1B+

12 bets, ~1,745 deterministic records, 7 held-out eval sets, all
reproducible from one CLI command. When ghost-base trains, every
claim is a `python -m ghostbench suite-compare --behavioral` away
from being defensibly significant.

---

## [0.9.7] — 2026-05-08 — GhostBench v0.3: behavioural tier with two-path validators

GhostBench's reserved `behavioral` tier moves from "slot reserved"
to "fully implemented." The tier asks a strictly stronger question
than `parse`: would a real downstream tool actually accept this
artifact? Each validator has a two-path design.

### Two-path validators

**Real-library path:** lazy-import the canonical reference parser
when available. STIX 2.1 via `stix2.parse(blob)`, YARA via
`yara.compile(source=blob)` (libyara binding), Sigma via
`sigma.collection.SigmaCollection.from_yaml()` from pysigma, MISP
via jsonschema validation. Catches edge cases the structural
parser doesn't (invalid UUID4 in STIX `id`, malformed YARA
condition trees, Sigma logsource taxonomy violations, MISP
attribute types outside the controlled vocab).

**Enhanced-structural fallback:** when the reference library isn't
installed, fall back to a deeper structural check than the v0.1
`parsers.py` ones. STIX: UUID4 format, RFC3339 timestamps,
`modified >= created` ordering, indicator-specific pattern + label
checks. YARA: rule-name validity, string-def shape, condition
references at least one defined string OR uses a wildcard, paren /
brace / bracket balance. Sigma: logsource has at least one of
(category, product, service); detection has selection blocks and a
condition that references at least one selection; level vocabulary
check. MISP: threat_level_id / analysis / distribution range
checks, every Attribute type in a curated subset of the MISP
controlled vocabulary, every Attribute has a non-empty value.
Provenance: every cite tag's source_id matches a plausible
identifier shape (CVE / CWE / T-code / passage / generic).

The fallbacks are still a strict upgrade over the parse tier; they
just don't catch every edge case the real library would.

### CLI integration

Every subcommand (`score`, `summary`, `compare`, `suite-compare`)
gets a common `--behavioral` flag that opts every record into the
behavioural tier at score time:

```bash
python -m ghostbench summary \
    --eval-dir data/raw \
    --predictions-dir logs/baselines_v09_chat \
    --run-name v09_chat --behavioral
```

Verified end-to-end on Mac. The behavioural tier appears as a
distinct row in the per-tier breakdown for the structurally
validated bets (bet 6 format-aware, bet 9 provenance), giving
operators the diagnostic view: how often does a prediction parse
but fail at the deeper validation layer?

### API additions

  - `ghostbench.behavioral` module with `BEHAVIORAL_VALIDATORS`
    public registry.
  - `score_record()` gains `behavioral_validators=` kwarg; eval
    records can request the tier via `behavioral: true`.
  - `Bench.score()` gains `behavioral_validators=` and
    `force_behavioral=True` kwargs for bulk override.
  - `__version__` bumped to `0.3.0`.

### Test stats

**205 total tests passing** (94 GhostLM + 111 ghostbench). 31 new
behavioural tests cover STIX UUID / timestamp / modified-vs-created
/ indicator pattern presence / external_ref shape; YARA
condition-references-strings / wildcard support / unbalanced
parens; Sigma logsource required keys / condition refs / level
vocabulary; MISP threat_level_id / analysis / distribution /
attribute-type vocabulary / value presence; provenance source_id
plausibility / field whitelist; registry coverage. Plus integration
tests showing the asymmetry: parse can pass while behavioural
fails (the strict-stricter property).

---

## [0.9.6] — 2026-05-08 — GhostBench: a packaged eval suite turns the project into a research artifact

The release that converts the v0.9.5 in-script eval scaffolding
into a properly-packaged, statistically-rigorous, reusable
benchmark suite anyone can pip-install (eventually) and point at
any small open LM. Eight commits since v0.9.5; 80 new ghostbench
tests on top of the existing 94 GhostLM tests = 174 green.

### `ghostbench/` v0.1 + v0.2: new package

Module layout:

  - **`ghostbench/__init__.py`**: public API: `Bench`, `Suite`,
    `EvalRecord`, `Prediction`, `Score`, `RunReport`, `wilson_ci`,
    `mcnemar_test`, `cohen_h`, `paired_diff_ci`.
  - **`ghostbench/stats.py`**: Wilson 95% CI, exact two-sided
    McNemar's binomial test, Cohen's h effect size with arcsine
    transform, Wilson-shifted paired-difference CI. Stdlib only,
    pickle-safe, json-portable.
  - **`ghostbench/scoring.py`**: `Score` data class with multi-tier
    pass/fail (parse / fields / substrings / reserved semantic +
    behavioural). `Score.passed` is strict-AND across the
    *requested* tiers, not all possible ones. `score_record()` is
    the operator-facing entry point.
  - **`ghostbench/parsers.py`**: `DEFAULT_PARSERS` for the five
    bets that have a structural validator (STIX / YARA / Sigma /
    MISP / provenance). Bets 7 + 8 deliberately have no parser;
    the scorer treats parse as vacuously True for them and the
    substring tier carries the score.
  - **`ghostbench/bench.py`**: `Bench` (one bet) + `Suite`
    (collection) + `EvalRecord` / `Prediction` data classes.
    `Suite.from_dir(eval_dir, parsers)` discovers benches by the
    canonical filename convention.
  - **`ghostbench/reports.py`**: `render_run_report`,
    `render_per_format_breakdown`, `render_paired_comparison`
    (with McNemar p / Cohen h / paired-diff CI / honest
    interpretation paragraph), `render_suite_summary`,
    `render_suite_paired_comparison` (one-row-per-bench with
    significance markers).
  - **`ghostbench/plot.py`**: matplotlib visualisations:
    `plot_run_report`, `plot_suite_summary`,
    `plot_paired_comparison` (per-tier bars + significance
    markers + Cohen-h size labels), `plot_suite_paired_comparison`
    (forest plot with accent on significant rows),
    `plot_projections` (per-bench projection chart with two
    distinct uncertainty layers). Lazy-imported so the package
    is import-safe without matplotlib installed.
  - **`ghostbench/projections.py`**: scaling-law-based forecasts:
    exposure curve `asymptote * (1 - exp(-records / saturation_n))`
    with per-bench priors, +/-30% asymptote credibility band, and
    Wilson 95% statistical CI at the eval n. Output is a
    `Projection` dataclass; `render_projection_table()` produces
    the markdown table.
  - **`ghostbench/__main__.py`**: CLI entry: `python -m ghostbench
    [score | summary | compare | suite-compare]`.
  - **`ghostbench/tests/`**: 80 unit tests covering Wilson CI /
    Cohen's h / McNemar / paired-diff CI / multi-tier scoring /
    suite discovery / paired comparison / suite paired
    comparison / projections / plotting (skipped cleanly if
    matplotlib missing).
  - **`ghostbench/examples/`**: five PNG plots showing the full
    visualisation toolkit on synthetic data.

### Statistical rigour beyond the v0.9.5 in-script eval

  - Wilson CI for binomial proportions (right at small n, doesn't
    blow up at p near 0/1, less conservative than Clopper-Pearson).
  - Cohen's h with the standard small/medium/large cuts. Keeps
    the interpretation honest when a 6x relative lift at p=0.01
    is actually a "small" effect.
  - Exact two-sided McNemar's binomial test for paired comparisons
    (n_discordant <= 25 typical). Right tool when the same eval
    prompts are scored under two checkpoints.
  - Newcombe-style paired-difference Wilson-shifted CI on the
    proportion delta. Tighter than two independent Wilson intervals
    when the data are paired.

### `scripts/build_v15_combined_synth.py` + `scripts/run_all_baselines.py`

Two infrastructure scripts shipped earlier in the v0.9.6 cycle:

  - `build_v15_combined_synth.py` merges the five templated-synth
    JSONL outputs into one corpus tagged by training-time use
    (587 pretrain + 918 SFT = 1,505 total). The output is what
    the ghost-base trainer will read.
  - `run_all_baselines.py` is a one-command reproducer: runs all
    four held-out evals (bet 6 / 7 / 8 / 9) against any checkpoint
    and writes per-bet scoring reports plus a combined summary.
    Verified to reproduce the v0.9.5 0/87 baseline exactly.

### `docs/ghost_base_projections.md`

A standalone doc that pulls the v0.9.5 record counts through
`ghostbench.projections.project_suite()` and renders both the
markdown table and the chart. Sets calibrated expectations for
the GPU run:

  - bet 6 (well-resourced, 560 records): projected 61% [42.7-79.4]
  - bet 7 (under-resourced, 36 records):  projected  6% [4.4-8.1]
  - bet 8 (under-resourced, 29 records):  projected  3% [2.0-3.6]
  - bet 9 (well-resourced, 429 records):  projected 75% [52.8-98.0]

The doc explicitly calls out what the projections do NOT claim
(they're projections, not predictions; assumptions about training-
quality may not hold; the actual ghost-base measurement could fall
outside the band, in which case the gap itself is a finding worth
investigating). Two recommended pre-GPU interventions for bet 7
and bet 8 are documented.

### Why v0.9.6 matters

Before this release, GhostLM was a credible small-model project
with eval scaffolding embedded in scripts. After this release, the
eval layer is a properly-packaged, model-agnostic, statistically-
rigorous benchmark suite. The same `Suite` machinery that scores
GhostLM checkpoints can score SmolLM2, Qwen2.5-0.5B, Llama-3.2-1B
side-by-side; the `compare` and `suite-compare` CLI commands
produce the publication-grade artifact for "did the new
checkpoint actually beat the baseline at p<0.05."

That converts the project from "interesting work" to "reusable
research infrastructure." Reviewers / researchers / companies
landing on GhostLM see a benchmark they could adopt for their
own small-LM work, not just a model they can download.

---

The release that converts "six bets, three measured" into "nine
bets, all shipped, 1,505 deterministic templated SFT records ready
for the v1.0 GPU run." 11 commits since v0.9.4. The new bets answer
a strategic question: not "what makes GhostLM narrowly competent?"
but **"what makes GhostLM exceptional, beyond what general-purpose
small LMs offer?"**

### What's new vs v0.9.4

**Bet 1 (tool-use SFT) now has training data**, not just a
distillation pipeline. [`scripts/synth_tool_use.py`](scripts/synth_tool_use.py)
emits 424 parser-valid four-message traces (`USER -> ASSISTANT
tool_call -> TOOL response -> ASSISTANT answer`) seeded from the
existing CVE / MITRE / CWE / RAG corpus. 98.6% acceptance on the
same `trace_quality_ok` filter the LLM-distilled flow uses; the
~10% "not found" injection trains the model to acknowledge lookup
failures rather than confabulate. Detail in
[docs/tool_use_synth.md](docs/tool_use_synth.md).

**Bet 6 (format-aware) now has a held-out eval set**, not just a
gold few-shot bank. Eval grew 8 -> 32 records with no overlap
with the few-shot file, fixing the train-on-test leak.
[`scripts/eval_format_compliance.py`](scripts/eval_format_compliance.py)
now emits Wilson 95% CIs for binomial pass rates: at n=32 with 0
hits the upper bound is **10.7%** (vs 32.4% at n=8), so any
future ghost-base score above ~11% is statistically separated
from the v0.9 baseline. Re-baselined v0.9 chat lands at
**0/32 = 0.0% [0.0-10.7]** on the held-out eval.

**Bet 7 (code-for-security)**, NEW: hand-curated bank of 12
vulnerability patterns covering OWASP-Top-10-shaped CWE classes
(CWE-89, CWE-78, CWE-22, CWE-79, CWE-502, CWE-120, CWE-798,
CWE-330, CWE-327, CWE-347, CWE-611, CWE-918) across Python /
JavaScript / C. [`scripts/synth_code_security.py`](scripts/synth_code_security.py)
emits 4 record variants per pattern (pretrain prose / identify-
and-fix Q&A / explain-the-diff Q&A / CWE-mapping Q&A) = 48
records, 100% parser-pass. Detail in
[docs/code_security_synth.md](docs/code_security_synth.md).

**Bet 8 (binary / hex literacy)**, NEW and **the most novel of
the three**: hand-curated bank of 15 patterns across five
categories (file_magic / packer / shellcode / pe_field /
disassembly) covering PE / ELF / Mach-O / ZIP / PDF / OLE2 / PNG
file magic, UPX and Themida packer signatures, NOP sleds and x64
syscall patterns, PE Optional Header Magic and Machine fields,
and a canonical Linux x64 execve('/bin/sh') 28-byte shellcode.
[`scripts/synth_binary_literacy.py`](scripts/synth_binary_literacy.py)
emits 44 records (pretrain prose + identify-hex Q&A +
show-magic Q&A) at 100% parser-pass. **No other small cybersec
LM trains on this distribution; even GPT-4 fails on real
obfuscated shellcode without prompt engineering.** Detail in
[docs/binary_literacy_synth.md](docs/binary_literacy_synth.md).

**Bet 9 (provenance / cite tags)**, NEW: trains ghost-base to
emit `<|cite|>{source_type}:{source_id}#field<|/cite|>` tags
inline in tool-use answers, attaching every factual claim to the
specific tool-response field that justifies it.
[`scripts/synth_tool_use_provenance.py`](scripts/synth_tool_use_provenance.py)
emits 429 cite-augmented traces over the same seeds as bet 1,
99.8% acceptance under the new `trace_with_cites_quality_ok`
filter (requires a valid cite tag in the assistant's final
answer). Stacks on top of bet 1's 424 plain traces for an
~853-record SFT corpus that teaches both the four-message
tool-use rhythm AND the citation discipline. Detail in
[docs/provenance_synth.md](docs/provenance_synth.md).

### Combined templated-synth corpus (deterministic floor)

| Bet | Records | Acceptance | Doc |
|---|---:|---:|---|
| 1 (tool-use, plain) | 424 | 98.6% | [tool_use_synth.md](docs/tool_use_synth.md) |
| 6 (STIX / YARA / Sigma / MISP) | 560 | 99.8% | [format_synth.md](docs/format_synth.md) |
| 7 (code-for-security) | 48 | 100.0% | [code_security_synth.md](docs/code_security_synth.md) |
| 8 (binary / hex literacy) | 44 | 100.0% | [binary_literacy_synth.md](docs/binary_literacy_synth.md) |
| 9 (cite-augmented tool-use) | 429 | 99.8% | [provenance_synth.md](docs/provenance_synth.md) |
| **TOTAL** | **1,505** | **99.4%** | |

That's the deterministic floor. LLM-distilled records on top
(bet 1 production at ~$200, bet 6 production at ~$50-100 on
Anthropic) bring the realistic ghost-base SFT mix to ~10K records
for a few hundred dollars, with no GPU spend until the actual
pretrain run.

### Strategic frame: nine bets, multi-modal-in-security

The first six bets ([0.9.4]) made GhostLM tool-grounded,
continuously updated, cybersec-tokenized, long-context, sparsely-
activated, structurally literate. Bets 7-9 add: code-aware,
binary-aware, provenance-aware. The combined nine-bet identity is
**"the only small open-source LM designed end-to-end for the
security analyst workflow,"** which is a defensible position big
general-purpose small LMs structurally cannot occupy because
their pretrain mix dilutes these signals and their RLHF removes
exploit-shaped content. Bet 8 specifically (hex / PE / ELF
reading) is genuinely first-of-kind; reading a hex dump is a
clean eval and no other small cybersec LM does this natively.

Strategic frame in full at
[docs/differentiation.md](docs/differentiation.md).

---

The release that converts "v0.9.3 saturated at the small-model
plateau" into "here are six concrete, code-backed bets to be
genuinely different from the from-scratch-cybersec-LM crowd, three
of them already measured, the other three waiting on GPU." 24 commits
since v0.9.3, 94 tests in the suite (71 pre-existing + 23 new for
the differentiation artifacts), all green.

### Headline measurements

| Result | Number | Doc |
|---|---|---|
| Bet 3: v1 BPE compression on mixed corpus, vs GPT-2 BPE | **+1.6%** | [docs/differentiation.md](docs/differentiation.md) |
| Bet 3: v1 BPE on cybersec-only subset, vs GPT-2 BPE | **+4.0%** | [docs/bpe_corpus_ablation.md](docs/bpe_corpus_ablation.md) |
| Bet 3: v1_cyber BPE on cybersec subset, vs GPT-2 BPE | **+4.3%** (general regresses to -7.6%) | [docs/bpe_corpus_ablation.md](docs/bpe_corpus_ablation.md) |
| Bet 5: MoE 100-step smoke, all 4 pass criteria | **PASS** | [docs/moe_training_smoke.md](docs/moe_training_smoke.md) |
| Bet 6: v0.9 chat structural-compliance baseline (held-out) | **0/8 = 0%** | [docs/format_baseline_v09.md](docs/format_baseline_v09.md) |

The three measured bets all moved from "hypothesis" to "quantified
fact." Bet 3's +25-35% target is falsified; the real cap is around
+4% on cybersec text and the recommendation is GPT-2 BPE default,
v1 mixed as opt-in. Bet 5's MoE wiring trains under real backprop,
not just compiles. Bet 6's baseline is locked at 0%, so the lift
from any future format-aware training will be a real measurement
on unseen prompts.

### The six bets

Doc: [docs/differentiation.md](docs/differentiation.md). Strategic
frame: the v0.9.3 RAG diagnostic identified a real bottleneck (81M
extracts from supplied context 1% of the time), the parameter-count
escape hatch is expensive, and the more interesting moves are
architectural / training-recipe / ecosystem-level changes that other
from-scratch projects aren't attempting. Six bets, each with code
already in the repo:

Doc: [docs/differentiation.md](docs/differentiation.md). Strategic
frame: the v0.9.3 RAG diagnostic identified a real bottleneck (81M
extracts from supplied context 1% of the time), the parameter-count
escape hatch is expensive, and the more interesting moves are
architectural / training-recipe / ecosystem-level changes that other
from-scratch projects aren't attempting. Six bets, each with code
already in the repo:

1. **Tool-grounded model (bet 1).** [scripts/distill_tool_use.py](scripts/distill_tool_use.py).
   Train ghost-base on tool-use traces (`question -> tool_call ->
   tool_response -> answer`) so it learns "lookup before answering"
   instead of "guess from memory". 4 tools wired (NVD, MITRE, CWE,
   RAG). Quality filter requires literal tag strings + parseable
   tool-call JSON. Cost: ~$200 on Sonnet for 10K traces.

2. **Continuously-updated model (bet 2).** [scripts/daily_finetune.py](scripts/daily_finetune.py).
   Nightly LoRA tune over the previous 24h of fresh threat-intel,
   pushed to a date-stamped HF Models repo
   (`Ghostgim/GhostLM-daily-YYYY-MM-DD`). Base checkpoint stays
   fixed; consumers download adapter and merge at load time. Cost:
   ~1-2 GPU hours per day.

3. **Custom 32K BPE (bet 3).** [scripts/train_v1_bpe.py](scripts/train_v1_bpe.py).
   **Settled at +4.0% on cyber text, -2.5% on general text** (vs
   GPT-2 BPE 50K), measured on 500-record subsets via
   [scripts/score_tokenizer.py](scripts/score_tokenizer.py). A
   followup retrain on cyber-only corpus pushes cyber to +4.3% but
   regresses general to -7.6%; per-domain ablation in
   [docs/bpe_corpus_ablation.md](docs/bpe_corpus_ablation.md). The
   bet 3 hypothesis ("+25-35% on cybersec text") is falsified at
   the magnitude claimed; the real cap is around +4%. Recommendation:
   ghost-base default to GPT-2 BPE; v1 mixed BPE stays on the shelf
   as `GhostTokenizerV1` opt-in for cyber-only inference paths;
   v1_cyber not productised.

4. **Long context via RoPE NTK rebase (bet 4).** [scripts/extend_context_ntk.py](scripts/extend_context_ntk.py).
   Code Llama-style non-linear scaling so high-frequency RoPE
   components stay sharp while low-frequency ones stretch to 16K.
   Two modes: `--rebase-only` for zero-shot extension testing,
   full mode for production-grade fine-tune. Cost: ~3-5 GPU hours.

5. **MoE architecture for ghost-1B+ (bet 5).** SparseMoE class in
   [ghostlm/model.py](ghostlm/model.py), config flags in
   [ghostlm/config.py](ghostlm/config.py). 4 experts top-2 routing,
   parallel SwiGLU experts, Switch-Transformer load-balancing aux
   loss wired into `GhostLM.forward()` (trainer stays
   architecture-agnostic). Two new presets in `from_preset()`:
   `ghost-1b` (1536d / 24L / 24h / 4 experts = 2.1B total /
   ~1.2B active) and `ghost-3b` (2048d / 32L / 32h / 4 experts =
   6.0B total / ~3.3B active). **100-step training smoke PASS**
   ([docs/moe_training_smoke.md](docs/moe_training_smoke.md)): CE
   drops 5.55 -> 0.76, aux loss stays glued to 2.0 (uniform-routing
   equilibrium for n=4 K=2), gate gradients grow 10x as the
   optimizer shapes the router. The wiring is correct end-to-end,
   not just compiles.

6. **Format-aware pretrain (bet 6).** [scripts/distill_format_aware.py](scripts/distill_format_aware.py).
   Synthesize (natural_language to structured_artifact and back)
   pairs across four format families: STIX 2.1 indicators, YARA
   rules, Sigma detection rules, MISP event JSON. Each ships its
   own syntactic validator (`parse_stix`, `parse_yara`,
   `parse_sigma`, `parse_misp`) so unparseable teacher outputs get
   filtered before write. The structural lever (different *kinds*
   of text the model sees) is complementary to the bet 3
   token-density lever. ~$50-100 on Sonnet for 1K clean traces;
   free Ollama smoke-test path. **End-to-end measurable**: 8-record
   gold few-shot bank ([data/raw/format_aware_seeds.jsonl](data/raw/format_aware_seeds.jsonl)),
   8-record held-out eval set
   ([data/raw/format_aware_eval.jsonl](data/raw/format_aware_eval.jsonl),
   no overlap with the few-shot bank),
   [scripts/eval_format_compliance.py](scripts/eval_format_compliance.py)
   for scoring, [scripts/run_format_baseline.py](scripts/run_format_baseline.py)
   for one-shot inference over a checkpoint. **v0.9 chat baseline:
   0/8 parse, 0/8 fields = 0.0%**
   ([docs/format_baseline_v09.md](docs/format_baseline_v09.md)). The
   floor is 0%; any non-zero number after bet 6 lands is measured
   differentiation no other small from-scratch cybersec LM reports.

The strategic claim isn't that any one bet definitely works; it's
that the **combination** of six reasonable bets gives GhostLM a
defensible identity that parameter-scale-only roadmaps don't.

---

## [0.9.3] — 2026-05-07 — pre-GPU push: RAG layer, fact-recall v2, distillation pipeline, retrieval-vs-generation diagnostic

The release that closed out a long pre-GPU work session. No model
training; no GPU spend; one published HF dataset; one diagnostic
finding strong enough to validate the parameter-scaling thesis the
project has been pursuing since v0.6.0.

### Headline finding

Three numbers from running the new RAG layer end-to-end on the v0.9
chat checkpoint plus a separate retrieval-quality diagnostic that
strips the language model out of the loop. **Retrieval works,
generation doesn't.**

```
Retrieval@4 (no LM):              41 / 100  (41.0%)
v0.9-bare fact-recall v2:          1 / 100  (1.0%)
v0.9+RAG fact-recall v2:           0 / 100  (0.0%)
```

The retriever surfaces a passage containing the canonical answer
for ~41% of fact-recall questions, but the 81M chat-tuned model
can extract those facts in ~1% of cases. Worse: the longer
RAG-augmented prompt destabilizes generation into mode-collapse
("X X X X X" repetition), dropping the score from 1/100 to 0/100.
Adding the right context to the prompt makes the 81M model
*worse*, not better. At this parameter scale the model has not
just failed to memorize facts; it has failed to learn the
meta-skill of "use the context window to answer". Per-topic
retrieval@4 (mitre 93%, tool 83%, cwe 67%, owasp 0%, protocol 9%)
is itself diagnostic; the BGE-small embedder fails on short-label
queries. Full investigation in
[`docs/rag_diagnostic_findings.md`](docs/rag_diagnostic_findings.md).

This is the cleanest evidence to date for the parameter-scaling
diagnosis. Validates the RAG infrastructure (retrieval works);
validates the parameter-scaling thesis (generation fails); confirms
the v1.0 ghost-base GPU run is the right next move.

### Hardware pathway documented

`docs/hardware_pathway.md` ships the multi-year scale-ladder
hardware recommendation: **RTX 6000 Pro Blackwell 96GB** (~$10K used)
for a workstation that carries the project through ghost-7B with
fp8 native training; corpus is the harder ceiling than hardware
past ghost-3B (Chinchilla-optimal scales linearly: ghost-7B wants
140B tokens, current corpus is 363M, 480x short). 100B+ documented
as cluster territory; the realistic path past ghost-7B is
continued-pretrain on a borrowed base. ROADMAP cross-links the
new doc from Phase 5 / Phase 6 hardware-target rows.

### Pre-GPU artifacts

Eight discrete artifacts shipped without GPU spend:

- **Corpus contamination audit.**
  `scripts/audit_corpus_contamination.py`. Two-tier check (exact
  substring + 12-word shingle overlap) gating the v1.0 GPU spend.
  Smoke run confirmed clean; full 2500-question audit running on
  M4 with results in `docs/contamination_audit.md`.
- **Free-form fact-recall benchmark v2** (n=100 seed, growing to
  200). Three schema additions over v1: `boundary_match` (rejects
  "10" matching inside "100"), `disqualifiers` (voids credit if
  listed phrase appears, catches question echoing), and
  `must_appear` (composite-fact AND-semantics). Doc at
  `docs/fact_recall_v2.md`. Published as a public HF Dataset at
  `Ghostgim/cybersec-fact-recall` for other small-cybersec-LM
  projects to use as a measurable ruler. Baseline numbers across
  v0.4 / v0.7 / v0.9 chat: 0/100, 1/100, 1/100 respectively.
- **RAG layer wired into the demo Space.**
  `huggingface.co/spaces/Ghostgim/ghostlm` chat now runs in
  retrieval-augmented mode by default when the index is loaded.
  Embeds queries with BAAI/bge-small-en-v1.5, retrieves top-4
  from a 83K-chunk index over the cybersec corpus, prepends as
  "Reference passages" before generation. Gracefully falls back
  to bare chat with an honest "RAG: OFF" note when the index
  isn't loaded.
- **Streaming chat in the Space.** `chat_fn` is now a generator;
  the Space yields tokens as they're sampled instead of blocking
  for 15-25 s per reply. Same total wall-clock, immediate first-
  token, far better perceived UX.
- **HF Models repo card.** `Ghostgim/GhostLM-v0.9-experimental` now
  has proper README.md frontmatter with bench numbers in the
  model-index schema (CTIBench 28.9% / SecQA 39.3% / in-repo CTF
  59.2% / fact recall 1/50 v1, 1/100 v2). Surfaces in HF model
  search.
- **Distillation pipeline scaffold for ghost-3B+ corpus.**
  `scripts/distill_common.py` (provider abstractions for Ollama /
  Anthropic / OpenAI-compatible, resume-safe writer, 5-shingle
  dedup, quality filters) plus four per-type scripts:
  `distill_ctf_walkthroughs.py` (offensive register from
  MITRE / CAPEC), `distill_threat_modeling.py` (STRIDE from
  OWASP / CWE), `distill_deobfuscation.py` (RE walkthroughs from
  exploitdb / security_code), `distill_malware_analysis.py`
  (IR-style writeups from MITRE / CISA-KEV). Doc at
  `docs/distillation.md`: target volume 130K records / ~65M
  synthetic tokens, provider cost envelope ~$400-2000.
- **MCP tool harness expansion.** Three new tools:
  `ghostlm_search_cve_nvd` (live REST API to NIST NVD,
  deterministic), `ghostlm_lookup_mitre_technique` (local-corpus
  MITRE lookup, also deterministic), `ghostlm_rag_query`
  (retrieval-augmented chat using the same RAG index). `docs/mcp.md`
  splits tools into model-backed vs deterministic categories.
- **Quantization script for v0.9 chat.** `scripts/quantize_v09.py`
  produces fp16 (~162 MB) and int8 (`torch.ao.quantization.quantize_dynamic`,
  ~80-110 MB) artifacts from the bf16 checkpoint. GGUF export
  documented as ~1 week of future work.

### Threat-intel corpus expansion (ongoing)

Three new collectors landed:

- `scripts/collect_vendor_research.py` for 11 vendor TI feeds:
  Cisco Talos, Palo Alto Unit 42, CrowdStrike, Mandiant, Rapid7,
  Tenable, Sophos, ESET, Trend Micro, SANS ISC, Recorded Future.
- `scripts/collect_cisa_advisories.py` for CISA Cybersecurity
  Advisories (separate from KEV; technical bulletins, joint
  advisories with FBI/NSA/MI5, ICS-CERT alerts).
- `scripts/collect_misp_feeds.py` for open MISP OSINT feeds (CIRCL,
  BotvrijEU). Renders structured threat-intel events as prose.

First batch of an ongoing collector series. Future batches: FIRST
PSIRT, vendor whitepapers, paid threat-intel feeds.

### Diagnostic-only artifact

- `scripts/eval_rag_recall.py` measures retrieval@K independent of
  the language model. Distinguishes the two failure modes that look
  identical on the generation bench: retriever-broken vs LM-can't-
  extract. The 41/100 vs 0/100 split documented above is from this
  script.

### What's still pending after this release

- **Rented GPU access** for the ~26h ghost-base pretrain.
- **RAG index rebuild** over the v1.0 corpus (current index was
  built 2026-05-01 against the v0.4-era corpus; rebuilding will
  fix the owasp 0% retrieval@4 result).
- **BM25 sparse-retrieval fallback** for short-label queries
  (protocol 9% retrieval@4 is the symptom).
- **GGUF export** for llama.cpp / Ollama (~1 week of careful
  mapping work).
- **fact-recall v2 expansion to n=200** (more handwriting; n=100
  seed is broad enough to detect the parameter gate).
- **Context-extension fine-tune** to ctx-1024 for long-form CTI
  inputs (carried over from prior [Unreleased]).

---

### v1.0 corpus + ghost-base launcher (carried into 0.9.3 from earlier)

### v1.0 corpus expansion (2026-05-06)

Ghost-small saturated as a register-matching parrot at 81M params
(0-2% on free-form fact recall across the whole line; v0.9.2
postmortem). The v1.0 lever is parameter count plus corpus
diversity, since the previous corpus was cybersec-writeup-only.
Five new collectors landed on M4, all running in parallel, all
auto-merged via the existing `scripts/rebuild_corpus.py --max-cve-tokens 6000000`:

- `scripts/collect_security_code.py` + `data/security_code_repos.json`
  — shallow-clones 30 curated cybersec tool repos (pwntools,
  impacket, scapy, sqlmap, volatility3, capa, plaso, AFL++, nuclei,
  trivy, prowler, paramiko, pyca/cryptography, etc.) and walks
  source files matching .py / .c / .h / .cpp / .js / .ts / .go /
  .rs / .sh, capped at 2000 files per repo. SPDX license per
  record. **6,235 records / ~9M tokens** in the merged corpus.
- `scripts/collect_fineweb_edu.py` — streams
  `HuggingFaceFW/fineweb-edu` (ODC-BY, classifier-filtered
  educational subset of CommonCrawl) at edu-score >= 3, target 50K
  records. **47,510 records / ~46M tokens** of textbook-style
  general-language web text.
- `scripts/collect_nist_sp800.py` — pulls 26 curated NIST SP 800
  publications (RMF, controls, identity, IDS, zero trust, secure
  SDF, etc.) from nvlpubs.nist.gov, extracts text via pymupdf,
  chunks at 12K chars. US gov public domain. **1,001 chunks /
  ~2.6M tokens**.
- `scripts/collect_security_blogs.py` — RSS/Atom puller for 11
  curated security research blogs (Project Zero, PortSwigger,
  Trail of Bits, Google Security, GitHub SecurityLab, NCC Group,
  Doyensec, Krebs, DFIR Report, Ret2 Systems, MSRC). stdlib HTML
  body extractor strips chrome. **199 records / ~0.6M tokens** in
  the merged corpus.
- `scripts/collect_math_reasoning.py` — streams
  `open-web-math/open-web-math` (ODC-BY, math-filtered web subset)
  for chain-of-thought capability. **18,991 records / ~21M
  tokens**.

Plus a Wikipedia cybersec resume run (existing collector), now
**730 records / ~1M tokens** in the merged corpus.

**Final v1.0 corpus: 516,736 train / 27,049 val / ~363M tokens
across 26 sources spanning six domains** (cybersec writeup 73%,
general language 13%, math/reasoning 6%, code 2.4%, plus the
PRIMUS-FineWeb crawl mix at 27% baseline). 0 leakage between
train and val. Per-source breakdown in `CORPUS.md`.

### Ghost-base launcher shipped

`scripts/train_ghost_base.py` — clones the train_v07.py pattern,
deepens to 12 layers (keeps d_model 768, d_ff 3072, 12 heads).
Estimated ~360M params, in the parameter range where SmolLM2-360M
and Phi-3.5-mini report factual recall on cybersec MCQ starting to
emerge. Recipe defaults assume H100 / bf16 territory: per-device
batch 16, 30K steps with 2K warmup, lr 2e-4 cosine.

Acceptance gate per `docs/ghost_base_spec.md`:

> ≥40% per-perm avg on debiased CTIBench (n=2500), OR
> ≥65% on the in-repo CTF eval (n=30), OR
> ≥30% on the 50-question fact-recall set
>
> Passing any one validates the rung. Fact-recall is the truth
> metric: that's where ghost-small fails today.

### Hardware pathway documented (2026-05-06)

`docs/hardware_pathway.md` ships the multi-year scale-ladder
hardware recommendation: **RTX 6000 Pro Blackwell 96GB** (~$10K used)
for a workstation that carries the project through ghost-7B with
fp8 native training. Per-rung capability matrix from ghost-base
through ghost-13B; explicit framing that **corpus is the harder
ceiling than hardware past ghost-3B** (Chinchilla-optimal scales
linearly: ghost-7B wants 140B tokens, current corpus is 363M, 480x
short). 100B+ is documented as cluster territory, not viable
single-workstation; the realistic path past ghost-7B is
continued-pretrain on a borrowed base. ROADMAP cross-links the new
doc from Phase 5 / Phase 6 hardware-target rows.

### Pre-GPU work that doesn't need rented compute (2026-05-06 / 07)

Eight discrete artifacts shipped in a single push to make the v1.0
ghost-base run more meaningful and unblock follow-on capabilities
without spending GPU money:

- **Corpus contamination audit.**
  `scripts/audit_corpus_contamination.py`. Two-tier check: tier-1 is
  exact normalized-question substring (smoking-gun direct
  contamination); tier-2 is `>= 3x 12-word shingle overlap` (catches
  near-paraphrases). Scans all 35 corpus shards including the
  516K-record processed train. Gates the GPU spend per the risk
  register in `docs/ghost_base_spec.md`. Result lands in
  `docs/contamination_audit.md`.
- **Free-form fact-recall benchmark v2** (n=100 seed, growing to 200).
  `data/raw/fact_recall_bench_v2.jsonl` + `scripts/eval_fact_recall_v2.py`
  + `docs/fact_recall_v2.md`. Three schema additions over v1:
  `boundary_match` (rejects "10" matching inside "100"),
  `disqualifiers` (voids credit if listed phrase appears, catches
  question echoing), and `must_appear` (composite-fact AND-semantics).
  Topic distribution: 30 cve, 15 mitre, 15 cwe, 11 protocol, 10 owasp,
  10 crypto, 6 tool, 3 misc. Becomes the truth metric for the
  ghost-base acceptance gate; v0.9 chat scored 1/50 on v1, expected
  to remain near floor on v2 with the false-positive cleanup.
  Grader has 11 unit tests covering boundary edges, disqualifier wins,
  and must_appear semantics.
- **RAG layer wired into the demo Space.**
  `huggingface.co/spaces/Ghostgim/ghostlm` chat now runs in
  retrieval-augmented mode by default when the index is available.
  Embeds queries with BAAI/bge-small-en-v1.5 (~30 MB on disk, fp16 on
  the Space), retrieves top-4 from a 83K-chunk index over the
  cybersec corpus, prepends as "Reference passages" before the model
  generates. The model is not RAFT-trained yet so it just sees
  retrieved context as part of the user message. Gracefully falls
  back to bare chat with an honest "RAG: OFF" note when the index
  isn't loaded. Index files (rag/index.npy fp16, rag/chunks.jsonl,
  rag/meta.json) host alongside weights at
  `Ghostgim/GhostLM-v0.9-experimental` Models repo so the Space
  stays inside HF's 1 GB free-tier LFS cap.
- **Streaming chat in the Space.**
  `chat_fn` is now a generator; the Space yields tokens as they're
  sampled instead of blocking for 15-25 s per reply. New helper
  `generate_until_end_stream` yields the growing token list after
  every iteration; chat_fn decodes and yields the running text
  snapshot per Gradio's API contract. Same total wall-clock,
  immediate first-token, far better perceived UX.
- **HF Models repo card.**
  `Ghostgim/GhostLM-v0.9-experimental` now has proper README.md
  frontmatter with bench numbers in the model-index schema (CTIBench
  28.9% / SecQA 39.3% / in-repo CTF 59.2% / free-form fact recall
  1/50). Surfaces in HF model search.
- **Distillation pipeline scaffold for ghost-3B+ corpus.**
  `scripts/distill_common.py` (provider abstractions for Ollama /
  Anthropic / OpenAI-compatible), `scripts/distill_ctf_walkthroughs.py`
  (first per-type distill script), `docs/distillation.md` (target
  volume 130K records ≈ 65M synthetic tokens, provider cost envelope,
  smoke-test recipe). The corpus expansion lever past ghost-3B is
  synthetic data from a strong teacher model conditioned on existing
  GhostLM corpus seeds; this commit ships the plumbing.
- **MCP tool harness expansion.** Three new tools in
  `scripts/mcp_server.py`: `ghostlm_search_cve_nvd` (live REST API
  to NIST NVD, deterministic / no model invocation),
  `ghostlm_lookup_mitre_technique` (local-corpus MITRE lookup, also
  deterministic), `ghostlm_rag_query` (retrieval-augmented chat using
  the same RAG index the Space uses). `docs/mcp.md` updated to split
  tools into model-backed vs deterministic categories.
- **Quantization script for v0.9 chat.**
  `scripts/quantize_v09.py` produces fp16 (~162 MB) and int8
  (`torch.ao.quantization.quantize_dynamic`, ~80-110 MB) artifacts
  from the bf16 checkpoint. Inference-only; faster CPU latency on
  cpu-basic Spaces, fits a 4 GB VPS or Raspberry Pi. GGUF export
  for llama.cpp / Ollama is documented as ~1 week of future work
  (GhostLM's SwiGLU + RMSNorm + RoPE layout doesn't directly map
  to LLaMA-2's canonical GGUF tensor naming).

### Threat-intel corpus expansion (ongoing)

`scripts/collect_vendor_research.py` adds 11 vendor TI research
feeds missing from the existing security_blogs collector: Cisco
Talos, Palo Alto Unit 42, CrowdStrike, Mandiant, Rapid7, Tenable,
Sophos, ESET, Trend Micro, SANS ISC, Recorded Future. Output goes
to `data/raw/vendor_research.jsonl` with source-tag
`vendor_research`. First of an ongoing collector series; future
batches: CISA cybersecurity advisories beyond KEV, MISP feeds,
FIRST.org PSIRT.

(The earlier "What's still pending" list from when this content lived
under [Unreleased] is superseded by the consolidated post-release list
above; the RAG-augmented v0.9 cross-bench rerun item it referenced
has now been completed and produced the headline finding for this
release.)
