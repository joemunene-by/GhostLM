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

## [Unreleased] — Upcoming

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
