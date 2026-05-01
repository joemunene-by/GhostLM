# GhostLM Roadmap

GhostLM is a multi-year, from-scratch effort. The released ghost-tiny is a 14.7M-param model on ~30M tokens — a learning artifact and a working pipeline, not a useful cyber-task model. The path to "useful" is the scale ladder below.

This roadmap is honest about what each rung needs (compute, corpus, time) and what each rung is expected to deliver. There are no shortcuts for "from scratch" at scale; the alternative path — fine-tuning a strong open base model — is acknowledged in the README and explicitly rejected for this project. Patience is a feature.

---

## Where we are: v0.5.0 — chat-v3 canonical chat model (36.9% on CTIBench MCQ)

**Current canonical model for chat / instruction following:** `checkpoints/phase5_chat_v3/best_model.pt` — Phase 4 ghost-small base + supervised fine-tune on a chat-format dataset that mixes templated cybersec instructions with 1,802 MCQ-format examples (2× oversampled). Lifts CTIBench MCQ accuracy from 17.8% (pretrain) → 19.0% (chat-v2 free-form) → **36.9% (chat-v3 MCQ-tuned)**, +19.1 pp / +447 questions correct over the pretrain baseline. Random baseline on 4-way MCQ is 25%; v3 is **1.48× random**.

**Current canonical base model for density / generation:** `checkpoints/phase4_ghost_small/best_model.pt` — ghost-small (~45M params) trained for 30k steps on the 12.56M-token Phase 3.6 corpus. Final val_loss 2.3535, a 1.20-nat (~3.3× perplexity) drop relative to Phase 3.5 ghost-tiny. Per-source perplexity dominates Phase 3.5 by 59–78% across every existing source and the new Exploit-DB source. The capacity-reallocation hypothesis is confirmed: 14.7M params couldn't hold seven sources at once; 45M params hold all seven without the tradeoff.

**v0.5 architecture switches (RoPE / SwiGLU / RMSNorm) are wired and gated on corpus expansion.** A `ghost-small-v0.5` preset in `GhostLMConfig.from_preset` flips all three on, and a forward+loss pass is verified end-to-end (45.0M params, matched parameter budget vs v0.4's 45.2M). The retrain that uses these switches doesn't ship until the v0.4.2 corpus expansion lands — there's no point retraining on the same 12.56M tokens when the architecture can absorb meaningfully more.

| Item | v0.3.5 ghost-tiny (historical canonical) | **v0.4.0 ghost-small (current canonical)** |
|---|---|---|
| Variant | ghost-tiny | **ghost-small** |
| Params | 14.7M | **~45M** |
| Training tokens | ~8.8M (NVD 65%, balanced) | **~12.56M** (NVD 46%, +Exploit-DB 30%) |
| Steps | 30,000 | **30,000** |
| Final val_loss | 3.5518 | **2.3535** (−1.20 nats vs P3.5) |
| Per-source PPL (overall) | 66.05 | **11.12** (−83%) |
| Security task eval, PMI | **39/125 (31.2%)** | 29/125 (23.2%) |
| Security task eval, logp | 22/125 (17.6%) | **24/125 (19.2%)** |
| Hardware | Mac M4 (CPU), ~3h13m | Mac M4 (MPS), ~15h |

**Why ghost-small is canonical despite the PMI dip:** for any density / generation use (the actual product) the model is unambiguously better — overall val PPL dropped by 5.9× (66.05 → 11.12), and **every single source improved 59–78% relative to the Phase 3.5 canonical**. The PMI scoring quirk that flatters Phase 3.5 vanishes under conservative logp scoring, where Phase 4 wins. See `CHANGELOG.md` v0.4.0 for the methodology analysis.

**The trajectory of training-recipe wins:**
- Phase 2→3 (3× training volume, fixed model+corpus mix): +1.6 pp on the suite
- Phase 3→3.5 (corpus rebalance, fixed model+steps): +11.2 pp
- Phase 3.5→3.6 (corpus volume, fixed model+steps): **−14.4 pp** (capacity ceiling)
- Phase 3.6→4 (model capacity, fixed corpus+steps): per-source PPL **−75% across the board**

**Capability characterization (v0.4.0):** ghost-small produces sharper register-matched prose than ghost-tiny — the same kinds of completions but with substantially fewer artifacts and broken token streams. Hallucinations are still rampant; form is right, facts are not. See MODEL_CARD's Sample Generations.

**Checkpoints on disk (in order of release):**
- Phase 1 / 2 archived: `checkpoints/best_model_phase{1,2}.pt`
- Phase 3: `checkpoints/phase3_refresh/best_model.pt`
- Phase 3.5 (historical canonical, better PMI scorer): `checkpoints/phase3.5_balanced/best_model.pt`
- Phase 3.6 (preserved learning artifact, capacity-ceiling diagnosis): `checkpoints/phase3.6_exploitdb/best_model.pt`
- **Phase 4 (current canonical):** `checkpoints/phase4_ghost_small/best_model.pt`

---

## Phase 3.5 — Corpus rebalance (complete, 2026-04-26)

Corpus is the long-term moat. Phase 3 brought volume (~30M tokens) but with NVD at ~90% token share, which made every other source statistically irrelevant. Phase 3.5 fixed the structural problem first; volume comes next.

**What landed:**
- **MITRE ATT&CK collector** — 691 enterprise techniques (Apache 2.0)
- **CAPEC collector** — 609 attack patterns (Apache 2.0)
- **CTFtime real-writeup collector** — 473 inline writeups across 28 curated 2020-2024 events; per-record attribution; off-site links deliberately not followed (per-page licensing not auditable)
- **GitHub-CTF-repos collector** — config-driven, JSON list of repos with explicit SPDX license per entry
- **NVD subsampling** — `rebuild_corpus.py --max-cve-tokens N` deterministically caps NVD's contribution by content-hash prefix. Without it, NVD owns ~90%; at `--max-cve-tokens 6000000` NVD owns 65.3% with diversity sources at ~35%.

**Current corpus state:**

| Source | Records | Tokens | Share |
|---|---|---|---|
| NVD CVE (subsampled) | 71,828 / 333,540 | 5.74M | 65.3% |
| Synthetic CTF (placeholder) | 3,000 | 1.51M | 17.2% |
| arXiv cs.CR abstracts | 2,000 | 0.74M | 8.4% |
| CTFtime real writeups | 467 | 0.47M | 5.3% |
| MITRE ATT&CK | 691 | 0.26M | 2.9% |
| CAPEC | 609 | 0.07M | 0.9% |
| **Total (post-dedup)** | **74,635** | **~8.8M** | |

Per-source license notes: see [CORPUS.md](CORPUS.md). Reproducible via `python3 scripts/rebuild_corpus.py --max-cve-tokens 6000000` (deterministic by content hash).

---

## Phase 3.6 — Corpus volume (attempted, capped by model capacity)

The structural rebalance worked; the volume add did not — at this rung. Phase 3.6 added Exploit-DB (~3.77M tokens, 30% of the new corpus mix) and re-trained ghost-tiny at the same 30K-step recipe. Eval suite regressed 14.4 pp (31.2% → 16.8%) and every existing per-source PPL got 28–42% worse. ghost-tiny at 14.7M params is at capacity — adding 43% more diverse text forces parameter reallocation away from the existing seven sources.

**What landed in v0.3.7 (corpus-side):**
- Exploit-DB collector hardened (persistent mirror, resume, Metasploit filter, structured metadata, date-desc CSV sort)
- 5,000 Exploit-DB records pulled (~3.77M tokens, GPL-2.0, mostly PHP webapps + Linux locals + Python PoCs from 2019–2025)
- arXiv full-text PDF collector built (uses pymupdf; not yet pulled at scale)
- CTFtime event-discovery script built (queries CTFtime API, filters by weight + participants; not yet run)
- v0.4.0 corpus-target tracker added to `scripts/data_audit.py`

**Status of the originally-planned Phase 3.6 sources:**

| Source | Phase 3.5 | v0.3.7 status |
|---|---|---|
| Exploit-DB | 0 | ~3.77M tokens pulled, 30% of corpus |
| arXiv cs.CR | 2,000 abstracts (~0.74M) | Full-text scaffolding shipped, not yet pulled |
| CTFtime real writeups | 473 | Discovery script shipped, not yet expanded |
| Security research blogs | 0 | Not yet built |
| Tool docs | 0 | Not yet built |
| **Total** | **~8.8M** | **~12.56M (Phase 3.6 corpus, sitting in `data/processed/`)** |

The Phase 3.6 corpus is reproducible: `python3 scripts/rebuild_corpus.py --max-cve-tokens 6000000`. It's the cleanest training target for the ghost-small run.

**The lesson:** more corpus at fixed model size has hit diminishing returns. The path forward is the model, not the data.

---

## Phase 4 — ghost-small (complete, 2026-04-30)

| Item | Value |
|---|---|
| Layers / d_model / heads | 6 / 512 / 8 |
| Params | ~45M actual (the 55M estimate was the config-side hand-waved figure; real count from `model.num_params()` is 45.17M) |
| Hardware | Mac Mini M4 (MPS), batch 8 × grad_accum 4, ~15h wall-clock |
| Training tokens | 12.56M (Phase 3.6 corpus, unchanged) |
| Steps | 30,000 |
| Final val_loss | **2.3535** |
| Best checkpoint | `checkpoints/phase4_ghost_small/best_model.pt` |

The first scale-up rung delivered cleanly. Loss curve was still descending at step 30k (train ~2.17, val 2.35) — no overfitting plateau visible at this step budget on this corpus, despite the corpus being well below Chinchilla-optimal for 45M params (~1.1B tokens would be the formal target).

**All four gates closed positively:**
1. ✓ Recipe-scales-with-data validated (Phase 2→3, +1.6 pp eval).
2. ✓ Source-mix structurally balanced (Phase 3→3.5, +11.2 pp eval, NVD 90% → 65%).
3. ✓ ghost-tiny capacity ceiling found (Phase 3.6, −14.4 pp + per-source PPL 28–42% worse on every source).
4. ✓ **Capacity-reallocation hypothesis confirmed.** ghost-small at 45M absorbed the same Phase 3.6 corpus that broke ghost-tiny, with **every existing source improving 59–78%** relative to the Phase 3.5 canonical. Overall val PPL dropped 66.05 → 11.12 (−83%).

**Eval methodology finding (worth flagging for v0.4.x):** the existing PMI security suite favored Phase 3.5 (31.2% vs Phase 4's 23.2%), but with conservative logp scoring Phase 4 wins (19.2% vs 17.6%). PMI subtracts unconditional candidate log-prob to break ties; a higher-capacity model with a tighter probability distribution gives PMI less separation. The 25-sample-per-task suite is small enough that this calibration asymmetry can flip headlines. Resolving cleanly probably requires both: (a) a bigger eval set, (b) a calibration-stable scoring rule.

---

## Phase 4.x — extension and corpus expansion (next on the critical path)

Phase 4 left two cheap, valuable follow-ups before the ghost-base jump:

- **v0.4.1 — extension run.** Train the existing ghost-small for another 30k–60k steps on the Phase 3.6 corpus and observe whether val_loss continues past 2.35 or plateaus into overfit. Cheap (one more overnight on M4). Tells us whether the 30k recipe was meaningfully undertrained at this corpus size, and gives a credible upper bound on what ghost-small can squeeze out of 12.56M tokens.
- **v0.4.2 — corpus expansion (Phase 4.5 corpus).** Run the arXiv full-text PDF collector at scale (collector landed in v0.3.6 but data not pulled), expand CTFtime past the curated 28-event seed via the discovery script, and ship the security-research-blogs collector. Target: ~50M-token corpus that ghost-small can train on without the per-source mode collapse Phase 3.6 forced on ghost-tiny. This is the corpus that makes Phase 5 (ghost-base) actually informative rather than overfit-prone.

Both are doable on local hardware over weeks. ghost-base is gated on at least the corpus expansion landing.

---

## Phase 5 — ghost-base (~350M params)

| Item | Value |
|---|---|
| Layers / d_model / heads | 12 / 768 / 12 |
| Params | ~350M |
| Hardware target | Rented GPU (A100 / H100 hours, ~hundreds of hours) |
| Training tokens (Chinchilla-optimal) | ~7B |

The first rung that needs rented GPU compute. This is where domain-coherent generation should start to emerge — the model should be able to produce a few sentences of structurally correct cyber-text without falling apart. Still not factually reliable.

**Gating (after v0.4.0):**
1. ✓ ghost-small recipe validated (Phase 4 confirmed scaling works).
2. ⚠ Corpus volume — at 12.56M tokens, ~7B Chinchilla-optimal is 550× away. Phase 4.5 expansion needs to land first; otherwise ghost-base will overfit a tiny corpus before generalization kicks in.
3. ⚠ External GPU compute or owned 4090/5090-class hardware.

**Cost estimate:** at ~$2–3/H100-hour, a Chinchilla-optimal run is on the order of low-thousand-dollar compute. Doable as a focused-burst project; not casual.

---

## Phase 6 — ghost-1B (long-term goal)

| Item | Value |
|---|---|
| Layers / d_model / heads | 24 / 1024 / 16 |
| Params | ~1B |
| Hardware target | Rented H100 cluster, or owned GPU (RTX 4090/5090 class for slow-but-feasible) |
| Training tokens (Chinchilla-optimal) | ~20B |

The smallest scale at which a from-scratch cyber LM has a real shot at being **genuinely useful** for tasks like CVE-to-exploit explanation, CTF challenge reasoning, or structured log analysis. Note that "useful" does not mean "competitive with general-purpose 7B+ models" — those have ~20× the params and ~100× the training data. ghost-1B's value proposition is *narrow domain depth*, not breadth.

**Cost estimate:** Chinchilla-optimal training of a 1B model is in the ten-thousand-dollar range on rented compute. Or several months on a single owned 4090/5090. This is the rung where the project either gets serious external support, gets done over years on consumer hardware, or stalls.

---

## Realistic timeline

A useful from-scratch 1B cyber LM is **2–3 years of sustained evenings/weekends work** — not because the steps are hard individually, but because corpus curation is slow, compute access at scale is gated by money or patience, and each scale rung needs the previous rung's recipe to be validated first.

This is the actual shape of the work. There are no shortcuts for "from scratch."

What that timeline does *not* require:
- New architecture inventions (the recipe is stable)
- A team (single-maintainer is feasible at this pace)
- Continuous compute (corpus and eval work fills the gaps)

What it does require:
- Corpus curation as a first-class, ongoing track (see CONTRIBUTING.md)
- Eval harness built before scale-up so improvements are measurable
- Patience.

---

## Adjacent tracks (not on the critical path)

- **Eval harness expansion** — held-out CVE→description, vuln-type classification, exploit-vs-benign code, CTF-challenge classification. Build before scaling so we can detect real progress vs. memorization.
- **HuggingFace Hub publication** — once ghost-small has a checkpoint worth publishing, push safetensors weights + config sidecar.
- **Gradio web demo** — for ghost-small or above. Not worth doing on ghost-tiny.
- **Fine-tuning scripts** — once ghost-base or ghost-1B exists, expose adapters / LoRA pipelines so users can specialize the base model further.

These all become valuable at the upper rungs. Doing them on ghost-tiny would be premature.
