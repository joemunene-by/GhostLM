# GhostLM Roadmap

GhostLM is a multi-year, from-scratch effort. The released ghost-tiny is a 14.7M-param model on ~30M tokens — a learning artifact and a working pipeline, not a useful cyber-task model. The path to "useful" is the scale ladder below.

This roadmap is honest about what each rung needs (compute, corpus, time) and what each rung is expected to deliver. There are no shortcuts for "from scratch" at scale; the alternative path — fine-tuning a strong open base model — is acknowledged in the README and explicitly rejected for this project. Patience is a feature.

---

## Where we are: Phase 3.5 canonical (v0.3.5); Phase 3.6 attempted, found ghost-tiny capacity ceiling (v0.3.7)

The current canonical model is **v0.3.5** (Phase 3.5 ghost-tiny on the rebalanced 8.8M-token corpus, val_loss 3.5518). The Phase 3.6 attempt added Exploit-DB (~3.77M tokens, 30% of the new corpus, total 12.56M) and re-trained ghost-tiny at the same 30K-step recipe. Result was a 14.4 pp regression on the eval suite (31.2% → 16.8%) with every existing per-source PPL 28–42% worse — ghost-tiny at 14.7M params is at capacity. The Phase 3.6 weights are preserved at `checkpoints/phase3.6_exploitdb/best_model.pt` as the cleanest ghost-small training target rather than promoted to canonical.

| Item | v0.3.5 (canonical) | v0.3.7 / Phase 3.6 (preserved) |
|---|---|---|
| Variant | ghost-tiny | ghost-tiny |
| Params | 14.7M | 14.7M |
| Training tokens | ~8.8M (NVD 65%, balanced) | ~12.56M (NVD 46%, +Exploit-DB 30%) |
| Steps | 30,000 | 30,000 |
| Final val_loss | **3.5518** | 3.8556 (different val distribution) |
| Cyber-text perplexity | **96.24** | _not benchmarked — regression already clear from per-source PPL_ |
| Security task eval (5×25=125) | **39/125 (31.2%)** | 21/125 (16.8%) — mode collapse on Vuln Type at 96% |

**The corpus-first thesis from Phase 3.5 ran out of headroom.** Phase 2→3 (3× training volume): +1.6 pp. Phase 3→3.5 (corpus rebalance, fixed steps): +11.2 pp. Phase 3.5→3.6 (corpus volume, fixed steps): −14.4 pp. More corpus at fixed model size doesn't keep paying. The next training rung is the model, not the data.

**Capability characterization (v0.3.5):** produces multi-register prose — CVE descriptions for CVE prompts, MITRE narrative for MITRE prompts, CTF writeup-style for CTF prompts. v0.3.3 collapsed everything to CVE register; v0.3.5 picks up source-specific cues. Hallucinations are still rampant — form is right, facts are not. See MODEL_CARD's Sample Generations.

**Phase 1 + Phase 2 archived** as `checkpoints/best_model_phase{1,2}.pt` for archaeological reference. **Phase 3** at `checkpoints/phase3_refresh/best_model.pt`. **Phase 3.6** at `checkpoints/phase3.6_exploitdb/best_model.pt`.

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

## Phase 4 — ghost-small (~55M params)

| Item | Value |
|---|---|
| Layers / d_model / heads | 6 / 512 / 8 |
| Params | ~55M (already wired in `GhostLMConfig.from_preset("ghost-small")`) |
| Hardware target | Mac M4 GPU/MPS (feasible on local hardware) |
| Training tokens (Chinchilla-optimal) | ~1.1B (20 tokens / param) |

The first scale-up rung. Validates whether the recipe scales — same architecture, same training loop, more layers, more dim, more data. Expected to produce noticeably more coherent generation than ghost-tiny but still well below "useful."

**Gating:**
1. ✓ **Recipe-scales-with-data validated** — Phase 2→3 ghost-tiny refresh dropped val_loss 0.34 nats at fixed model size, same recipe + more data.
2. ✓ **Source-mix structurally balanced** — Phase 3.5 brought NVD share from 90% to 65% with real diversity sources at 35%. The model can no longer learn "complete-the-CVE-template" as the dominant objective.
3. ✓ **ghost-tiny capacity ceiling found** — Phase 3.6 attempted volume-add (~12.56M tokens) at fixed model size, regressed 14.4 pp on the eval suite. This *unblocks* the ghost-small jump rather than gating it: there's no more headroom in ghost-tiny to extract from corpus work, so the next training rung is the model, not more data.
4. ⚠️ **Corpus volume vs. Chinchilla-optimal** — the 12.56M-token Phase 3.6 corpus is well below the ~1.1B target for Chinchilla-optimal training at 55M params. ghost-small on this corpus will likely overfit at long step counts; the goal of the first ghost-small run isn't to be Chinchilla-optimal — it's to test the capacity-reallocation hypothesis. If 55M params absorb the corpus without the per-source regression ghost-tiny showed, the hypothesis is confirmed and corpus expansion (toward 50–100M tokens) becomes the right path. If 55M still regresses, the diagnosis was wrong and we go back to the drawing board.

**Immediate next move:** ghost-small on the Phase 3.6 corpus, GPU-required. The same corpus that broke ghost-tiny is the cleanest test case.

---

## Phase 5 — ghost-base (~350M params)

| Item | Value |
|---|---|
| Layers / d_model / heads | 12 / 768 / 12 |
| Params | ~350M |
| Hardware target | Rented GPU (A100 / H100 hours, ~hundreds of hours) |
| Training tokens (Chinchilla-optimal) | ~7B |

The first rung that needs rented GPU compute. This is where domain-coherent generation should start to emerge — the model should be able to produce a few sentences of structurally correct cyber-text without falling apart. Still not factually reliable.

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
