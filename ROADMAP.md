# GhostLM Roadmap

GhostLM is a multi-year, from-scratch effort. The released ghost-tiny is a 14.7M-param model on ~30M tokens — a learning artifact and a working pipeline, not a useful cyber-task model. The path to "useful" is the scale ladder below.

This roadmap is honest about what each rung needs (compute, corpus, time) and what each rung is expected to deliver. There are no shortcuts for "from scratch" at scale; the alternative path — fine-tuning a strong open base model — is acknowledged in the README and explicitly rejected for this project. Patience is a feature.

---

## Where we are: Phase 3.5 corpus rebalanced (v0.3.5), ghost-tiny refresh in flight

The released checkpoint is still v0.3.3 (Phase 3 ghost-tiny on the post-NVD-pull corpus, val_loss 3.4458). Phase 3.5 landed today as **corpus work** — the structural rebalance and three new diversity collectors. The ghost-tiny refresh on the rebalanced corpus is currently training on the Mac M4 (run-name `phase3.5_balanced`); when it finishes, the eval/benchmark pass will populate the v0.3.5 numbers below.

| Item | v0.3.3 (released) | v0.3.5 (training) |
|---|---|---|
| Variant | ghost-tiny | ghost-tiny |
| Params | 14.7M | 14.7M |
| Training tokens | ~30M (NVD 90%) | ~8.8M (NVD 65%, balanced) |
| Steps | 30,000 | 30,000 |
| Final val_loss | **3.4458** | _(in flight)_ |
| Cyber-text perplexity | 142.09 | _(pending eval)_ |
| Security task eval | 4/30 (13.3%, mode-collapsed) | _(pending eval)_ |

**Phase 2→3 headline still holds:** the recipe scales with data at fixed model size. Phase 3→3.5 is a *different* test — same model size, fewer total tokens but more balanced source mix. The val_loss number is not directly comparable (different val distribution), but the eval-task numbers will be — that's the cleaner read on whether diversity beats raw NVD volume at fixed parameters.

**Capability characterization (v0.3.3):** produces CVE-database register — proper CVE-style descriptions, security-prose grammar, real CVE phrasing in roughly the right context. Hallucinations are still rampant — form is right, facts are not. See MODEL_CARD's Sample Generations.

**Phase 1 + Phase 2 archived** as `checkpoints/best_model_phase{1,2}.pt` for archaeological reference.

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

## Phase 3.6 — Corpus volume (next)

The structural rebalance is done; the next track is volume on the non-NVD side. The corpus is currently 8.8M tokens — well below Chinchilla-optimal for ghost-small at 55M params (~1.1B tokens). Filling that gap means growing the diversity sources, not pulling more NVD.

| Source | Phase 3.5 | Phase 3.6 target |
|---|---|---|
| CTFtime real writeups | 473 | 3,000+ (expand event list, add CTFtime crawl beyond curated 28-event seed) |
| arXiv cs.CR | 2,000 abstracts | 5,000+ abstracts + selected full-text PDFs |
| Security research blogs | 0 | Project Zero, PortSwigger Research, Trail of Bits, Google Security blog (license-gated per source) |
| Tool docs | 0 | nmap, metasploit, burp, ghidra, pwntools (license per upstream tool) |
| Exploit-DB | partial (PR #19) | Full corpus with PoC code linked to CVE descriptions |
| Drop synthetic CTF | 3,000 (placeholder) | 0 (drop once real CTFtime + GitHub-CTF-repos exceed it in token volume) |
| **Total tokens** | **~8.8M** | **~50–100M** |

This is realistically a 3–6 month track. It does not require new compute — it runs in parallel with continued ghost-tiny iteration.

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
1. ✓ **Recipe-scales-with-data validated** — Phase 2→3 ghost-tiny refresh dropped val_loss 0.34 nats at fixed model size, same recipe + more data. Done.
2. ✓ **Source-mix structurally balanced** — Phase 3.5 brought NVD share from 90% to 65% with real diversity sources at 35%. The model can no longer learn "complete-the-CVE-template" as the dominant objective. Done.
3. ✗ **Corpus volume** — at 8.8M tokens post-rebalance, well below the ~1.1B target for Chinchilla-optimal training at 55M params. Phase 3.6 (corpus volume) needs to land first; otherwise ghost-small will overfit a small corpus and the comparison vs ghost-tiny won't be informative.

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
