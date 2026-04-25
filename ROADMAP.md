# GhostLM Roadmap

GhostLM is a multi-year, from-scratch effort. Where we are today (a 14.7M-param model on ~2.7M tokens) is the starting line — not a useful cyber-task model. The path to "useful" is the scale ladder below.

This roadmap is honest about what each rung needs (compute, corpus, time) and what each rung is expected to deliver. There are no shortcuts for "from scratch" at scale; the alternative path — fine-tuning a strong open base model — is acknowledged in the README and explicitly rejected for this project. Patience is a feature.

---

## Where we are: Phase 2 complete (v0.3.0)

| Item | Value |
|---|---|
| Variant | ghost-tiny |
| Params | 14.7M |
| Training tokens | ~2.66M (post-dedup) |
| Steps | 10,000 (Phase 1 + Phase 2 resumed on rebalanced corpus) |
| Final val_loss | 3.7813 (perplexity ≈ 44) |
| Compute used | Mac Mini M4 (CPU), ~70 min wall-clock for the Phase 2 portion |

Capability characterization: vocabulary is on-domain (CTF terms, exploit techniques, CVE format, common vuln types). Grammar is broken; semantics are absent; topic-binding fails. See MODEL_CARD's Sample Generations.

This is the expected level for the params/tokens budget. The fix is scale — both model and corpus — not more steps at this scale.

---

## Phase 3 — Corpus expansion (next)

Corpus is the long-term moat. It compounds across every future scale rung and is the work that pays off even when compute is the bottleneck. Target growth: **10–100× the current corpus size.**

| Source | Current | Target |
|---|---|---|
| NVD CVE | 19,925 | Full NVD dump (millions, properly chunked) |
| arXiv cs.CR | 1,000 | 10,000+ (broader date range, full abstracts + selected full-text) |
| CTF writeups | 3,000 (synthetic) | Replace with real corpus from CTFtime + GitHub writeup repos |
| Security research blogs | 0 | Project Zero, PortSwigger Research, Trail of Bits, Google Security blog, etc. |
| MITRE ATT&CK | 0 | Full structured + unstructured |
| Tool docs | 0 | nmap, metasploit, burp, ghidra, pwntools, etc. |
| **Total tokens** | **~2.66M** | **~1B** |

Tracking and per-source license notes: see [CORPUS.md](CORPUS.md).

This is realistically a 6–12 month effort done well. It does not require new compute — it can run in parallel with continued ghost-tiny iteration.

---

## Phase 4 — ghost-small (~55M params)

| Item | Value |
|---|---|
| Layers / d_model / heads | 6 / 512 / 8 |
| Params | ~55M (already wired in `GhostLMConfig.from_preset("ghost-small")`) |
| Hardware target | Mac M4 GPU/MPS (feasible on local hardware) |
| Training tokens (Chinchilla-optimal) | ~1.1B (20 tokens / param) |

The first scale-up rung. Validates whether the recipe scales — same architecture, same training loop, more layers, more dim, more data. Expected to produce noticeably more coherent generation than ghost-tiny but still well below "useful."

**Gating:** Phase 3 corpus expansion needs to be reasonably far along before this run starts (need at least ~500M–1B training tokens for the Chinchilla-optimal budget). Otherwise the larger model will overfit a small corpus and regress.

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
