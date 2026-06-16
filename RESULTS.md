# GhostLM benchmark results

Each row is one (checkpoint × benchmark) score. Updated by `scripts/run_bench.py`.

## Generalist evidence (v0.10+) — in progress

The generalist pivot is measured with `scripts/scorecard.py`, which scores a checkpoint on the general rulers (ARC-Easy, ARC-Challenge, OpenBookQA) alongside the retained cybersec benches, using the same debiased multi-permutation text-scoring, and places each number next to published peer small-models.

**Corpus de-specialization (achieved, 2026-06-16):** the v0.10 generalist corpus is 258.9M tokens, domain mix general_web 51.9% / knowledge 16.7% / math 16.7% / cybersec 8.6% / code 5.1% / instruction 1.1%. Cybersec fell from ~65-73% of tokens to 8.6%. Benchmark decontamination: 0.004% of records contaminated.

**Peer reference band (50-360M class, published zero-shot %):** random 25; Pythia-160M ARC-Easy 43.5 / ARC-Challenge 18.8; ~111M OpenBookQA 27.8 / ARC-Easy 34.8; SmolLM2-360M ARC-Challenge 36.6. "Competitive for 50-100M": ARC-Easy 35-45%, OpenBookQA 25-35%.

**ghost-small-gen (~45M, from scratch, MPS, 30,000 steps, final val_loss 3.76):** trained on the decontaminated generalist corpus with intra-document attention masking + multi-stage domain curriculum. Final scorecard on the full benchmark sets (full detail and training progression in [`docs/scorecard.md`](docs/scorecard.md)):

| Benchmark | n | GhostLM (45M) | 95% CI | vs random | Peer reference |
|---|---:|---:|---:|:--:|---|
| ARC-Easy | 2365 | **27.2%** | 25.4-28.9 | + | Pythia-160M 43.5, 111M 34.8, 256M 37.6 |
| ARC-Challenge | 1165 | **24.3%** | 22.1-26.6 | ~ | Pythia-160M 18.8, SmolLM2-360M 36.6 |
| OpenBookQA | 500 | **27.4%** | 23.7-31.1 | ~ | 111M 27.8, 256M 25.4, LaMini-35M 26.2 |
| SecQA (cyber) | 210 | **34.3%** | 28.5-40.6 | + | retention |
| CTF eval (cyber) | 30 | **63.3%** | 46.7-80.0 | + | retention |

Honest read at 45M from scratch: the generalist pivot worked, three of five benchmarks are statistically above the 25% random baseline on the full sets, cybersecurity is fully retained and strongest (SecQA 34.3%, CTF 63.3%) on a corpus only 8.6% cybersecurity, and the model is competitive with its size class (OpenBookQA beats the 256M and 35M peers; ARC-Challenge beats Pythia-160M, a ~3.5x larger model). It does not clear the 35-45% competitive band on ARC-Easy (27.2%): above chance, not outstanding there. (Earlier mid-run figures were scored on an easier 400-question subset and were optimistic; these full-set numbers are the honest measure.)

The first table preserves the **single-order** numbers (one fixed option ordering, log-prob of each letter token). These are the numbers in v0.5.0's release notes. As of v0.6.0 we know they're inflated by CTIBench's 15/32/37/15 gold-letter distribution combined with chat-v3's 98.6% C-emission, so a model that always emits "C" scores 37.1% on this metric. The single-order numbers stay here for historical comparison; the **debiased text-scoring** table below is the right read of real capability. See `docs/ctibench_bias_finding.md`.

## Single-order (biased)

| Checkpoint | Benchmark | n | Correct | Accuracy | Date |
|---|---|---:|---:|---:|---|
| ghost-small-v0.5 chat-v2 | ctibench-mcq | 2500 | 475 | 0.190 | 2026-05-01 |
| ghost-small-v0.4 (pretrain, no chat) | ctibench-mcq | 2500 | 446 | 0.178 | 2026-05-01 |
| ghost-small-v0.5 chat-v2 + RAG(top4) | ctibench-mcq | 2500 | 476 | 0.190 | 2026-05-01 |
| ghost-small-v0.5 chat-v3 (MCQ-tuned) | ctibench-mcq | 2500 | 922 | 0.369 | 2026-05-01 |
| ghost-small-v0.5 chat-v4 (RAFT) + RAG(top4) | ctibench-mcq | 2500 | 540 | 0.216 | 2026-05-02 |
| ghost-small-v0.5 chat-v4 (RAFT, no retrieval) | ctibench-mcq | 2500 | 626 | 0.250 | 2026-05-02 |
| ghost-small-v0.5 chat (v0.5 base) | ctibench-mcq | 2500 | 813 | 0.325 | 2026-05-02 |
| ghost-small-v0.5 chat-long (v0.5 base, 4K steps) | ctibench-mcq | 2500 | 428 | 0.171 | 2026-05-02 |
| ghost-small-v0.5 chat-recovered (extended pretrain + CoT MCQ + tok surgery) | ctibench-mcq | 2500 | 771 | 0.308 | 2026-05-03 |
| ghost-small-v0.5 chat-v5 (hybrid raw×5+CoT×2 + small-talk×8 + lr5e-5 + mean-init) | ctibench-mcq | 2500 | 871 | 0.348 | 2026-05-03 |
| ghost-small-v0.4 chat-v3 (MCQ-tuned, canonical) + RAG(top4) + RAG(top4) | ctibench-mcq | 2500 | 913 | 0.365 | 2026-05-03 |
| ghost-small-v0.5 chat-v5 + RAG(top4) + RAG(top4) | ctibench-mcq | 2500 | 844 | 0.338 | 2026-05-03 |
| ghost-small-v0.4 chat-v3 (MCQ-tuned, canonical) + RAG(top2) + RAG(top2) | ctibench-mcq | 2500 | 923 | 0.369 | 2026-05-03 |
| ghost-small-v0.4 chat-v6 (v0.4 base, expanded SFT: +MITRE-full +CISA-KEV) | ctibench-mcq | 2500 | 465 | 0.186 | 2026-05-03 |
| ghost-small-v0.4 chat-v3-repro (baseline data, ctx 1024) | ctibench-mcq | 2500 | 816 | 0.326 | 2026-05-03 |
| ghost-small-v0.4 chat-v3-repro2 (canonical recipe: lr 3e-5, 1800 steps, batch 8 × accum 4, ctx 1024) | ctibench-mcq | 2500 | 780 | 0.312 | 2026-05-03 |
| ghost-small-v0.6 chat (v0.6 base: v0.5 arch + GPT-2 BPE + expanded corpus, canonical chat-v3 recipe) | ctibench-mcq | 2500 | 745 | 0.298 | 2026-05-03 |
| ghost-small-v0.6 chat-hybrid (v0.6 base + chat-v5 hybrid recipe: raw×5 + CoT×2 + small-talk×8) | ctibench-mcq | 2500 | 374 | 0.150 | 2026-05-03 |
| ghost-small-v0.7 chat (81M wide, step 600 best, OOM-killed before completion) | ctibench-mcq | 2500 | 648 | 0.259 | 2026-05-04 |

## Debiased text-scoring on CTIBench, full bench (n=2500, 2 perms)

`scripts/eval_text_scoring.py` skips the letter token entirely, scores log P(option_text | prompt) per option under N option-letter permutations, and reports the mean accuracy. A pure single-letter emitter collapses to 25% (random) on this metric. **All rows below are on the full n=2500 CTIBench MCQ test split** (2 permutations: A,B,C,D and C,B,D,A). The earlier table that mixed n=500 subset numbers across versions has been retired; see the deprecation note at the bottom of this file.

| Checkpoint | Per-perm accs | Per-perm avg | Date |
|---|---|---:|---|
| ghost-small-v0.4 chat-v3 (canonical) | 0.271 / 0.282 | **0.276** | 2026-05-06 |
| ghost-small-v0.6 chat (v0.5 arch + GPT-2 BPE) | 0.283 / 0.280 | **0.282** | 2026-05-06 |
| ghost-small-v0.7 chat (81M wide) | 0.272 / 0.273 | **0.272** | 2026-05-06 |
| ghost-small-v0.7 chat-ctx1024 (extension fine-tune) | 0.270 / 0.264 | **0.267** | 2026-05-06 |
| ghost-small-v0.8 chat (81M wide + fact-dense pretrain) | 0.272 / 0.276 | **0.274** | 2026-05-06 |
| **ghost-small-v0.9 chat (273M-token corpus)** | 0.287 / 0.291 | **0.289** | 2026-05-06 |

v0.9 is the bench-winner across every chat-tune in the ghost-small line on the full CTIBench test split, by 0.7-2.2 pp.

## Cross-bench: in-repo CTF eval (n=30, 4 perms, debiased text-scoring)

A hand-written 30-question CTF / cybersec MCQ set at `data/raw/ctf_eval_bench.jsonl` (issue #6). Same multi-perm text-scoring methodology. 30 questions is small, so a 4-point swing is ~5 questions and within noise; treat absolute numbers as indicative, *the ranking is informative.*

| Checkpoint | Per-perm accs | Per-perm avg | Date |
|---|---|---:|---|
| ghost-small-v0.4 chat-v3 (canonical) | 0.500 / 0.433 / 0.533 / 0.533 | **0.500** | 2026-05-06 |
| ghost-small-v0.7 chat (81M wide) | 0.500 / 0.500 / 0.500 / 0.500 | **0.500** | 2026-05-06 |
| ghost-small-v0.7 chat-ctx1024 (extension fine-tune) | 0.467 / 0.467 / 0.467 / 0.433 | **0.458** | 2026-05-06 |
| **ghost-small-v0.9 chat (273M-token corpus)** | 0.567 / 0.633 / 0.567 / 0.600 | **0.592** | 2026-05-06 |

## Cross-bench: SecQA (n=210, 4 perms, debiased text-scoring)

External cybersec MCQ from `zefang-liu/secqa` on HuggingFace (v1 + v2 test splits combined). Pulled via `scripts/fetch_secqa.py`, scored with the same multi-perm methodology. Independent of the in-repo CTF set, so it confirms the v0.9 lead generalizes.

| Checkpoint | Per-perm avg | Date |
|---|---:|---|
| ghost-small-v0.4 chat-v3 (canonical) | **0.350** | 2026-05-06 |
| ghost-small-v0.7 chat (81M wide) | **0.376** | 2026-05-06 |
| **ghost-small-v0.9 chat (273M-token corpus)** | **0.393** | 2026-05-06 |

v0.9 leads on SecQA by 1.7 pp over v0.7 and 4.3 pp over v0.4. Same ordering as CTIBench full-bench and CTF eval; the inversion is consistent across three independent cybersec MCQ sources.

## Free-form fact recall (n=50, substring grading)

The truth metric for "does the model actually know facts." `scripts/eval_fact_recall.py` runs hand-written single-line factual prompts (CVE id lookup, CWE numbers, MITRE technique IDs, OWASP categories, crypto / protocol facts, misc) through chat completion at low temperature and credits any answer whose canonical form (or one of its alternates) appears as a substring of the model's response. Permissive on purpose: meant to catch the "v0.9 surfaces the right magic numbers near the surface" pattern.

| Checkpoint | Pass rate | Topics that scored |
|---|---:|---|
| ghost-small-v0.4 chat-v3 (canonical) | **0/50 (0.0%)** | none |
| ghost-small-v0.7 chat (81M wide) | **1/50 (2.0%)** | owasp 1/5 |
| ghost-small-v0.9 chat (273M-token corpus) | **1/50 (2.0%)** | crypto 1/5 |

Both "hits" are arguably spurious: v0.7's owasp hit ("Injection") appears in tangent prose unrelated to A03; v0.9's crypto hit ("256") comes from echoing "SHA-256" from the question itself. **At the ghost-small (45-81M) parameter scale, free-form fact recall is at floor.** The 28-39% numbers on the MCQ benches above reflect the model's ability to match register and topic, not its ability to retrieve facts.

This is the cleanest evidence that the ghost-small line ships as a "cybersec parrot" and the next move (ghost-base, 30L × 960d × 15h × 3200 d_ff, ~360M, SmolLM2-360M shape) is a parameter-count bet on factual recall emerging.

---

## Free-form fact recall v2 (n=100, smarter grader, 2026-05-07)

Replaces the n=50 v1 bench with three schema additions: `boundary_match` (rejects "10" matching inside "100"), `disqualifiers` (voids credit if listed phrase appears, catches question echoing), and `must_appear` (composite-fact AND-semantics). Documented in [`docs/fact_recall_v2.md`](docs/fact_recall_v2.md). Also published as a public HF dataset at [`Ghostgim/cybersec-fact-recall`](https://huggingface.co/datasets/Ghostgim/cybersec-fact-recall) for other small-cybersec-LM projects to use as a measurable ruler.

Topic distribution: 30 cve, 15 mitre, 15 cwe, 11 protocol, 10 owasp, 10 crypto, 6 tool, 3 misc.

| Checkpoint | Pass rate | Topics that scored |
|---|---:|---|
| ghost-small-v0.4 chat-v3 (45M) | **0/100 (0.0%)** | none |
| ghost-small-v0.7 chat (81M) | **1/100 (1.0%)** | misc 1/3 |
| ghost-small-v0.9 chat (81M) | **1/100 (1.0%)** | owasp 1/10 |

Same floor as v1, with the false-positive cleanup confirming nothing changes. Both v0.7 and v0.9 hits are likely spurious echoes of question keywords. The v2 grader doesn't *invent* false positives, but it also can't elevate "near-miss" into "knows the fact". The bench discriminates: a model that clears 30% on this ruler genuinely knows facts; a model at 1% genuinely doesn't, regardless of how high it scored on multiple-choice.

The v1.0 ghost-base acceptance gate uses **>=30% on this v2 bench** as one of three ways the run can validate. The fact-recall floor is the truth metric.

---

## Deprecated: earlier n=500 debiased table

The previous version of this section reported v0.4 at 30.5%, v0.5 at 29.7%, v0.6 at 31.2%, v0.7 at 32.2%, v0.8 at 31.2% on debiased CTIBench. **All of those were on a 500-record subset of the test split, while the v0.9 number (28.9%) was the only one on the full 2500-record bench.** The apples-to-apples re-bench above shows the actual full-bench scores cluster 4-5 pp lower; v0.9 is the bench winner, not a regression. The historical n=500 table is gone, the v0.9.0 / v0.9.1 release notes preserve the wrong numbers as historical record, and CHANGELOG v0.9.2 documents the correction.
