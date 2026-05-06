# Ghost-base spec (v1.0 candidate)

This is the design doc for the next architectural rung after the
ghost-small (45-81M) line was diagnosed as plateauing at ~30% on
debiased CTIBench (see [`docs/ctibench_bias_finding.md`](ctibench_bias_finding.md)
for the eval methodology and [`CHANGELOG.md`](../CHANGELOG.md) v0.9.0
for the six attempts that hit the ceiling). Nothing here is built
yet; the doc exists so the v1.0 milestone has a clear scope before
we commit GPU budget.

## Why scale up at all

The ceiling diagnosis is firm at the ghost-small rung: BPE swap, full
architecture refresh (RoPE / SwiGLU / RMSNorm), parameter doubling
(45M → 81M), text-loss SFT, fact-density injection, and a 4×
corpus-density expansion (60M → 273M tokens) all landed inside a
4-point band on debiased CTIBench. Live testing is consistent across
every variant: the model has the *register* of cyber writing and
none of the *facts* (gets EternalBlue's CVE wrong, conflates MITRE
technique IDs, hallucinates CVE-to-CWE mappings).

The pattern matches the literature. SmolLM2-360M and Phi-3.5-mini
both report factual recall on cybersec MCQ emerging in the 300M-400M
parameter range. Below that, models can match register but can't
hold facts. Ghost-base is the rung where this should change, or where
the diagnosis flips and the eval methodology becomes the next
suspect.

## Architecture (matches SmolLM2-360M)

| Field | Value | Rationale |
|---|---|---|
| Layers | 30 | 5× v0.7's depth. Depth contributes more to factual recall than width per the SmolLM2 / Phi-3 ablations. |
| d_model | 960 | 1.25× v0.7's 768. |
| n_heads | 15 | head_dim 64 (= 960 / 15), unchanged from v0.7's head budget. |
| d_ff | 3200 | ~3.33× d_model. SwiGLU full width, sized to hit ~360M total. |
| Vocab | 50,264 | GPT-2 50K BPE + 7 special tokens (unchanged from v0.6+). |
| Context | 1024 train, 2048 inference | Same RoPE-extension path the v0.7 ctx-1024 fine-tune validated. |
| Norm | RMSNorm | Unchanged. |
| FFN | SwiGLU | Unchanged. |
| Position | RoPE base 10000 | Unchanged. |

Estimated parameter count: ~360M (verified to within 1% of
SmolLM2-360M's published 362M; an earlier draft of this spec
quoted "12L × 768d → ~360M" which an M4 smoke at that shape
revealed to be only 124M, so the launcher and spec were corrected
to the deeper SmolLM2-style shape that actually hits the target).
Within the 300-400M band where the literature reports MCQ-
factual-recall capability emerging.

## Corpus

The v0.9 corpus is at 273M train tokens (`data/processed/train.jsonl`,
669,085 records: PRIMUS-Seed 85K, PRIMUS-FineWeb 300K, NVD 71K,
Exploit-DB 5K, fact-QA 11K, MITRE/CWE/CAPEC/OWASP/RFCs/Wikipedia,
arXiv, CTFtime). Chinchilla-optimal for 360M params is ~7.2B tokens,
which is **25× short**. Three options for closing the gap:

1. **Train at 273M tokens with ~3 epochs (~820M tokens seen, ~10%
   of Chinchilla).** Fastest path. Risks overfitting on small
   sources (CWE, OWASP, RFCs are <100K tokens combined). Probably
   the v1.0 ship target if compute budget is tight.
2. **Train at 273M tokens for 1 epoch and accept undertraining.**
   Saves compute. Likely worse than option 1 unless we hit the
   Chinchilla efficient-frontier asymmetry (undertrain better than
   overfit at the small data scale).
3. **Expand corpus to 1B+ tokens before training.** Adds 6-12 weeks
   of corpus work (more PRIMUS-FineWeb shards, security blogs at
   scale, CTF writeup expansion, full-text academic papers). Right
   answer for a serious v1.0 but blocks shipping for months.

**Tentative call:** option 1 with a 3-epoch budget on the v0.9 corpus,
oversampling the high-quality non-FineWeb sources (CWE / NVD /
fact-QA / OWASP / MITRE) at ~5× weight. Documenting the undertraining
in the model card as a known limitation.

## Compute estimate

Chinchilla FLOPs ≈ 6 × params × tokens. For one full Chinchilla-optimal
run (360M × 7.2B): 6 × 3.6e8 × 7.2e9 ≈ **15.6 PFLOPs**.

| Option | Tokens trained | FLOPs | H100 wall-clock @ 67 TFLOPS bf16 sustained | Cost @ $2.50/h |
|---|---|---|---|---|
| 1 epoch, 273M tokens | 273M | 0.59 PFLOPs | ~9 hours | ~$23 |
| 3 epochs, 820M tokens | 820M | 1.77 PFLOPs | ~26 hours | ~$66 |
| Chinchilla-optimal, 7.2B tokens | 7.2B | 15.6 PFLOPs | ~233 hours | ~$580 |

Numbers above assume one H100, no mixed-precision FLOPs gain beyond
bf16, real-world utilization 60-80% of peak. A H100 at $2.50/h is on
the cheap end of current spot pricing; the high end is $4-5/h. The
3-epoch run at one H100 fits in a long weekend at <$100, which is
probably the right v1.0 target.

## Hardware path

- **Rented spot H100** (Lambda Labs / Vast.ai / RunPod): cheapest, but
  preemption-tolerant training is a separate engineering task (the
  current `trainer.py` checkpoints every save_interval steps and can
  resume, so cleanly exits already; pre-emption mid-step would lose
  ~50 steps which is fine).
- **Owned 4090 / 5090**: ~83 TFLOPS bf16 sustained, slower than H100
  but no rental cost. 3-epoch run in ~21 hours. Right answer if we
  expect multiple iterations.
- **TPU v4 / v5p via Trillium / Cloud Run**: cheaper than H100 but
  Flax port of GhostLM is not done. Would add weeks.

**Tentative call:** rent spot H100 for the first ghost-base run. If
the result is interesting enough to iterate on, then evaluate owning
a 4090 vs continuing to rent.

## Acceptance criteria

The ghost-base run is a success if it clears **at least one** of:

- **≥40% per-perm avg on debiased CTIBench (n=2500, full bench).**
  10pp above the 30% ceiling. Validates the param-count diagnosis
  and unblocks v1.1+ work on top of the same architecture.
- **≥50% on a hand-written cybersec MCQ benchmark designed to test
  fact recall (separate from CTIBench, drawn from CWE / MITRE /
  CISA / IETF RFC content).** Validates that the model has actually
  learned the facts, regardless of whether CTIBench specifically
  rewards that.
- **Qualitative fluency upgrade in live testing.** Live prompts get
  CVE bindings right ("EternalBlue is CVE-2017-0144" not "CVE-2018-9013"),
  bind techniques to tactics correctly, don't hallucinate version
  ranges. This is the "would I actually use it" bar.

If none of those clear, the diagnosis flips: at the ghost-base scale,
something other than parameter count is the bottleneck. Candidates
to investigate in that case (in priority order):

1. **CTIBench-specific eval bias.** Run CySecBench, SecQA, the
   in-repo CTF eval, and a hand-written fact-recall set on
   v0.7-chat / v0.9-chat / ghost-base-chat. If they all stall at
   ~30% on every benchmark, the eval framing might be the problem
   (text-scoring on 4-way MCQ might cap out where the model can't
   distinguish between option strings on subtle semantic differences,
   regardless of factual knowledge).
2. **Tokenizer mismatch.** GPT-2 50K BPE was not trained on
   cybersec text and wastes vocabulary on natural-English subwords
   that aren't load-bearing here. A custom 32K BPE retrained on
   the v0.9 corpus might give the model 30% more effective context
   per token. v0.5 tried this with limited corpus and saw no win,
   but ghost-base has 25× more cybersec data.
3. **SFT data quality.** The chat-v3 recipe was tuned for v0.4. The
   1,802 MCQ examples in `data/raw/chat/mcq.jsonl` may not exercise
   the longer factual chains a 360M model could learn. Larger
   high-quality SFT set (10-50K examples) is a real lever.

## Risk register

- **Eval methodology bias.** All ghost-small numbers come from
  text-scoring on 4-way MCQ. If that has a 30% ceiling for reasons
  unrelated to model capability (e.g., 4 plausible distractors
  on every question collapsing the per-token distribution), no
  amount of scaling fixes it. Mitigation: hand-write a 100-question
  free-form fact-recall benchmark (no MCQ) and rubric-grade with
  Qwen-14B. Ship before ghost-base run.
- **Spot preemption.** A 26-hour run on spot H100 has non-trivial
  preemption risk. Mitigation: save-interval already at 1500 steps,
  resume path validated on v0.9. Separate concern: if run reliably
  preempts every 6-12 hours, on-demand H100 at 2× cost may be
  cheaper net than spot in restart overhead.
- **Corpus contamination.** PRIMUS-FineWeb is CommonCrawl-derived;
  CTIBench questions or near-paraphrases may appear in there. The
  v0.9 chat regression vs v0.7 (28.9% < 32.2%) is consistent with
  FineWeb diluting the cyber-text register, but a contamination
  audit (CTIBench question hashes vs FineWeb shard contents) would
  rule it out as a confound. Right move before ghost-base spends
  $50+ on a corpus that might be poisoned.
- **Single-point-of-failure on M4 vs cloud.** All ghost-small work
  was done on a Mac M4 mini, with logs and checkpoints local. Once
  ghost-base ships to rented hardware, weights need to come back to
  M4 (or at least to cloud storage) for inference + eval. Set up
  rclone / S3 for the result artifacts before launching.

## Sequencing

1. **Now (already done):** v0.9.0 release tagged. Six-attempt ceiling
   diagnosis documented. Corpus on disk at 273M tokens.
2. **Next (M4-doable, this week):** v0.7 chat ctx-1024 extension
   fine-tune (running). Cross-bench validation against the in-repo
   CTF eval set + a free-form fact-recall benchmark (to ship). Both
   are gated on M4 wall-clock and can run while ghost-base prep
   continues.
3. **Then (cloud, when ready):** corpus contamination audit. CTIBench
   question hashes vs FineWeb shard contents. ~1 day.
4. **Then (cloud, the v1.0 run):** ghost-base 3-epoch run on rented
   H100. 26-hour single shot, ~$70, save-interval 1500 with cloud
   bucket sync. Chat-tune and bench on M4 after weights return.
5. **Decision point:** if ghost-base clears ≥40% on debiased
   CTIBench, ship as v1.0.0. If not, work through the diagnosis-flip
   priority list above.

## What this doc is not

- A commitment to ship v1.0. The ghost-small line ended with an
  honest negative result; ghost-base could too. The acceptance
  criteria above are the gate.
- A detailed run-book for the H100 spot procurement, cloud bucket
  setup, or weight-shuttling tooling. Those land when they're
  unblocked, not now.
- A pre-commitment to 12L × 768d. If a contamination audit, a
  better tokenizer, or a 100-question free-form benchmark moves the
  ghost-small numbers materially before we rent compute, the
  ghost-base scale-up may not be the next move at all.
