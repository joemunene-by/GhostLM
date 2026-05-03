# v0.5 Chat-Tune Postmortem (2026-05-03)

The canonical chat is `chat-v3 (MCQ-tuned)` at **36.9%** on CTIBench MCQ.
**Important:** chat-v3 sits on the v0.4 base (`phase4_ghost_small`, GPT-2 50K
BPE, no RoPE/SwiGLU/RMSNorm) — not the v0.5 base. The v0.5 base attempts
(chat-v05 / recovered / v5) all underperform it. This document records the
v0.5-base recovery attempts and what they actually changed about our
understanding.

## Result table

| Run | Base | Recipe | Steps | LR | Val | CTIBench MCQ |
|---|---|---|---|---|---|---|
| chat-v2 | v0.4 | Cybersec Q&A only, no MCQ | 1500 | 5e-5 | — | 19.0% |
| **chat-v3 (canonical)** | **v0.4** | **Raw letter-only MCQ × 5** | 1500 | 5e-5 | — | **36.9%** |
| chat-v4 (RAFT) | v0.4 | RAG-augmented chat-v3 mix | 1500 | 5e-5 | — | 25.0% |
| chat-v05 | v0.5 | chat-v3 recipe on v0.5 base | 1500 | 5e-5 | — | 32.5% |
| chat-v05-long | v0.5 | chat-v3 mix, 4000 steps | 4000 | 5e-5 | — | 17.1% |
| chat-recovered | v0.5 | CoT MCQ × 1 + small-talk × 30 | 1500 | 3e-5 | 2.808 | 30.8% |
| chat-v4-failed | v0.5 | Hybrid + lr 2e-4 | 300 | 2e-4 | diverged | killed |
| **chat-v5** | **v0.5** | **Hybrid raw × 5 + CoT × 2 + small-talk × 8** | 2000 | 5e-5 | 2.990 | **34.8%** |

### RAG at inference (verified 2026-05-03)

| Model | No RAG | + RAG(top2) | + RAG(top4) |
|---|---|---|---|
| chat-v3 (canonical, v0.4 base) | 36.9% | 36.9% | 36.5% |
| chat-v5 (v0.5 base) | 34.8% | — | 33.8% |

RAG is neutral or slightly negative on both. The 36M class can't usefully
exploit retrieved context for MCQ — confirmed across two distinct base
architectures and two top-K settings. **Inference-time RAG is dead at
this scale.**

### Corpus expansion + recipe repro (2026-05-03)

After expanding the corpus with MITRE full STIX (+1,110) and CISA KEV
(+1,587) and rebuilding `train.jsonl` to 307,375 records, we tried to
push the v0.4 base canonical further with chat-v6 (same recipe + new
sources in SFT mix) — **18.6%, below random**. The new-source Q&A
records actively confused the MCQ letter-shortcut signal.

To isolate the cause we then attempted to **reproduce** chat-v3's 36.9%
on today's data:

| Run | Recipe | Val | CTIBench MCQ |
|---|---|---|---|
| chat-v3 canonical (saved Apr 30) | (frozen) | (saved) | **36.9%** |
| chat-v3-repro | lr 5e-5, 1500 steps, batch 4×8, ctx 1024 | 1.973 | 32.6% |
| chat-v3-repro2 | **lr 3e-5, 1800 steps, batch 8×4, ctx 1024** ← exact saved recipe | 1.834 | **31.2%** |

Both reproductions land **5–6 points below canonical** despite lower val
loss. The 36.9% is a **frozen artifact** that cannot be reliably
recreated with current data. Likely causes: (a) the underlying source
files (`mcq.jsonl`, base corpus shards) drifted between Apr 30 and
May 03, (b) seed/shuffle stochasticity at this small scale produces
±6% bench variance.

**Conclusion:** the best *reproducible* result we have is **chat-v5 at
34.8%** (v0.5 base, hybrid SFT recipe). The 36.9% canonical chat-v3 is
preserved on HF as a frozen checkpoint but should not be treated as
the operating ceiling — it's a one-shot artifact.

## What we learned

### What chat-v3 actually does

The 36.9% canonical is a *pattern-match shortcut*, not reasoning. With raw
letter-only MCQ at × 5 multiplier, the model learns "after the prompt ends in
'Answer:', emit a single letter consistent with the surface features of the
options." This is a known class of MCQ artifact (Answer Matching > MCQ,
arXiv 2507.02856) — sub-100M models can hit reasonable MCQ scores by
exploiting the choice distribution without understanding the question.

### Why CoT-MCQ alone made it worse

`chat-recovered` (30.8%) replaced the letter-only MCQ × 5 with CoT MCQ × 1.
The CoT records have the format `"B. <1-2 sentence justification>"` — Qwen-14B
generated the reasoning. The hypothesis, from Phi-3.5-mini and OpenMath-Mini,
was that reasoning supervision should outperform pattern-match supervision
even at low multipliers.

It didn't — at 36M params, the model can't compress 1-2 sentences of cybersec
reasoning into useful weight updates, and it loses the letter-shortcut signal
in the process. Documented size effect: weaker students benefit from coarser
supervision; long rationales over-smooth gradients (Skip-Thinking, arXiv
2505.18642; Unveiling Key Factors for Distilling CoT, arXiv 2502.18001).

The 30 × small-talk multiplier compounded the damage by pushing task-data
share below 5% of the SFT mix — well outside the SmolLM2 reference of
≥ 20% task share.

### Why chat-v4 (lr 2e-4) diverged

Research said an undertrained backbone needs aggressive SFT lr to escape a
bad pretrain basin. SmolLM2 uses 3e-4 SFT lr at 135M params. Scaled down to
36M with mean-init new tokens, 2e-4 was still too hot — val climbed
monotonically across 3 evals (3.175 → 3.285 → 3.403) before we killed it at
step 300.

Lesson: the SmolLM2 lr reference doesn't transfer linearly to 36M with new
embedding rows. The safe range is closer to 5e-5.

### What chat-v5 got right (and didn't)

The hybrid recipe (raw × 5 + CoT × 2, small-talk × 8, lr 5e-5, mean-init
embeddings) lifted the score from 30.8% → 34.8% — a real **+4.0 point** gain
over the prior recovery attempt. But it still trails canonical by 2.1 points.

The hybrid was directionally right — keeping the letter-shortcut anchor
(raw × 5) preserved the discriminative signal, while CoT × 2 added some
reasoning supervision without over-rotating. Mean-init for new tokens kept
the residual stream stable.

What it didn't fix: the letter-shortcut at × 5 is still doing most of the
work, and there's no mechanism in this recipe that actually transfers
*knowledge* into the model — only better calibration on top of the shortcut.
To beat 36.9% durably, the lever isn't another SFT recipe — it's either:

1. **Bigger model** (ghost-base ~350M) so reasoning supervision actually fits.
2. **Better pretrain coverage** of the CTIBench knowledge domain (more
   cyber threat intel, MITRE corpus depth) so the shortcut isn't the only
   path to a correct answer.
3. **Proper retrieval at inference** (RAG done right, not the chat-v4 RAFT
   attempt that conflated training-time and inference-time augmentation).

## Decision

- **Canonical stays:** `chat-v3 (MCQ-tuned)` on **v0.4 base** at 36.9% on the
  main HF repo.
- **Ship chat-v5 separately:** push to `Ghostgim/GhostLM-v0.5-experimental`
  with this postmortem in the model card. Honest framing: "improved CoT
  hybrid recipe on the v0.5 architecture, still 2.1pt below the v0.4-base
  canonical — primarily of research interest."
- **No more chat-tune iterations on v0.5.** The 36.9% ceiling is a pretrain +
  capacity ceiling, not a recipe ceiling. v0.5 base may be a regression
  versus v0.4 for this MCQ task — possibly because the custom 32K BPE
  fragments cybersec terms that GPT-2's 50K BPE keeps whole. Next swing
  should be ghost-base or a corpus-side fix, not another SFT permutation.
- **No inference-time RAG.** Verified across both bases and two top-K values
  — the 36M class can't exploit retrieved context for MCQ.

## Sources

- [Answer Matching Outperforms MCQ, arXiv 2507.02856](https://arxiv.org/abs/2507.02856)
- [Skip-Thinking, arXiv 2505.18642](https://arxiv.org/html/2505.18642v1)
- [Unveiling Key Factors for Distilling CoT, arXiv 2502.18001](https://arxiv.org/html/2502.18001v1)
- [SmolLM2, arXiv 2502.02737](https://arxiv.org/html/2502.02737v1)
- [How Abilities in LLMs are Affected by SFT Data Composition](https://openreview.net/forum?id=6M5G5hNiAU)
