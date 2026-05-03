# CTIBench MCQ Single-Letter Bias and the Real 36-45M Ceiling (2026-05-04)

The CTIBench MCQ "ceiling" we have been chasing for a month is two
artifacts stacked: a positional bias in the bench, and a single-letter
collapse in our SFT objective. Once both are stripped away, every chat
tune we have ever shipped has the same real capability of around 30%
on a 25% baseline. The bottleneck is not recipe. It is capacity at
36-45M params on this corpus.

This document captures the full investigation that landed at that
conclusion, including the live-test that confirmed the model knows
cybersec vocabulary but not cybersec facts.

## The bench is biased

CTIBench MCQ gold-letter distribution (2,500 records):

| Letter | Count | Pct |
|---|---|---|
| A | 374 | 15.0% |
| B | 813 | 32.5% |
| C | 928 | **37.1%** |
| D | 385 | 15.4% |

A model that always picks C scores **37.1%** on single-order eval, which
is higher than every model we have actually trained. The "canonical"
chat-v3 at 36.9% is below this trivial baseline.

## Letter-scoring debiased eval

`scripts/eval_debiased.py` scores each of the 2,500 records under 4
different option-letter orderings and counts a record correct only when
the model picks the gold answer regardless of which letter the gold was
mapped to. A pure positional-bias model collapses to 25% (random).

| Model | Latched letter | Single-order | Per-perm avg | All-perm correct |
|---|---|---|---|---|
| chat-v3 canonical | C (98.6%) | 36.9% | 30.3% | **0 / 2500** |
| chat-v5 best repro | C-leaning (79.6%) | 34.8% | 29.3% | 1 / 2500 |
| chat-v3-repro2 | B/C dual (49/38) | 31.2% | 26.0% | **3 / 2500** |
| chat-v06 canonical | B (86.2%) | 29.8% | 23.4% | 0 / 2500 |
| chat-v06 hybrid | A (100%) | 15.0% | 20.7% | 0 / 2500 |

The single-order ranking is **inverted** from the all-perm ranking. The
"canonical" gets zero questions right under all permutations. The model
we dismissed for failing to reproduce 36.9% (chat-v3-repro2 at 31.2%)
gets three.

### Per-gold-letter accuracy reveals the bias mechanism

chat-v3 canonical:
```
gold=A:    7/ 374 (1.9%)
gold=B:    0/ 813 (0.0%)
gold=C:  915/ 928 (98.6%)
gold=D:    0/ 385 (0.0%)
```

98.6% on C-gold and 0% on every other letter. A "pick C" predictor.
The 36.9% accuracy decomposes to: (374 × 0.019) + (813 × 0) + (928 × 0.986) + (385 × 0) ≈ 922.

Same pattern across every model with a different letter as the attractor:
v0.4-base models latch onto C, v0.6-base canonical onto B, v0.6-hybrid
onto A. Cross-entropy SFT with letter-only assistant turns
("assistant: B") on a 36M-param model teaches the cheapest possible
loss-minimizing strategy: emit the most common letter the optimizer
happens to find.

## Text-scoring eval (option-content logprobs)

`scripts/eval_text_scoring.py` skips the letter-token entirely. For each
MCQ record it computes `log P(option_text | prompt)` for each option
(per-token average, length-normalized) and picks the highest. A model
that learned cybersec content but expressed it through letter emission
should suddenly score above chance.

500 records, 2 permutations each:

| Model | text per-perm avg | Real signal vs random |
|---|---|---|
| chat-v3 repro2 | **31.7%** | +6.7 |
| chat-v06 canonical | 31.2% | +6.2 |
| chat-v3 canonical | 30.5% | +5.5 |
| chat-v5 | 29.7% | +4.7 |

All four models cluster in the 29-32% range under text scoring. The
"36.9% canonical advantage" disappears entirely. **v0.6 base, which we
dismissed in letter scoring, is on par with v0.4 base in real
capability.** The BPE swap hypothesis was vindicated by debiased eval
and only looked like a failure under biased scoring.

## Text-loss SFT experiment (chat-text)

If letter-only SFT teaches the model to emit a single letter, training
on full option text should fix it. We tested by retraining with assistant
turns set to "B. <full option text>" instead of just "B".

Result: **30.1% per-perm avg.** Right in the same cluster.

| Model | SFT objective | text per-perm |
|---|---|---|
| chat-v3 repro2 | letter | 31.7% |
| chat-v06 canonical | letter | 31.2% |
| chat-v3 canonical | letter | 30.5% |
| **chat-text** | **text** | **30.1%** |
| chat-v5 | letter | 29.7% |

The SFT objective is not the bottleneck. Real capability is bounded at
~30% regardless of how the model is trained.

## Live test confirms capacity, not recipe

Free-form generation from chat-v3 canonical on 5 cybersec questions:

- "What is phishing?" → "CAPEC-5 — phishing attacks." (knows the
  vocabulary association)
- "What does CVE-2017-0144 (EternalBlue) exploit?" → describes a Linux
  mlx5e memory leak. **Wrong**. EternalBlue is Windows SMB. Pattern-
  matched the CVE prefix to NVD-style descriptions.
- "How does a SQL injection attack work?" → coherent description of
  unsanitized input vulnerabilities, framed as a CTF writeup.
- "What is the difference between symmetric and asymmetric encryption?"
  → defaults to a CTF writeup about RSA, never actually contrasts the
  two.
- "Explain MITRE ATT&CK technique T1059." → knows the URL format,
  conflates T1059 with RDP (which is T1021).

The model is a **cybersec parrot.** It has learned vocabulary patterns,
URL formats, and writing styles (heavily CTF-writeup-flavored, since
that dominates the training corpus). It has not learned cybersec facts.
The 30% real ceiling is exactly what you would expect from a model that
can sometimes pattern-match the right option text but cannot
distinguish factually correct from factually wrong cybersec sentences.

## What this overturns

1. **Recipe iteration was rearranging deck chairs.** chat-v3 / chat-v5 /
   chat-v06 / chat-v06-hybrid / chat-text all have effectively the same
   real capability (29-32%). The recipe variations changed which letter
   the model latched onto, not whether it learned anything.

2. **The seed-variance question is irrelevant.** The 5-7 point bench
   swing across seeds was bias-attractor variance, not capability
   variance. Different seeds locked onto slightly different letter
   distributions.

3. **The BPE swap experiment WAS successful.** v0.6 (v0.5 architecture
   + GPT-2 BPE + expanded corpus) is on par with v0.4 in real
   capability. Single-order letter scoring made it look like a
   regression.

4. **The "36.9% ceiling" does not exist as a capability number.** It is
   the always-most-common-letter baseline.

5. **Pretrain corpus expansion did not help capability.** v0.6 trained
   on 500M tokens lands at the same real capability as chat-v3 trained
   on 12M tokens. Capacity bottleneck, not data bottleneck.

6. **HF model card and Space description need correction.** Both
   currently report 36.9% as a capability number. Both should now
   report single-order alongside per-perm-avg with a footnote about
   single-order on this bench rewarding positional bias.

## What to do next

The 36-45M scale truly tops out at ~30% real CTIBench MCQ. To break
this, three real options:

1. **Scale to ghost-medium ~130M or ghost-base ~350M.** Per the SmolLM2
   and Phi-3.5-mini literature, factual recall on cybersec MCQ emerges
   around 130-300M params on a balanced corpus. This is the next real
   swing. Locally on M4 ghost-medium fits at ctx 512 / batch 4 / accum 8
   (effective batch 32) and trains at ~3-5s/step, putting an
   8K-step overnight run at ~8 hours. Real GPU access would unlock
   ghost-base.

2. **Curate a higher-fact-density training corpus.** Strip the CTF-
   writeup-heavy bias, replace with NVD descriptions, MITRE technique
   bodies, security textbook excerpts. Multi-day investment. May or may
   not help at 36-45M but should help at any scale.

3. **Tool use / retrieval at inference.** Train the model to emit a
   retrieval query, fetch facts, then answer. Removes the memorization
   requirement. Tool use itself emerges with scale though, so this
   path is contingent on (1).

## Files

- `scripts/eval_debiased.py` (multi-permutation letter eval)
- `scripts/eval_text_scoring.py` (multi-permutation text eval)
- `scripts/build_mcq_text_data.py` (text-loss MCQ data builder)
- `logs/debiased/*.json` (per-checkpoint letter-scoring results)
- `logs/text_scoring/*.json` (per-checkpoint text-scoring results)
- `RESULTS.md` (canonical table, single-order numbers; needs new column
  for `text_per_perm_avg` and a note that single-order is biased)

## Sources

- [Answer Matching Outperforms MCQ, arXiv 2507.02856](https://arxiv.org/abs/2507.02856)
- [SmolLM2: When Smol Goes Big, arXiv 2502.02737](https://arxiv.org/html/2502.02737v1)
- [Phi-3 Technical Report, arXiv 2404.14219](https://arxiv.org/abs/2404.14219)
- [Skip-Thinking: Chunk-wise CoT Distillation, arXiv 2505.18642](https://arxiv.org/html/2505.18642v1)
- [How Abilities in LLMs are Affected by SFT Data Composition](https://openreview.net/forum?id=6M5G5hNiAU)
