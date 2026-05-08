# Bets 7 / 8 / 9 baselines: v0.9 chat (2026-05-08)

The structural-compliance metric for bet 6 is locked
([docs/format_baseline_v09.md](format_baseline_v09.md)). This doc
captures the parallel measurement for the three multi-modal-in-
security bets shipped in v0.9.5: bet 7 (code-for-security),
bet 8 (binary / hex literacy), bet 9 (provenance / cite tags).

## Setup

- Checkpoint: `Ghostgim/GhostLM-v0.9-experimental` (a.k.a.
  `phase19_chat_v09/best_model.pt`, 81M params, 6L / 768d / 12h,
  GPT-2 BPE, trained on PRIMUS + CWE + OWASP + RFCs + fact-QA
  pretrain plus chat-tune).
- Eval sets:
  - [`data/raw/code_security_eval.jsonl`](../data/raw/code_security_eval.jsonl) (n=20)
  - [`data/raw/binary_literacy_eval.jsonl`](../data/raw/binary_literacy_eval.jsonl) (n=20)
  - [`data/raw/provenance_eval.jsonl`](../data/raw/provenance_eval.jsonl) (n=15)
- Inference: `scripts/run_format_baseline.py` against MPS on the M4
  with `temperature=0.7 top_k=50 top_p=0.95 max_tokens=400`.
- Scoring: `scripts/eval_format_compliance.py` with the new
  `parse_provenance` parser plus parser-less paths for `code_security`
  and `binary_literacy` (substring scoring only).

## Headline result

| Bet | Format | n | parse-pass % (Wilson 95% CI) | fields-pass % (Wilson 95% CI) |
|---|---|---:|---:|---:|
| 7 | `code_security` | 20 | 100.0% [83.9-100.0] | **0.0% [0.0-16.1]** |
| 8 | `binary_literacy` | 20 | 100.0% [83.9-100.0] | **0.0% [0.0-16.1]** |
| 9 | `provenance` | 15 | 0.0% [0.0-20.4] | **0.0% [0.0-20.4]** |

All three measurements lock at 0% fields-pass. Bets 7 and 8 have no
parser registered, so parse-pass is vacuously 100%; the substring
check is the actual scoring lever and it's where the failures land.
Bet 9's parser counts well-formed `<|cite|>{type}:{id}<|/cite|>`
tags in the answer — v0.9 emits zero of them across all 15 prompts,
so parse-pass is also 0%.

**Lift targets (95% CI separation):**
- Any future ghost-base score above **~16%** on bet 7 or 8 is
  statistically separated from the v0.9 baseline.
- Any score above **~20%** on bet 9 is similarly separated.

## What v0.9 actually produces (failure modes)

### Bet 7 (code-for-security)

20/20 prompts produced *some text* (parse-pass = 100%) but none
contained the right CWE id paired with the fix-property keyword.
Sampled outputs:

- Prompt: "A web app builds an LDAP search filter by concatenating
  user input..." Expected `CWE-90`, `LDAP`, `escap`. v0.9
  produced 1 token (`A`), then `<|ghost_end|>`.
- Prompt: "A Python web framework stores user IDs from POST bodies
  directly into the User model..." Expected `CWE-915`, `mass
  assignment`. v0.9 produced unrelated registry-style prose about
  HKLM keys.

Failure mode is the same as bet 6: when the question is outside
the prose register the chat-tune memorised, the model collapses
to `A` or drifts into unrelated CVE-style narrative.

### Bet 8 (binary / hex literacy)

20/20 prompts produced text (parse-pass = 100%); zero produced
the format name with required reasoning substrings. Sampled
failures:

- Prompt: hex `7F 45 4C 46 02 01 01 00`. Expected `ELF`, `64-bit`,
  `little-endian`. v0.9 produced 75 tokens of unrelated CVE prose.
- Prompt: hex `4D 5A 90 00 ...`. Expected `PE`, `MZ`, `e_lfanew`.
  v0.9 produced 1 token (`A`), end.

The model has zero binary literacy because v0.9's pretrain corpus
contains essentially zero hex content. The eval correctly registers
that gap.

### Bet 9 (provenance)

15/15 prompts produced text; zero contained any `<|cite|>` tag.
v0.9 has no exposure to cite-augmented traces in its training
corpus, so the structural pattern is not in the model's prior.
This is the most clean-cut 0% measurement of the three.

## Why these are the right baselines

For all three bets, v0.9's training corpus contains essentially
zero training signal for the capability being evaluated:

- **Bet 7**: v0.9 saw cybersec prose, not (vulnerable code +
  CWE-id-in-context) pairs.
- **Bet 8**: v0.9 saw essentially no hex / binary content.
- **Bet 9**: v0.9 saw no cite tags during training.

The 0% baselines are therefore not a "model failure" so much as
"capability not trained for, capability not present." That is
exactly the gap bets 7, 8, and 9 are designed to fill.

## Lift target table

If ghost-base trains on the templated-synth corpus (1,505 records)
and re-runs each eval, the lift expectations look like:

| Bet | v0.9 baseline | ghost-base lift target | Statistical separation |
|---|---|---|---|
| 7 | 0/20 [0.0-16.1] | 30-50% (early target) | Any score >16% |
| 8 | 0/20 [0.0-16.1] | 25-45% (binary literacy is harder) | Any score >16% |
| 9 | 0/15 [0.0-20.4] | 60-80% (most templated coverage) | Any score >20% |

Bet 9 has the highest lift potential because the cite-tag pattern
is highly structured and the templated corpus has 429 records
teaching exactly that shape. Bet 8 has the lowest because
binary literacy is the hardest capability for a small from-scratch
LM to acquire.

## Re-running

```bash
PYTHONPATH=. python3 scripts/run_format_baseline.py \
    --checkpoint <ckpt> \
    --seeds data/raw/<bet>_eval.jsonl \
    --out logs/baseline_<run>_<bet>.jsonl

PYTHONPATH=. python3 scripts/eval_format_compliance.py \
    --predictions logs/baseline_<run>_<bet>.jsonl
```

The Wilson 95% CIs in the report widen at small n and tighten at
larger n. The current eval-set sizes (n=15-20) give upper bounds
in the 16-20% range at p=0; growing each eval set to n=50-100 over
time would tighten the upper bound to ~7%.

## Comparison-rows table (this grows)

| Checkpoint | Date | bet 7 fields % | bet 8 fields % | bet 9 fields % |
|---|---|---:|---:|---:|
| v0.9 chat (81M) | 2026-05-08 | 0/20 [0.0-16.1] | 0/20 [0.0-16.1] | 0/15 [0.0-20.4] |
