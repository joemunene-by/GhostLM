# GhostLM differentiation strategy

Most from-scratch cybersec LMs in 2025-2026 follow the same recipe:
clone SmolLM2 architecture, train on PRIMUS + CTIBench-adjacent web
text, ship at 28-32% on debiased CTIBench MCQ, declare victory. The
benchmarks are crowded with near-identical artifacts. This document
captures GhostLM's six concrete bets to be **genuinely different**
rather than another point in that crowd, with a scaffold per bet
that's already in the repo.

The strategic frame: **the bottleneck the v0.9.3 RAG diagnostic
identified is real, the parameter-count escape hatch is expensive,
and the more interesting moves are architectural / training-recipe /
ecosystem-level changes that other from-scratch projects aren't
attempting.**

## The v0.9.3 diagnostic, restated

From `docs/rag_diagnostic_findings.md` (commit `f445d88`):

| Metric | Score |
|---|---:|
| Retrieval@4 (no LM) | 41/100 |
| v0.9-bare fact-recall v2 | 1/100 |
| v0.9+RAG fact-recall v2 | 0/100 |

**The retriever surfaces the right corpus passage 41% of the time;
the 81M model can extract that fact 1% of the time; adding the
retrieved context destabilizes the model into mode collapse.**

The obvious fix is parameter scaling (ghost-base, ghost-1B, etc).
That's the v1.0 GPU spend already planned at
`docs/ghost_base_spec.md`. The six bets below are different
moves: each is something a parameter-scaled-only roadmap doesn't
solve.

## Bet 1: tool-grounded model, not memorization-based

**Hypothesis.** A 360M model trained to ISSUE TOOL CALLS for
factual lookups crushes a 7B model trained to MEMORIZE the same
facts on real cybersec workflows. The v0.9.3 diagnostic identified
the exact failure: the 81M chat model can't extract from supplied
context, even when the right text is in the prompt. RAG layered
on top doesn't help because the model isn't TRAINED on the
supplied-context-extraction objective.

**Fix.** Train ghost-base directly on tool-use traces. Synthesize
10K examples of `(question -> tool_call -> tool_response -> answer)`
chains via Claude/Llama distillation, where the final answer
references ONLY facts that appeared in the tool response.
Fine-tune ghost-base on this data + the existing chat-v3 SFT set.
The model learns the meta-skill of "lookup before answering"
rather than "guess from memory".

**Scaffold.** `scripts/distill_tool_use.py` (commit `9b395a4`).
Four tools covered (search_cve_nvd, lookup_mitre_technique,
lookup_cwe, rag_retrieve), each generating ~2,500 traces.
Quality filter requires four literal tag strings AND parseable
JSON tool-call body. Cost: ~$200 on Anthropic Sonnet for 10K
traces, free smoke-test on Ollama.

**Why it's the differentiator.** Every other small cybersec LM
trains for memorization, hits the parameter-scale wall at 30%
MCQ / 1% fact-recall, ships register-shaped fiction. A
tool-using small model is a **different model-level objective**
that bypasses the wall entirely.

## Bet 2: continuously updated, not training-cutoff frozen

**Hypothesis.** Cybersec is uniquely time-sensitive. Today's CVE,
today's ransomware family, today's CISA advisory aren't in last
month's training data. Every other LLM is frozen at its cutoff
date; a continuously-updated cybersec LM is a different KIND of
product.

**Fix.** Nightly LoRA fine-tune over the previous 24h of fresh
threat-intel data (CISA KEV, vendor TI research, MISP feeds,
security blogs). Each night's run produces a small (~10-50 MB)
LoRA adapter that gets pushed to a date-stamped HF Models repo
(`Ghostgim/GhostLM-daily-2026-MM-DD`). The base checkpoint stays
fixed; consumers download the adapter and merge at load time.

**Scaffold.** `scripts/daily_finetune.py` (commit `8b755b5`).
Cron-friendly orchestrator. Runs all collectors, measures the
24h corpus delta, gates the tune on >= 50K new tokens, runs a
1-2h LoRA fine-tune, pushes the adapter to HF, retries on
failure. Cost: ~1-2 GPU hours per day. On a workstation
(RTX 6000 Ada) running 24/7 that's a tiny fraction of the
hardware envelope.

**Why it's the differentiator.** No other small cybersec LM
does daily updates on a public schedule. The competitive frame
shifts from "static benchmark numbers" to "knowing about
yesterday's threat".

## Bet 3: cybersec-native tokenizer

**Hypothesis.** GPT-2's 50K BPE was trained on general-English web
text. It allocates vocabulary to `the`, `and`, `tion` and splits
high-value cybersec sequences (`CVE-XXXX-`, `T1059`, `CWE-89`,
hex strings, `CVSS:3.1`) across many tokens. A 32K BPE retrained
on the v1.0 corpus should compress cybersec text by 25-35% in
tokens-per-byte, freeing context budget proportionally.

**Fix.** Train a fresh 32K BPE on the v1.0 corpus. Pre-allocate
the 11 GhostLM special tokens (chat roles + tool-use tags). Plug
the new tokenizer into `ghostlm/tokenizer.py` as an alternate
backend, replacing the tiktoken GPT-2 path. Rerun ghost-base
pretrain on the recompressed corpus.

**Scaffold.** `scripts/train_v1_bpe.py` (commit `5bd934c`).
Streams the 1.7 GB chunks corpus, trains BPE via the
`tokenizers` library, saves to `data/tokenizer/v1/`,
auto-generates a `compression_report.md` showing tokens-per-byte
vs GPT-2 BPE on a 100-record sample. Cost: ~30-60 min on M4
CPU, no GPU.

**Why it's the differentiator.** v0.5 attempted this on a 60M-
token corpus and the result tokenized cybersec densely but
fragmented out-of-domain English. With 6x more data + three
new domains (FineWeb-Edu, math, code), the tokenizer should
preserve general English while specializing on cybersec. The
compression report tells us honestly whether the bet pays off.

**Result (2026-05-08).** v1 BPE: **0.2190 tokens/byte** vs
GPT-2's **0.2225** on a 99-record sample → **+1.6% compression
win**, not the +25-35% the hypothesis projected. Per-record
distribution shows the expected split: cybersec-heavy logs and
incident reports compress 5-10% better, but large general-text
samples (FineWeb-Edu chunks) sometimes regress 0.5-5%. The bet
didn't pay off at the magnitude expected. Whether to ship v1
BPE in ghost-base / ghost-1B is now a 1.6% question, not a
25%+ question. **Recommendation:** keep v1 BPE on the shelf as
an opt-in alternate backend; default ghost-base to GPT-2 BPE
unless a downstream eval (CTIBench accuracy at fixed token
budget) shows the cybersec specialization translates to
benchmark improvement, not just compression. The honest result
is the result.

## Bet 4: long context for IR-style workflows

**Hypothesis.** Real cybersec workflows are long-context. An
incident-response analyst dumping a 50K-token threat report
into the chat box doesn't want a 1024-context model. ghost-base
at 32K is uniquely useful for IR triage even at the same
parameter count as a 4K competitor.

**Fix.** Extend the context length via RoPE NTK-aware
interpolation. The standard recipe since 2023's Code Llama
paper: scale `rope_base` non-linearly so high-frequency
components stay sharp while low-frequency components stretch to
cover the new context. After rebase, run a short fine-tune on
long-form corpus (arXiv full-text, NIST SP 800 chunks, full
security blog posts) at the new context length.

**Scaffold.** `scripts/extend_context_ntk.py` (commit `8b755b5`).
Computes the new rope_base via the standard formula, saves a
rebased checkpoint, and prints the canonical `finetune_chat.py`
invocation for the actual tune. Two modes: `--rebase-only` for
zero-shot extension testing (works for 2-4x), full mode for
production-grade 16x extension via fine-tune. Cost: ~3-5 GPU
hours.

**Why it's the differentiator.** Other small cybersec LMs cap
at 4K context. The chunk-the-document pattern they use loses
inter-document context. A 32K-context ghost-base reads the
whole report in one shot.

## Bet 5: MoE architecture for ghost-1B+

**Hypothesis.** Going dense at 1B is the obvious move. Going MoE
(4 experts × 500M, 2 active per token) gives 2B params learned
at 1B inference compute. From-scratch MoE at this scale is rare;
most cybersec LMs that ever reach 1B graft MoE in late or skip
it. Doing it right out of the gate makes ghost-1B a different
kind of artifact.

**Fix.** Add a `SparseMoE` FFN class to `ghostlm/model.py`,
gated on a new `use_moe` config flag. Standard Mixtral /
DeepSeek-MoE shape: linear router, top-K gating, parallel SwiGLU
experts, weighted sum, load-balancing aux loss. The trainer
reads the aux loss off each MoE layer after every forward and
adds the load-balancing term to the cross-entropy objective.

**Scaffold.** `ghostlm/model.py` + `ghostlm/config.py` (commit
`6d5a731`). Off by default; opt-in via:

```python
config.use_moe = True
config.n_experts = 4
config.n_experts_active = 2
config.moe_aux_loss_coef = 0.01
```

Smoke-validated on ghost-tiny config: 2 MoE layers, forward
pass clean, aux losses (~2.03, 2.08) indicate healthy uniform
routing across the 4 experts. Existing v0.4-v0.9 checkpoints
are unaffected (they don't set use_moe).

**Why it's the differentiator.** Most from-scratch cybersec
projects are dense; the few that explore MoE do it as a
post-hoc graft on a pretrained dense base. Ghost-1B trained
with native MoE from step 1 is a different artifact with
different compute/capacity tradeoffs.

## Bet 6: format-aware structured-data pretrain

**Hypothesis.** Other small cybersec LMs train almost entirely
on prose: blog posts, RFCs, MITRE technique descriptions, CVE
summaries. They get OK at *talking about* threat intel but
can't *produce* the structured artifacts real CTI workflows
exchange (STIX 2.1 bundles, YARA rules, Sigma detection rules,
MISP event JSON). A model that reads AND emits those formats
slots into existing pipelines without a translator. Bet 3's
+1.6% compression result already showed that "recompress the
same prose" is a small lever; the bigger lever is letting the
model see *different kinds of text* during pretrain.

**Fix.** Synthesize 1K (natural_language ↔ structured_artifact)
pairs across four format families via teacher distillation
(Sonnet / Qwen-14B), seeded from existing GhostLM corpus
shards (NVD for STIX, security blogs for YARA + MISP, MITRE for
Sigma). Each generation passes a syntactic validator
(`parse_stix`, `parse_yara`, `parse_sigma`, `parse_misp`)
before write so the corpus stays clean. Drops into
`data/processed/train.jsonl` like every other distill output;
ghost-base sees STIX-shaped JSON, YARA-shaped DSL, etc., during
pretrain and learns the structural vocabulary, not just the
prose vocabulary.

**Scaffold.** `scripts/distill_format_aware.py` (commit `XXX`).
Four format adapters: `stix_indicator`, `yara_rule`,
`sigma_rule`, `misp_event`. Free smoke-test path on Ollama
(`--provider ollama --model qwen2.5:14b --max-traces-per-format
10`); production path on Anthropic (~$50-100 budget for 1K
clean traces). Resume-safe via the shared `ResumeIndex` from
`scripts/distill_common.py`; reruns skip already-distilled
seed records.

**Why it's the differentiator.** No other from-scratch cybersec
LM trains on the structured-format vocabulary at pretrain time.
The few that handle YARA / Sigma do it as a downstream tool
integration (call the model, regex-extract, format separately).
Native structural literacy is a different capability. It also
compounds with bet 1 (tool-use SFT): tools that emit STIX or
YARA in their responses become first-class citizens of the
training distribution rather than out-of-domain artifacts.

## How the bets compose

The six bets are independent but mutually reinforcing:

| Bet | Pairs well with | Anti-pairs with |
|---|---|---|
| 1 (tool-use SFT) | RAG layer, MCP tools, daily updates, format-aware pretrain (structured tool outputs) | (none) |
| 2 (daily updates) | tool-use SFT (more tools to call), context extension | (none) |
| 3 (custom BPE) | every other bet (smaller tokens = more budget); +1.6% measured, optional default | (none) |
| 4 (long context) | tool-use SFT (longer tool responses), MoE (more attention compute amortized) | (none) |
| 5 (MoE) | scaling beyond 1B; less impact at 360M ghost-base scale | parameter-efficient fine-tunes (LoRA on MoE is finicky) |
| 6 (format-aware pretrain) | tool-use SFT (tools emit STIX/YARA/Sigma cleanly), daily updates (fresh threat-intel artifacts arrive in these formats) | (none) |

Recommended sequencing:

1. **Done (2026-05-08):** Bet 3 (custom BPE) ran in ~30 min on
   M4. Result: +1.6% vs GPT-2 BPE, well below the +25-35%
   projection. The artifact is committed at
   `data/tokenizer/v1/` and wired into `ghostlm/tokenizer.py`
   as the `GhostTokenizerV1` opt-in backend; ghost-base default
   stays GPT-2 BPE pending a downstream eval.
2. **Now (M4-doable, free Ollama):** Bet 6 (format-aware
   pretrain) smoke-test on the existing seed shards. ~10
   traces per format on Ollama validates the prompt + parser
   pipeline end to end before committing $50-100 to the
   Anthropic production run.
3. **After ghost-base v1.0 GPU run lands:** Bet 1 (tool-use SFT)
   on top of ghost-base. ~$200 distillation budget + 1-2 GPU
   hours to fine-tune. The point of the GPU spend.
4. **In parallel with bet 1:** Bet 6 production run (~1K traces,
   $50-100 on Sonnet) so the SFT data already includes
   structured-format examples.
5. **After bet 1:** Bet 4 (long context) extension on the
   tool-using ghost-base. ~3-5 GPU hours. Unlocks IR workflows.
6. **Once owned hardware (Blackwell 96GB) lands:** Bet 2 (daily
   cron) becomes practical at home; before that it's a rented-
   GPU expense.
7. **When ghost-1B planning starts:** Bet 5 (MoE) bakes into the
   architecture from step 0. No retrofit cost.

## What this is not

- **A claim that GhostLM will beat GPT-4 on general benchmarks.**
  It won't, and that isn't the goal. The goal is a from-scratch,
  fully-auditable cybersec specialist that does narrow tasks
  well. The differentiation is along axes (tool grounding,
  freshness, context, transparency) where 7B+ general models
  don't compete.
- **A guarantee any of these bets work.** Bet 3 has a known
  failure mode (v0.5 attempted it once). Bet 1's payoff depends
  on ghost-base actually being capable enough to use tools. Bet
  4 might compromise short-context quality in a fine-tune
  recipe that's not yet validated. Each scaffold runs and
  produces a real artifact; whether the artifact wins is an
  empirical question the eval harness answers.
- **An exhaustive list.** Other plausible bets we considered and
  did not scaffold: RAFT-style retrieval-aware fine-tune (variant
  of bet 1), RLHF on offensive-security correctness with a
  domain-expert reward model, distillation-built corpus expansion
  to ghost-7B scale. STIX/MISP-aware structured-format pretrain
  graduated from this list to a full bet (bet 6 above). The six
  we shipped are the ones with the cleanest ROI given current
  GhostLM state.

## Summary

The six scaffolds collectively shift GhostLM from "another point
on the small-cybersec-LM benchmark plot" to "an artifact with a
recognizable shape: tool-grounded, continuously updated, cybersec-
tokenized, long-context, sparsely-activated, structurally
literate". Each scaffold is already in the repo and runs as soon
as compute / budget / operator attention are available. The
strategic claim isn't that any one bet definitely works; it's that
the **combination** of six reasonable bets gives GhostLM a
defensible identity that parameter-scale-only roadmaps don't.
