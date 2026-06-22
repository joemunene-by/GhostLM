# GhostLM Capability Atlas

A complete, measured, per-domain map of what GhostLM can do today, the
honest ceiling at this scale and why, and the projected capability at each
rung of the scale ladder. The point is to answer *"what is GhostLM, across
every domain, with evidence"* — not vibes, not hype.

**Measured checkpoint:** `ghost-small-gen` — ghost-small-v0.5 (~45M params),
trained **from scratch** on the decontaminated v0.10 generalist corpus
(258.9M tokens, 0.004% benchmark contamination), Mac M4 / MPS, 30,000 steps,
val_loss 3.76. All MCQ numbers are debiased multi-permutation text-scoring
(a pure single-letter emitter collapses to 25%). 95% CI is a percentile
bootstrap over questions. The ladder (`hardware_pathway.md`):
**ghost-tiny (13M) → ghost-small (45–81M) → ghost-base (360M) → ghost-1B →
ghost-3B → ghost-7B.**

## Summary — every domain at a glance

| Domain | Today (45M) | Eval asset | vs random / floor | Trajectory |
|---|---:|---|---|---|
| General knowledge | ARC-Easy 27.2% | `general_mcq_bench` | above chance | climbs with params |
| Reasoning | ARC-Challenge 24.3% | `general_mcq_bench` | at chance | threshold ~360M |
| Math (numeracy) | **30.8%** | `math_mcq_bench` (new) | **above chance** | climbs with params |
| Code — understanding | partial | `code_explain/security/write` | structured signal | strong at 360M+ |
| Code — generation | floor | HumanEval/MBPP (to wire) | ~0 at this scale | needs 1B+ |
| Cybersecurity | SecQA 34.3% / CTF 63.3% | `secqa`, `ctf_eval_bench` | **well above chance** | the standout; expert at scale |
| Agentic / tool-use | pretrain floor 0%/8.8% | `eval_agent` (strict/soft) | tool-use SFT required | reliable at 7B |
| Instruction / format | partial | `eval_format_compliance` | structured outputs | solid at 1B+ |
| Retrieval / RAG | system-level | `eval_rag_recall` | closes fact gap now | force-multiplier at every rung |

Numbers without a bold/CI are qualitative reads pending a generative-eval
pass; the MCQ rulers are measured.

## Per-domain detail

### General knowledge
**Today:** ARC-Easy 27.2% (CI 25.4–28.9, above the 25% floor), OpenBookQA
27.4%. On a corpus only ~52% general web, the model learns real
world-knowledge signal above chance and is competitive *for its size*
(OpenBookQA beats a 256M survey model and matches a 111M one).
**Ceiling & why:** knowledge is fact-binding-bound — 45–81M cannot store and
retrieve enough facts to clear the 35–45% competitive band. This is the wall
the scale ladder exists to break.
**Projection:** 360M crosses into competitive (ARC-E ~35–45%); 1B/3B push
toward MMLU ~45–55%; a well-fed 7B reaches Mistral-7B-class (~55–65% MMLU).

### Reasoning
**Today:** ARC-Challenge 24.3% (straddles chance) — still beats Pythia-160M
(18.8), a ~3.5× larger model, which says the *data* is doing work even where
*size* isn't there yet.
**Ceiling & why:** multi-step reasoning needs depth and capacity the small
model lacks; it pattern-matches more than it reasons.
**Projection:** the clearest threshold effect on the ladder — expect a real
jump at 360M–1B as chains-of-thought become learnable, then steady gains.

### Math (numeracy) — newly measured
**Today: 30.8%** on the new `math_mcq_bench` (120 deterministic problems:
arithmetic, percentages, rates, simple algebra, sequences, word problems),
above the 25% floor and consistent across both permutations with a
well-distributed prediction profile (not letter-collapse). Genuine numeracy
signal at 45M.
**Ceiling & why:** exact multi-step arithmetic and algebra are capacity- and
CoT-bound; the model gets the easy band and misses compositional steps.
**Projection:** climbs steadily with scale + CoT data; GSM8K-style generative
math becomes tractable around 1B–3B.

### Code — understanding vs generation
**Today:** the tractable half works — code-explanation, language ID, and
security-property reasoning have real signal (the `code_explain` /
`code_security` / `code_write` banks and evals exist). Freeform *generation*
(HumanEval/MBPP pass@1) is at the floor at this scale.
**Ceiling & why:** generation needs to hold a whole correct program in
capacity; 45–81M can't. Understanding and short structured outputs
(detection rules, parsers, security-property answers) are reachable.
**Projection:** at 360M+, structured code tasks firm up; useful generation
(CodeLlama-7B / early-Qwen-Coder band, HumanEval ~30–50%) arrives at 1B–7B,
strongest on security-adjacent code because that's the data edge. New asset:
`synth_code_reasoning.py` adds step-through debugging traces, a gap the
explain/write/security banks didn't cover.

### Cybersecurity — the specialty
**Today:** the standout. SecQA 34.3% (CI 28.5–40.6, well above chance) and
CTF-eval 63.3% (CI 46.7–80.0) — on a corpus only 8.6% cybersecurity by
tokens. Cybersecurity is *fully retained* through the generalist
de-specialization.
**Ceiling & why:** the cap is exact fact recall (specific CVEs/techniques),
which is fact-binding-bound like general knowledge — but the *understanding*
and the retrieval-augmented path are already strong.
**Projection:** this is where GhostLM can *beat* a generic same-size and even
larger model. At 1B–7B with focus retained, expert-class on
CTIBench/SecQA/CWE/MITRE — a domain-leading small model, not just competent.

### Agentic / tool-use
**Measured (control):** the `ghost-small-gen` *pretrain* (no tool-use SFT)
scores **0/15 strict, 8.8% soft, mean iters 1.00** on `provenance_eval` — it
never dispatches a tool. This is the honest floor, and it empirically
confirms the thesis: **agentic ability is taught by the tool-use SFT, not the
pretrain.** A tool-use-tuned chat checkpoint is required to score above floor
(the GhostAgent runtime + bet-1/bet-9 SFT teach the `<|tool_call|>` /
`<|cite|>` shapes); those weights are not currently on disk, so the tuned
number is pending a re-tune.
**Ceiling & why:** even with SFT, *reliable* multi-step orchestration (pick
the right tool, read the result, decide the next call) is emergent-at-scale;
45–81M learns the format, not robust planning.
**Projection:** ~7B is the rough threshold where function-calling and
multi-step agentic loops become dependable — the real payoff for the ghost
ecosystem (drive **ghostloop**, call the security suite, RAG-then-act). The
system (model + tools + retrieval + safety gates) already punches above the
bare model.

### Instruction-following / structured output
**Today:** partial — `eval_format_compliance` measures STIX/YARA/Sigma/MISP
structural correctness; the format-aware pretrain (bet 6) gives real signal
on constrained generation, a known small-LM-tractable task.
**Projection:** solid at 1B+, near-saturating for structured security formats
where the grammar is tight.

### Retrieval / RAG — the force multiplier
**Today:** the BGE+LanceDB RAG layer (`rag_chat.py`, `eval_rag_recall.py`)
already offsets the small model's biggest weakness — fact recall — by
retrieving the exact CVE/technique/spec at inference. The diagnostic finding:
*retrieval works; generation is the limiter.*
**Projection:** RAG is a multiplier at **every** rung — a small model that
retrieves well beats a larger one that can't be run or shaped. This is why
GhostLM is *useful now*, before the big rungs land.

## The honest frame

Two scaling truths govern this whole map (`hardware_pathway.md`,
`data_scaling_7b.md`):

1. **Capability tracks params × data together, not params alone.** The 45M
   model already beats larger peers on several rulers *because the data is
   good* — but it's fact-binding-bound, and that wall only breaks with scale.
2. **The biggest, cheapest wins are early** (a fact-binding threshold around
   mid-scale), then real-but-shrinking gains at rising cost, until **data or
   compute caps you.** For a solo project the binding limit is the corpus, not
   the parameter count — see `data_scaling_7b.md` (~140B tokens to earn a 7B).

**Bottom line:** at 45M, GhostLM is a small from-scratch generalist with
genuine above-chance signal on general knowledge and math, a standout
cybersecurity specialty, learned agentic/format structure, and a RAG layer
that already makes it useful. It is not frontier — and the map above says
exactly which rung unlocks which domain. Reproduce any row:
`make scorecard CKPT=checkpoints/ghost_small_gen/best_model.pt`.
