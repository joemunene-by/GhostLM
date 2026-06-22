# Data scaling plan: feeding a GhostLM-7B

The scale ladder (`hardware_pathway.md`) ends at **ghost-7B**. The honest
blocker for 7B is not the parameters, it is the **token budget**. This doc
sets concrete per-domain token targets, sources, and phasing so a 7B run is
*earned*, not undertrained.

## The target

Chinchilla-optimal is ~20 tokens/param, so a 7B wants **~140B training
tokens**. The current generalist corpus is ~**259M tokens** (`RESULTS.md`,
2026-06-16 snapshot), so this is a **~500× climb** — the real work of the 7B
rung. Over-training past Chinchilla (Llama-3 did ~1,875× for an 8B) buys
better inference-time quality, but 140B is the floor that makes 7B params pay
off, and it is already a serious pipeline.

## Target mix and token budget

GhostLM's edge is a security specialty inside a competent generalist. So the
mix weights security well above a generic model while keeping a strong general
+ code backbone (vs the de-specialized 8.6% security in the current generalist
snapshot):

| Domain | Target % | Tokens | Rationale |
|---|---:|---:|---|
| General web + knowledge | 47% | ~66B | Reasoning/knowledge backbone — the biggest unlock vs today. |
| Code | 19% | ~27B | Up from ~11% — coding + tool-use are target capabilities. |
| Cybersecurity (the edge) | 14% | ~20B | The differentiator; far above a generic 7B's ~0%. |
| Math + reasoning | 11% | ~15B | CoT, structured reasoning. |
| Instruction + tool-use | 9% | ~13B | Chat, RAFT, agentic ghostloop traces, distillation. |

## Sources per domain

- **General (66B)** — `collect_fineweb_edu.py` (FineWeb-Edu, quality-classifier
  filtered, hundreds of B available) is the bulk; + Wikipedia, StackExchange,
  open books, C4. The easiest target to hit.
- **Code (27B)** — **The Stack v2 / StarCoderData** (permissive, deduped) is
  the realistic bulk. `collect_code_corpus.py` (105 repos / ~42M tokens) is the
  wrong tool at this scale — keep it for a curated security-code subset only.
- **Cybersecurity (20B)** — the hard one: high-quality security text is scarce,
  there are not 20B unique tokens in the wild. Path: a few B of real corpus
  (PRIMUS, full NVD/CVE via `collect_nvd_full.py`, MITRE via
  `collect_mitre_full.py`, CWE, OWASP, ExploitDB, CTFtime, CISA/vendor
  advisories, RFCs, arXiv cs.CR full-text — most collectors already exist) **+
  heavy synthetic generation** (the Qwen-distilled Q&A and templated synth
  banks) **+ upsampling** (2–4 epochs on security while general sees 1). The 20B
  is *effective* tokens, not unique.
- **Math (15B)** — OpenWebMath, FineMath, proof-pile-2, AlgebraicStack, plus
  CoT/reasoning traces (`collect_math_reasoning.py` is the seed).
- **Instruction/tool (13B)** — the SFT corpus, bet-1/bet-9 tool-use traces,
  RAFT, distillation traces (`distill_*` pipeline), and agentic ghostloop
  episodes.

## The parts that bite

1. **Security scarcity → synth dependency.** Most of the 20B security target is
   synthetic/upsampled. Fine (the synth banks are good), but it caps *novel*
   fact density and risks collapse if overdone — hold a real:synth ratio you
   trust and decontaminate hard.
2. **Dedup + decontamination at 140B.** MinHash near-dedup + benchmark
   decontamination (currently 0.004%) become mandatory, the dominant quality
   lever.
3. **Storage + tokenize.** ~140B tokens is ~300–500 GB tokenized; the prep
   pipeline (download → filter → dedup → tokenize → shard memmap) is itself a
   multi-day, big-disk job.
4. **Tokenizer.** v1 BPE is 32K; for a general+code+security 7B, retrain at
   **48–64K vocab** for efficiency across the mix.
5. **Compute.** 140B through a 7B is the weeks-on-a-card / multi-GPU regime
   (fp8 + grad-checkpointing per `hardware_pathway.md`). The M4 caps at 81M —
   this rung is gated on real silicon.

## Phasing

- **A — Backbone:** FineWeb-Edu (66B) + The Stack (27B) → download / filter /
  dedup. The bulk, mostly mechanical.
- **B — Specialty:** scale security to 20B effective (collectors + synth +
  upsample) and math to 15B.
- **C — Instruction/tool:** expand SFT + tool-use + distillation to 13B.
- **D — Assemble:** retrain tokenizer (48–64K), set curriculum (general early,
  specialty + instruction late), shard, pretrain.

## Bottom line

The params are the easy part. ~90B of this (general + code) is download-and-
filter; the real craft is the ~20B security layer (scarcity + synth) and the
dedup/decontam discipline at scale. Nail those and a GhostLM-7B is a credible
Mistral-7B-class generalist with a real security edge — see
`capability_atlas.md` for the per-domain capability projection across the rungs.
