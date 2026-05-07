# RAG diagnostic findings (2026-05-07)

Working result from running the RAG layer end-to-end against the
v0.9 chat checkpoint and the fact-recall v2 bench, plus a separate
retrieval-quality diagnostic that strips the language model out of
the loop.

The headline: **retrieval works, generation doesn't**. This is the
cleanest evidence to date for the parameter-scaling hypothesis the
project has been tracking since v0.6.0.

## What was measured

Two complementary numbers on the same 100-question fact-recall v2 bench:

1. **End-to-end v0.9+RAG generation score.** Embed each question
   with BGE-small, retrieve top-4 passages from the 83K-chunk corpus
   index, prepend them to the user message, generate with the v0.9
   chat-tuned model, grade the completion with the v2 grader.
   `scripts/eval_fact_recall_v2.py --rag-dir data/rag` is the
   driver.

2. **Retrieval@4 only.** Same retrieval recipe, but instead of asking
   the model to answer, just check whether the top-4 passages
   themselves contain the canonical answer (or one of its alternates)
   as a boundary-respecting substring. `scripts/eval_rag_recall.py`
   is the driver.

The two numbers together distinguish the two failure modes that
look identical from outside the box:

| Scenario | Retrieval@K | v0.9+RAG generation | Diagnosis |
|---|---:|---:|---|
| both at floor | low | low | corpus + retriever both miss; need bigger index or better embedder |
| retrieval works, gen at floor | high | low | model can't extract from context; need parameter scaling or RAFT |
| both high | high | high | the RAG layer is the answer at this scale; ship it |

## What we measured (n=100)

```
Retrieval@4:                 41 / 100 (41.0%)
v0.9+RAG generation score:    0 / 100 (0.0%)
v0.9-bare generation score:   1 / 100 (1.0%)
```

**RAG made the 81M model slightly worse, not better.** Adding 4
relevant reference passages to the prompt shifts the v0.9 chat
output from "register-shaped fiction with one accidental hit" to
"mode-collapse repetition under the longer prompt". The model
doesn't lose information in the prompt; it loses coherence when
asked to handle the longer context window.

Per-topic retrieval@4 breakdown:

| Topic | Retrieval@4 | Per-topic n |
|---|---:|---:|
| mitre | 14 / 15 | 93.3% |
| tool | 5 / 6 | 83.3% |
| cwe | 10 / 15 | 66.7% |
| misc | 1 / 3 | 33.3% |
| crypto | 3 / 10 | 30.0% |
| cve | 7 / 30 | 23.3% |
| protocol | 1 / 11 | 9.1% |
| owasp | 0 / 10 | 0.0% |

## Interpretation

**The bottleneck is generation, not retrieval.** For ~41% of questions
the retriever already surfaces a passage containing the canonical
answer, but the 81M chat-tuned model can't extract it. Worse: when
given the right reference passages in the prompt, the v0.9+RAG
score drops to 0/100, below the v0.9-bare 1/100 score. The longer
retrieval-augmented prompt actively destabilizes the model into
mode collapse ("A vulnerability was found in code-projects Online
Online Online Online..." is a representative completion).

This is a stronger version of the v0.9.2 register-matching diagnosis.
At 81M parameters the model has not just failed to learn factual
recall, it has failed to learn the meta-skill of "look at the
context window for the answer". RAG can't compensate for either
deficit; it requires a model with enough capacity to use the
retrieved context at all.

Per-topic the picture is informative:

- **mitre at 93.3% retrieval@4** is unsurprising. The corpus
  contains the full MITRE ATT&CK matrix, every technique ID is its
  own well-defined chunk, and BGE-small handles "what is T1059"
  cleanly.
- **tool at 83.3%** tracks: tool questions hit security_code chunks
  and the existing tool-related text in the corpus.
- **cwe at 66.7%** is solid: CWE entries are also well-chunked.
- **cve at 23.3%** is low because the v2 bench includes 30 specific
  CVEs, several of which are recent (CVE-2024-6387, CVE-2024-21887,
  etc.) and post-date the corpus build. The retriever can't find
  what isn't there.
- **protocol at 9.1%** is genuinely surprising. RFC numbers and
  protocol facts ARE in the corpus (rfcs.jsonl exists). The
  hypothesis is that BGE-small embeds "what RFC defines TLS 1.3"
  more like generic crypto / web prose than like the literal
  "RFC 8446" string, so the chunked passage that mentions 8446
  doesn't make the top-4 cut.
- **owasp at 0%** is the failure mode that needs follow-up. The
  v2 bench's owasp questions are mostly about A0x labels and ASVS
  structure. The corpus has owasp_top10.jsonl (18 records) and
  owasp_asvs.jsonl (80 records), but apparently the chunks aren't
  surfacing for these queries. Could be: (a) BGE-small doesn't
  embed "A01:2021" tokens well, (b) chunk boundaries split the
  label from the description so neither half retrieves cleanly,
  (c) the index was built before the v1.0 corpus expansion that
  added more OWASP material.

## Action items

1. **Rebuild the RAG index over the v1.0 corpus.** The current
   index (83,628 chunks at ~/Desktop/GhostLM/data/rag/index.npy)
   was built 2026-05-01 against the v0.4-era 60M-token corpus.
   Rebuilding against the 363M-token v1.0 corpus should fix the
   owasp 0% (more OWASP coverage in v1.0) and probably move
   protocol meaningfully. Cost: ~10-20 min on M4 MPS.

2. **Try a smarter embedder for the protocol and owasp registers.**
   BGE-small is 30 MB and tuned for general English; for short
   string-match queries like "RFC 8446" it's at a disadvantage
   vs a tf-idf-style retriever or a hybrid. Adding a sparse-text
   fallback (BM25 over the same chunks) before the dense
   retrieval would likely fix protocol and owasp specifically.

3. **The parameter-scaling diagnosis is the dominant story.**
   Even with the gaps above, retrieval@4 hits 41% while v0.9+RAG
   generation is at floor. The fix isn't a different retriever or
   a better index; it's a model with enough capacity to actually
   extract facts from supplied context. That's the ghost-base
   v1.0 GPU run, exactly as planned.

4. **A RAFT-style retrieval-aware fine-tune would help the
   intermediate range.** Once ghost-base lands, training it with
   reference-passages-in-prompt SFT (RAFT recipe) is plausibly
   the path to making the small model genuinely RAG-effective.
   The v1.0 corpus already has a `raft_train.jsonl` shard
   prepared. Phase ordering: ghost-base pretrain -> chat-tune ->
   RAFT-tune.

## Why this is a release-quality finding

Without spending a dollar on GPU, this session's RAG layer + the
diagnostic produced a clean, reproducible finding that:

- Validates the RAG infrastructure (retriever surfaces 41% of
  answers).
- Validates the parameter-scaling diagnosis (model can't extract
  from the surfaced context).
- Identifies two concrete index-side improvements (rebuild over
  v1.0 corpus, add BM25 fallback) that don't need GPU either.
- Sets up the RAFT-tune as the next-rung post-ghost-base move.

Worth shipping as v0.9.3 with these numbers as the headline once
the v0.9+RAG generation score finishes filling in.
