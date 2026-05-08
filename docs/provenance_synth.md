# Bet 9: provenance-aware tool-use templated synth

## Why this is the deployment-grade differentiator

In a SOC context, wrong-but-confident is worse than honest-uncertain.
Operators need every claim traceable to its source: the CVE entry,
the MITRE technique, the RAG passage, the specific field of the
tool response that justifies the assistant's statement. Big general-
purpose models do not do this consistently because their RLHF reward
favours fluency over auditability.

A small from-scratch LM trained day-one on cite-mandatory traces is
a demonstrably different deployment artifact. This is the bet that
matters most when GhostLM is being evaluated for adoption inside an
actual security operations workflow.

## Trace shape

The provenance trace extends the bet 1 four-message format with
inline `<|cite|>...<|/cite|>` tags in the final assistant answer:

```
USER:      What does ATT&CK technique T1059.001 do?
ASSISTANT: <|tool_call|>{"name": "lookup_mitre_technique",
                         "args": {"technique_id": "T1059.001"}}<|/tool_call|>
TOOL:      <|tool_response|>{"id": "T1059.001",
                              "name": "Command and Scripting Interpreter: PowerShell",
                              "tactic": "Execution",
                              "platform": "windows",
                              "summary": "..."}<|/tool_response|>
ASSISTANT: T1059.001 (Command and Scripting Interpreter: PowerShell)
           <|cite|>mitre:T1059.001#name<|/cite|> is an Execution
           technique on windows <|cite|>mitre:T1059.001#tactic<|/cite|>:
           Adversaries use PowerShell for execution; commonly via
           -EncodedCommand or remote download.
           <|cite|>mitre:T1059.001#summary<|/cite|>.
```

Every factual claim in the answer is followed by a cite tag pointing
at the specific field of the tool response that justifies it. The
model trained on this corpus learns "claim, cite, claim, cite" as
its default rhythm.

## Cite tag scheme

```
<|cite|>{source_type}:{source_id}[#field]<|/cite|>
```

| source_type | source_id meaning | example |
|---|---|---|
| `nvd` | CVE identifier | `<|cite|>nvd:CVE-2017-0144#description<|/cite|>` |
| `mitre` | ATT&CK T-code (or sub-technique) | `<|cite|>mitre:T1059.001#tactic<|/cite|>` |
| `cwe` | CWE identifier | `<|cite|>cwe:CWE-89#description<|/cite|>` |
| `rag` | RAG passage id (e.g. `passage_0`) | `<|cite|>rag:passage_0#owasp_top10.jsonl<|/cite|>` |

The optional `#field` segment names which specific field of the
source the claim refers to. This is what separates "the model knows
to cite something" from "the model knows precisely what part of the
something it is citing." Training on the latter makes the cite tags
useful for downstream verification.

## Quality filter

`trace_with_cites_quality_ok` extends the existing `trace_quality_ok`
filter from `distill_tool_use.py` with two additional checks:

1. At least one `<|cite|>...<|/cite|>` tag must appear AFTER the
   closing `<|/tool_response|>` tag (i.e. the cite is in the final
   answer, not inside the tool response).
2. Every cite tag body must match the `source_type:source_id` shape
   (contains a colon, both halves non-empty).

This rejects traces that either drop cites entirely or emit malformed
cite tags.

## First run (2026-05-08)

```bash
PYTHONPATH=. python3 scripts/synth_tool_use_provenance.py \
    --max-cve 200 --max-cwe 100 --max-rag 100
```

| Tool | Accepted | Rejected | Notes |
|---|---:|---:|---|
| search_cve_nvd | 200 | 0 | Cite tags add words that push borderline traces above the 40-word floor |
| lookup_mitre_technique | 30 | 0 | |
| lookup_cwe | 100 | 0 | |
| rag_retrieve | 99 | 1 | One sparse-text seed |
| **TOTAL** | **429** | **1** | 99.8% acceptance |

Higher acceptance than the plain (no-cite) bet 1 run (424/430 =
98.6%) because the cite tags themselves contribute words; some
short CVE descriptions that fell under the word floor in the plain
flow now clear it after cite augmentation.

## Composition: stacking with bet 1

Bet 1 produces 424 plain tool-use traces. Bet 9 produces 429 cite-
augmented traces over the same seeds. Stacked, that's an **~853-
record SFT corpus** with two phases:

1. Plain traces teach the four-message tool-use rhythm (USER asks,
   ASSISTANT issues tool call, TOOL responds, ASSISTANT answers).
2. Cite-augmented traces teach the *provenance* rhythm on top:
   every claim in the final answer carries a cite, and the cite
   names the field of the tool response that justifies the claim.

A model trained on both phases learns the four-message shape AND
the citation discipline. Operators can verify any claim by reading
the cite, looking up the named field in the tool response, and
confirming the assistant's restatement.

## What this enables at eval time

Once ghost-base trains on this corpus, the eval becomes:

- **Coverage**: of N factual claims in the assistant answer, how
  many are covered by a cite? (target: 100%)
- **Accuracy**: of M cite tags, how many resolve to a field that
  actually appears in the tool response? (target: 100%)
- **Specificity**: of K cite tags, how many use the optional
  `#field` segment to name the specific field? (target: ~80%)

These are clean, automatable evals (the harness parses cite tags,
matches against the tool response, scores). Versus the v0.9
baseline which has zero citation discipline, any non-zero coverage
score is a measured win.

## Reproducing

```bash
PYTHONPATH=. python3 scripts/synth_tool_use_provenance.py \
    --cve data/raw/cve_full.jsonl \
    --cwe data/raw/cwe.jsonl \
    --out data/processed/synth_tool_use_provenance.jsonl \
    --max-cve 200 --max-cwe 100 --max-rag 100
```

Deterministic: same corpus + same script produces byte-identical
output.
