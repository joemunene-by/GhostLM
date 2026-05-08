# Bet 1 templated synth corpus

## Purpose

Bet 1 ([docs/differentiation.md](differentiation.md) §"Bet 1: tool-grounded
model, not memorization-based") needs ghost-base to learn the meta-skill
of "lookup before answering" by training on traces of the form:

```
USER:      <cybersec question>
ASSISTANT: <|tool_call|>{"name": "<TOOL>", "args": {...}}<|/tool_call|>
TOOL:      <|tool_response|>{...}<|/tool_response|>
ASSISTANT: <answer that uses ONLY the tool response>
```

The canonical pipeline is
[`scripts/distill_tool_use.py`](../scripts/distill_tool_use.py) calling
an LLM teacher (Anthropic ~$200, free Ollama). This doc captures the
parallel deterministic-template path in
[`scripts/synth_tool_use.py`](../scripts/synth_tool_use.py) that
produces parser-valid traces with no LLM spend and no GPU. Same
quality bar (`trace_quality_ok` from `distill_tool_use`) so the
templated records are evaluated identically to the LLM flow.

## Run + result (2026-05-08)

```bash
PYTHONPATH=. python3 scripts/synth_tool_use.py \
    --max-cve 200 --max-cwe 100 --max-rag 100
```

| Tool | Accepted | Rejected | Seed |
|---|---:|---:|---|
| search_cve_nvd | 195 | 5 | First 200 non-rejected CVE-2020+ entries from `data/raw/cve_full.jsonl` |
| lookup_mitre_technique | 30 | 0 | Hand-curated 30-technique bank covering all 12 ATT&CK tactics |
| lookup_cwe | 100 | 0 | First 100 entries from `data/raw/cwe.jsonl` |
| rag_retrieve | 99 | 1 | First 100 entries from `owasp_top10` + `owasp_asvs` + `rfcs` + `security_blogs` |
| **TOTAL** | **424** | **6** | (~98.6% acceptance) |

**Every accepted trace passes `trace_quality_ok`**: contains the four
literal tag strings, has a parseable JSON tool_call body with `name`
and `args` fields, and the `name` is in the `TOOLS` dict from
`distill_tool_use`. The 6 rejections are mostly CVE entries where the
description is too short to clear the 40-word minimum after templating.

## "Not found" injection (~10%)

Every 10th trace per tool returns an empty / not-found tool response,
and the assistant answer acknowledges the gap rather than fabricating
content. This trains the model to say "I don't know based on this
lookup" when the tool yields nothing useful, exactly per the bet 1
specification at the top of `distill_tool_use.py`. Examples in the
output JSONL:

```
USER: What is CVE-2024-12345 about?
ASSISTANT: <|tool_call|>{"name":"search_cve_nvd","args":{"q":"CVE-2024-12345"}}<|/tool_call|>
TOOL: <|tool_response|>{"cve":"CVE-2024-12345","found":false,"matches":[]}<|/tool_response|>
ASSISTANT: The lookup for CVE-2024-12345 returned no matches in NVD. I don't
know what this CVE is about based on this search alone...
```

The not-found answer enumerates plausible reasons (deprecated id,
stale snapshot, narrow query) so it stays informative without
confabulating facts.

## Why this matters for bet 1

The v0.9.3 RAG diagnostic ([`docs/rag_diagnostic_findings.md`](../docs/rag_diagnostic_findings.md))
found that v0.9 chat extracts the right fact from a supplied passage
**1% of the time** even when the retriever surfaces it 41% of the
time. The model isn't trained on the supplied-context-extraction
objective; the chat-tune was on prose Q&A.

Bet 1's training-time fix is exactly that objective: every assistant
answer in a tool-use trace is *constrained* to facts that appeared in
the tool response. Ghost-base trained on these traces learns
"answer = restate-the-tool-output" as the default behavior, not
"answer = paraphrase-from-memory."

The 424 templated records give ghost-base a structural floor for that
objective. When the LLM-distilled records arrive (~$200 budget for
~10K traces from `distill_tool_use.py`), they layer on idiomatic
variety on top of the floor.

## Per-tool template detail

### search_cve_nvd
- **Question template**: `What is {CVE-id} about?`
- **Tool args**: `{"q": "{CVE-id}"}`
- **Tool response**: `{"cve": "{id}", "description": "{text[:600]}", "cvss": <regex-extracted or null>, "source": "nvd"}`
- **Answer**: First sentence of CVE description plus CVSS line if found.

### lookup_mitre_technique
- **Question template**: `What does ATT&CK technique {T-code} do?`
- **Tool args**: `{"technique_id": "{T-code}"}`
- **Tool response**: `{"id": "{T-code}", "name": "{name}", "tactic": "{tactic}", "platform": "{platform}", "summary": "{summary}", "url": "https://attack.mitre.org/..."}`
- **Answer**: One-sentence summary referencing the technique name + tactic + platform.

### lookup_cwe
- **Question template**: `What is {CWE-id}?`
- **Tool args**: `{"cwe_id": "{CWE-id}"}`
- **Tool response**: `{"id": "{id}", "name": "{first 80 chars of first line}", "description": "{text[:500]}", "url": "https://cwe.mitre.org/..."}`
- **Answer**: One-sentence definition derived from the corpus text.

### rag_retrieve
- **Question template**: `From the cybersec corpus, what does this say about {topic_hint[:60]}?`
- **Tool args**: `{"query": "{topic_hint}", "k": 4}`
- **Tool response**: `{"query": "{topic}", "passages": [{"text": "{seed_text[:600]}", "source": "{filename}", "score": 0.87}, ...]}`
- **Answer**: First-passage paraphrase, capped at 300 chars.

## Why 98.6% > LLM teacher's typical 60-80%

LLM distillation in practice loses ~20-40% of generations to:
- Forgotten tag strings (`<|tool_call|>` paraphrased to `[tool_call]`)
- Tool_call body that's not valid JSON (extra prose, trailing commas)
- Tool name typos or made-up tools not in the registry
- Answer that confabulates beyond the tool response

Templated synth lands 98.6% because the script only emits text that
matches the parser; failures land at template-edit time, not training
time. The remaining 1.4% rejections are seed-driven (CVE descriptions
under the 40-word floor after templating).

## Output schema

Every trace lands as a `DistillRecord`-shaped JSONL entry:

```json
{
  "id": "synth_tool_use#<seed_id>#<hash>",
  "source": "synth_tool_use",
  "teacher": "templated",
  "seed_source": "search_cve_nvd | lookup_mitre_technique | lookup_cwe | rag_retrieve",
  "seed_id": "<original CVE/T-code/CWE id or seed_path>",
  "text": "<the four-message trace string>"
}
```

Drops into the SFT training data identically to the LLM-distilled
output of `distill_tool_use.py`. The trainer does not care which
path produced the trace.

## Reproducing

```bash
PYTHONPATH=. python3 scripts/synth_tool_use.py \
    --cve data/raw/cve_full.jsonl \
    --cwe data/raw/cwe.jsonl \
    --out data/processed/synth_tool_use.jsonl \
    --max-cve 200 --max-cwe 100 --max-rag 100
```

Deterministic: same corpus + same script produces byte-identical
output. The output JSONL is gitignored under `data/processed/*`;
regenerate as needed.

## Scaling

The `--max-cve` / `--max-cwe` / `--max-rag` knobs scale the volumes
linearly with the seed corpora. CVE corpus has ~186K usable entries,
CWE has ~969 entries, RAG seed shards have ~400 records combined.
Realistic max without re-curating: ~5K CVE + ~969 CWE + ~400 RAG +
30 MITRE = ~6.4K traces. That's ~64% of the 10K target volume that
the LLM-distilled flow aims for; the templated path is well-suited
to do most of the heavy lifting and reserve LLM budget for variety.

## What this does NOT replace

Templated synth produces rigid, low-diversity traces. A production
SFT mix should pair:

1. ~5K templated records (this script): structural floor, 98.6%
   parser-valid, deterministic.
2. ~5K LLM-distilled records (`distill_tool_use.py`): idiomatic
   variety, ~60-80% parser-valid, costs ~$200 on Sonnet.

The combination gives ghost-base both the *shape* (templates) and
the *idiom* (LLM teacher) of tool-use traces. Ship one, the other
follows.
