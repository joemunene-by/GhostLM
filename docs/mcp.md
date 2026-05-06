# GhostLM MCP server

Exposes the chat-tuned model as a [Model Context Protocol](https://modelcontextprotocol.io/)
server so Claude Desktop / Claude Code users can query it as a local
cybersecurity-knowledge tool from inside any conversation.

## Why this exists

GhostLM is a 45M-parameter specialist; the big general-purpose models are
strong on reasoning but lack the depth of a curated security corpus. The
"small specialist + big generalist" pattern is exactly what MCP was designed
to enable. Running GhostLM as an MCP tool means Claude can offload narrow
security questions ("explain CVE-2021-44228", "map this incident-report
excerpt to ATT&CK techniques") to a model that has actually seen the
relevant material in training.

## Requirements

- **Python ≥ 3.10** (the official `mcp` SDK requires it; the rest of GhostLM
  runs on 3.9+ but the MCP server doesn't).
- The MCP SDK and the GhostLM dependencies::

      pip install mcp torch tiktoken

- A chat-tuned checkpoint, e.g. `checkpoints/phase5_chat_v2/best_model.pt`.

## Install

From the GhostLM repo root:

```bash
claude mcp add ghostlm -- python3 /absolute/path/to/GhostLM/scripts/mcp_server.py \
    --checkpoint /absolute/path/to/GhostLM/checkpoints/phase5_chat_v2/best_model.pt
```

Verify it loaded:

```bash
claude mcp list
```

You should see `ghostlm` listed. Restart any running Claude Code / Claude
Desktop session for it to pick up the new server.

## Tools

The server exposes six tools, split between **model-backed** (subject to
hallucination at 81M scale, prefer for prose-style questions) and
**deterministic / fact-grounded** (no model invocation, prefer for factual
lookups).

### Model-backed

#### `ghostlm_query(question)`

Free-form security Q&A. Use for general explanations, walkthroughs, "how
does X work" questions, comparisons.

#### `ghostlm_explain_cve(cve_id)`

CVE-specific explainer. Pass an ID like `CVE-2021-44228`; returns the
affected product, vulnerability class, impact, and (where the model has
them) mitigations. For canonical CVE data, use `ghostlm_search_cve_nvd`
instead and treat this as the editorial-summary alternative.

#### `ghostlm_map_to_attack(description)`

Take a free-text description of an observed attack or capability and return
the most likely MITRE ATT&CK technique IDs, names, and a short justification.
Useful for incident-response triage and CTI-workflow integration.

#### `ghostlm_rag_query(question, top_k=4)`

Retrieval-augmented version of `ghostlm_query`. Embeds the question with
BGE-small, retrieves the top-K most-similar passages from the GhostLM corpus
index (~83K cybersec chunks), and conditions generation on those passages.
Substantially reduces hallucination on factual questions vs the bare
`ghostlm_query` tool. Falls back to bare query if the index is unavailable
(offline, Hub down, etc).

### Deterministic / fact-grounded

#### `ghostlm_search_cve_nvd(cve_id)`

Live lookup against NIST's National Vulnerability Database REST API.
Returns the canonical description, CVSS v3 + v2 base scores, CWE
references, and publication dates. The model is not invoked. Use this
whenever you need authoritative CVE data; reach for `ghostlm_explain_cve`
only when you want a model-style summary.

#### `ghostlm_lookup_mitre_technique(technique_id)`

Local-corpus lookup of a MITRE ATT&CK technique by ID (e.g. `T1059`,
`T1059.001`, `TA0001`). Reads from the bundled `data/raw/mitre_attack.jsonl`
and `mitre_full.jsonl` shards; returns the canonical technique text exactly
as it appears in the GhostLM corpus. Deterministic; the model is not
invoked.

## Caveats

The model is small and will be wrong. Use it as a fast-first-pass tool, not
an authoritative source. Verify CVE numbers, exact CVSS scores, and dates
against NVD; verify technique mappings against MITRE ATT&CK Navigator. The
goal is to keep Claude focused on reasoning over the right material rather
than guessing at security specifics from general pretraining.

## Architecture

The MCP server is a stdio transport process — Claude launches it as a
subprocess and communicates via JSON-RPC on stdin/stdout. The model is
loaded once at startup and stays resident; subsequent tool calls reuse the
same warm model. No network traffic, no telemetry, no third-party.
