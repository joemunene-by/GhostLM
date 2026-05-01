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

The server exposes three:

### `ghostlm_query(question)`

Free-form security Q&A. Use for general explanations, walkthroughs, "how
does X work" questions, comparisons.

### `ghostlm_explain_cve(cve_id)`

CVE-specific explainer. Pass an ID like `CVE-2021-44228`; returns the
affected product, vulnerability class, impact, and (where the model has
them) mitigations.

### `ghostlm_map_to_attack(description)`

Take a free-text description of an observed attack or capability and return
the most likely MITRE ATT&CK technique IDs, names, and a short justification.
Useful for incident-response triage and CTI-workflow integration.

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
