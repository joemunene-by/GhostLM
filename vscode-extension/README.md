# GhostLM Security Review — VS Code Extension

AI-powered code security review using GhostLM, a cybersecurity-focused language model.

## Features

- **Review File** — Scan entire file for common security vulnerabilities
- **Review Selection** — Right-click selected code to check for issues
- **Explain Vulnerability** — Get AI-powered explanations (requires GhostLM API server)
- **Inline Diagnostics** — Issues appear as warnings/errors in the editor

## Detected Vulnerability Categories

| Category | Severity | Examples |
|----------|----------|---------|
| SQL Injection | Critical | String concatenation in queries, f-strings in SQL |
| XSS | High | innerHTML, document.write, dangerouslySetInnerHTML |
| Command Injection | Critical | User input in exec/spawn/system calls |
| Hardcoded Secrets | High | API keys, passwords, private keys in source |
| Insecure Config | Medium | Disabled SSL, wildcard CORS, debug mode |
| Path Traversal | High | User input in file operations |

## Setup

```bash
cd vscode-extension
npm install
npm run compile
```

To test locally, press `F5` in VS Code to launch an Extension Development Host.

### With GhostLM API (optional, for AI explanations)

```bash
# From the GhostLM root directory
python scripts/api.py --checkpoint checkpoints/best_model.pt
```

Then use the "Explain Vulnerability" command for AI-powered analysis.

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `ghostlm.apiEndpoint` | `http://localhost:8000` | GhostLM API server URL |
| `ghostlm.maxTokens` | `200` | Max tokens for AI generation |
| `ghostlm.temperature` | `0.7` | Sampling temperature |

## Commands

- `Ctrl+Shift+P` → "GhostLM: Review File for Security Issues"
- `Ctrl+Shift+P` → "GhostLM: Explain Vulnerability"
- Right-click selection → "GhostLM: Review Selection for Security Issues"

## Built by

Joe Munene — [github.com/joemunene-by/GhostLM](https://github.com/joemunene-by/GhostLM)
