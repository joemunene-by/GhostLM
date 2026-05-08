# Bet 7: code-for-security templated synth

## Why this exists

GhostLM has been trained almost entirely on cybersec *prose*: CVE
descriptions, MITRE writeups, security blogs, OWASP guides. It has
seen vanishingly little *code* in security context. Generalist small
models (Llama-3.2-1B, Qwen2.5-0.5B) do see code in pretrain, but
their mix dilutes security-relevant code with general code, and
their RLHF often filters out exploit-shaped content. **A small
from-scratch LM trained natively on security-context code is a
different artifact** and the right shape for a security analyst
workflow that's frequently looking at vulnerable / patched code,
exploit POCs, and patches.

Bet 7 is the templated synthesis path. The LLM-distilled equivalent
(GitHub patch-corpus mining, exploit-DB POC explanation distillation)
is a follow-on that pairs naturally with this baseline.

## Pattern bank

[`data/raw/code_security_patterns.jsonl`](../data/raw/code_security_patterns.jsonl)
is the hand-curated source of truth. Each entry has:

```json
{
  "id": "PAT-NNN",
  "cwe": "CWE-NN",
  "name": "Human-readable vulnerability class",
  "language": "python | javascript | c",
  "vulnerable": "<code snippet>",
  "patched":    "<code snippet>",
  "explanation": "<why vuln is exploitable, why fix works>",
  "cve_examples": ["CVE-YYYY-NNNN", ...]
}
```

The initial bank ships with 12 patterns covering the OWASP Top
10-shaped vulnerability classes plus a few canonical CWEs:

| Pattern | CWE | Language |
|---|---|---|
| SQL Injection via string concatenation | CWE-89 | Python |
| OS command injection via shell=True | CWE-78 | Python |
| Path traversal via unsanitised filename | CWE-22 | Python |
| Reflected XSS via innerHTML | CWE-79 | JavaScript |
| Insecure deserialisation via pickle | CWE-502 | Python |
| Stack buffer overflow via strcpy | CWE-120 | C |
| Hard-coded credentials in source | CWE-798 | Python |
| Predictable random for security tokens | CWE-330 | Python |
| AES-ECB mode leaks plaintext patterns | CWE-327 | Python |
| JWT none-algorithm acceptance | CWE-347 | Python |
| XXE via default XML parser | CWE-611 | Python |
| SSRF via unrestricted outbound fetch | CWE-918 | Python |

The bank is plain JSONL. Adding patterns is just appending records;
the synth script has no Python-coded knowledge of which patterns
exist. Realistic goal: grow to 50-100 patterns covering the full
OWASP Top 10 + CWE Top 25 across Python / JS / C / Go / Java /
PowerShell / shell.

## Output formats per pattern

[`scripts/synth_code_security.py`](../scripts/synth_code_security.py)
emits four record variants per pattern:

1. **`pretrain_prose`**: flat markdown article with title (CWE),
   vulnerable code, patched code, explanation, real-world CVEs.
   Right shape for pretrain corpus mixing.

2. **`identify_and_fix`**: chat Q&A. USER shows vulnerable code
   and asks "what is wrong + how would you fix it"; ASSISTANT
   identifies the CWE, explains the bug, shows the patched version,
   references CVEs.

3. **`explain_the_diff`**: chat Q&A. USER shows both versions and
   asks "why is the second safer"; ASSISTANT explains the security
   property each version has / lacks.

4. **`cwe_mapping`**: chat Q&A. USER shows vulnerable code and
   asks "which CWE class"; ASSISTANT names the CWE with a two-
   sentence rationale plus CVE examples.

12 patterns × 4 variants = 48 records on the initial bank, all
parser-clean (every record passes the lightweight word-count plus
code-fence-presence filter).

## Run

```bash
PYTHONPATH=. python3 scripts/synth_code_security.py \
    --bank data/raw/code_security_patterns.jsonl \
    --out data/processed/synth_code_security.jsonl
```

Deterministic. Same bank + same script produces byte-identical
output. The output JSONL lands in `data/processed/` and is
gitignored along with the other generated training files;
regenerate as needed.

## Composition with other bets

| Bet | Trains | Pairs with bet 7 by |
|---|---|---|
| Bet 1 (tool-use SFT) | "lookup before answering" | bet 7 records can include `<|tool_call|>` to fetch CWE / CVE detail before answering |
| Bet 6 (format-aware) | STIX / YARA / Sigma / MISP emission | bet 7 vuln descriptions can serve as the natural-language input for STIX-indicator generation |
| Bet 9 (provenance) | citation tags | bet 7 explanations can carry `<|cite|>{cwe_id}<|/cite|>` to teach source attribution |

## Why this bet earns big-company / research attention

GPT-4 / Claude / Llama do CWE explanations *adequately* but not
better than a junior security engineer. A small from-scratch LM
that does this comparably at 1-3% the inference cost AND reads
exploit-shaped content without RLHF refusals is a genuinely
different artifact. The reproducibility (every record is a
deterministic template + curated bank) means anyone reading the
GhostLM paper can re-derive the training data exactly, which is
the academic-publishing bar that closed-model recipes can't meet.

## What this does NOT replace

Templated synth gives the model *patterns*. Real understanding of
novel vulnerabilities needs:

- LLM-distilled records of *real* GitHub commits that fix CVEs
  (mining the patch corpus is bet 7's planned LLM-distill phase)
- POC explanation traces from Exploit-DB content
  (`data/raw/exploitdb.jsonl` already in repo as a seed)
- Long-context vulnerability reports where the bug crosses files

Templated synth + LLM-distilled records + corpus-aware long-context
fine-tune (bet 4) is the full bet 7 picture. This commit ships the
deterministic floor.
