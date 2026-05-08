# Bet 8: binary / hex literacy templated synth

## Why this is the most novel bet

Big LMs cannot read a hex dump. Their pretrain saw vanishingly little
of it; their tokenizers fragment hex bytes into 1-2 character tokens
that wreck context efficiency; their RLHF preference data has no
signal for "did the model correctly identify a PE header." A small
from-scratch LM trained natively on binary-as-text is a fundamentally
different artifact.

This is the bet most likely to reach research-community attention
because it's measurably first-of-kind. Reading a hex dump is a clean
eval (give unannotated bytes, ask for the byte-signature). No other
small cybersec LM does this; even GPT-4 fails on real obfuscated
shellcode without explicit prompt engineering. A small open-source
LM that handles this natively is a genuine first.

## Pattern bank

[`data/raw/binary_literacy_patterns.jsonl`](../data/raw/binary_literacy_patterns.jsonl)
holds 15 hand-curated patterns across five categories:

| Category | n | Coverage |
|---|---:|---|
| `file_magic` | 7 | PE / ELF / Mach-O / ZIP / PDF / OLE2 / PNG |
| `packer` | 2 | UPX section signatures, Themida / VMProtect markers |
| `shellcode` | 3 | NOP sled, x64 function prologue, Linux x64 syscall pattern |
| `pe_field` | 2 | Optional Header Magic (PE32 vs PE32+), Machine field |
| `disassembly` | 1 | Linux x64 execve('/bin/sh') canonical 28-byte payload |

Each entry carries:

```json
{
  "id": "BIN-NNN",
  "category": "file_magic | packer | shellcode | pe_field | disassembly",
  "name": "Human-readable label",
  "hex_at_offset_0": "<bytes at the canonical anchor point>",
  "ascii_decode": "<ASCII representation, with escapes for non-printable>",
  "longer_pattern": "<extended sequence showing context>",
  "explanation": "<paragraph explaining what the bytes mean and why it matters>",
  "examples": ["<real-world artefact 1>", "<artefact 2>", ...]
}
```

The bank is plain JSONL. Adding patterns is appending records. The
synth script has no Python-coded knowledge of which patterns exist;
it iterates whatever the bank contains.

## Output formats per pattern

[`scripts/synth_binary_literacy.py`](../scripts/synth_binary_literacy.py)
emits 2-3 record variants per pattern depending on category:

1. **`pretrain_prose`** (every pattern): flat markdown article with
   the bytes at offset 0, ASCII decode, longer signature, and prose
   explanation. Right shape for pretrain corpus mixing.

2. **`identify_hex`** (every pattern): chat Q&A. USER pastes a hex
   sequence and asks "what is this and how do you know"; ASSISTANT
   names the pattern and walks through the byte-level reasoning.

3. **`show_magic`** (file_magic / packer / shellcode / pe_field
   only; skipped for disassembly): chat Q&A. USER asks "show me the
   magic bytes of `<X>`"; ASSISTANT gives hex + ASCII + structural
   context.

15 patterns produce 44 records on the initial bank: 15 pretrain +
15 identify + 14 show_magic (the disassembly entry has no "magic"
concept). 100% parser-pass under the lightweight word-count filter.

## Run

```bash
PYTHONPATH=. python3 scripts/synth_binary_literacy.py \
    --bank data/raw/binary_literacy_patterns.jsonl \
    --out data/processed/synth_binary_literacy.jsonl
```

Deterministic. The output JSONL is gitignored under `data/processed/`;
regenerate as needed.

## Why the bank is intentionally curated

Auto-extracting hex dumps from binary corpus would produce far more
volume but with no human verification of the explanation that
accompanies each. Bet 8's value is *teaching the model to reason
about bytes*, not just exposing it to bytes. A 15-pattern hand-
curated bank where every explanation has been written by a human
who understands the format is higher signal per token than 1000
auto-extracted dumps.

The realistic next-stage growth path is:
- Expand to ~50 patterns (add Linux ELF Section header layouts,
  Mach-O fat-binary header, more shellcode variants, more disassembly
  prologues across calling conventions, more packer signatures).
- Add a separate auto-extraction pipeline for raw hex dumps from
  curated binaries (system DLLs, public-domain malware samples)
  with LLM-distilled annotations.
- Pair with bet 4 (long context) so a 32K window can hold a full
  hex-and-disassembly dump of a small function for the model to
  reason over end-to-end.

## What bet 8 buys at evaluation time

Once ghost-base trains on these records, a held-out eval becomes
straightforward:

- Parse-pass: given hex bytes, does the model name the format
  correctly? (binary classification per pattern)
- Field-pass: given a hex sequence, does the model identify the
  specific field at the right offset? (e.g., "what is the e_machine
  value in this ELF header")
- Disassembly-pass: given a hex byte sequence that is x86_64 code,
  does the model correctly identify the operation? (e.g., "this is
  a syscall to write")

Versus the v0.9 baseline (which has zero binary literacy by
construction, since none of its training corpus was hex), any
non-zero pass rate is a clean win and a publishable result.
