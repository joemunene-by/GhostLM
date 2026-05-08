# Format-compliance baseline: v0.9 chat (2026-05-08)

The structural-compliance metric is bet 6's
([docs/differentiation.md](differentiation.md) §"Bet 6: format-aware
structured-data pretrain") acceptance gate. To know whether bet 6
moves the number, we need the "before bet 6" floor on the same
checkpoint we plan to lift. This file captures that floor.

## Setup

- Checkpoint: `Ghostgim/GhostLM-v0.9-experimental` (a.k.a.
  `phase19_chat_v09/best_model.pt`, 81M params, 6L / 768d / 12h, GPT-2 BPE,
  trained on PRIMUS + CWE + OWASP + RFCs + fact-QA pretrain plus chat-tune).
- Eval set: the 8-record held-out eval at
  [`data/raw/format_aware_eval.jsonl`](../data/raw/format_aware_eval.jsonl)
  (2 records per format across STIX 2.1 indicators, YARA rules, Sigma
  rules, MISP events). Each record has the natural-language prompt and
  per-record `required_fields` / `required_substrings` tags. **The
  eval set is deliberately disjoint from the few-shot bank at
  `format_aware_seeds.jsonl`**: distillation reads from the few-shot
  bank, eval reads from this file, and the two share zero prompts. Bet
  6 lift numbers are therefore measured on unseen examples.

  The original baseline (commit `349c29f`) was scored on the few-shot
  bank itself, before the train-on-test fix in `bbe34c9` separated the
  files. Re-running on the held-out eval set reproduces the same
  headline number, so this doc captures both rows in the comparison
  table below.
- Inference: `scripts/run_format_baseline.py` against MPS on the M4
  with `temperature=0.7 top_k=50 top_p=0.95 max_tokens=600`.
- Scoring: `scripts/eval_format_compliance.py`.

## Headline result

| Format | n | parse-pass | fields-pass | parse % | fields % |
|---|---:|---:|---:|---:|---:|
| stix_indicator | 2 | 0 | 0 | 0.0% | 0.0% |
| yara_rule | 2 | 0 | 0 | 0.0% | 0.0% |
| sigma_rule | 2 | 0 | 0 | 0.0% | 0.0% |
| misp_event | 2 | 0 | 0 | 0.0% | 0.0% |
| **OVERALL** | **8** | **0** | **0** | **0.0%** | **0.0%** |

**v0.9 chat scores 0% parse, 0% fields on the structural-compliance
benchmark.** That is the baseline bet 6 has to beat.

## What v0.9 actually produces

The format ledger is consistent across all four families: when asked
to write a STIX bundle / YARA rule / Sigma rule / MISP event, v0.9
chat falls back to one of three failure modes.

**Failure mode 1: terminal collapse.** The first YARA prompt produced
the single token `A` followed by `<|ghost_end|>`. Two of the eight
prompts went this way. The chat-tune objective rewarded brevity in
the small-talk regime, and "I don't know how to start" lands at
maximally-short.

**Failure mode 2: register-shaped prose.** The first STIX prompt
produced free-form vulnerability description text: *"A vulnerability
was found in the SMBv1 service. A successful exploit of this issue
may be used to cause a denial-of-service condition."* Plausible NVD-
style sentence, zero JSON, zero STIX field. v0.9 has memorised CVE
prose patterns; the structural format request slides off.

**Failure mode 3: hallucinated narrative.** The Sigma T1059.001
prompt produced *"T1048.001 — Malicious Files or (ex: [File](https://
attack.mitre.org/techniques/T1553/002) and ..."* with a runaway tail
of `(Citation: Malicious File)` tokens. The model has seen MITRE
markdown narrative in the corpus and reaches for it when asked about
a T-code; it does not know that the *prompt* asked for YAML. (This
is also what triggered the `parse_sigma → yaml.safe_load → YAMLError`
crash that fix `cd48582` resolved.)

The other prompts produced similar prose-or-collapse outputs. None of
the 8 generations contained the literal `rule X {`, `title:`,
`"type": "indicator"`, or `"Event":` shells the parsers look for.

## Why this is the right baseline

A 0% floor sounds dramatic; in this context it's exactly the point.
v0.9 chat was trained on a corpus that contains ~zero structurally-
formatted CTI artifacts. STIX bundles, YARA rules, Sigma rules, MISP
events were not part of the FineWeb-Edu / NVD prose / arXiv / OWASP
text the model saw. The model has no structural prior to draw on.
**That is the lever bet 6 is designed to pull.**

If after a bet-6 distill run (1K traces from
`scripts/distill_format_aware.py`) followed by ghost-base pretrain
plus a brief format-aware SFT phase, the v1.0 number moves from 0%
to even ~30% parse / ~15% fields, that is a measured capability that
no other from-scratch cybersec LM at this scale reports.

If after the same intervention the number is still 0%, that's a real
falsifying result for bet 6's hypothesis and we discard the bet
honestly.

## Re-running

```bash
PYTHONPATH=. python3 scripts/run_format_baseline.py \
    --checkpoint <ckpt> \
    --seeds data/raw/format_aware_seeds.jsonl \
    --out logs/format_baseline_<run_name>.jsonl

PYTHONPATH=. python3 scripts/eval_format_compliance.py \
    --predictions logs/format_baseline_<run_name>.jsonl
```

The seed JSONL can be extended with new records as eval-set growth
continues; the harness scores whatever the seed file contains.

## Comparison rows (this table grows)

| Checkpoint | Date | Eval set | Pretrain notes | parse-pass % | fields-pass % | n |
|---|---|---|---|---:|---:|---:|
| v0.9 chat (81M) | 2026-05-08 | format_aware_seeds (leaky) | PRIMUS + CWE + OWASP + RFCs + fact-QA, no structured-format data | 0.0% | 0.0% | 8 |
| v0.9 chat (81M) | 2026-05-08 | format_aware_eval (held-out) | same checkpoint, same pretrain | 0.0% | 0.0% | 8 |
