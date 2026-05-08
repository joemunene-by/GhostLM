# v0.9.5 templated-synth corpus + reproducer

Two infrastructure pieces that close the v0.9.5 deliverable.

## 1. Combined synth corpus

[`scripts/build_v15_combined_synth.py`](../scripts/build_v15_combined_synth.py)
merges the five templated-synth JSONL outputs into one corpus
file, tagged by intended training-time use.

### Run

```bash
PYTHONPATH=. python3 scripts/build_v15_combined_synth.py
```

### Result (2026-05-08)

```
=== per (source, seed_source) ===
  synth_format_aware / stix_indicator: 500
  synth_tool_use / search_cve_nvd: 195
  synth_tool_use_provenance / search_cve_nvd: 200
  synth_tool_use / lookup_cwe: 100
  synth_tool_use_provenance / lookup_cwe: 100
  synth_tool_use / rag_retrieve: 99
  synth_tool_use_provenance / rag_retrieve: 99
  synth_format_aware / sigma_rule: 30
  synth_format_aware / yara_rule: 30
  synth_format_aware / misp_event: 30
  synth_tool_use / lookup_mitre_technique: 30
  synth_tool_use_provenance / lookup_mitre_technique: 30
  synth_binary_literacy / pretrain_prose: 15
  synth_binary_literacy / identify_hex: 15
  synth_binary_literacy / show_magic: 14
  synth_code_security / pretrain_prose: 12
  synth_code_security / identify_and_fix: 12
  synth_code_security / explain_the_diff: 12
  synth_code_security / cwe_mapping: 12

=== by format_type ===
  sft: 918
  pretrain: 587

Wrote 1505 records to data/processed/synth_v15_combined.jsonl
(0 dropped to unknown source pairs)
```

### Schema

Each output record adds one new field, `format_type`, on top of the
existing synth-record schema:

```json
{
  "id": "synth_<bet>#<seed_id>#<hash>",
  "source": "synth_<bet>",
  "teacher": "templated",
  "seed_source": "<variant_or_tool_name>",
  "seed_id": "<original id>",
  "text": "<full record text>",
  "format_type": "pretrain | sft"   <- NEW
}
```

The trainer paths consume the file by filtering on `format_type`:

- **Pretrain mix:** records where `format_type == "pretrain"` get
  appended to the pretrain corpus alongside `data/processed/train.jsonl`.
  Their `text` is a flat blob (markdown article shape).
- **SFT mix:** records where `format_type == "sft"` get parsed into
  USER / ASSISTANT / TOOL turns and appended to the chat-tune SFT
  corpus alongside `data/processed/chat_train.jsonl`.

### Categorisation rules

Pretrain shape (one big text blob, ~587 records):
- `synth_format_aware:*` (all four format families)
- `synth_code_security:pretrain_prose`
- `synth_binary_literacy:pretrain_prose`

SFT shape (USER/ASSISTANT/TOOL traces, ~918 records):
- `synth_tool_use:*` (all four tool types)
- `synth_tool_use_provenance:*` (all four tool types)
- `synth_code_security:identify_and_fix / explain_the_diff / cwe_mapping`
- `synth_binary_literacy:identify_hex / show_magic`

The mapping lives in `CATEGORY_RULES` inside the script. Adding a
new variant to any synth pipeline is a one-line change.

## 2. One-command baseline reproducer

[`scripts/run_all_baselines.py`](../scripts/run_all_baselines.py)
walks the four held-out eval sets against any GhostLM checkpoint
and produces a combined summary table.

### Run

```bash
PYTHONPATH=. python3 scripts/run_all_baselines.py \
    --checkpoint checkpoints/phase19_chat_v09/best_model.pt \
    --run-name v09_chat
```

### Verified result (2026-05-08, against v0.9 chat)

```
| Bet | n | parse % (95% CI) | fields % (95% CI) |
|---|---:|---|---|
| bet6_format_aware    | 32 |   0.0% [0.0-10.7] |   0.0% [0.0-10.7] |
| bet7_code_security   | 20 | 100.0% [83.9-100.0] |   0.0% [0.0-16.1] |
| bet8_binary_literacy | 20 | 100.0% [83.9-100.0] |   0.0% [0.0-16.1] |
| bet9_provenance      | 15 |   0.0% [0.0-20.4] |   0.0% [0.0-20.4] |
```

Reproduces the per-bet numbers documented in
[`docs/format_baseline_v09.md`](format_baseline_v09.md) and
[`docs/baselines_v09_bets789.md`](baselines_v09_bets789.md).

### Output structure

```
logs/baselines_<run_name>/
  bet6_format_aware.jsonl       raw predictions per bet
  bet6_format_aware_score.md    scoring report per bet
  bet7_code_security.jsonl
  bet7_code_security_score.md
  bet8_binary_literacy.jsonl
  bet8_binary_literacy_score.md
  bet9_provenance.jsonl
  bet9_provenance_score.md
  summary.md                    combined headline table
  summary.json                  machine-readable summary
```

The `summary.md` is what feeds the comparison-rows tables in the
per-bet docs. Run after every checkpoint that should be measured;
diff `summary.json` between runs to see lift on each bet.

## Why these two pieces matter

Before this commit, the v0.9.5 work was credible but **not yet
research-reproducible**:

- Five synth files lived separately in `data/processed/` with no
  unified consumer.
- Four eval baselines existed but each one needed its own
  invocation of `run_format_baseline.py` plus
  `eval_format_compliance.py`.

After this commit:

- One file (`synth_v15_combined.jsonl`) is the canonical training
  corpus the GPU run will read; trainer code paths can branch on
  `format_type`.
- One command (`run_all_baselines.py --checkpoint <ckpt> --run-name
  <name>`) reproduces every locked baseline measurement plus the
  comparison-table summary. Anyone who clones the repo can
  re-derive the v0.9 numbers.

That is the academic-bar work that converts the v0.9.5 release
from a code dump into a research artifact.

## Lift expectations after ghost-base trains

Once ghost-base trains on `synth_v15_combined.jsonl` (combined
pretrain + SFT corpus), re-run:

```bash
PYTHONPATH=. python3 scripts/run_all_baselines.py \
    --checkpoint checkpoints/<ghost_base_run>/best_model.pt \
    --run-name ghost_base_v1
```

Diff against the v09_chat summary to read the lift. The Wilson
95% CIs in each report tell you which deltas are statistically
significant at small n. Targets per bet:

| Bet | v0.9 floor | Statistical-significance threshold |
|---|---|---|
| 6 | 0/32 [0.0-10.7] | any score >11% |
| 7 | 0/20 [0.0-16.1] | any score >16% |
| 8 | 0/20 [0.0-16.1] | any score >16% |
| 9 | 0/15 [0.0-20.4] | any score >20% |

The ghost-base run is now one CLI command away from a measurable,
publishable result on every differentiation bet.
