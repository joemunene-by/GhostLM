# GhostBench

A statistically-rigorous evaluation suite for small cybersecurity
language models.

GhostBench is the evaluation half of the GhostLM project. It is
designed to be model-agnostic and extractable as a standalone
library: the same suite that measures GhostLM checkpoints can be
pointed at any open small LM (SmolLM2, Qwen2.5-0.5B, Llama-3.2-1B,
gpt-2-medium) for a head-to-head comparison.

## Why GhostBench exists

Existing small-LM benchmarks measure general capability: HumanEval
for code, MMLU for knowledge, GSM8K for math. None of them target
the *security analyst* workflow: read vulnerable code and identify
the CWE; recognise a hex byte sequence as a PE / ELF / packer
signature; emit STIX 2.1 indicators or YARA rules with valid
structure; cite the tool response that justifies each factual claim.

A small LM that excels at those tasks is materially useful inside
a SOC and is also a research artifact in its own right (no one has
published competitive small-model numbers on these capabilities).
GhostBench is the measurement layer that makes the comparison
possible.

## What it measures

Nine bets, each backed by a held-out eval set:

| Bench | Capability | Held-out n |
|---|---|---:|
| bet6_format_aware    | STIX 2.1 / YARA / Sigma / MISP emission | 32 |
| bet7_code_security   | CWE-class identification + fix on held-out vulns | 20 |
| bet8_binary_literacy | Hex / file-magic / disassembly recognition | 20 |
| bet9_provenance      | Inline `<\|cite\|>` tag emission with valid format | 15 |

Bets 1, 2, 4, 5 from the
[GhostLM differentiation strategy](../docs/differentiation.md) do
not yet have held-out evals because they target training-recipe
or architecture properties rather than per-prediction outcomes.
Their evaluation will be added in a future GhostBench release.

## What's distinctive about the scoring

Multi-tier scoring per prediction. Each bet's eval records can
request any subset of:

- **`parse`** — structural validity. STIX 2.1 SDO has the right
  shape and `spec_version: 2.1`. YARA has rule + strings + condition.
  Sigma is loadable YAML with the required top-level keys. MISP has
  `Event.Attribute` array of typed IOCs. Provenance has at least
  one well-formed `<\|cite\|>{type}:{id}<\|/cite\|>` tag. (Bets 7
  and 8 have no parser; this tier is vacuously True.)

- **`fields`** — dotted-path field checks against the parsed object.
  Right for STIX / Sigma / MISP / provenance. E.g. for a STIX
  indicator, `required_fields = [{"path": "type", "value":
  "indicator"}]`.

- **`substrings`** — required substrings in the raw prediction text.
  Right for YARA, code-security, binary-literacy, plus a useful
  complement for the others. E.g. for a code-security record,
  `required_substrings = ["CWE-89", "parameterized"]`.

- **`semantic`** (reserved) — LLM-as-judge tier. Not implemented in
  v0.1; the slot is reserved so future versions can add it without
  breaking the API.

- **`behavioral`** (reserved) — end-to-end task-completion tier.
  E.g. does `yara -p file rule.yar` actually compile the predicted
  rule? Not implemented in v0.1.

`Score.passed` is the strict-AND across the *requested* tiers, not
all possible tiers, so a record with only `required_substrings` is
not penalised for not having a parser.

## Statistical rigour

Small-n binomial-proportion analysis is the central problem (15-32
records per bench, pass-rates near 0 or 1 are common when measuring
novel capabilities on small LMs):

- **`wilson_ci(k, n)`**: Wilson 95% binomial proportion interval.
  Right at small n, doesn't blow up at p near 0 or 1, less
  conservative than Clopper-Pearson.

- **`mcnemar_test(b, c)`**: exact two-sided binomial McNemar's test
  for paired binary outcomes. The right tool when comparing two
  checkpoints on the SAME eval prompts.

- **`cohen_h(p1, p2)`**: arcsine-transform effect size with the
  standard small/medium/large cuts. Keeps interpretation honest
  when a 6x relative lift at p=0.01 is actually a "small" effect.

- **`paired_diff_ci(b, c, n)`**: Newcombe-style Wilson-shifted CI on
  the proportion difference under paired sampling. Tighter than two
  independent Wilson CIs.

The CLI's `compare` and `suite-compare` commands wrap all four into
a single paired-comparison report that is the publication-grade
artifact for "did the new checkpoint actually beat the baseline."

## Quick start

```python
from ghostbench import Bench, Suite, Prediction
from ghostbench.parsers import DEFAULT_PARSERS

# Build a Suite from a directory of eval JSONLs.
suite = Suite.from_dir("data/raw", parsers=DEFAULT_PARSERS)

# For each Bench, score the predictions you generated.
for bench in suite:
    preds = [Prediction.from_dict(rec) for rec in
             load_jsonl(f"logs/{bench.name}.jsonl")]
    report = bench.score(preds, run_name="ghost_base_v1")
    print(report.summary())
```

Or via CLI:

```bash
# Score a single bench
python -m ghostbench score \
    --eval data/raw/format_aware_eval.jsonl \
    --predictions logs/baselines_v09_chat/bet6_format_aware.jsonl \
    --bench-name bet6_format_aware --run-name v09_chat

# Whole-suite summary
python -m ghostbench summary \
    --eval-dir data/raw \
    --predictions-dir logs/baselines_v09_chat \
    --run-name v09_chat \
    --out logs/baselines_v09_chat/ghostbench_summary.md

# Paired comparison (the bet that matters once ghost-base lands)
python -m ghostbench compare \
    --eval data/raw/format_aware_eval.jsonl \
    --a-predictions logs/baselines_v09_chat/bet6_format_aware.jsonl \
    --a-name v09_chat \
    --b-predictions logs/baselines_ghost_base_v1/bet6_format_aware.jsonl \
    --b-name ghost_base_v1 \
    --bench-name bet6_format_aware

# Suite-level paired comparison: which bets does ghost_base_v1 win on?
python -m ghostbench suite-compare \
    --eval-dir data/raw \
    --a-predictions-dir logs/baselines_v09_chat --a-name v09_chat \
    --b-predictions-dir logs/baselines_ghost_base_v1 \
    --b-name ghost_base_v1
```

## v0.9 baseline (verified 2026-05-08)

Whole-suite numbers from the v0.9 chat checkpoint, scored via
GhostBench:

| Bench | n | passed | rate | 95% CI |
|---|---:|---:|---:|---|
| bet6_format_aware | 32 | 0 | 0.0% | [0.0-10.7] |
| bet7_code_security | 20 | 0 | 0.0% | [0.0-16.1] |
| bet8_binary_literacy | 20 | 0 | 0.0% | [0.0-16.1] |
| bet9_provenance | 15 | 0 | 0.0% | [0.0-20.4] |
| **OVERALL** | **87** | **0** | **0.0%** | **[0.0-4.2]** |

Per-tier breakdown surfaces detail the legacy eval didn't: bet 6
substrings tier passes 1/32 (3.1%) even though `passed` is 0/32,
because the structural parse and field checks fail on that
prediction. This per-tier diagnostic is what makes GhostBench
useful for *targeted* training-recipe iteration, not just
overall-pass-rate tracking.

## Design constraints

- **Stdlib only.** No numpy, no scipy, no pandas. Pickle-safe,
  json-portable. The whole package imports in ~50 ms.
- **Test-first.** 61 unit tests covering Wilson CI / Cohen's h /
  McNemar / paired-diff CI / multi-tier scoring / suite discovery /
  paired comparison / suite paired comparison. Edge cases (n=0,
  perfect pass, perfect fail, balanced discordant) are explicitly
  tested.
- **No model-specific code.** GhostBench knows nothing about
  GhostLM's architecture, tokenizer, or chat format. It scores
  prediction strings against eval records. Inference is the
  caller's responsibility (see `scripts/run_format_baseline.py`
  and `scripts/run_all_baselines.py` for the GhostLM-specific
  inference layer).

## Roadmap

The slot for `semantic` and `behavioral` tiers is reserved in
`Score.tier_results` but not implemented in v0.1. v0.2 candidates:

- **Semantic tier** via LLM-as-judge with cheap models (Claude
  Haiku, gpt-4o-mini). Score a prediction's *meaning* against an
  expected rationale, not just substring presence.
- **Behavioral tier** for executable artifacts: actually run YARA
  / Sigma / STIX through their reference parsers and score the
  binary outcome.
- **Coverage / accuracy / specificity** sub-metrics for bet 9's
  cite tags: don't just count cites, verify each cite resolves
  to a field that actually appears in the tool response.
- **Plotting** module: matplotlib helpers for lift charts, paired-
  comparison forest plots, per-bet bar charts with CIs.

Contributions welcome via the
[GhostLM repo](https://github.com/joemunene-by/GhostLM).

## License

Apache 2.0. Same license as the rest of GhostLM.
