"""GhostBench: a statistically-rigorous evaluation suite for small
cybersecurity language models.

GhostBench is the evaluation half of the GhostLM project, designed to
be model-agnostic and extractable as a standalone library. The same
suite that measures GhostLM checkpoints can be pointed at any open
small LM (SmolLM2, Qwen2.5-0.5B, Llama-3.2-1B, etc.) for a head-to-
head comparison on the nine differentiation bets formalised in
``docs/differentiation.md``.

Public API surface:

    from ghostbench import (
        Bench,           # one bet's eval set + scoring config
        Suite,           # collection of Benches that share a checkpoint
        EvalRecord,      # one (prompt, expected) pair from an eval JSONL
        Prediction,      # one (prompt, predicted) pair from a model run
        Score,           # outcome of scoring one prediction against
                         # its eval record
        RunReport,       # aggregated scores across all predictions in
                         # a Bench, with Wilson CIs + paired-comparison
                         # statistics
        wilson_ci,       # 95% binomial proportion interval
        mcnemar_test,    # paired-comparison test for two checkpoints
        cohen_h,         # effect-size estimator for proportion delta
    )

The library is designed around the principle that scoring a small LM
is fundamentally a binomial-proportion problem: each prediction either
passes the bet's structural / content / behavioural checks or it does
not. Sample sizes are small (n=15-32 per bet), so confidence intervals
matter. The Wilson interval is right at small n; the normal-approximation
interval breaks at p near 0 or 1 (which is exactly the regime small LMs
occupy on novel capabilities); Clopper-Pearson is too conservative.
McNemar's test handles paired comparisons (same eval prompts, two
different checkpoints) correctly.

Multi-tier scoring is supported via ``Score.tiers``: a single
prediction can have lexical (substring), structural (parser),
semantic (LLM-judge), and behavioural (downstream task) tiers, each
with its own pass/fail. ``Score.tier_pass(name)`` returns the per-tier
verdict; ``Score.passed`` returns the strict-AND across all tiers
that the eval record actually requested.
"""

from __future__ import annotations

from .bench import Bench, Suite, EvalRecord, Prediction
from .scoring import Score, RunReport, score_record
from .stats import wilson_ci, mcnemar_test, cohen_h, paired_diff_ci

__all__ = [
    "Bench",
    "Suite",
    "EvalRecord",
    "Prediction",
    "Score",
    "RunReport",
    "score_record",
    "wilson_ci",
    "mcnemar_test",
    "cohen_h",
    "paired_diff_ci",
]

__version__ = "0.1.0"
