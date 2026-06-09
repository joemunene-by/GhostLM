"""Performance projection module for ghost-base.

Given a v0.9 baseline measurement (typically 0/n on every bet, since
v0.9 has zero training signal for the new capabilities), this module
projects what ghost-base could realistically score after training on
the templated-synth corpus, with confidence bounds.

The projection model is intentionally simple: it uses a coverage-and-
exposure heuristic, not a learned regression. The explicit model is:

    expected_pass_rate = capability_baseline
                       * (1 - exp(-records_seen / saturation_n))

where:

  ``capability_baseline``   the asymptotic pass rate the bet's
                             capability could plausibly hit if a
                             3B-5B model were trained on the same
                             data. We set this per-bet from the
                             literature on small-LM capabilities.
                             For STIX / YARA / Sigma / MISP emission
                             at ~360M parameters, asymptote sits
                             around 60-75% based on prior work on
                             format-constrained generation. For
                             provenance / cite tags, asymptote is
                             higher (~80-90%) because the structural
                             pattern is highly cued by training data.
                             For binary literacy, asymptote is lower
                             (~35-50%) because the underlying
                             capability is harder.

  ``records_seen``          how many SFT records of this bet's
                             capability the model trains on.

  ``saturation_n``          the n at which 63% of the asymptote is
                             reached (the e-fold). Calibrated from
                             prior work on format-constrained SFT:
                             ~150 records for "easy" structural
                             tasks, ~400 for "harder" ones.

The projection comes with TWO confidence layers:

1. **Statistical**: Wilson 95% CI around the projected pass rate
   given the eval set's n.
2. **Methodological**: a +/- 30% credibility interval on the
   asymptote, reflecting that the asymptote is itself a guess.

The output of ``project()`` is therefore a dict with both layers
visible so consumers can communicate the uncertainty honestly.

This is NOT a learned regression. It's a sanity-check projection
that says: 'given a reasonable model of what training data does,
ghost-base should land roughly here.' The actual measurement comes
from running ghost-base through GhostBench. The projection's value
is setting expectations BEFORE the GPU run so a result that's
wildly off the projection (in either direction) is itself a
finding worth investigating.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

from .stats import wilson_ci


# Default per-bet asymptotes. These are the ``capability_baseline``
# values in the projection formula. They reflect literature priors
# on what a ~360M-param model trained on the right SFT data could
# reasonably achieve on each capability. Override at projection
# time via the ``asymptote_overrides`` arg.
DEFAULT_ASYMPTOTES: Dict[str, float] = {
    "bet6_format_aware":   0.65,   # STIX / YARA / Sigma / MISP emission
    "bet7_code_security":  0.55,   # CWE identification on novel code
    "bet8_binary_literacy": 0.40,  # hex / file-magic recognition
    "bet9_provenance":     0.80,   # cite-tag emission (highly cued)
}


# Default per-bet saturation_n values. Calibrated from prior work
# on format-constrained SFT: ~150 records for "easy" tasks where
# the pattern is highly structural, ~400 for harder tasks where
# generalisation is needed.
DEFAULT_SATURATION_N: Dict[str, int] = {
    "bet6_format_aware":   200,    # 4 format families, structural-but-rich
    "bet7_code_security":  300,    # CWE generalisation needed
    "bet8_binary_literacy": 400,   # byte-level recognition is hard
    "bet9_provenance":     150,    # cite-tag pattern is highly cued
}


# Asymptote credibility interval factor. We project a low-, mid-,
# and high- variant per bet to communicate that the asymptote
# itself is uncertain.
_ASYMPTOTE_BAND = 0.30


@dataclass
class Projection:
    """One bet's projected pass rate with statistical + methodological
    uncertainty layers."""
    bench_name: str
    eval_n: int
    records_seen: int
    asymptote: float
    saturation_n: int

    point_estimate: float       # expected pass rate as percentage
    methodological_lo: float    # asymptote band at 1-band
    methodological_hi: float    # asymptote band at 1+band
    wilson_lo: float             # 95% CI lower at point_estimate
    wilson_hi: float             # 95% CI upper at point_estimate

    def to_row(self) -> Dict:
        return {
            "bench": self.bench_name,
            "eval_n": self.eval_n,
            "records_seen": self.records_seen,
            "asymptote": self.asymptote,
            "saturation_n": self.saturation_n,
            "point_estimate_pct": round(self.point_estimate, 1),
            "methodological_band": [
                round(self.methodological_lo, 1),
                round(self.methodological_hi, 1),
            ],
            "wilson_95_ci": [
                round(self.wilson_lo, 1), round(self.wilson_hi, 1),
            ],
        }


def _exposure_curve(records_seen: int, asymptote: float,
                     saturation_n: int) -> float:
    """Exponential approach to the asymptote. Returns a fraction in
    [0, asymptote]."""
    if saturation_n <= 0:
        return asymptote
    return asymptote * (1.0 - math.exp(-records_seen / saturation_n))


def project_bet(bench_name: str, eval_n: int, records_seen: int,
                 asymptote: float = None,
                 saturation_n: int = None) -> Projection:
    """Project one bet's expected pass rate.

    Args:
        bench_name: Bench identifier (e.g. "bet7_code_security").
        eval_n: Number of held-out eval records the projected score
                will be measured on. Used for the Wilson CI width.
        records_seen: Number of training records for this bet's
                      capability. From the templated-synth corpus
                      categorisation.
        asymptote: Override the per-bet default asymptote (see
                   DEFAULT_ASYMPTOTES). Useful for "what if I assume
                   a more / less ambitious ceiling."
        saturation_n: Override the per-bet default saturation_n.

    Returns:
        A Projection with point estimate + statistical CI +
        methodological credibility band.
    """
    a = asymptote if asymptote is not None else \
        DEFAULT_ASYMPTOTES.get(bench_name, 0.5)
    s = saturation_n if saturation_n is not None else \
        DEFAULT_SATURATION_N.get(bench_name, 250)

    point = 100.0 * _exposure_curve(records_seen, a, s)
    band_lo = 100.0 * _exposure_curve(records_seen, a * (1 - _ASYMPTOTE_BAND), s)
    band_hi = 100.0 * _exposure_curve(records_seen, a * (1 + _ASYMPTOTE_BAND), s)

    # Wilson 95% CI at the point estimate, given the eval set's n.
    expected_passes = round(point / 100.0 * eval_n)
    wlo, whi = wilson_ci(expected_passes, eval_n)

    return Projection(
        bench_name=bench_name, eval_n=eval_n,
        records_seen=records_seen, asymptote=a,
        saturation_n=s,
        point_estimate=point,
        methodological_lo=band_lo, methodological_hi=band_hi,
        wilson_lo=wlo, wilson_hi=whi,
    )


def project_suite(records_per_bet: Dict[str, int],
                   eval_n_per_bet: Dict[str, int],
                   asymptote_overrides: Dict[str, float] = None,
                   saturation_overrides: Dict[str, int] = None
                   ) -> List[Projection]:
    """Project all four GhostBench benches at once.

    Args:
        records_per_bet: {bench_name: records_seen} from the
                         templated-synth corpus split.
        eval_n_per_bet: {bench_name: eval_set_n} from the held-out
                        eval JSONL files.
        asymptote_overrides: Optional per-bet asymptote overrides.
        saturation_overrides: Optional per-bet saturation_n overrides.

    Returns:
        List of Projection, one per bench in the input.
    """
    asymptote_overrides = asymptote_overrides or {}
    saturation_overrides = saturation_overrides or {}
    out: List[Projection] = []
    for bench_name in sorted(records_per_bet):
        records = records_per_bet[bench_name]
        eval_n = eval_n_per_bet.get(bench_name, 20)
        out.append(project_bet(
            bench_name=bench_name,
            eval_n=eval_n,
            records_seen=records,
            asymptote=asymptote_overrides.get(bench_name),
            saturation_n=saturation_overrides.get(bench_name),
        ))
    return out


# ---------------------------------------------------------------------------
# Markdown rendering of the projection
# ---------------------------------------------------------------------------


def render_projection_table(projections: List[Projection],
                              run_name: str = "ghost_base_v1_projected") -> str:
    """Markdown table of projected pass rates for each bench.

    Suitable for "Expected Results" section of the v0.9.5 release
    notes / ghost-base spec doc."""
    lines = [f"# Ghost-base projections: {run_name}", ""]
    lines.append(
        "| Bench | eval_n | records_seen | point | "
        "methodological band | Wilson 95% CI |"
    )
    lines.append("|---|---:|---:|---:|---|---|")
    for p in projections:
        lines.append(
            f"| {p.bench_name} | {p.eval_n} | {p.records_seen} | "
            f"**{p.point_estimate:.1f}%** | "
            f"[{p.methodological_lo:.1f}%-{p.methodological_hi:.1f}%] | "
            f"[{p.wilson_lo:.1f}%-{p.wilson_hi:.1f}%] |"
        )
    lines.append("")
    lines.append(
        "**Methodology.** Point estimate uses an exposure curve of "
        "the form `asymptote * (1 - exp(-records_seen / saturation_n))`. "
        "Asymptote is a literature-prior guess (60-75% for STIX/YARA/Sigma/MISP, "
        "55% for CWE-on-novel-code, 40% for binary-literacy, 80% for "
        "highly-cued cite-tag emission). Methodological band is "
        "+/- 30% on the asymptote. Wilson 95% CI is the statistical "
        "uncertainty around the point estimate at the eval's n. "
        "These are projections, not predictions. Real measurement "
        "comes from running ghost-base through GhostBench."
    )
    return "\n".join(lines) + "\n"
