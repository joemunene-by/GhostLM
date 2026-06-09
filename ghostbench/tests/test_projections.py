"""Tests for ghostbench.projections."""

import pytest

from ghostbench.projections import (
    DEFAULT_ASYMPTOTES, DEFAULT_SATURATION_N, project_bet, project_suite, render_projection_table,
)


# ---------------------------------------------------------------------------
# project_bet
# ---------------------------------------------------------------------------


def test_project_bet_zero_records_zero_lift():
    """records_seen = 0 should project zero pass rate (model has
    seen nothing)."""
    p = project_bet("bet7_code_security", eval_n=20, records_seen=0)
    assert p.point_estimate == pytest.approx(0.0, abs=0.01)


def test_project_bet_high_records_approaches_asymptote():
    """Many records should put the projection near the asymptote."""
    asymptote = 0.5
    p = project_bet("bet_x", eval_n=20, records_seen=10_000,
                     asymptote=asymptote, saturation_n=200)
    # 1 - exp(-50) ≈ 1, so we should land within 1% of asymptote.
    assert abs(p.point_estimate - 50.0) < 1.0


def test_project_bet_methodological_band_brackets_point():
    """The methodological band should bracket the point estimate."""
    p = project_bet("bet7_code_security", eval_n=20, records_seen=300)
    assert p.methodological_lo < p.point_estimate < p.methodological_hi


def test_project_bet_wilson_ci_brackets_point_estimate():
    """The Wilson CI should bracket the point estimate (or land at
    one edge if k is at the boundary)."""
    p = project_bet("bet9_provenance", eval_n=15, records_seen=429)
    assert p.wilson_lo <= p.point_estimate <= p.wilson_hi + 0.5
    # Wilson CI width should be reasonable for n=15.
    assert (p.wilson_hi - p.wilson_lo) < 60


def test_project_bet_records_more_helps_more():
    """More training records → higher projected score (monotone)."""
    a = project_bet("bet6_format_aware", eval_n=32, records_seen=100)
    b = project_bet("bet6_format_aware", eval_n=32, records_seen=560)
    assert b.point_estimate > a.point_estimate


def test_project_bet_higher_asymptote_higher_projection():
    """A more ambitious asymptote → higher projection at fixed n."""
    a = project_bet("bet_x", eval_n=20, records_seen=200,
                     asymptote=0.4, saturation_n=200)
    b = project_bet("bet_x", eval_n=20, records_seen=200,
                     asymptote=0.7, saturation_n=200)
    assert b.point_estimate > a.point_estimate


def test_default_asymptotes_cover_all_benches():
    """DEFAULT_ASYMPTOTES has an entry for each of the four benches."""
    expected = {
        "bet6_format_aware", "bet7_code_security",
        "bet8_binary_literacy", "bet9_provenance",
    }
    assert expected.issubset(DEFAULT_ASYMPTOTES.keys())
    assert expected.issubset(DEFAULT_SATURATION_N.keys())


# ---------------------------------------------------------------------------
# project_suite
# ---------------------------------------------------------------------------


def test_project_suite_returns_one_per_bench():
    """One Projection per bench in the input."""
    projs = project_suite(
        records_per_bet={
            "bet6_format_aware": 560,
            "bet7_code_security": 36,
            "bet8_binary_literacy": 29,
            "bet9_provenance": 429,
        },
        eval_n_per_bet={
            "bet6_format_aware": 32,
            "bet7_code_security": 20,
            "bet8_binary_literacy": 20,
            "bet9_provenance": 15,
        },
    )
    names = sorted(p.bench_name for p in projs)
    assert names == [
        "bet6_format_aware", "bet7_code_security",
        "bet8_binary_literacy", "bet9_provenance",
    ]


def test_project_suite_at_v15_record_counts():
    """End-to-end smoke against the v0.9.5 record counts. No
    assertions on the exact numbers (those are projections, not
    predictions); just that the projections are non-degenerate."""
    projs = project_suite(
        records_per_bet={
            "bet6_format_aware": 560,
            "bet7_code_security": 36,
            "bet8_binary_literacy": 29,
            "bet9_provenance": 429,
        },
        eval_n_per_bet={
            "bet6_format_aware": 32,
            "bet7_code_security": 20,
            "bet8_binary_literacy": 20,
            "bet9_provenance": 15,
        },
    )
    for p in projs:
        # Point estimate should be in (0, 100) range and below the
        # asymptote (per the exposure-curve formula).
        assert 0 < p.point_estimate < 100 * p.asymptote + 0.1


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_projection_table_includes_all_benches():
    projs = project_suite(
        records_per_bet={"bet6_format_aware": 560, "bet9_provenance": 429},
        eval_n_per_bet={"bet6_format_aware": 32, "bet9_provenance": 15},
    )
    md = render_projection_table(projs, run_name="ghost_base_v1_projected")
    assert "bet6_format_aware" in md
    assert "bet9_provenance" in md
    assert "ghost_base_v1_projected" in md
    # Methodology paragraph is included.
    assert "exposure curve" in md.lower()


# ---------------------------------------------------------------------------
# Override semantics
# ---------------------------------------------------------------------------


def test_project_suite_respects_overrides():
    """Asymptote / saturation overrides shift the projection."""
    base = project_suite(
        records_per_bet={"bet7_code_security": 36},
        eval_n_per_bet={"bet7_code_security": 20},
    )[0]
    optimistic = project_suite(
        records_per_bet={"bet7_code_security": 36},
        eval_n_per_bet={"bet7_code_security": 20},
        asymptote_overrides={"bet7_code_security": 0.85},
    )[0]
    assert optimistic.point_estimate > base.point_estimate
    assert optimistic.asymptote == 0.85
