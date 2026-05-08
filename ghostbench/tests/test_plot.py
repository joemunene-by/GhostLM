"""Tests for ghostbench.plot.

Plotting tests don't render actual PNGs (pytest CI environments
often lack a display backend). They verify the API is callable,
returns Figure objects, and gracefully errors on bad input.
"""

import pytest

# Skip ALL plotting tests if matplotlib isn't installed; the package
# is import-safe without it.
plt = pytest.importorskip("matplotlib.pyplot")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")   # non-interactive backend for CI

from ghostbench.plot import (
    plot_paired_comparison, plot_run_report,
    plot_suite_paired_comparison, plot_suite_summary,
)
from ghostbench.scoring import RunReport, Score


def _make_score(seed_id: str, passed: bool, fmt: str = "f",
                tiers=("substrings",)) -> Score:
    return Score(
        seed_id=seed_id, fmt=fmt, requested_tiers=tiers,
        tier_results={t: passed for t in tiers},
    )


def _make_report(name, run, scores) -> RunReport:
    return RunReport(bench_name=name, run_name=run, n=len(scores),
                     scores=list(scores))


def test_plot_run_report_returns_figure():
    """plot_run_report returns a matplotlib Figure."""
    scores = [_make_score(f"s{i}", i % 3 == 0) for i in range(15)]
    report = _make_report("bench_x", "v09", scores)
    fig = plot_run_report(report)
    assert fig is not None
    assert hasattr(fig, "savefig")
    plt.close(fig)


def test_plot_run_report_empty_raises():
    """Reports with no per-tier data must raise."""
    report = RunReport(bench_name="b", run_name="r", n=0, scores=[])
    with pytest.raises(ValueError):
        plot_run_report(report)


def test_plot_suite_summary_returns_figure():
    r1 = _make_report("bench_a", "v09",
                      [_make_score(f"a{i}", i % 2 == 0) for i in range(10)])
    r2 = _make_report("bench_b", "v09",
                      [_make_score(f"b{i}", True) for i in range(8)])
    fig = plot_suite_summary([r1, r2])
    assert fig is not None
    plt.close(fig)


def test_plot_suite_summary_empty_raises():
    with pytest.raises(ValueError):
        plot_suite_summary([])


def test_plot_paired_comparison_returns_figure():
    a_scores = [_make_score(f"s{i}", False) for i in range(10)]
    b_scores = [_make_score(f"s{i}", True) for i in range(10)]
    a_report = _make_report("b", "v09", a_scores)
    b_report = _make_report("b", "ghost", b_scores)
    fig = plot_paired_comparison(a_report, b_report)
    assert fig is not None
    plt.close(fig)


def test_plot_paired_comparison_no_overlap_raises():
    a_scores = [_make_score(f"a{i}", True) for i in range(3)]
    b_scores = [_make_score(f"b{i}", True) for i in range(3)]
    a_report = _make_report("b", "v09", a_scores)
    b_report = _make_report("b", "ghost", b_scores)
    with pytest.raises(ValueError):
        plot_paired_comparison(a_report, b_report)


def test_plot_suite_paired_comparison_returns_figure():
    a1 = _make_report("bet1", "v09",
                      [_make_score(f"s{i}", False) for i in range(10)])
    b1 = _make_report("bet1", "ghost",
                      [_make_score(f"s{i}", True) for i in range(10)])
    a2 = _make_report("bet2", "v09",
                      [_make_score(f"t{i}", True) for i in range(8)])
    b2 = _make_report("bet2", "ghost",
                      [_make_score(f"t{i}", True) for i in range(8)])
    fig = plot_suite_paired_comparison([a1, a2], [b1, b2])
    assert fig is not None
    plt.close(fig)


def test_plot_suite_paired_comparison_length_mismatch_raises():
    r = _make_report("b", "x", [_make_score("s1", True)])
    with pytest.raises(ValueError):
        plot_suite_paired_comparison([r, r], [r])
