"""Tests for ghostbench.reports."""

import pytest

from ghostbench.reports import (
    render_paired_comparison, render_run_report,
    render_suite_paired_comparison, render_suite_summary,
)
from ghostbench.scoring import RunReport, Score


def _make_score(seed_id: str, passed: bool, fmt: str = "test_fmt") -> Score:
    """Helper: build a Score that requests a single 'substrings' tier."""
    return Score(
        seed_id=seed_id, fmt=fmt,
        requested_tiers=("substrings",),
        tier_results={"substrings": passed},
    )


def _make_report(name: str, run: str, scores) -> RunReport:
    return RunReport(bench_name=name, run_name=run, n=len(scores),
                     scores=list(scores))


def test_render_run_report_smoke():
    scores = [_make_score(f"s{i}", i % 2 == 0) for i in range(8)]
    report = _make_report("bench_x", "run_y", scores)
    md = render_run_report(report)
    assert "bench_x" in md
    assert "run_y" in md
    assert "n = **8**" in md
    # 4 of 8 passed.
    assert "4 / 8" in md
    assert "substrings" in md


def test_render_paired_comparison_significant():
    """A clear win for B should show ✓ significance and a positive
    Cohen's h."""
    a_scores = [_make_score(f"s{i}", False) for i in range(10)]
    b_scores = [_make_score(f"s{i}", True) for i in range(10)]
    a_report = _make_report("b", "v09_chat", a_scores)
    b_report = _make_report("b", "ghost_base", b_scores)
    md = render_paired_comparison(a_report, b_report)
    # All 10 prompts switched from fail to pass; clear significance.
    assert "significantly outperforms" in md
    assert "ghost_base" in md


def test_render_paired_comparison_no_difference():
    """When both runs match perfectly, no significance is detected."""
    scores_a = [_make_score(f"s{i}", i < 3) for i in range(8)]
    scores_b = [_make_score(f"s{i}", i < 3) for i in range(8)]
    md = render_paired_comparison(
        _make_report("b", "a", scores_a),
        _make_report("b", "b", scores_b),
    )
    assert "no significant difference" in md.lower() or \
        "comparable" in md.lower()


def test_render_paired_comparison_no_overlap_raises():
    """Disjoint seed_ids should raise ValueError."""
    a_scores = [_make_score(f"a{i}", True) for i in range(3)]
    b_scores = [_make_score(f"b{i}", True) for i in range(3)]
    with pytest.raises(ValueError):
        render_paired_comparison(
            _make_report("b", "a", a_scores),
            _make_report("b", "b", b_scores),
        )


def test_render_suite_summary():
    r1 = _make_report("bet1", "v09",
                      [_make_score("a", True), _make_score("b", False)])
    r2 = _make_report("bet2", "v09",
                      [_make_score("c", True), _make_score("d", True)])
    md = render_suite_summary([r1, r2], "v09")
    assert "bet1" in md and "bet2" in md
    assert "OVERALL" in md
    # Combined: 3 / 4 passed.
    assert "3" in md and "4" in md


def test_render_suite_paired_comparison_marks_wins():
    """Suite-level table flags significant bets with ✓."""
    a1 = _make_report("bet1", "v09",
                      [_make_score(f"s{i}", False) for i in range(10)])
    b1 = _make_report("bet1", "ghost_base",
                      [_make_score(f"s{i}", True) for i in range(10)])
    a2 = _make_report("bet2", "v09",
                      [_make_score(f"t{i}", True) for i in range(8)])
    b2 = _make_report("bet2", "ghost_base",
                      [_make_score(f"t{i}", True) for i in range(8)])
    md = render_suite_paired_comparison([a1, a2], [b1, b2])
    # bet1 should be significant; bet2 shouldn't (both at 100%).
    assert "✓" in md
