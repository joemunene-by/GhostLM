"""Tests for ghostbench.stats."""

import pytest

from ghostbench.stats import (
    cohen_h, mcnemar_test, paired_diff_ci, wilson_ci,
)


# ---------------------------------------------------------------------------
# Wilson CI
# ---------------------------------------------------------------------------


def test_wilson_ci_zero_hits_zero_n():
    """n=0 returns (0, 0) without dividing by zero."""
    lo, hi = wilson_ci(0, 0)
    assert lo == 0.0 and hi == 0.0


def test_wilson_ci_zero_hits_small_n():
    """At n=8 with 0 hits, upper bound is ~32.4% (the well-known
    small-n value)."""
    lo, hi = wilson_ci(0, 8)
    assert lo == 0.0
    assert 32.0 < hi < 33.0


def test_wilson_ci_zero_hits_n_32():
    """At n=32 with 0 hits, upper bound is ~10.7% (matches what
    bet 6's expanded eval set shipped with)."""
    lo, hi = wilson_ci(0, 32)
    assert lo == 0.0
    assert 10.0 < hi < 11.5


def test_wilson_ci_full_pass():
    """At n=20 with 20 hits, lower bound ~83.9%, upper 100%."""
    lo, hi = wilson_ci(20, 20)
    assert hi == pytest.approx(100.0, abs=1e-9)
    assert 83.0 < lo < 85.0


def test_wilson_ci_midrange():
    """At n=32 with 16 hits (50%), CI brackets the point estimate."""
    lo, hi = wilson_ci(16, 32)
    assert lo < 50.0 < hi
    # Width should be reasonable: ~ ±17 percentage points at this n.
    assert 30 < hi - lo < 40


def test_wilson_ci_rejects_invalid():
    """k > n or k < 0 must raise."""
    with pytest.raises(ValueError):
        wilson_ci(-1, 10)
    with pytest.raises(ValueError):
        wilson_ci(11, 10)


# ---------------------------------------------------------------------------
# Cohen's h
# ---------------------------------------------------------------------------


def test_cohen_h_zero_diff():
    """Equal proportions give h = 0."""
    assert cohen_h(0.5, 0.5) == pytest.approx(0.0, abs=1e-10)
    assert cohen_h(0.1, 0.1) == pytest.approx(0.0, abs=1e-10)


def test_cohen_h_sign():
    """h is positive when p1 > p2."""
    assert cohen_h(0.6, 0.4) > 0
    assert cohen_h(0.4, 0.6) < 0


def test_cohen_h_lift_from_low_p():
    """A lift from 1% to 6% is small (~0.27) under Cohen's h, even
    though it's a 6x relative increase. This is the property that
    keeps the metric honest at small p."""
    h = cohen_h(0.06, 0.01)
    assert 0.20 < h < 0.35


def test_cohen_h_rejects_out_of_range():
    with pytest.raises(ValueError):
        cohen_h(1.5, 0.5)
    with pytest.raises(ValueError):
        cohen_h(0.5, -0.1)


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------


def test_mcnemar_no_discordant_pairs():
    """When both checkpoints agree on every prompt, p = 1.0 (no
    evidence of difference) and n_discordant = 0."""
    p, n = mcnemar_test(0, 0)
    assert p == 1.0 and n == 0


def test_mcnemar_perfect_separation_small():
    """5 discordant pairs all in the same direction. p should be
    small (well under 0.05 even at n_discordant = 5)."""
    p, n = mcnemar_test(5, 0)
    assert n == 5
    # Two-sided exact p = 2 * (0.5)^5 = 0.0625
    assert 0.05 < p < 0.07


def test_mcnemar_balanced_split():
    """Balanced discordant counts give p = 1.0 (no evidence of
    asymmetry)."""
    p, n = mcnemar_test(3, 3)
    assert n == 6
    assert p == pytest.approx(1.0, abs=1e-6)


def test_mcnemar_clear_significance():
    """10 vs 0 discordant pairs is clearly significant."""
    p, _ = mcnemar_test(10, 0)
    assert p < 0.01


def test_mcnemar_rejects_negatives():
    with pytest.raises(ValueError):
        mcnemar_test(-1, 5)


# ---------------------------------------------------------------------------
# Paired difference CI
# ---------------------------------------------------------------------------


def test_paired_diff_ci_zero_diff():
    """Equal discordant counts give a CI centred on 0."""
    lo, hi = paired_diff_ci(3, 3, 20)
    assert lo < 0.0 < hi


def test_paired_diff_ci_perfect_gain():
    """All 5 discordant pairs favour B (b=0, c=5): the function
    computes p_A - p_B, which is negative when B is better, so
    the CI sits on the negative side and excludes 0."""
    lo, hi = paired_diff_ci(0, 5, 20)
    assert hi < 0   # CI is fully negative => B significantly outperforms A
    assert lo >= -100.0


def test_paired_diff_ci_perfect_loss():
    """Mirror of the above: 5 discordant pairs favour A (b=5, c=0).
    p_A - p_B is positive => CI sits on the positive side."""
    lo, hi = paired_diff_ci(5, 0, 20)
    assert lo > 0
    assert hi <= 100.0


def test_paired_diff_ci_zero_n():
    """n=0 returns (0, 0)."""
    assert paired_diff_ci(0, 0, 0) == (0.0, 0.0)


def test_paired_diff_ci_rejects_overflow():
    """b + c > n is impossible by definition; must raise."""
    with pytest.raises(ValueError):
        paired_diff_ci(10, 10, 5)
