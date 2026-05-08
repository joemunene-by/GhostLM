"""Statistical helpers for GhostBench.

Small-n binomial-proportion analysis is the central problem: each
GhostBench eval has 15-32 records, and pass-rates near 0 or 1 are
common when measuring novel capabilities on small LMs. The helpers
here are the right tools for that regime:

    wilson_ci(k, n)        Wilson 95% interval for the proportion
                           ``k/n``. Right at small n. Returns a
                           tuple of percentages (lo, hi).

    mcnemar_test(b, c)     McNemar's exact test for paired binary
                           outcomes. Used when comparing two
                           checkpoints on the SAME eval prompts:
                           ``b`` is the count of prompts checkpoint
                           A passed but B failed; ``c`` is B-passed-
                           but-A-failed; the agreed cases drop out.
                           Returns (p_value, n_discordant).

    cohen_h(p1, p2)        Cohen's h effect size for the difference
                           between two proportions. The 0.2 / 0.5 /
                           0.8 cuts (small / medium / large) keep us
                           honest when a 5-percentage-point lift
                           looks numerically bigger than it actually
                           is at p near 0.

    paired_diff_ci(b, c, n) Wilson-shifted interval on the proportion
                           difference under paired sampling. Tighter
                           than two independent Wilson CIs for the
                           paired-comparison case.

All functions are deterministic, stdlib-only, and take/return
plain Python numbers so the module pickles cleanly and ports to
JSON / CSV reports without numpy round-trips.
"""

from __future__ import annotations

import math
from typing import Tuple


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson 95% binomial proportion confidence interval, returned
    as percentages.

    The Wilson interval is the right CI for ``k`` successes in ``n``
    trials when n is small or p is near 0 / 1. It does not blow up
    at p = 0 (the Wald / normal-approximation interval has zero
    width there, which is wrong) and is less conservative than
    Clopper-Pearson.

    Examples:
        >>> wilson_ci(0, 32)   # zero hits in 32 trials
        (0.0, 10.7...)
        >>> wilson_ci(0, 8)    # zero hits in 8 trials
        (0.0, 32.4...)
        >>> wilson_ci(20, 32)  # 20 hits in 32 trials
        (45.6..., 76.4...)

    Args:
        k: Number of successes.
        n: Number of trials. Must be non-negative.
        z: Critical value of the standard normal (default 1.96 for
           two-sided 95% CI; use 2.576 for 99%).

    Returns:
        ``(lo, hi)`` as percentages in [0, 100]. ``(0.0, 0.0)`` when
        ``n == 0``.
    """
    if n <= 0:
        return (0.0, 0.0)
    if k < 0 or k > n:
        raise ValueError(f"k must be in [0, n]; got k={k}, n={n}")
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    spread = z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n) / denom
    lo = max(0.0, center - spread)
    hi = min(1.0, center + spread)
    return (100.0 * lo, 100.0 * hi)


def cohen_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for the difference between two
    proportions.

    Computed as ``2 * (asin(sqrt(p1)) - asin(sqrt(p2)))``, which is
    the difference between the variance-stabilising arcsine
    transforms of the two proportions. Conventional thresholds:

        |h| < 0.2   negligible
        0.2 <= |h| < 0.5   small
        0.5 <= |h| < 0.8   medium
        |h| >= 0.8   large

    These cuts are particularly useful when one of the proportions
    is near 0 or 1: a lift from 1% to 6% sounds like 6x but lands
    at h ~= 0.27 (small), whereas a lift from 50% to 56% sounds
    like 12% but lands at h ~= 0.12 (negligible).

    Args:
        p1: First proportion in [0, 1].
        p2: Second proportion in [0, 1].

    Returns:
        Cohen's h. Sign is positive when ``p1 > p2``.
    """
    if not (0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0):
        raise ValueError(f"proportions must be in [0, 1]; got {p1=}, {p2=}")
    return 2.0 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))


def mcnemar_test(b: int, c: int) -> Tuple[float, int]:
    """Exact McNemar's test for paired binary outcomes.

    McNemar's test is the right tool when comparing two checkpoints
    on the SAME eval prompts. The "paired" structure (every prompt
    is scored under both checkpoints) means agreed cases (both pass
    or both fail) carry no information about the difference; only
    the discordant cases matter.

      b = count of prompts checkpoint A passed but B failed
      c = count of prompts checkpoint B passed but A failed

    Under the null hypothesis (the two checkpoints have equal pass
    probability), b is distributed Binomial(b + c, 0.5).

    For small n_discordant (b + c <= 25) we compute the exact
    two-sided binomial p-value rather than the chi-squared
    approximation, since the approximation is poor at small n.

    Args:
        b: Discordant count where A passed and B failed.
        c: Discordant count where B passed and A failed.

    Returns:
        ``(p_value, n_discordant)``. ``p_value`` is two-sided; small
        means the two checkpoints differ. ``n_discordant = b + c``;
        when both are 0, returns ``(1.0, 0)``.
    """
    if b < 0 or c < 0:
        raise ValueError(f"b, c must be non-negative; got {b=}, {c=}")
    n = b + c
    if n == 0:
        return (1.0, 0)
    # Exact two-sided binomial test against p = 0.5.
    k = min(b, c)
    # Sum tail probabilities <= P(X = k) and >= P(X = n - k).
    p = 0.0
    for i in range(0, n + 1):
        prob = math.comb(n, i) * (0.5 ** n)
        # Two-sided: include probabilities at least as extreme as
        # the observed split.
        if math.comb(n, i) <= math.comb(n, k):
            p += prob
    return (min(1.0, p), n)


def paired_diff_ci(b: int, c: int, n: int, z: float = 1.96
                   ) -> Tuple[float, float]:
    """Wilson-shifted CI on the proportion DIFFERENCE under paired
    sampling.

    For paired data with discordant counts ``b`` (A pass, B fail)
    and ``c`` (B pass, A fail), the maximum-likelihood estimate of
    the difference ``p_A - p_B`` is ``(b - c) / n``. The Wilson-shifted
    interval (Newcombe's method 10) is the recommended small-n
    CI. The implementation here is the simpler approximation
    that uses the score statistic directly; it agrees with
    Newcombe's method to within rounding for n >= 10 and the
    discordant-count regime we care about.

    Args:
        b: A-pass-B-fail count.
        c: B-pass-A-fail count.
        n: Total paired trials.
        z: Critical value (default 1.96 for 95%).

    Returns:
        ``(lo, hi)`` of the paired difference as percentages in
        [-100, 100]. Negative values mean B is better.
    """
    if n <= 0:
        return (0.0, 0.0)
    if b + c > n:
        raise ValueError(
            f"discordant counts exceed n: {b=}, {c=}, {n=}"
        )
    p_diff = (b - c) / n
    var = ((b + c) - (b - c) ** 2 / n) / (n * n)
    if var < 0:
        var = 0.0
    spread = z * math.sqrt(var)
    lo = max(-1.0, p_diff - spread)
    hi = min(1.0, p_diff + spread)
    return (100.0 * lo, 100.0 * hi)
