"""Matplotlib visualizations for GhostBench reports.

These are the publication-grade visuals: forest plots with Wilson 95%
CI error bars, per-bench bar charts, suite paired-comparison forest
plots showing which bets a new checkpoint won on with statistical
significance, and projection charts showing expected ghost-base
scores with both methodological and statistical uncertainty layers.

All plots are designed to drop straight into a paper or blog post;
the styling is intentionally restrained (single colour family, no
chartjunk) so the data carries the visual weight.

Module is import-safe without matplotlib installed: every public
function does a lazy import and emits a clear error if the user
hasn't installed it. matplotlib is a heavy dep we don't want at
``ghostbench`` package import time.

Public API:

    plot_run_report(report, out_path=None, ...)
        Single-run bar chart with per-tier rates + CI error bars.

    plot_suite_summary(reports, out_path=None, ...)
        Multi-bench bar chart showing the strict-AND ``passed`` rate
        per bench with CI error bars.

    plot_paired_comparison(a_report, b_report, out_path=None, ...)
        Forest plot of the per-tier paired difference (B − A) with
        Wilson-shifted CIs, clearly marking which tiers crossed
        significance.

    plot_suite_paired_comparison(a_reports, b_reports, out_path=None, ...)
        One row per bench, paired difference with CI, highlighted
        rows for statistically-significant lifts.

    plot_projections(projections, out_path=None, ...)
        Per-bench projection chart showing point estimate +
        methodological credibility band + Wilson 95% statistical CI.
        The visualisation that says 'here's what ghost-base is
        likely to score, with uncertainty.'

All functions return the matplotlib Figure so callers can add
custom annotations or save in any format.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from .scoring import RunReport
from .stats import cohen_h, mcnemar_test, paired_diff_ci, wilson_ci


# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------


# Single colour family. The first two are the "two checkpoints"
# colours used in paired plots; the third + fourth are accents for
# significance markers / CI fills.
_PALETTE = {
    "a": "#3D5A80",   # deep blue (baseline / older checkpoint)
    "b": "#EE6C4D",   # warm orange (new / candidate checkpoint)
    "fill": "#E0FBFC",
    "sig": "#293241",
    "grid": "#CCD0D5",
    "text": "#1A1D24",
}


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        from matplotlib import patches  # noqa: F401
        return plt
    except ImportError as e:
        raise ImportError(
            "ghostbench.plot requires matplotlib; install with "
            "'pip install matplotlib'"
        ) from e


def _save_or_return(fig, out_path: Optional[str]):
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Single-run report
# ---------------------------------------------------------------------------


def plot_run_report(report: RunReport, out_path: Optional[str] = None,
                    title: Optional[str] = None,
                    figsize: Tuple[float, float] = (8.0, 4.0)):
    """Bar chart of per-tier pass rates for one RunReport.

    Each bar shows the pass rate; an error bar shows the Wilson
    95% CI; the n is annotated above each bar.
    """
    plt = _require_matplotlib()

    summary = report.summary()
    tiers = sorted(summary["per_tier"].keys())
    if not tiers:
        raise ValueError("report has no per-tier data to plot")

    rates = []
    ci_lo = []
    ci_hi = []
    ns = []
    for t in tiers:
        info = summary["per_tier"][t]
        passes, n = info["passes"], info["n"]
        rate = 100 * passes / max(1, n)
        lo, hi = wilson_ci(passes, n)
        rates.append(rate)
        ci_lo.append(rate - lo)
        ci_hi.append(hi - rate)
        ns.append(n)

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        tiers, rates,
        yerr=[ci_lo, ci_hi],
        color=_PALETTE["b"], edgecolor=_PALETTE["sig"],
        capsize=4, error_kw={"ecolor": _PALETTE["sig"], "elinewidth": 1.2},
    )
    for bar, rate, n in zip(bars, rates, ns):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(rate + 6, 95),
            f"n={n}",
            ha="center", va="bottom",
            fontsize=9, color=_PALETTE["text"],
        )

    ax.set_ylim(0, 105)
    ax.set_ylabel("Pass rate (%) with Wilson 95% CI")
    ax.set_xlabel("Tier")
    ax.set_title(title or f"{report.bench_name} ({report.run_name})")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=_PALETTE["grid"], linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return _save_or_return(fig, out_path)


# ---------------------------------------------------------------------------
# Suite summary
# ---------------------------------------------------------------------------


def plot_suite_summary(reports: List[RunReport],
                        out_path: Optional[str] = None,
                        title: Optional[str] = None,
                        figsize: Tuple[float, float] = (9.0, 4.5)):
    """One bar per Bench showing the strict-AND ``passed`` rate.

    Right plot for the top-of-paper headline that says 'here's how
    a single checkpoint scored on every bench.'
    """
    plt = _require_matplotlib()

    if not reports:
        raise ValueError("no reports to plot")

    names = [r.bench_name for r in reports]
    rates = []
    ci_lo = []
    ci_hi = []
    ns = []
    for r in reports:
        passed = r.passed_count()
        n = r.n
        rate = 100 * passed / max(1, n)
        lo, hi = wilson_ci(passed, n)
        rates.append(rate)
        ci_lo.append(rate - lo)
        ci_hi.append(hi - rate)
        ns.append(n)

    fig, ax = plt.subplots(figsize=figsize)
    run_name = reports[0].run_name if reports else ""
    bars = ax.bar(
        range(len(names)), rates,
        yerr=[ci_lo, ci_hi],
        color=_PALETTE["b"], edgecolor=_PALETTE["sig"],
        capsize=4, error_kw={"ecolor": _PALETTE["sig"], "elinewidth": 1.2},
    )
    for bar, rate, n in zip(bars, rates, ns):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(rate + 6, 95),
            f"n={n}",
            ha="center", va="bottom",
            fontsize=9, color=_PALETTE["text"],
        )

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("strict-AND pass rate (%) with Wilson 95% CI")
    ax.set_title(title or f"GhostBench suite: {run_name}")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=_PALETTE["grid"], linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return _save_or_return(fig, out_path)


# ---------------------------------------------------------------------------
# Paired comparison (one bench)
# ---------------------------------------------------------------------------


def plot_paired_comparison(a_report: RunReport, b_report: RunReport,
                             out_path: Optional[str] = None,
                             title: Optional[str] = None,
                             figsize: Tuple[float, float] = (8.0, 5.0)):
    """Side-by-side bars (A and B) per tier with paired-difference
    annotations + significance markers.

    The plot doubles as the publication-grade artifact for "did B
    actually beat A on this bench". McNemar p-values that cross
    α=0.05 get a star; effect sizes are labelled small/medium/large
    by Cohen's h convention.
    """
    plt = _require_matplotlib()

    # Pair scores by seed_id so the McNemar / paired-diff stats are
    # well-defined.
    a_by_id = {s.seed_id: s for s in a_report.scores}
    b_by_id = {s.seed_id: s for s in b_report.scores}
    common_ids = sorted(set(a_by_id) & set(b_by_id))
    if not common_ids:
        raise ValueError("no overlapping seed_ids between the two runs")

    # Determine the union of tiers requested by the eval.
    tiers = sorted({
        t
        for sid in common_ids
        for t in (a_by_id[sid].requested_tiers
                  + b_by_id[sid].requested_tiers)
    })
    if not tiers:
        raise ValueError("paired runs have no requested tiers in common")

    n = len(common_ids)
    fig, ax = plt.subplots(figsize=figsize)

    width = 0.36
    a_rates, a_lo, a_hi = [], [], []
    b_rates, b_lo, b_hi = [], [], []
    annotations = []
    for t in tiers:
        a_pass = sum(1 for sid in common_ids if a_by_id[sid].tier_pass(t))
        b_pass = sum(1 for sid in common_ids if b_by_id[sid].tier_pass(t))
        d_b = sum(
            1 for sid in common_ids
            if a_by_id[sid].tier_pass(t) and not b_by_id[sid].tier_pass(t)
        )
        d_c = sum(
            1 for sid in common_ids
            if b_by_id[sid].tier_pass(t) and not a_by_id[sid].tier_pass(t)
        )
        p_value, _ = mcnemar_test(d_b, d_c)
        h = cohen_h(b_pass / n, a_pass / n)
        a_r = 100 * a_pass / n
        b_r = 100 * b_pass / n
        a_low, a_high = wilson_ci(a_pass, n)
        b_low, b_high = wilson_ci(b_pass, n)
        a_rates.append(a_r); a_lo.append(a_r - a_low); a_hi.append(a_high - a_r)
        b_rates.append(b_r); b_lo.append(b_r - b_low); b_hi.append(b_high - b_r)
        sig_marker = "**" if p_value < 0.01 else ("*" if p_value < 0.05 else "")
        h_label = (
            "neg" if abs(h) < 0.2 else
            "sml" if abs(h) < 0.5 else
            "med" if abs(h) < 0.8 else
            "lrg"
        )
        annotations.append((sig_marker, h_label, p_value))

    x = list(range(len(tiers)))
    bars_a = ax.bar(
        [xi - width / 2 for xi in x], a_rates, width,
        yerr=[a_lo, a_hi],
        label=a_report.run_name, color=_PALETTE["a"],
        edgecolor=_PALETTE["sig"], capsize=3,
        error_kw={"ecolor": _PALETTE["sig"], "elinewidth": 1.0},
    )
    bars_b = ax.bar(
        [xi + width / 2 for xi in x], b_rates, width,
        yerr=[b_lo, b_hi],
        label=b_report.run_name, color=_PALETTE["b"],
        edgecolor=_PALETTE["sig"], capsize=3,
        error_kw={"ecolor": _PALETTE["sig"], "elinewidth": 1.0},
    )

    # Significance markers above each pair.
    for i, (sig, h_label, p) in enumerate(annotations):
        ymax = max(a_rates[i] + a_hi[i], b_rates[i] + b_hi[i])
        if sig:
            ax.text(
                i, min(ymax + 4, 100),
                sig + f"\nh={h_label}",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=_PALETTE["sig"],
            )

    ax.set_xticks(x)
    ax.set_xticklabels(tiers, fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Tier pass rate (%) with Wilson 95% CI")
    ax.set_title(title or f"{a_report.bench_name}: "
                          f"{a_report.run_name} vs {b_report.run_name} "
                          f"(n={n})")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=_PALETTE["grid"], linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    # Caption explaining the markers.
    ax.text(
        0.5, -0.18,
        "* p<0.05  ** p<0.01 (McNemar exact paired)   "
        "h: neg/sml/med/lrg by Cohen convention",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=8, color=_PALETTE["text"],
    )
    fig.tight_layout()
    return _save_or_return(fig, out_path)


# ---------------------------------------------------------------------------
# Suite paired comparison (forest plot)
# ---------------------------------------------------------------------------


def plot_projections(projections, out_path: Optional[str] = None,
                       title: Optional[str] = None,
                       figsize: Tuple[float, float] = (9.0, 5.0)):
    """Per-bench projection chart with two uncertainty layers.

    Each bench gets:
      - point estimate: a coloured marker at the projected pass rate
      - methodological band: a wide light bar on the +/-30%-asymptote
        credibility interval (the "we're guessing the asymptote"
        uncertainty)
      - Wilson 95% CI: a narrower error bar overlaid on the marker
        (the "small-n statistical" uncertainty at the eval set's n)

    Right plot for the 'expected results' section of the ghost-base
    spec doc: communicates two distinct sources of uncertainty
    without conflating them.

    Args:
        projections: Iterable of Projection objects from
                     ``ghostbench.projections.project_suite()``.
    """
    plt = _require_matplotlib()

    projs = list(projections)
    if not projs:
        raise ValueError("no projections to plot")

    fig, ax = plt.subplots(figsize=figsize)
    y_positions = list(range(len(projs)))

    for i, p in enumerate(projs):
        # Wide band: methodological credibility interval (asymptote +/- 30%).
        ax.plot(
            [p.methodological_lo, p.methodological_hi],
            [i, i],
            color=_PALETTE["fill"], linewidth=14, solid_capstyle="round",
        )
        # Narrower band: Wilson statistical CI at projected score.
        ax.plot(
            [p.wilson_lo, p.wilson_hi],
            [i, i],
            color=_PALETTE["b"], linewidth=4, solid_capstyle="round",
        )
        # Point estimate marker.
        ax.scatter([p.point_estimate], [i],
                    color=_PALETTE["sig"], s=80, zorder=4,
                    edgecolor=_PALETTE["sig"])
        # Right-margin annotation with the records-seen number.
        ax.text(
            105, i,
            f"  records: {p.records_seen}, "
            f"asymptote: {100 * p.asymptote:.0f}%",
            va="center", ha="left", fontsize=9, color=_PALETTE["text"],
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([p.bench_name for p in projs], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(-5, 200)
    ax.set_xlabel("Projected pass rate (%)")
    ax.set_title(
        title or "Ghost-base projections (point + methodological band + Wilson 95% CI)"
    )
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=_PALETTE["grid"], linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.text(
        0.5, -0.15,
        "Light wide band: methodological credibility (asymptote ± 30%).   "
        "Solid narrow band: Wilson 95% CI at eval n.   "
        "Marker: point estimate.",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=8, color=_PALETTE["text"],
    )
    fig.tight_layout()
    return _save_or_return(fig, out_path)


def plot_suite_paired_comparison(a_reports: List[RunReport],
                                  b_reports: List[RunReport],
                                  out_path: Optional[str] = None,
                                  title: Optional[str] = None,
                                  figsize: Tuple[float, float] = (8.0, 5.5)):
    """Forest plot: one row per bench, showing the paired difference
    (B − A) with a Wilson-shifted CI. Significant rows get the
    accent colour.

    This is the headline visualisation for 'here's where ghost-base
    actually moves the needle vs v0.9 chat.' If 4/4 benches show
    a CI excluding zero, that's the publishable headline plot.
    """
    plt = _require_matplotlib()

    if len(a_reports) != len(b_reports):
        raise ValueError(
            f"report list length mismatch: "
            f"{len(a_reports)} vs {len(b_reports)}"
        )
    if not a_reports:
        raise ValueError("no reports to plot")

    rows = []
    for a, b in zip(a_reports, b_reports):
        a_by_id = {s.seed_id: s for s in a.scores}
        b_by_id = {s.seed_id: s for s in b.scores}
        common_ids = sorted(set(a_by_id) & set(b_by_id))
        if not common_ids:
            rows.append((a.bench_name, 0.0, (0.0, 0.0), 1.0))
            continue
        n = len(common_ids)
        d_b = sum(1 for sid in common_ids
                  if a_by_id[sid].passed and not b_by_id[sid].passed)
        d_c = sum(1 for sid in common_ids
                  if b_by_id[sid].passed and not a_by_id[sid].passed)
        a_pass = sum(1 for sid in common_ids if a_by_id[sid].passed)
        b_pass = sum(1 for sid in common_ids if b_by_id[sid].passed)
        diff = 100 * (b_pass - a_pass) / n
        diff_lo, diff_hi = paired_diff_ci(d_c, d_b, n)
        p_value, _ = mcnemar_test(d_b, d_c)
        rows.append((a.bench_name, diff, (diff_lo, diff_hi), p_value))

    fig, ax = plt.subplots(figsize=figsize)
    y_positions = list(range(len(rows)))
    a_name = a_reports[0].run_name
    b_name = b_reports[0].run_name

    for i, (name, diff, (lo, hi), p_value) in enumerate(rows):
        sig = p_value < 0.05
        colour = _PALETTE["b"] if sig else _PALETTE["a"]
        # Plot CI bar.
        ax.plot([lo, hi], [i, i], color=colour, linewidth=2.5, alpha=0.8)
        # Plot point estimate.
        ax.scatter([diff], [i], color=colour, s=80, zorder=3,
                    edgecolor=_PALETTE["sig"])
        # Label.
        marker = "**" if p_value < 0.01 else ("*" if p_value < 0.05 else " ")
        ax.text(
            105, i,
            f"{marker} Δ={diff:+.1f}%  p={p_value:.3f}",
            va="center", ha="left", fontsize=9, color=_PALETTE["text"],
        )

    ax.axvline(0, color=_PALETTE["sig"], linewidth=0.8, linestyle="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(-110, 200)
    ax.set_xlabel(f"Paired difference: {b_name} − {a_name} (%) "
                   f"with Wilson-shifted 95% CI")
    ax.set_title(title or f"Suite paired comparison: {b_name} vs {a_name}")
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=_PALETTE["grid"], linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.text(
        0.5, -0.15,
        "* p<0.05  ** p<0.01 (McNemar exact paired)   "
        "Filled in accent: significant lift",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=8, color=_PALETTE["text"],
    )
    fig.tight_layout()
    return _save_or_return(fig, out_path)
