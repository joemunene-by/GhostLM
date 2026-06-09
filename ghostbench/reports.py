"""Reporting and rendering for GhostBench runs.

The ``render_run_report`` function produces the markdown table format
used in ``docs/baselines_v09_bets789.md`` and
``docs/format_baseline_v09.md``. The ``render_paired_comparison``
function produces a paired-comparison table when two checkpoints
have been scored against the same eval set; this is the right tool
for "did ghost-base actually beat v0.9 on bet 6, with statistical
significance?"

Reports are pure-Python, no matplotlib dep at import time. Plotting
helpers live in ``ghostbench.plot`` for callers that want PNG output.
"""

from __future__ import annotations

from typing import List, Optional

from .scoring import RunReport
from .stats import wilson_ci, mcnemar_test, cohen_h, paired_diff_ci


# ---------------------------------------------------------------------------
# Single-run rendering
# ---------------------------------------------------------------------------


def render_run_report(report: RunReport) -> str:
    """Markdown table for one RunReport. Per-tier counts plus Wilson
    95% CIs. Used for the baseline-row format in the per-bet docs."""
    summary = report.summary()
    lines = [f"# {report.bench_name} run: {report.run_name}", ""]
    lines.append(f"n = **{report.n}** predictions; "
                 f"passed (strict-AND across requested tiers): "
                 f"**{report.passed_count()} / {report.n}** "
                 f"({100*report.passed_count()/max(1, report.n):.1f}%)")
    lines.append("")
    lines.append("| Tier | passes | n | rate | 95% CI |")
    lines.append("|---|---:|---:|---:|---|")
    for tier, info in summary["per_tier"].items():
        passes, n = info["passes"], info["n"]
        rate = 100 * passes / max(1, n)
        lo, hi = wilson_ci(passes, n)
        lines.append(
            f"| {tier} | {passes} | {n} | {rate:.1f}% | [{lo:.1f}-{hi:.1f}] |"
        )
    return "\n".join(lines) + "\n"


def render_per_format_breakdown(report: RunReport) -> str:
    """Per-format rows of a RunReport. Used when one Bench's records
    span multiple format values (e.g. bet 6's eval has stix /
    yara / sigma / misp slices)."""
    by_fmt = report.by_format()
    if not by_fmt:
        return "(no per-format breakdown available)"
    lines = ["| Format | n | passed | rate | 95% CI |",
             "|---|---:|---:|---:|---|"]
    total_n = total_passed = 0
    for fmt in sorted(by_fmt.keys()):
        sub = by_fmt[fmt]
        passes = sub.passed_count()
        n = sub.n
        total_n += n
        total_passed += passes
        rate = 100 * passes / max(1, n)
        lo, hi = wilson_ci(passes, n)
        lines.append(
            f"| {fmt} | {n} | {passes} | {rate:.1f}% | [{lo:.1f}-{hi:.1f}] |"
        )
    if total_n:
        lo, hi = wilson_ci(total_passed, total_n)
        lines.append(
            f"| **OVERALL** | **{total_n}** | **{total_passed}** | "
            f"**{100*total_passed/total_n:.1f}%** | "
            f"**[{lo:.1f}-{hi:.1f}]** |"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Paired comparison (one bench, two runs)
# ---------------------------------------------------------------------------


def render_paired_comparison(a: RunReport, b: RunReport,
                              title: Optional[str] = None) -> str:
    """Markdown table comparing two checkpoints on the same Bench.

    Reports per-tier:
      - Pass count for each run + Wilson CI.
      - McNemar's test p-value on the paired discordant counts.
      - Cohen's h effect size on the proportion difference.
      - Paired-difference Wilson-shifted 95% CI.

    Both reports must have the same set of seed_ids so the pairing
    is well-defined; raises ValueError otherwise.
    """
    if title is None:
        title = f"{a.bench_name}: {a.run_name} vs {b.run_name}"

    # Index by seed_id for pairing.
    a_by_id = {s.seed_id: s for s in a.scores}
    b_by_id = {s.seed_id: s for s in b.scores}
    common_ids = sorted(set(a_by_id) & set(b_by_id))
    if not common_ids:
        raise ValueError("no overlapping seed_ids between the two runs")

    # We compare on the strict-AND ``passed`` outcome. This is the
    # right thing if both runs requested the same tiers; if they
    # diverge, render_per_tier_comparison below is the safer call.
    n = len(common_ids)
    a_pass = sum(1 for sid in common_ids if a_by_id[sid].passed)
    b_pass = sum(1 for sid in common_ids if b_by_id[sid].passed)
    discordant_b = sum(1 for sid in common_ids
                       if a_by_id[sid].passed and not b_by_id[sid].passed)
    discordant_c = sum(1 for sid in common_ids
                       if b_by_id[sid].passed and not a_by_id[sid].passed)
    p_value, n_disc = mcnemar_test(discordant_b, discordant_c)
    h = cohen_h(b_pass / n, a_pass / n)
    diff_lo, diff_hi = paired_diff_ci(discordant_c, discordant_b, n)

    a_lo, a_hi = wilson_ci(a_pass, n)
    b_lo, b_hi = wilson_ci(b_pass, n)

    h_label = (
        "negligible" if abs(h) < 0.2 else
        "small" if abs(h) < 0.5 else
        "medium" if abs(h) < 0.8 else
        "large"
    )

    lines = [
        f"# Paired comparison: {title}",
        "",
        f"n = **{n}** paired predictions on the same eval prompts.",
        "",
        "| Run | passed | rate | 95% CI |",
        "|---|---:|---:|---|",
        f"| {a.run_name} | {a_pass} | {100*a_pass/n:.1f}% | "
        f"[{a_lo:.1f}-{a_hi:.1f}] |",
        f"| {b.run_name} | {b_pass} | {100*b_pass/n:.1f}% | "
        f"[{b_lo:.1f}-{b_hi:.1f}] |",
        "",
        "## Statistical comparison",
        "",
        f"- Discordant pairs: **{n_disc}** "
        f"({a.run_name}-only: {discordant_b}, "
        f"{b.run_name}-only: {discordant_c})",
        f"- McNemar's two-sided exact p: **{p_value:.4f}**",
        f"- Cohen's h ({b.run_name} vs {a.run_name}): "
        f"**{h:+.3f}** ({h_label})",
        f"- Paired difference 95% CI "
        f"({b.run_name} − {a.run_name}): "
        f"**[{diff_lo:+.1f}%, {diff_hi:+.1f}%]**",
        "",
        "## Interpretation",
        "",
    ]

    # Honest interpretation. Avoids overclaiming.
    if p_value < 0.05 and (b_pass - a_pass) > 0:
        lines.append(
            f"At α=0.05, **{b.run_name} significantly outperforms "
            f"{a.run_name}** on this Bench. The {h_label} effect size "
            f"corroborates this is a meaningful difference, not just "
            f"a statistically detectable one."
        )
    elif p_value < 0.05 and (b_pass - a_pass) < 0:
        lines.append(
            f"At α=0.05, **{a.run_name} significantly outperforms "
            f"{b.run_name}** on this Bench."
        )
    else:
        lines.append(
            f"At α=0.05, no significant difference detected (p="
            f"{p_value:.3f}). Either the two checkpoints are "
            f"comparable on this Bench, or the eval set is too small "
            f"(n={n}) to detect the true difference. Cohen's h of "
            f"{h:+.3f} ({h_label}) suggests the underlying difference "
            f"is below the bench's current statistical power."
        )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Suite-level summary across multiple Benches
# ---------------------------------------------------------------------------


def render_suite_summary(reports: List[RunReport],
                          run_name: str) -> str:
    """One-row-per-Bench summary table for a single run across the
    whole Suite. Used as the top-of-page table in the v0.9.5 baseline
    docs and in CHANGELOG."""
    lines = [f"# Suite summary: {run_name}", ""]
    lines.append("| Bench | n | passed | rate | 95% CI |")
    lines.append("|---|---:|---:|---:|---|")
    total_n = total_passed = 0
    for r in reports:
        passed = r.passed_count()
        rate = 100 * passed / max(1, r.n)
        lo, hi = wilson_ci(passed, r.n)
        total_n += r.n
        total_passed += passed
        lines.append(
            f"| {r.bench_name} | {r.n} | {passed} | {rate:.1f}% | "
            f"[{lo:.1f}-{hi:.1f}] |"
        )
    if total_n:
        lo, hi = wilson_ci(total_passed, total_n)
        lines.append(
            f"| **OVERALL** | **{total_n}** | **{total_passed}** | "
            f"**{100*total_passed/total_n:.1f}%** | "
            f"**[{lo:.1f}-{hi:.1f}]** |"
        )
    return "\n".join(lines) + "\n"


def render_suite_paired_comparison(a_reports: List[RunReport],
                                    b_reports: List[RunReport]) -> str:
    """One-row-per-Bench paired comparison summary. Highlights which
    bets the new checkpoint wins on with statistical significance.

    Both lists must be in the same Bench order; pairs are matched by
    list index, not Bench name. (The runner in
    ``scripts/run_all_baselines.py`` already preserves order.)"""
    if len(a_reports) != len(b_reports):
        raise ValueError(
            f"report list length mismatch: {len(a_reports)} vs {len(b_reports)}"
        )
    a_name = a_reports[0].run_name if a_reports else "A"
    b_name = b_reports[0].run_name if b_reports else "B"

    lines = [f"# Suite paired comparison: {b_name} vs {a_name}", ""]
    lines.append(
        "| Bench | A passed / n | B passed / n | A rate | B rate | "
        "diff CI | McNemar p | Cohen h | sig |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---|---:|---:|:---:|"
    )
    for a, b in zip(a_reports, b_reports):
        a_by_id = {s.seed_id: s for s in a.scores}
        b_by_id = {s.seed_id: s for s in b.scores}
        common_ids = sorted(set(a_by_id) & set(b_by_id))
        if not common_ids:
            lines.append(
                f"| {a.bench_name} | {a.passed_count()}/{a.n} | "
                f"{b.passed_count()}/{b.n} | n/a | n/a | n/a | n/a | n/a | n/a |"
            )
            continue
        n = len(common_ids)
        a_pass = sum(1 for sid in common_ids if a_by_id[sid].passed)
        b_pass = sum(1 for sid in common_ids if b_by_id[sid].passed)
        d_b = sum(1 for sid in common_ids
                  if a_by_id[sid].passed and not b_by_id[sid].passed)
        d_c = sum(1 for sid in common_ids
                  if b_by_id[sid].passed and not a_by_id[sid].passed)
        p_value, _ = mcnemar_test(d_b, d_c)
        h = cohen_h(b_pass / n, a_pass / n)
        diff_lo, diff_hi = paired_diff_ci(d_c, d_b, n)
        sig = "✓" if p_value < 0.05 else "·"
        lines.append(
            f"| {a.bench_name} | {a_pass}/{n} | {b_pass}/{n} | "
            f"{100*a_pass/n:.1f}% | {100*b_pass/n:.1f}% | "
            f"[{diff_lo:+.1f}, {diff_hi:+.1f}] | "
            f"{p_value:.4f} | {h:+.3f} | {sig} |"
        )
    return "\n".join(lines) + "\n"
