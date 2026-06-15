"""Tests for the generalist scorecard's table assembly and bench specs."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.scorecard import (
    COMPETITIVE_BAND,
    PEER_REFERENCE,
    bootstrap_ci,
    default_benches,
    render_scorecard,
)


def test_peer_reference_has_general_and_cyber_benches():
    assert "arc_easy" in PEER_REFERENCE
    assert "openbookqa" in PEER_REFERENCE
    assert "secqa" in PEER_REFERENCE
    # Every bench has a random baseline.
    for ref in PEER_REFERENCE.values():
        assert ref["random"] == 25.0


def test_default_benches_split_general_mcq():
    specs = default_benches(Path("data/raw"))
    keys = {s.key for s in specs}
    assert {"arc_easy", "arc_challenge", "openbookqa", "secqa", "ctf_eval_bench"} <= keys
    arc = next(s for s in specs if s.key == "arc_easy")
    assert arc.bench_filter == "arc_easy"
    assert arc.prompt_style == "general"
    secqa = next(s for s in specs if s.key == "secqa")
    assert secqa.prompt_style == "cybersec"


def test_render_scorecard_includes_scores_and_peers():
    results = {
        "arc_easy": {"acc": 38.2, "ci": (36.1, 40.3), "n": 2365, "perms": 4, "label": "ARC-Easy"},
        "openbookqa": {"acc": 28.5, "ci": (25.5, 31.4), "n": 500, "perms": 4, "label": "OpenBookQA"},
    }
    md = render_scorecard("ghost-small-gen", results)
    assert "ghost-small-gen" in md
    assert "**38.2%**" in md          # filled-in GhostLM score
    assert "36.1-40.3" in md          # CI rendered
    assert "pythia_160m=43.5" in md   # peer reference rendered
    assert "| arc_challenge |" in md  # unscored bench still listed with em dash
    assert "—" in md


def test_render_marks_significance_above_random():
    # CI lower bound above 25 -> '+'; straddling -> '~'; below -> '-'.
    results = {
        "arc_easy": {"acc": 38.0, "ci": (35.0, 41.0), "n": 100, "perms": 4, "label": "ARC-Easy"},
        "openbookqa": {"acc": 26.0, "ci": (23.0, 29.0), "n": 100, "perms": 4, "label": "OpenBookQA"},
    }
    md = render_scorecard("x", results)
    rows = {line.split("|")[1].strip(): line for line in md.splitlines() if line.startswith("| arc_easy") or line.startswith("| openbookqa")}
    assert " + " in rows["arc_easy"]      # 35 > 25 -> significant
    assert " ~ " in rows["openbookqa"]    # 23 < 25 < 29 -> straddles


def test_bootstrap_ci_basic():
    # All-correct -> CI pinned near 100; all-wrong -> near 0.
    lo, hi = bootstrap_ci([1.0] * 50, seed=0)
    assert lo == 100.0 and hi == 100.0
    lo, hi = bootstrap_ci([0.0] * 50, seed=0)
    assert lo == 0.0 and hi == 0.0
    # Mixed -> CI brackets the mean and is ordered.
    lo, hi = bootstrap_ci([1.0, 0.0] * 50, seed=0)
    assert lo < 50.0 < hi


def test_bootstrap_ci_deterministic():
    accs = [1.0, 0.0, 1.0, 1.0, 0.0] * 20
    assert bootstrap_ci(accs, seed=7) == bootstrap_ci(accs, seed=7)


def test_competitive_band_covers_general_benches():
    assert "arc_easy" in COMPETITIVE_BAND
    assert "openbookqa" in COMPETITIVE_BAND
