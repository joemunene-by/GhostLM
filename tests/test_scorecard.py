"""Tests for the generalist scorecard's table assembly and bench specs."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.scorecard import (
    COMPETITIVE_BAND,
    PEER_REFERENCE,
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
        "arc_easy": {"acc": 38.2, "n": 2365, "perms": 4, "label": "ARC-Easy"},
        "openbookqa": {"acc": 28.5, "n": 500, "perms": 4, "label": "OpenBookQA"},
    }
    md = render_scorecard("ghost-small-gen", results)
    assert "ghost-small-gen" in md
    assert "**38.2%**" in md          # filled-in GhostLM score
    assert "pythia_160m=43.5" in md   # peer reference rendered
    assert "| arc_challenge |" in md  # unscored bench still listed with em dash
    assert "—" in md


def test_competitive_band_covers_general_benches():
    assert "arc_easy" in COMPETITIVE_BAND
    assert "openbookqa" in COMPETITIVE_BAND
