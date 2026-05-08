"""Tests for ghostbench.bench.Bench, Suite, EvalRecord, Prediction."""

import json
from pathlib import Path

import pytest

from ghostbench.bench import Bench, EvalRecord, Prediction, Suite
from ghostbench.parsers import DEFAULT_PARSERS


# ---------------------------------------------------------------------------
# EvalRecord / Prediction round-trip
# ---------------------------------------------------------------------------


def test_eval_record_from_dict_defaults():
    """Missing fields default to empty lists / None."""
    er = EvalRecord.from_dict({"format": "x", "prompt": "y"})
    assert er.fmt == "x"
    assert er.prompt == "y"
    assert er.required_fields == []
    assert er.required_substrings == []
    assert er.seed_id is None


def test_eval_record_to_score_dict_uses_prompt_for_seed():
    """When seed_id is missing, the score dict uses the first 60
    chars of the prompt as the seed id."""
    er = EvalRecord(fmt="x", prompt="A" * 80)
    d = er.to_score_dict()
    assert d["seed_id"] == "A" * 60


def test_prediction_from_dict():
    """Predictions deserialise predicted_artifact + tag fields."""
    p = Prediction.from_dict({
        "format": "code_security",
        "prompt": "q?",
        "predicted_artifact": "a",
        "required_substrings": ["s"],
    })
    assert p.fmt == "code_security"
    assert p.predicted_artifact == "a"
    assert p.required_substrings == ["s"]


# ---------------------------------------------------------------------------
# Bench scoring
# ---------------------------------------------------------------------------


def test_bench_score_smoke(tmp_path: Path):
    """Round-trip a tiny eval JSONL through Bench.from_jsonl + score()."""
    eval_path = tmp_path / "tiny_eval.jsonl"
    eval_path.write_text(
        json.dumps({
            "format": "code_security",
            "prompt": "what CWE?",
            "required_substrings": ["CWE-89"],
            "seed_id": "t1",
        }) + "\n"
    )
    bench = Bench.from_jsonl(
        name="tiny", description="d",
        path=eval_path, parsers=DEFAULT_PARSERS,
    )
    assert len(bench) == 1

    preds = [Prediction(
        fmt="code_security", prompt="what CWE?",
        predicted_artifact="The answer is CWE-89.",
        required_substrings=["CWE-89"], seed_id="t1",
    )]
    report = bench.score(preds, run_name="r1")
    assert report.n == 1
    assert report.passed_count() == 1


def test_bench_score_failure_path(tmp_path: Path):
    """Wrong predicted output gives passed_count = 0."""
    eval_path = tmp_path / "tiny_eval.jsonl"
    eval_path.write_text(
        json.dumps({
            "format": "code_security",
            "prompt": "what CWE?",
            "required_substrings": ["CWE-89"],
            "seed_id": "t1",
        }) + "\n"
    )
    bench = Bench.from_jsonl(
        name="tiny", description="d",
        path=eval_path, parsers=DEFAULT_PARSERS,
    )
    preds = [Prediction(
        fmt="code_security", prompt="what CWE?",
        predicted_artifact="unrelated text",
        required_substrings=["CWE-89"], seed_id="t1",
    )]
    report = bench.score(preds, run_name="r1")
    assert report.passed_count() == 0


# ---------------------------------------------------------------------------
# Suite discovery
# ---------------------------------------------------------------------------


def test_suite_discovers_known_evals(tmp_path: Path):
    """Suite.from_dir picks up the canonical eval filenames."""
    eval_dir = tmp_path / "evals"
    eval_dir.mkdir()
    (eval_dir / "code_security_eval.jsonl").write_text(
        json.dumps({"format": "code_security", "prompt": "q",
                     "required_substrings": ["x"]}) + "\n"
    )
    (eval_dir / "format_aware_eval.jsonl").write_text(
        json.dumps({"format": "stix_indicator", "prompt": "q",
                     "required_substrings": ["x"]}) + "\n"
    )
    suite = Suite.from_dir(eval_dir, parsers=DEFAULT_PARSERS)
    names = sorted(b.name for b in suite)
    assert names == ["bet6_format_aware", "bet7_code_security"]


def test_suite_keyed_lookup(tmp_path: Path):
    """Suite[name] returns the named Bench."""
    eval_dir = tmp_path / "evals"
    eval_dir.mkdir()
    (eval_dir / "code_security_eval.jsonl").write_text(
        json.dumps({"format": "code_security", "prompt": "q",
                     "required_substrings": ["x"]}) + "\n"
    )
    suite = Suite.from_dir(eval_dir, parsers=DEFAULT_PARSERS)
    bench = suite["bet7_code_security"]
    assert bench.name == "bet7_code_security"
    with pytest.raises(KeyError):
        suite["nonexistent"]
