"""Tests for the v0.9.23 code-explain synth pipeline."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PATTERNS = REPO_ROOT / "data" / "raw" / "code_explain_patterns.jsonl"
EVAL_PATH = REPO_ROOT / "data" / "raw" / "code_explain_eval.jsonl"


def _load_jsonl(path):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class TestPatternsBank:
    def test_loads(self):
        recs = _load_jsonl(PATTERNS)
        assert len(recs) >= 40

    def test_record_shape(self):
        recs = _load_jsonl(PATTERNS)
        for r in recs:
            for f in ("id", "language", "snippet", "purpose",
                       "explanation"):
                assert r.get(f), f"{r.get('id')} missing {f}"
            # Snippet should be non-trivial.
            assert len(r["snippet"]) >= 10
            assert len(r["explanation"]) >= 80

    def test_languages_diverse(self):
        recs = _load_jsonl(PATTERNS)
        langs = {r["language"] for r in recs}
        # The bank should cover at least 6 languages.
        assert len(langs) >= 6, f"only {len(langs)} languages: {langs}"
        for must in ("python", "go", "rust"):
            assert must in langs

    def test_unique_ids(self):
        recs = _load_jsonl(PATTERNS)
        ids = [r["id"] for r in recs]
        assert len(ids) == len(set(ids)), "duplicate ids in bank"


class TestEval:
    def test_eval_loads(self):
        recs = _load_jsonl(EVAL_PATH)
        assert len(recs) >= 14

    def test_eval_shape(self):
        recs = _load_jsonl(EVAL_PATH)
        for r in recs:
            assert "prompt" in r
            assert isinstance(r.get("required_substrings"), list)
            assert len(r["required_substrings"]) >= 1


class TestSynthIntegration:
    def test_synth_runs(self, tmp_path):
        out = tmp_path / "synth.jsonl"
        result = subprocess.run(
            [sys.executable, "scripts/synth_code_explain.py",
             "--bank", str(PATTERNS),
             "--out", str(out)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr
        recs = _load_jsonl(out)
        # 40 patterns × 5 variants = 200 records minimum.
        assert len(recs) >= 195, f"got only {len(recs)}"
        sources = {r.get("seed_source") for r in recs}
        for v in ("pretrain_prose", "identify_lang",
                   "explain_purpose", "walkthrough", "concepts"):
            assert v in sources, f"variant missing: {v}"

    def test_record_format(self, tmp_path):
        out = tmp_path / "synth.jsonl"
        subprocess.run(
            [sys.executable, "scripts/synth_code_explain.py",
             "--bank", str(PATTERNS),
             "--out", str(out)],
            cwd=REPO_ROOT, check=True, timeout=60,
        )
        recs = _load_jsonl(out)
        for r in recs:
            assert r["source"] == "synth_code_explain"
            assert r["teacher"] == "templated"
            assert r["text"]
            assert len(r["text"]) >= 100
