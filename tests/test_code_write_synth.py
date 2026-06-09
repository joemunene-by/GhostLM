"""Tests for the v0.9.24 code-write synth pipeline."""

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

PATTERNS = REPO_ROOT / "data" / "raw" / "code_write_patterns.jsonl"
EVAL_PATH = REPO_ROOT / "data" / "raw" / "code_write_eval.jsonl"


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
        assert len(recs) >= 193

    def test_record_shape(self):
        recs = _load_jsonl(PATTERNS)
        for r in recs:
            for f in ("id", "language", "description",
                       "implementation", "explanation"):
                assert r.get(f), f"{r.get('id')} missing {f}"
            assert len(r["implementation"]) >= 10
            assert len(r["explanation"]) >= 60

    def test_languages_diverse(self):
        recs = _load_jsonl(PATTERNS)
        langs = {r["language"] for r in recs}
        # Bank should cover at least 4 languages.
        assert len(langs) >= 4
        for must in ("python", "go", "rust", "javascript"):
            assert must in langs

    def test_unique_ids(self):
        recs = _load_jsonl(PATTERNS)
        ids = [r["id"] for r in recs]
        assert len(ids) == len(set(ids))


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
            [sys.executable, "scripts/synth_code_write.py",
             "--bank", str(PATTERNS),
             "--out", str(out)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr
        recs = _load_jsonl(out)
        # 195 patterns × 3 reliable variants minimum = 585 records.
        assert len(recs) >= 580, f"got only {len(recs)}"
        sources = {r.get("seed_source") for r in recs}
        for v in ("pretrain_prose", "write_function",
                   "write_idiomatic"):
            assert v in sources, f"variant missing: {v}"

    def test_record_format(self, tmp_path):
        out = tmp_path / "synth.jsonl"
        subprocess.run(
            [sys.executable, "scripts/synth_code_write.py",
             "--bank", str(PATTERNS),
             "--out", str(out)],
            cwd=REPO_ROOT, check=True, timeout=60,
        )
        recs = _load_jsonl(out)
        for r in recs:
            assert r["source"] == "synth_code_write"
            assert r["teacher"] == "templated"
            assert r["text"]
