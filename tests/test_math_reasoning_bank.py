"""Tests for the v0.9.21 math + reasoning SFT bank."""

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

MR_PATH = REPO_ROOT / "data" / "raw" / "chat" / "math_reasoning.jsonl"


def _load_jsonl(path):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


class TestBankShape:
    def test_file_exists(self):
        assert MR_PATH.exists()

    def test_loads(self):
        recs = _load_jsonl(MR_PATH)
        assert len(recs) >= 50

    def test_record_shape(self):
        recs = _load_jsonl(MR_PATH)
        for r in recs:
            assert r.get("source") == "math_reasoning"
            assert "topic" in r
            assert isinstance(r["turns"], list)
            assert len(r["turns"]) >= 2
            assert r["turns"][0]["role"] == "user"
            assert r["turns"][1]["role"] == "assistant"
            assert r["turns"][0]["content"].strip()
            assert r["turns"][1]["content"].strip()


class TestTopicCoverage:
    def test_topics_span_categories(self):
        recs = _load_jsonl(MR_PATH)
        topics = {r["topic"] for r in recs}
        for t in ("arithmetic", "algebra", "geometry",
                   "word_problems", "probability", "statistics",
                   "logic", "proof", "combinatorics"):
            assert t in topics, f"missing topic: {t}"

    def test_arithmetic_well_represented(self):
        recs = _load_jsonl(MR_PATH)
        arith = [r for r in recs if r["topic"] == "arithmetic"]
        assert len(arith) >= 8

    def test_word_problems_well_represented(self):
        recs = _load_jsonl(MR_PATH)
        wp = [r for r in recs if r["topic"] == "word_problems"]
        assert len(wp) >= 8

    def test_step_by_step_examples_present(self):
        """At least a few records should show explicit step-by-step
        reasoning so the model learns to walk through computations."""
        recs = _load_jsonl(MR_PATH)
        step_by_step = [r for r in recs
                         if "step" in r["turns"][1]["content"].lower()]
        assert len(step_by_step) >= 3


class TestBuildChatDatasetCLI:
    def test_help_includes_new_flags(self):
        result = subprocess.run(
            [sys.executable, "scripts/build_chat_dataset.py", "--help"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0, result.stderr
        out = result.stdout
        assert "--math-reasoning" in out
        assert "--math-reasoning-multiplier" in out
        assert "--math-reasoning-val-frac" in out

    def test_default_path_in_script(self):
        src = (REPO_ROOT / "scripts" / "build_chat_dataset.py").read_text()
        assert 'default="data/raw/chat/math_reasoning.jsonl"' in src
