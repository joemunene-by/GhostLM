"""Tests for the v0.9.20 programming Q&A SFT bank.

Bank shape, language coverage, topic coverage, and CLI wiring.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


PQ_PATH = REPO_ROOT / "data" / "raw" / "chat" / "programming_qa.jsonl"


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
        assert PQ_PATH.exists()

    def test_loads_as_jsonl(self):
        recs = _load_jsonl(PQ_PATH)
        assert len(recs) >= 60

    def test_record_shape(self):
        recs = _load_jsonl(PQ_PATH)
        for r in recs:
            assert r.get("source") == "programming_qa"
            assert "topic" in r
            assert isinstance(r.get("turns"), list)
            assert len(r["turns"]) >= 2
            assert r["turns"][0]["role"] == "user"
            assert r["turns"][1]["role"] == "assistant"
            assert r["turns"][0]["content"].strip()
            assert r["turns"][1]["content"].strip()


class TestTopicCoverage:
    def test_topic_categories(self):
        """The bank should cover programming basics across multiple
        languages plus how-to / debug / refactor / concepts."""
        recs = _load_jsonl(PQ_PATH)
        topics = {r["topic"] for r in recs}
        # Categories the bank was designed to cover.
        for t in ("python_basics", "javascript_basics", "go_basics",
                   "rust_basics", "code_explain", "debug_help",
                   "refactor", "concepts", "tooling"):
            assert t in topics, f"missing topic: {t}"

    def test_python_well_represented(self):
        """Python should have the most records (it's the project's
        primary language)."""
        recs = _load_jsonl(PQ_PATH)
        py = [r for r in recs if r["topic"] == "python_basics"]
        assert len(py) >= 15

    def test_multiple_languages(self):
        """The bank should mention Python, Go, Rust, JavaScript,
        Java explicitly somewhere in the content."""
        recs = _load_jsonl(PQ_PATH)
        text = " ".join(t["content"] for r in recs
                          for t in r["turns"]).lower()
        for lang in ("python", "javascript", "go", "rust", "java"):
            assert lang in text, f"language missing from bank: {lang}"


class TestBuildChatDatasetCLI:
    def test_help_includes_new_flags(self):
        result = subprocess.run(
            [sys.executable, "scripts/build_chat_dataset.py", "--help"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0, result.stderr
        out = result.stdout
        assert "--programming-qa" in out
        assert "--programming-qa-multiplier" in out
        assert "--programming-qa-val-frac" in out

    def test_default_path_in_script(self):
        src = (REPO_ROOT / "scripts" / "build_chat_dataset.py").read_text()
        assert 'default="data/raw/chat/programming_qa.jsonl"' in src
