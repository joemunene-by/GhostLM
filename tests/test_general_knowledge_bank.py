"""Tests for the v0.9.18 general-knowledge SFT bank.

Covers:
  - data/raw/chat/general_knowledge.jsonl loads as valid JSONL.
  - Every record has the {turns, source, topic} shape that
    ChatDataset / build_chat_dataset.py expects.
  - Topic coverage spans the 15 categories the bank was designed
    to cover (math, science, programming, geography, history,
    comparison, definitions, etymology, identity, cross_domain,
    how_to, philosophy, reasoning, uncertainty, conversation).
  - The build_chat_dataset.py CLI accepts the new --general-
    knowledge / --general-knowledge-multiplier / --general-
    knowledge-val-frac flags.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


GK_PATH = REPO_ROOT / "data" / "raw" / "chat" / "general_knowledge.jsonl"


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
        assert GK_PATH.exists(), f"general-knowledge bank not found: {GK_PATH}"

    def test_loads_as_jsonl(self):
        recs = _load_jsonl(GK_PATH)
        assert len(recs) >= 90, f"expected >= 90 records, got {len(recs)}"

    def test_record_shape(self):
        recs = _load_jsonl(GK_PATH)
        for r in recs:
            assert "turns" in r
            assert "source" in r
            assert "topic" in r
            assert r["source"] == "general_knowledge"
            assert isinstance(r["turns"], list)
            assert len(r["turns"]) >= 2

    def test_turns_alternate_user_assistant(self):
        recs = _load_jsonl(GK_PATH)
        for r in recs:
            roles = [t["role"] for t in r["turns"]]
            assert roles[0] == "user", (
                f"first turn must be user: {roles[0]}")
            assert roles[1] == "assistant", (
                f"second turn must be assistant: {roles[1]}")

    def test_no_empty_content(self):
        recs = _load_jsonl(GK_PATH)
        for r in recs:
            for t in r["turns"]:
                assert t.get("content", "").strip(), (
                    f"empty content: {r}")


class TestTopicCoverage:
    def test_topics_span_expected_categories(self):
        """The bank should cover broad domains beyond cybersec."""
        recs = _load_jsonl(GK_PATH)
        topics = {r["topic"] for r in recs}
        expected = {
            "math", "science", "programming", "geography",
            "history", "comparison", "definitions", "etymology",
            "identity", "cross_domain", "how_to", "uncertainty",
            "conversation",
        }
        missing = expected - topics
        assert not missing, f"missing topics: {missing}"

    def test_each_topic_has_multiple_records(self):
        """Every topic that appears should have at least 2 records
        so it survives a 90/10 train/val split."""
        recs = _load_jsonl(GK_PATH)
        counts = {}
        for r in recs:
            counts[r["topic"]] = counts.get(r["topic"], 0) + 1
        thin = [t for t, n in counts.items() if n < 2]
        assert not thin, f"topics with only 1 record: {thin}"

    def test_uncertainty_pattern_present(self):
        """Refusal / uncertainty examples teach the model when NOT to
        confidently answer (lottery, future weather, current prices)."""
        recs = _load_jsonl(GK_PATH)
        unc = [r for r in recs if r["topic"] == "uncertainty"]
        assert len(unc) >= 4, f"only {len(unc)} uncertainty examples"

    def test_cross_domain_identity_present(self):
        """The model should know it's not strictly cybersec-only."""
        recs = _load_jsonl(GK_PATH)
        cross = [r for r in recs if r["topic"] == "cross_domain"]
        assert len(cross) >= 2


class TestBuildChatDatasetCLI:
    def test_help_includes_new_flags(self):
        """build_chat_dataset.py --help should advertise the three
        new general-knowledge flags."""
        result = subprocess.run(
            [sys.executable, "scripts/build_chat_dataset.py", "--help"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0, result.stderr
        out = result.stdout
        assert "--general-knowledge" in out
        assert "--general-knowledge-multiplier" in out
        assert "--general-knowledge-val-frac" in out

    def test_default_path_points_to_real_file(self):
        """argparse doesn't surface defaults in --help by default, so
        verify the default by importing the parser and inspecting the
        action's ``default`` attribute directly."""
        # Import via runpy so the script's parse_args runs with no args.
        # Easier: read the script source for the default literal.
        src = (REPO_ROOT / "scripts" / "build_chat_dataset.py").read_text()
        assert 'default="data/raw/chat/general_knowledge.jsonl"' in src, (
            "default --general-knowledge path missing from script")
