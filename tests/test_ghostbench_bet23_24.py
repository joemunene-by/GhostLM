"""Tests that bets 23 + 24 are wired into Suite.from_dir discovery."""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ghostbench.bench import Suite
from ghostbench.parsers import DEFAULT_PARSERS


def test_default_mapping_includes_code_explain_and_write():
    """Suite.from_dir's default mapping must know about bet 23 + 24."""
    src = (REPO_ROOT / "ghostbench" / "bench.py").read_text()
    assert '"code_explain_eval.jsonl": "bet23_code_explain"' in src
    assert '"code_write_eval.jsonl": "bet24_code_write"' in src


def test_descriptions_include_bet23_and_bet24():
    src = (REPO_ROOT / "ghostbench" / "bench.py").read_text()
    assert '"bet23_code_explain":' in src
    assert '"bet24_code_write":' in src


def test_suite_discovers_bet23_24_when_eval_files_present(tmp_path):
    """Build a tiny eval dir with both files; Suite.from_dir picks them up."""
    rec = {"prompt": "what does it do", "required_substrings": ["sort"]}
    (tmp_path / "code_explain_eval.jsonl").write_text(json.dumps(rec) + "\n")
    (tmp_path / "code_write_eval.jsonl").write_text(json.dumps(rec) + "\n")
    suite = Suite.from_dir(tmp_path, parsers=DEFAULT_PARSERS)
    names = {b.name for b in suite}
    assert "bet23_code_explain" in names
    assert "bet24_code_write" in names
