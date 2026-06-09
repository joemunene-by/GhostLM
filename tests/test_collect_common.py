"""Tests for the shared collector infrastructure (scripts/collect_common.py)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from collect_common import JsonlWriter, iter_jsonl


def test_jsonl_writer_standard_schema(tmp_path):
    path = tmp_path / "out.jsonl"
    with JsonlWriter(path, source="unit_test") as out:
        assert out.write(rec_id="A-1", text="alpha " * 10, title="Alpha")
    recs = list(iter_jsonl(path))
    assert len(recs) == 1
    assert recs[0]["id"] == "A-1"
    assert recs[0]["source"] == "unit_test"
    assert recs[0]["title"] == "Alpha"
    assert recs[0]["text"].startswith("alpha")


def test_jsonl_writer_skips_short_and_dupes(tmp_path):
    path = tmp_path / "out.jsonl"
    with JsonlWriter(path, source="unit_test", min_chars=20) as out:
        assert not out.write(rec_id="short", text="tiny")
        assert out.write(rec_id="ok-1", text="x" * 30)
        assert not out.write(rec_id="ok-2", text="x" * 30)  # exact dupe
    assert out.skipped_short == 1
    assert out.skipped_dupe == 1
    assert out.written == 1
    assert len(list(iter_jsonl(path))) == 1


def test_jsonl_writer_truncates_on_paragraph_boundary(tmp_path):
    path = tmp_path / "out.jsonl"
    text = "para one\n\npara two\n\n" + "para three " * 50
    with JsonlWriter(path, source="unit_test", max_chars=25) as out:
        out.write(rec_id="long", text=text)
    assert out.truncated == 1
    rec = next(iter_jsonl(path))
    assert rec["text"] == "para one\n\npara two"


def test_jsonl_writer_append_mode(tmp_path):
    path = tmp_path / "out.jsonl"
    with JsonlWriter(path, source="unit_test") as out:
        out.write(rec_id="r1", text="first record body")
    with JsonlWriter(path, source="unit_test", append=True) as out:
        out.write(rec_id="r2", text="second record body")
    ids = [r["id"] for r in iter_jsonl(path)]
    assert ids == ["r1", "r2"]


def test_failure_accounting(tmp_path):
    path = tmp_path / "out.jsonl"
    with JsonlWriter(path, source="unit_test") as out:
        out.count_failure()
        out.count_failure()
    assert out.failed == 2
    assert "Failed 2" in out.summary()
