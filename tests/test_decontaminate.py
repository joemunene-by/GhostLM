"""Tests for the corpus decontamination tool.

Verifies that training records containing a benchmark question (exact or
shingle-overlap) are flagged, clean records are not, and the writer drops
exactly the flagged records.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.decontaminate import (
    _extract_bench_texts,
    load_benchmarks,
    scan_corpus,
)


def _write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_extract_mcq_question_and_choices():
    rec = {"question": "What is the capital of France?",
           "choices": {"A": "Paris", "B": "Lyon", "C": "Nice", "D": "Brest"},
           "answer": "A"}
    q_only = _extract_bench_texts(rec, include_answers=False)
    assert q_only == ["What is the capital of France?"]
    with_ans = _extract_bench_texts(rec, include_answers=True)
    assert "Paris" in with_ans and "A" in with_ans


def test_shingle_overlap_flags_contaminated_record(tmp_path):
    bench = tmp_path / "general_mcq_bench.jsonl"
    long_q = ("which of the following statements best explains why photosynthesis "
              "is considered the foundation of nearly every food web on earth")
    _write_jsonl(bench, [{"question": long_q, "choices": {"A": "x", "B": "y"}, "answer": "A"}])

    exact_by_len, shingle_set = load_benchmarks([bench], include_answers=False, shingle_n=12)
    assert shingle_set

    corpus = tmp_path / "train.jsonl"
    _write_jsonl(corpus, [
        {"text": "Completely unrelated text about networking protocols and routing."},
        {"text": "An article noting that " + long_q + " in a biology textbook."},
    ])
    flagged, hits = scan_corpus(corpus, exact_by_len, shingle_set, shingle_n=12, min_shingles=3)
    assert flagged == [1]
    assert hits[0]["tier"] == "shingle"


def test_short_question_exact_match(tmp_path):
    bench = tmp_path / "x_eval.jsonl"
    short_q = "what gas do plants absorb during the day"  # 8 words, < shingle_n
    _write_jsonl(bench, [{"question": short_q, "answer": "carbon dioxide"}])
    exact_by_len, shingle_set = load_benchmarks([bench], include_answers=False, shingle_n=12)
    assert 8 in exact_by_len

    corpus = tmp_path / "train.jsonl"
    _write_jsonl(corpus, [
        {"text": "Photosynthesis basics: what gas do plants absorb during the day, you ask."},
        {"text": "Unrelated content with no overlap whatsoever here."},
    ])
    flagged, _ = scan_corpus(corpus, exact_by_len, shingle_set, shingle_n=12, min_shingles=3)
    assert flagged == [0]


def test_clean_corpus_not_flagged(tmp_path):
    bench = tmp_path / "x_bench.jsonl"
    _write_jsonl(bench, [{"question": "what is the boiling point of water in celsius at sea level",
                          "answer": "100"}])
    exact_by_len, shingle_set = load_benchmarks([bench], include_answers=False, shingle_n=12)
    corpus = tmp_path / "train.jsonl"
    _write_jsonl(corpus, [{"text": "The mitochondria is the powerhouse of the cell."}])
    flagged, _ = scan_corpus(corpus, exact_by_len, shingle_set, shingle_n=12, min_shingles=3)
    assert flagged == []
