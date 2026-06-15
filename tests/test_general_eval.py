"""Tests for the general-domain eval harness additions.

Covers the pure logic that de-cyber-frames the MCQ eval and normalizes
general benchmark answer keys, so the generalist pivot has a measurable,
non-cybersec ruler.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.fetch_general_mcq import _norm_answer
from scripts.eval_text_scoring import format_prompt


# ---------- _norm_answer ----------

def test_norm_answer_letters_passthrough():
    for letter in ("A", "B", "C", "D"):
        assert _norm_answer(letter) == letter


def test_norm_answer_numeric_remap():
    assert _norm_answer("1") == "A"
    assert _norm_answer("2") == "B"
    assert _norm_answer("3") == "C"
    assert _norm_answer("4") == "D"


def test_norm_answer_unmappable_is_empty():
    assert _norm_answer("E") == ""
    assert _norm_answer("") == ""
    assert _norm_answer(None) == ""


# ---------- format_prompt framing ----------

def _fake_tokenizer():
    """A tokenizer stub whose encode echoes the text, so we can assert on framing."""
    tok = MagicMock()
    tok.encode.side_effect = lambda s: [("TEXT", s)]
    return tok


_REC = {
    "question": "Which gas do plants primarily absorb for photosynthesis?",
    "choices": {"A": "Oxygen", "B": "Carbon dioxide", "C": "Nitrogen", "D": "Hydrogen"},
    "answer": "B",
}


def _rendered(record, **kw):
    tok = _fake_tokenizer()
    format_prompt(record, tok, chat_format=False, **kw)
    return tok.encode.call_args[0][0]


def test_cybersec_style_keeps_cyber_framing():
    body = _rendered(_REC, prompt_style="cybersec")
    assert "cybersecurity question" in body
    assert _REC["question"] in body


def test_general_style_drops_cyber_framing():
    body = _rendered(_REC, prompt_style="general")
    assert "cybersecurity" not in body
    assert "multiple-choice question" in body


def test_domain_general_record_forces_general_framing():
    rec = dict(_REC, domain="general")
    # Even with the default cybersec style, a general-domain record is reframed.
    body = _rendered(rec, prompt_style="cybersec")
    assert "cybersecurity" not in body


def test_prompt_ends_with_answer_cue():
    body = _rendered(_REC, prompt_style="general")
    assert body.rstrip().endswith("Answer:")
