"""GhostLM's identity answers: intent matching, app routing, and the chat fine-tuning rows."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghostlm import identity  # noqa: E402


@pytest.mark.parametrize("question,kind", [
    ("Who built you?", "builder"),
    ("who made you", "builder"),
    ("Who is your creator?", "builder"),
    ("Are you ChatGPT?", "origin"),
    ("what model are you based on?", "origin"),
    ("how many parameters do you have?", "size"),
    ("What were you trained on?", "training"),
    ("what can you do?", "abilities"),
    ("What's your name?", "who"),
    ("tell me about yourself", "who"),
    ("who is Joe Munene?", "maker_work"),
    ("what else has Joe built?", "maker_work"),
    ("what is the ghostsuite?", "maker_work"),
])
def test_intents(question, kind):
    assert identity.intent(question) == kind


@pytest.mark.parametrize("question", [
    "What is SQL injection?", "Who discovered Log4Shell?", "how big is a TCP header?",
    "What can you tell me about CVE-2021-44228?", "who made the Mirai botnet?",
])
def test_ordinary_questions_are_not_identity(question):
    assert identity.intent(question) is None


def test_answers_name_the_builder_and_match_the_model():
    assert "Joe Munene" in identity.answer("builder")
    assert "349 million" in identity.answer("size", "ghost-base-live")
    assert "45 million" in identity.answer("size", "ghost-small-gen")


def test_sft_rows_are_well_formed_and_plain():
    rows = identity.sft_examples()
    assert len(rows) >= 25
    for r in rows:
        user, assistant = r["turns"]
        assert user["role"] == "user" and assistant["role"] == "assistant"
        assert identity.intent(user["content"]) is not None
        assert not any(c in assistant["content"] for c in "—–→~")


def test_app_answers_identity_without_loading_a_model(tmp_path):
    from ghostlm.app.engine import Engine
    eng = Engine(tmp_path, rag_dir=tmp_path)
    events = list(eng.stream_answer("Who built you?", "ghost-base-live", []))
    assert events[-1]["type"] == "done" and "Joe Munene" in events[-1]["answer"]
    assert eng._model is None
