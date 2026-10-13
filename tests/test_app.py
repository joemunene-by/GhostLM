"""Tests for the GhostLM web app: prompt building, key facts and the HTTP API."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from ghostlm.app.engine import build_prompt, key_facts  # noqa: E402
from ghostlm.app.server import create_app  # noqa: E402

PASSAGES = [
    {"source": "cisa_kev", "ref": "CVE-2021-44228", "match": "id", "score": 1.0,
     "text": "Apache Log4j2 contains a JNDI flaw. It allows remote code execution. Patch now."},
    {"source": "primus", "ref": "doc_9", "match": "dense", "score": 0.6, "text": "Unrelated web text."},
]


def test_key_facts_quote_only_exact_matches():
    facts = key_facts(PASSAGES)
    assert facts == [{"cite": 1, "ref": "CVE-2021-44228",
                      "text": "Apache Log4j2 contains a JNDI flaw. It allows remote code execution."}]


def test_build_prompt_numbers_passages_and_keeps_recent_history():
    history = [{"user": f"q{i}", "assistant": f"a{i}"} for i in range(6)]
    prompt = build_prompt("What is it?", PASSAGES, history, passage_chars=40)
    assert "[1] (cisa_kev CVE-2021-44228)" in prompt and "[2] (primus doc_9)" in prompt
    assert "q1" not in prompt and "q5" in prompt
    assert prompt.endswith("Question: What is it?\nAnswer:")


class FakeEngine:
    models = {"ghost-small-gen": object()}

    def model_list(self):
        return [{"key": "ghost-small-gen", "label": "small", "available": True}]

    def stream_answer(self, question, model_key, history, use_retrieval=True):
        yield {"type": "tool", "name": "exact_lookup", "detail": "CVE-2021-44228"}
        yield {"type": "sources", "passages": PASSAGES[:1]}
        yield {"type": "facts", "facts": key_facts(PASSAGES)}
        yield {"type": "token", "text": "It is "}
        yield {"type": "token", "text": f"Log4Shell ({len(history)} prior turns)."}
        yield {"type": "done", "answer": f"It is Log4Shell ({len(history)} prior turns).", "model": model_key}


def _events(response):
    return [json.loads(line[6:]) for line in response.iter_lines() if line.startswith("data: ")]


def test_chat_streams_saves_and_continues(tmp_path):
    client = TestClient(create_app(engine=FakeEngine(), chat_dir=tmp_path, training_url="http://127.0.0.1:9/x"))
    assert client.get("/").status_code == 200
    with client.stream("POST", "/api/chat", json={"message": "What is CVE-2021-44228?"}) as r:
        events = _events(r)
    chat_id = events[0]["id"]
    assert [e["type"] for e in events] == ["chat", "tool", "sources", "facts", "token", "token", "done"]

    with client.stream("POST", "/api/chat", json={"message": "And the fix?", "chat_id": chat_id}) as r:
        done = _events(r)[-1]
    assert "(1 prior turns)" in done["answer"]

    chat = client.get(f"/api/chats/{chat_id}").json()
    assert len(chat["turns"]) == 2 and chat["turns"][0]["facts"][0]["ref"] == "CVE-2021-44228"
    assert [c["id"] for c in client.get("/api/chats").json()] == [chat_id]
    assert client.get("/api/training").json() == {"state": "offline"}

    client.delete(f"/api/chats/{chat_id}")
    assert client.get(f"/api/chats/{chat_id}").status_code == 404


def test_delete_others_keeps_the_named_chat_and_delete_all_clears_everything(tmp_path):
    client = TestClient(create_app(engine=FakeEngine(), chat_dir=tmp_path))
    ids = []
    for question in ("What is CVE-2021-44228?", "What is CWE-79?", "Explain T1059"):
        with client.stream("POST", "/api/chat", json={"message": question}) as r:
            ids.append(_events(r)[0]["id"])
    assert client.delete("/api/chats", params={"keep": ids[1]}).json() == {"ok": True, "removed": 2}
    assert [c["id"] for c in client.get("/api/chats").json()] == [ids[1]]
    assert client.delete("/api/chats", params={"keep": "../etc"}).status_code == 400
    assert client.delete("/api/chats").json()["removed"] == 1 and client.get("/api/chats").json() == []


def test_rejects_bad_input(tmp_path):
    client = TestClient(create_app(engine=FakeEngine(), chat_dir=tmp_path))
    assert client.post("/api/chat", json={"message": "  "}).status_code == 400
    assert client.post("/api/chat", json={"message": "hi", "model": "nope"}).status_code == 400
    assert client.get("/api/chats/..%2Fetc").status_code in (400, 404)


def test_small_talk_detection():
    from ghostlm.app.engine import is_small_talk
    assert is_small_talk("hey") and is_small_talk("Thanks a lot!") and is_small_talk("ok")
    assert not is_small_talk("How does cross-site scripting work?")
    assert not is_small_talk("CVE-2021-44228")


def test_best_sentences_picks_relevant_and_skips_questions():
    import numpy as np
    from ghostlm.app.engine import best_sentences
    passages = [{"ref": "a", "text": "What is XSS really about here? XSS lets attackers inject script into pages users view. "
                 "The weather was nice that day and nothing else happened at all."}]

    def embed(text, query=True):
        return np.array([1.0, 0.0]) if ("XSS" in text or "inject" in text) else np.array([0.0, 1.0])

    out = best_sentences("XSS", passages, embed)
    assert [f["text"] for f in out] == ["XSS lets attackers inject script into pages users view."]
    assert out[0]["cite"] == 1


JAVA = """explain this code (import java.util.Scanner; public class Library { static final int MAX_BOOKS = 10;
static int available = MAX_BOOKS; public static void main(String[] args) { Scanner input = new Scanner(System.in);
int choice; do { System.out.println("1. Borrow Book"); choice = input.nextInt(); switch (choice) { case 1: borrow();
break; case 2: System.out.println("Thank you for using the library system."); break; default:
System.out.println("Invalid choice."); } } while (choice != 2); } static void borrow() { if (available > 0) {
available--; System.out.println("Book borrowed successfully."); } else { System.out.println("None left."); } } })"""


def test_code_detection_ignores_ordinary_questions():
    from ghostlm.app import codeview
    assert codeview.looks_like_code(JAVA)
    assert codeview.looks_like_code("import os\ndef f(path):\n    data = open(path).read()\n    total = len(data)\n"
                                    "    for line in data.splitlines():\n        total += 1\n    return total\n")
    for question in ("How does SQL injection work?", "What is CVE-2021-44228?", "explain this code",
                     "Explain ATT&CK T1059.001 and how PowerShell is used; what mitigations exist for it in detail?"):
        assert not codeview.looks_like_code(question), question


def test_code_outline_reads_structure_and_each_function():
    from ghostlm.app import codeview
    text = "\n".join(codeview.outline(JAVA))
    assert "This is Java code defining `Library`." in text
    assert "It has 2 functions: `main` and `borrow`." in text
    assert "It imports `java.util.Scanner`." in text and "`the`" not in text
    assert '`borrow` checks `available > 0`, takes 1 from `available` and prints "Book borrowed successfully.".' in text
    assert "`main` reads input, loops, calls `borrow`" in text
    assert "`MAX_BOOKS` = `10`" in text and "a switch with 2 cases and a default" in text
    assert "no risky calls stand out" in text


def test_code_outline_flags_risks_and_hides_credentials():
    from ghostlm.app import codeview
    code = ("import subprocess\nfrom flask import request\nAPI_KEY = \"sk-abcdefghijklmnop\"\ndef ping():\n"
            "    host = request.args.get(\"host\")\n    if host:\n"
            "        return subprocess.run(f\"ping {host}\", shell=True)\n"
            "    cur.execute(\"SELECT * FROM users WHERE name = '\" + name + \"'\")\n")
    text = "\n".join(codeview.outline(code))
    assert "This is Python code." in text and "It imports `subprocess` and `flask`." in text
    assert "runs shell commands" in text and "SQL injection" in text and "hardcoded credential" in text
    assert "sk-abcdefghijklmnop" not in text and "value hidden" in text


def test_page_logo_matches_the_mascot_grid_and_hides_scrollbars():
    import re

    from ghostlm.app.server import STATIC
    from ghostlm.code import mascot
    page = (STATIC / "index.html").read_text()
    rows = re.findall(r'"([.#]{11})"', page.split("const GHOST = [")[1].split("];")[0])
    assert rows == ["".join("#" if cell else "." for cell in line) for line in mascot.grid()]
    assert "scrollbar-width: none" in page and "::-webkit-scrollbar { display: none; }" in page
