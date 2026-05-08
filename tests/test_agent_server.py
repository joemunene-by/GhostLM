"""Tests for the GhostAgent HTTP server (multi-vendor compatible).

Covers OpenAI, Anthropic, Gemini, Ollama endpoint families plus the
native /v1/agent/run and introspection routes. Uses FastAPI
TestClient with a stub generator, so no checkpoint required.
"""

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ["GHOST_AGENT_OFFLINE"] = "1"

# Skip the whole module if FastAPI isn't installed (it's an optional dep).
fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from ghostlm.agent import (  # noqa: E402
    GhostAgent,
    MessageRole,
    RuntimeConfig,
)
from ghostlm.agent.server import create_app  # noqa: E402


def _two_step_gen(history):
    """Round 1: tool call. Round 2: cite-tagged answer."""
    n = sum(1 for m in history if m.role == MessageRole.ASSISTANT)
    if n == 0:
        return ('<|tool_call|>{"name": "search_cve_nvd", '
                '"args": {"q": "CVE-2017-0144"}}<|/tool_call|>')
    return ('CVE-2017-0144 is EternalBlue '
            '<|cite|>nvd:CVE-2017-0144<|/cite|>.')


def _plain_gen(history):
    """Always emit a plain answer with no tool calls."""
    return "plain answer"


@pytest.fixture
def two_step_client():
    app = create_app(_two_step_gen,
                      RuntimeConfig(max_iters=4),
                      model_name="ghostlm-stub")
    return TestClient(app)


@pytest.fixture
def plain_client():
    app = create_app(_plain_gen,
                      RuntimeConfig(max_iters=2),
                      model_name="ghostlm-plain")
    return TestClient(app)


# ---------------------------------------------------------------------------
# Health + introspection
# ---------------------------------------------------------------------------


class TestStaticUI:
    def test_index_serves_html(self, two_step_client):
        r = two_step_client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")
        body = r.text
        assert "<html" in body and "</html>" in body
        assert "<form" in body
        assert "/v1/agent/run" in body  # JS calls into the native API

    def test_index_lists_tool_examples(self, two_step_client):
        r = two_step_client.get("/")
        # The UI ships canned example queries; spot-check one.
        assert "CVE-2017-0144" in r.text


class TestIntrospection:
    def test_healthz(self, two_step_client):
        r = two_step_client.get("/healthz")
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "ok"
        assert d["model"] == "ghostlm-stub"
        assert "search_cve_nvd" in d["tools"]

    def test_v1_models(self, two_step_client):
        r = two_step_client.get("/v1/models")
        assert r.status_code == 200
        d = r.json()
        assert d["object"] == "list"
        assert d["data"][0]["id"] == "ghostlm-stub"

    def test_api_tags_ollama(self, two_step_client):
        r = two_step_client.get("/api/tags")
        assert r.status_code == 200
        d = r.json()
        assert d["models"][0]["name"] == "ghostlm-stub"


# ---------------------------------------------------------------------------
# Native /v1/agent/run
# ---------------------------------------------------------------------------


class TestAgentRun:
    def test_returns_trace_and_metadata(self, two_step_client):
        r = two_step_client.post("/v1/agent/run",
                                   json={"query": "What is CVE-2017-0144?"})
        assert r.status_code == 200
        d = r.json()
        assert d["model"] == "ghostlm-stub"
        assert d["terminated_reason"] == "answer_emitted"
        assert d["iterations"] == 2
        assert "EternalBlue" in d["final_answer"]
        assert "trace" in d
        assert d["trace"]["query"] == "What is CVE-2017-0144?"

    def test_max_iters_override(self, two_step_client):
        r = two_step_client.post("/v1/agent/run",
                                   json={"query": "loop test",
                                         "max_iters": 1})
        assert r.status_code == 200
        # max_iters=1 forces termination after the first assistant
        # message, even though _two_step_gen would have continued.
        assert r.json()["iterations"] == 1

    def test_include_trace_false(self, two_step_client):
        r = two_step_client.post("/v1/agent/run",
                                   json={"query": "x",
                                         "include_trace": False})
        assert r.status_code == 200
        assert "trace" not in r.json()


# ---------------------------------------------------------------------------
# OpenAI Chat Completions
# ---------------------------------------------------------------------------


class TestOpenAI:
    def test_basic_completion(self, two_step_client):
        r = two_step_client.post("/v1/chat/completions", json={
            "model": "ghostlm-stub",
            "messages": [{"role": "user",
                           "content": "What is CVE-2017-0144?"}],
        })
        assert r.status_code == 200
        d = r.json()
        assert d["object"] == "chat.completion"
        assert d["choices"][0]["finish_reason"] == "stop"
        assert d["choices"][0]["message"]["role"] == "assistant"
        assert "EternalBlue" in d["choices"][0]["message"]["content"]

    def test_tool_calls_surfaced(self, two_step_client):
        r = two_step_client.post("/v1/chat/completions", json={
            "model": "ghostlm-stub",
            "messages": [{"role": "user", "content": "lookup"}],
        })
        msg = r.json()["choices"][0]["message"]
        assert "tool_calls" in msg
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "search_cve_nvd"
        # Arguments are a JSON string per OpenAI spec.
        args = json.loads(tc["function"]["arguments"])
        assert args["q"] == "CVE-2017-0144"

    def test_no_tool_calls_when_plain_answer(self, plain_client):
        r = plain_client.post("/v1/chat/completions", json={
            "model": "ghostlm-plain",
            "messages": [{"role": "user", "content": "hi"}],
        })
        msg = r.json()["choices"][0]["message"]
        assert msg.get("tool_calls") is None
        assert msg["content"] == "plain answer"

    def test_no_user_message_400(self, two_step_client):
        r = two_step_client.post("/v1/chat/completions", json={
            "messages": [{"role": "assistant", "content": "x"}],
        })
        assert r.status_code == 400

    def test_streaming_yields_done(self, two_step_client):
        r = two_step_client.post("/v1/chat/completions", json={
            "model": "ghostlm-stub",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        assert r.status_code == 200
        chunks = [ln for ln in r.text.split("\n") if ln.strip()]
        assert "data: [DONE]" in chunks[-1]
        # First chunk is the role-assistant delta.
        first = json.loads(chunks[0][len("data: "):])
        assert first["choices"][0]["delta"].get("role") == "assistant"


# ---------------------------------------------------------------------------
# Anthropic Messages
# ---------------------------------------------------------------------------


class TestAnthropic:
    def test_string_content(self, two_step_client):
        r = two_step_client.post("/v1/messages", json={
            "model": "claude-shim", "max_tokens": 256,
            "messages": [{"role": "user",
                           "content": "What is CVE-2017-0144?"}],
        })
        assert r.status_code == 200
        d = r.json()
        assert d["type"] == "message"
        assert d["role"] == "assistant"
        assert d["model"] == "claude-shim"
        assert d["stop_reason"] == "end_turn"

    def test_content_block_format(self, two_step_client):
        """Anthropic clients can send content as a list of typed blocks."""
        r = two_step_client.post("/v1/messages", json={
            "model": "claude-shim", "max_tokens": 256,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "What is CVE-2017-0144?"},
            ]}],
        })
        assert r.status_code == 200
        assert r.json()["stop_reason"] == "end_turn"

    def test_tool_use_blocks(self, two_step_client):
        r = two_step_client.post("/v1/messages", json={
            "model": "claude-shim", "max_tokens": 256,
            "messages": [{"role": "user", "content": "lookup"}],
        })
        d = r.json()
        tool_uses = [b for b in d["content"] if b.get("type") == "tool_use"]
        assert len(tool_uses) == 1
        assert tool_uses[0]["name"] == "search_cve_nvd"
        assert tool_uses[0]["input"] == {"q": "CVE-2017-0144"}

    def test_text_block_present(self, two_step_client):
        r = two_step_client.post("/v1/messages", json={
            "model": "claude-shim", "max_tokens": 256,
            "messages": [{"role": "user", "content": "lookup"}],
        })
        d = r.json()
        text_blocks = [b for b in d["content"] if b.get("type") == "text"]
        assert len(text_blocks) >= 1
        assert "EternalBlue" in text_blocks[-1]["text"]

    def test_no_user_message_400(self, two_step_client):
        r = two_step_client.post("/v1/messages", json={
            "max_tokens": 256,
            "messages": [{"role": "assistant", "content": "x"}],
        })
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------


class TestGemini:
    def test_basic_generate_content(self, two_step_client):
        r = two_step_client.post(
            "/v1beta/models/gemini-pro:generateContent", json={
                "contents": [{"role": "user", "parts": [
                    {"text": "What is CVE-2017-0144?"},
                ]}],
            })
        assert r.status_code == 200
        d = r.json()
        assert d["candidates"][0]["finishReason"] == "STOP"
        assert (d["candidates"][0]["content"]["role"] == "model")
        text = d["candidates"][0]["content"]["parts"][0]["text"]
        assert "EternalBlue" in text

    def test_no_user_content_400(self, two_step_client):
        r = two_step_client.post(
            "/v1beta/models/gemini-pro:generateContent", json={
                "contents": [{"role": "model", "parts": [{"text": "x"}]}],
            })
        assert r.status_code == 400

    def test_usage_metadata(self, two_step_client):
        r = two_step_client.post(
            "/v1beta/models/gemini-pro:generateContent", json={
                "contents": [{"role": "user",
                               "parts": [{"text": "hi"}]}],
            })
        usage = r.json()["usageMetadata"]
        assert usage["promptTokenCount"] >= 1
        assert "totalTokenCount" in usage


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------


class TestOllama:
    def test_chat(self, two_step_client):
        r = two_step_client.post("/api/chat", json={
            "model": "ghostlm-stub",
            "messages": [{"role": "user",
                           "content": "What is CVE-2017-0144?"}],
        })
        assert r.status_code == 200
        d = r.json()
        assert d["done"] is True
        assert d["done_reason"] == "stop"
        assert d["message"]["role"] == "assistant"
        assert "EternalBlue" in d["message"]["content"]

    def test_generate(self, two_step_client):
        r = two_step_client.post("/api/generate", json={
            "model": "ghostlm-stub",
            "prompt": "What is CVE-2017-0144?",
        })
        assert r.status_code == 200
        d = r.json()
        assert d["done"] is True
        assert "EternalBlue" in d["response"]

    def test_chat_no_user_400(self, two_step_client):
        r = two_step_client.post("/api/chat", json={
            "messages": [{"role": "assistant", "content": "x"}],
        })
        assert r.status_code == 400
