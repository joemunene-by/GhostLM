"""Tests for the agent-distillation pipeline.

Covers:
  - OpenAICompatGenerator: request shape, role mapping, response
    extraction, error handling. Uses httpx.MockTransport so no
    network is needed.
  - distill_agent_traces.py: trace_to_bet1_text, trace_has_cite_tag,
    end-to-end via a stub teacher that produces a valid trace.
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ["GHOST_AGENT_OFFLINE"] = "1"

httpx = pytest.importorskip("httpx")

from ghostlm.agent import (  # noqa: E402
    AgentMessage,
    AgentTrace,
    GhostAgent,
    MessageRole,
    RuntimeConfig,
)
from ghostlm.agent.teacher import OpenAICompatGenerator  # noqa: E402
from scripts.distill_agent_traces import (  # noqa: E402
    trace_has_cite_tag,
    trace_to_bet1_text,
)


# ---------------------------------------------------------------------------
# OpenAICompatGenerator
# ---------------------------------------------------------------------------


def _mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    """Build an httpx.Client bound to a MockTransport with the given handler."""
    return httpx.Client(transport=httpx.MockTransport(handler))


class TestOpenAICompatGenerator:
    def test_builds_request_with_messages(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["body"] = json.loads(request.content)
            captured["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "ok"}}],
            })

        gen = OpenAICompatGenerator(
            base_url="http://localhost:11434/v1",
            api_key="ollama-test", model="qwen2.5:14b",
            client=_mock_client(handler),
        )
        history = [
            AgentMessage(role=MessageRole.SYSTEM, content="sys"),
            AgentMessage(role=MessageRole.USER, content="hi"),
        ]
        out = gen(history)
        assert out == "ok"
        assert captured["url"].endswith("/chat/completions")
        body = captured["body"]
        assert body["model"] == "qwen2.5:14b"
        assert [m["role"] for m in body["messages"]] == ["system", "user"]
        assert body["temperature"] == 0.6  # default
        assert captured["auth"] == "Bearer ollama-test"

    def test_tool_role_maps_to_user(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "x"}}],
            })

        gen = OpenAICompatGenerator(
            base_url="http://x/v1", api_key="", model="m",
            client=_mock_client(handler),
        )
        history = [
            AgentMessage(role=MessageRole.USER, content="q"),
            AgentMessage(role=MessageRole.ASSISTANT, content="a"),
            AgentMessage(role=MessageRole.TOOL,
                          content="<|tool_response|>{}<|/tool_response|>"),
        ]
        gen(history)
        roles = [m["role"] for m in captured["body"]["messages"]]
        assert roles == ["user", "assistant", "user"]

    def test_non_200_raises(self):
        def handler(request):
            return httpx.Response(500, text="server error")

        gen = OpenAICompatGenerator(
            base_url="http://x/v1", api_key="", model="m",
            client=_mock_client(handler),
        )
        with pytest.raises(RuntimeError, match="HTTP 500"):
            gen([AgentMessage(role=MessageRole.USER, content="q")])

    def test_malformed_response_raises(self):
        def handler(request):
            return httpx.Response(200, json={"unexpected": "shape"})

        gen = OpenAICompatGenerator(
            base_url="http://x/v1", api_key="", model="m",
            client=_mock_client(handler),
        )
        with pytest.raises(RuntimeError, match="unexpected teacher response"):
            gen([AgentMessage(role=MessageRole.USER, content="q")])

    def test_no_api_key_no_auth_header(self):
        captured = {}

        def handler(request):
            captured["auth"] = request.headers.get("authorization")
            return httpx.Response(200, json={
                "choices": [{"message": {"content": "x"}}]})

        gen = OpenAICompatGenerator(
            base_url="http://x/v1", api_key="", model="m",
            client=_mock_client(handler),
        )
        gen([AgentMessage(role=MessageRole.USER, content="q")])
        assert captured["auth"] is None


# ---------------------------------------------------------------------------
# trace_to_bet1_text
# ---------------------------------------------------------------------------


class TestTraceToBet1Text:
    def _build_trace(self, *contents) -> AgentTrace:
        roles = [MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT,
                 MessageRole.TOOL, MessageRole.ASSISTANT]
        t = AgentTrace(query="q")
        for content, role in zip(contents, roles):
            t.add(AgentMessage(role=role, content=content))
        return t

    def test_valid_four_message_trace(self):
        t = self._build_trace(
            "system",
            "What is CVE-2017-0144?",
            ('<|tool_call|>{"name": "search_cve_nvd", '
             '"args": {"q": "CVE-2017-0144"}}<|/tool_call|>'),
            '<|tool_response|>{"cve": "CVE-2017-0144"}<|/tool_response|>',
            'EternalBlue <|cite|>nvd:CVE-2017-0144<|/cite|>.',
        )
        text = trace_to_bet1_text(t)
        assert text is not None
        lines = text.strip().split("\n")
        assert lines[0].startswith("USER: ")
        assert lines[1].startswith("ASSISTANT: <|tool_call|>")
        assert lines[2].startswith("TOOL: <|tool_response|>")
        assert lines[3].startswith("ASSISTANT: ")
        assert "EternalBlue" in lines[3]

    def test_no_tool_call_returns_none(self):
        t = AgentTrace(query="q")
        t.add(AgentMessage(role=MessageRole.USER, content="hi"))
        t.add(AgentMessage(role=MessageRole.ASSISTANT,
                            content="plain answer no tool"))
        assert trace_to_bet1_text(t) is None

    def test_no_final_answer_returns_none(self):
        t = AgentTrace(query="q")
        t.add(AgentMessage(role=MessageRole.USER, content="hi"))
        t.add(AgentMessage(role=MessageRole.ASSISTANT,
                            content='<|tool_call|>{"name": "x", '
                                    '"args": {}}<|/tool_call|>'))
        t.add(AgentMessage(role=MessageRole.TOOL,
                            content="<|tool_response|>{}<|/tool_response|>"))
        # No final assistant turn.
        assert trace_to_bet1_text(t) is None


# ---------------------------------------------------------------------------
# trace_has_cite_tag
# ---------------------------------------------------------------------------


class TestTraceHasCiteTag:
    def test_present(self):
        t = AgentTrace(query="q")
        t.add(AgentMessage(role=MessageRole.ASSISTANT,
                            content="answer <|cite|>nvd:CVE-X<|/cite|>"))
        assert trace_has_cite_tag(t) is True

    def test_absent(self):
        t = AgentTrace(query="q")
        t.add(AgentMessage(role=MessageRole.ASSISTANT,
                            content="answer with no cite"))
        assert trace_has_cite_tag(t) is False

    def test_only_in_user_does_not_count(self):
        t = AgentTrace(query="q")
        t.add(AgentMessage(role=MessageRole.USER,
                            content="<|cite|>x:y<|/cite|>"))
        t.add(AgentMessage(role=MessageRole.ASSISTANT,
                            content="no cite"))
        assert trace_has_cite_tag(t) is False


# ---------------------------------------------------------------------------
# End-to-end: stub teacher -> GhostAgent -> bet-1 record
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_perfect_teacher_produces_distilled_record(self):
        """A teacher that always emits a perfect bet-1 + bet-9 trace
        through the GhostAgent runtime should produce a valid bet-1
        record convertable via trace_to_bet1_text."""

        # The handler counts calls so it can return tool-call on
        # round 1 and the cite-tagged answer on round 2.
        call_count = {"n": 0}

        def handler(request):
            call_count["n"] += 1
            n = call_count["n"]
            if n == 1:
                content = ('<|tool_call|>{"name": "search_cve_nvd", '
                           '"args": {"q": "CVE-2017-0144"}}<|/tool_call|>')
            else:
                content = ('CVE-2017-0144 is EternalBlue '
                           '<|cite|>nvd:CVE-2017-0144<|/cite|>.')
            return httpx.Response(200, json={
                "choices": [{"message": {"content": content}}],
            })

        teacher = OpenAICompatGenerator(
            base_url="http://stub/v1", api_key="", model="stub-teacher",
            client=_mock_client(handler),
        )
        agent = GhostAgent(teacher, RuntimeConfig(max_iters=4))
        trace = agent.run("What is CVE-2017-0144?")

        assert trace.terminated_reason == "answer_emitted"
        assert trace_has_cite_tag(trace)
        text = trace_to_bet1_text(trace)
        assert text is not None
        assert "<|tool_call|>" in text
        assert "<|cite|>" in text
        assert "EternalBlue" in text


# ---------------------------------------------------------------------------
# CLI subprocess: distill_agent_traces.py with stub teacher
# ---------------------------------------------------------------------------


class TestDistillCLI:
    def test_cli_runs_with_unreachable_teacher(self, tmp_path):
        """When the teacher is unreachable the script should report
        errors gracefully, not crash. We use a localhost port no
        teacher is listening on and assert the script exits cleanly
        (errors are logged per-prompt; the script returns 0)."""
        prompts_path = tmp_path / "prompts.jsonl"
        prompts_path.write_text(
            json.dumps({"prompt": "What is CVE-2017-0144?",
                         "seed_id": "p1"}) + "\n"
            + json.dumps({"prompt": "What is CWE-89?",
                            "seed_id": "p2"}) + "\n"
        )
        out_path = tmp_path / "out.jsonl"
        result = subprocess.run(
            [sys.executable, "scripts/distill_agent_traces.py",
             "--teacher-base-url", "http://127.0.0.1:1/v1",
             "--teacher-model", "fake",
             "--prompts", str(prompts_path),
             "--out", str(out_path),
             "--max-records", "2",
             "--offline"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        # Output JSONL exists but is empty (every prompt errored).
        assert out_path.exists()
        assert out_path.read_text().strip() == ""
        # Stdout reports the error count.
        assert "errors:" in result.stdout
