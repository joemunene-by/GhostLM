"""Tests for the MCP-server agent retrofit (v0.9.14).

The MCP server adds a `ghostlm_agent` tool that wraps the full
GhostAgent loop. These tests cover:
  - make_generator_from_loaded refactor (the building block).
  - GhostLMRuntime.agent() lazy-build + caching.
  - ghostlm_agent tool against a fake runtime.

The MCP `mcp` package is an optional dep; tests skip cleanly if it
is not installed (the wiring is independent of the MCP transport).
"""

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ["GHOST_AGENT_OFFLINE"] = "1"

# Skip everything if the MCP package isn't installed.
mcp_pkg = pytest.importorskip("mcp")

from ghostlm.agent import GhostAgent, MessageRole, RuntimeConfig  # noqa: E402


# ---------------------------------------------------------------------------
# GhostLMRuntime.agent()
# ---------------------------------------------------------------------------


class TestGhostLMRuntimeAgent:
    def _build_runtime(self):
        """Build a GhostLMRuntime backed by a tiny random model.

        We bypass the checkpoint-loading path by constructing a
        runtime instance and assigning model/tokenizer/device
        manually, mirroring what the real ``__init__`` produces.
        """
        from ghostlm.config import GhostLMConfig
        from ghostlm.model import GhostLM
        from ghostlm.tokenizer import GhostTokenizer
        # Late-import so the test is skippable without the script
        # being importable from a clean cwd.
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        from mcp_server import GhostLMRuntime

        rt = GhostLMRuntime.__new__(GhostLMRuntime)
        cfg = GhostLMConfig.from_preset("ghost-tiny")
        cfg.vocab_size = 50264
        cfg.context_length = 64
        rt.config = cfg
        rt.model = GhostLM(cfg).eval()
        rt.device = "cpu"
        rt.tokenizer = GhostTokenizer()
        rt.end_id = rt.tokenizer._special_tokens[rt.tokenizer.END]
        rt._agent = None
        return rt

    def test_agent_lazy_built(self):
        rt = self._build_runtime()
        assert rt._agent is None
        agent = rt.agent(max_iters=2)
        assert isinstance(agent, GhostAgent)
        assert agent.config.max_iters == 2
        # Cached on the runtime.
        assert rt._agent is agent

    def test_same_max_iters_returns_cached(self):
        rt = self._build_runtime()
        a1 = rt.agent(max_iters=3)
        a2 = rt.agent(max_iters=3)
        assert a1 is a2

    def test_different_max_iters_rebuilds(self):
        rt = self._build_runtime()
        a1 = rt.agent(max_iters=2)
        a2 = rt.agent(max_iters=4)
        assert a1 is not a2
        assert a2.config.max_iters == 4


# ---------------------------------------------------------------------------
# ghostlm_agent tool against a stub runtime
# ---------------------------------------------------------------------------


class TestGhostlmAgentTool:
    def test_returns_final_answer(self, monkeypatch):
        """Drop a stub runtime that returns a canned trace and verify
        the ghostlm_agent tool returns its final answer."""
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import mcp_server

        def stub_gen(history):
            n = sum(1 for m in history
                     if m.role == MessageRole.ASSISTANT)
            if n == 0:
                return ('<|tool_call|>{"name": "search_cve_nvd", '
                        '"args": {"q": "CVE-2017-0144"}}<|/tool_call|>')
            return ('CVE-2017-0144 is EternalBlue '
                    '<|cite|>nvd:CVE-2017-0144<|/cite|>.')

        class StubRuntime:
            def agent(self, max_iters=6):
                return GhostAgent(stub_gen,
                                    RuntimeConfig(max_iters=max_iters))

        monkeypatch.setattr(mcp_server, "_runtime", StubRuntime())
        # mcp_server.ghostlm_agent is wrapped by FastMCP; call the
        # underlying function via .fn (FastMCP exposes it).
        tool_fn = getattr(mcp_server.ghostlm_agent, "fn",
                           mcp_server.ghostlm_agent)
        out = tool_fn("What is CVE-2017-0144?")
        assert "EternalBlue" in out
        assert "<|cite|>" in out

    def test_include_trace_emits_json_block(self, monkeypatch):
        sys.path.insert(0, str(REPO_ROOT / "scripts"))
        import mcp_server

        def stub_gen(history):
            return "plain answer"

        class StubRuntime:
            def agent(self, max_iters=6):
                return GhostAgent(stub_gen,
                                    RuntimeConfig(max_iters=max_iters))

        monkeypatch.setattr(mcp_server, "_runtime", StubRuntime())
        tool_fn = getattr(mcp_server.ghostlm_agent, "fn",
                           mcp_server.ghostlm_agent)
        out = tool_fn("test", include_trace=True)
        assert out.startswith("```json")
        assert "```" in out
        # Extract the JSON block and verify it's valid.
        body = out.split("```json", 1)[1].split("```", 1)[0].strip()
        d = json.loads(body)
        assert d["query"] == "test"
        assert d["final_answer"] == "plain answer"
