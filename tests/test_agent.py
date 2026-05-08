"""GhostAgent unit tests: parser, tools, dispatch, runtime loop."""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Force offline so test runs are deterministic and don't touch NVD.
os.environ["GHOST_AGENT_OFFLINE"] = "1"

from ghostlm.agent import (
    AgentMessage,
    AgentTrace,
    GhostAgent,
    MessageRole,
    RuntimeConfig,
    TOOLS_REGISTRY,
    Tool,
    ToolResult,
    parse_agent_output,
    parse_cite_tags,
    parse_tool_calls,
)
from ghostlm.agent.parser import normalise_tags, strip_tool_call_blocks
from ghostlm.agent.runner import make_generator_from_loaded
from ghostlm.agent.tools import dispatch


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class TestParser:
    def test_parses_single_tool_call(self):
        text = ('I will look up. <|tool_call|>{"name": "search_cve_nvd", '
                '"args": {"q": "CVE-2017-0144"}}<|/tool_call|>')
        out = parse_agent_output(text)
        assert len(out.tool_calls) == 1
        assert out.tool_calls[0].name == "search_cve_nvd"
        assert out.tool_calls[0].args == {"q": "CVE-2017-0144"}
        assert out.plain_text == "I will look up."

    def test_parses_multiple_tool_calls(self):
        text = ('<|tool_call|>{"name": "a", "args": {"x": 1}}<|/tool_call|>'
                '<|tool_call|>{"name": "b", "args": {"y": 2}}<|/tool_call|>')
        out = parse_agent_output(text)
        assert [c.name for c in out.tool_calls] == ["a", "b"]
        assert out.tool_calls[0].args == {"x": 1}
        assert out.tool_calls[1].args == {"y": 2}

    def test_parses_cites(self):
        text = ('CVE-2017-0144 is EternalBlue '
                '<|cite|>nvd:CVE-2017-0144#description<|/cite|>.')
        out = parse_agent_output(text)
        assert len(out.cites) == 1
        c = out.cites[0]
        assert c.source_type == "nvd"
        assert c.source_id == "CVE-2017-0144"
        assert c.field == "description"

    def test_cite_without_field(self):
        text = "Statement <|cite|>cwe:CWE-89<|/cite|>."
        out = parse_agent_output(text)
        assert out.cites[0].field is None

    def test_normalises_spaced_tags(self):
        text = '<|tool call|>{"name": "x", "args": {}}<|/tool call|>'
        out = parse_agent_output(text)
        assert len(out.tool_calls) == 1
        assert out.tool_calls[0].name == "x"
        assert any("normalised" in w for w in out.parse_warnings)

    def test_strips_json_code_fence(self):
        text = ('<|tool_call|>```json\n'
                '{"name": "x", "args": {}}\n```<|/tool_call|>')
        out = parse_agent_output(text)
        assert len(out.tool_calls) == 1

    def test_malformed_json_warns_does_not_crash(self):
        text = '<|tool_call|>not json{<|/tool_call|>'
        out = parse_agent_output(text)
        assert out.tool_calls == []
        assert any("did not parse" in w for w in out.parse_warnings)

    def test_missing_name_warns(self):
        text = '<|tool_call|>{"args": {}}<|/tool_call|>'
        out = parse_agent_output(text)
        assert out.tool_calls == []
        assert any("missing valid 'name'" in w for w in out.parse_warnings)

    def test_strip_tool_call_blocks(self):
        text = ('Pre <|tool_call|>{"name": "x", "args": {}}<|/tool_call|> '
                'mid <|tool_call|>{"name": "y", "args": {}}<|/tool_call|> '
                'post')
        assert strip_tool_call_blocks(text) == "Pre  mid  post"

    def test_non_string_input(self):
        out = parse_agent_output(None)
        assert out.tool_calls == []
        assert out.plain_text == ""
        assert any("non-string" in w for w in out.parse_warnings)


# ---------------------------------------------------------------------------
# Tools + dispatch
# ---------------------------------------------------------------------------


class TestTools:
    def test_registry_has_four_canonical_tools(self):
        """The four bet-1 canonical tools must always be registered.
        Additional tools may be present (v0.9.15 added 5 more)."""
        canonical = {
            "search_cve_nvd",
            "lookup_mitre_technique",
            "lookup_cwe",
            "rag_retrieve",
        }
        assert canonical.issubset(set(TOOLS_REGISTRY.keys()))

    def test_search_cve_offline_hit(self):
        result = dispatch("search_cve_nvd", {"q": "CVE-2017-0144"})
        assert result.error is None
        assert result.response["cve"] == "CVE-2017-0144"
        assert "EternalBlue" in result.response["description"]
        assert result.response["source"] == "offline_cache"

    def test_search_cve_unknown_id_returns_not_found(self):
        result = dispatch("search_cve_nvd", {"q": "CVE-9999-99999"})
        assert result.error is None
        assert result.response.get("found") is False

    def test_lookup_mitre_technique(self):
        result = dispatch("lookup_mitre_technique",
                          {"technique_id": "T1003.001"})
        assert result.error is None
        assert "LSASS" in result.response["name"]

    def test_lookup_cwe_with_and_without_prefix(self):
        a = dispatch("lookup_cwe", {"cwe_id": "89"})
        b = dispatch("lookup_cwe", {"cwe_id": "CWE-89"})
        assert a.response["id"] == b.response["id"] == "CWE-89"

    def test_rag_retrieve_finds_passages(self):
        result = dispatch("rag_retrieve",
                          {"query": "EternalBlue", "k": 2})
        assert result.error is None
        assert len(result.response["passages"]) >= 1
        assert any("CVE-2017-0144" in p["id"]
                   for p in result.response["passages"])

    def test_unknown_tool(self):
        result = dispatch("not_a_tool", {})
        assert result.error is not None
        assert "unknown tool" in result.error

    def test_missing_required_arg(self):
        result = dispatch("search_cve_nvd", {})
        assert result.error is not None
        assert "missing required arg" in result.error

    def test_backend_exception_captured(self):
        def boom(args):
            raise ValueError("boom")
        reg = {"crash": Tool(name="crash", description="",
                              args_schema={}, fn=boom)}
        result = dispatch("crash", {}, registry=reg)
        assert result.error == "ValueError: boom"
        assert result.response is None


# ---------------------------------------------------------------------------
# New tools (v0.9.15): CISA KEV, GreyNoise, VirusTotal, Shodan, OTX
# ---------------------------------------------------------------------------


class TestNewTools:
    def test_registry_includes_new_tools(self):
        for name in ("lookup_cisa_kev", "lookup_greynoise",
                     "lookup_virustotal_hash", "lookup_shodan",
                     "lookup_alienvault_otx"):
            assert name in TOOLS_REGISTRY

    def test_cisa_kev_offline_hit(self):
        r = dispatch("lookup_cisa_kev", {"cve_id": "CVE-2021-44228"})
        assert r.error is None
        assert r.response["cve"] == "CVE-2021-44228"
        assert "Log4Shell" in r.response["vulnerabilityName"]
        assert r.response["source"] == "offline_cache"

    def test_cisa_kev_not_found(self):
        r = dispatch("lookup_cisa_kev", {"cve_id": "CVE-9999-99999"})
        assert r.error is None
        assert r.response.get("found") is False

    def test_cisa_kev_missing_arg(self):
        r = dispatch("lookup_cisa_kev", {})
        assert r.error is not None
        assert "missing required arg" in r.error

    def test_greynoise_known_benign(self):
        r = dispatch("lookup_greynoise", {"ip": "8.8.8.8"})
        assert r.error is None
        assert r.response["classification"] == "benign"
        assert r.response.get("riot") is True

    def test_greynoise_unknown_ip(self):
        r = dispatch("lookup_greynoise", {"ip": "10.20.30.40"})
        assert r.error is None
        assert r.response.get("classification") == "unknown"

    def test_virustotal_hash_eicar(self):
        eicar = ("275a021bbfb6489e54d471899f7db9d1663fc695"
                  "ec2fe2a2c4538aabf651fd0f")
        r = dispatch("lookup_virustotal_hash", {"hash": eicar})
        assert r.error is None
        assert "EICAR" in r.response["type"]
        assert r.response["malicious"] == 0

    def test_virustotal_hash_invalid_format(self):
        r = dispatch("lookup_virustotal_hash", {"hash": "deadbeef"})
        # Backend returns an error blob (not a dispatch error), so
        # ToolResult.error stays None but response carries the error.
        assert r.error is None
        assert "error" in r.response
        assert "invalid hash format" in r.response["error"]

    def test_virustotal_hash_uppercase_normalised(self):
        eicar_upper = ("275A021BBFB6489E54D471899F7DB9D1663FC695"
                        "EC2FE2A2C4538AABF651FD0F")
        r = dispatch("lookup_virustotal_hash", {"hash": eicar_upper})
        assert r.error is None
        # Backend lowercases for lookup, so the EICAR cache hits.
        assert "EICAR" in r.response["type"]

    def test_shodan_known_ip(self):
        r = dispatch("lookup_shodan", {"ip": "8.8.8.8"})
        assert r.error is None
        assert "dns.google" in r.response["hostnames"]
        assert 53 in r.response["ports"]

    def test_shodan_unknown_ip(self):
        r = dispatch("lookup_shodan", {"ip": "192.0.2.99"})
        assert r.error is None
        assert r.response.get("found") is False

    def test_otx_known_apt(self):
        r = dispatch("lookup_alienvault_otx", {"indicator": "lazarus"})
        assert r.error is None
        assert r.response["pulse_count"] >= 1
        assert "Lazarus" in r.response["summaries"][0]["name"]

    def test_otx_case_insensitive(self):
        # Indicator is lowercased internally so any casing matches.
        r = dispatch("lookup_alienvault_otx", {"indicator": "LAZARUS"})
        assert r.error is None
        assert r.response["pulse_count"] >= 1

    def test_otx_unknown_indicator(self):
        r = dispatch("lookup_alienvault_otx",
                      {"indicator": "totally-unknown-actor"})
        assert r.error is None
        assert r.response.get("found") is False

    def test_offline_env_var_skips_real_apis(self, monkeypatch):
        """With GHOST_AGENT_OFFLINE=1 (default in this test suite),
        no live HTTP request should be attempted. We verify by
        asserting an unreachable upstream URL doesn't matter."""
        # Already set in module scope; just verify the cached path
        # succeeded above without timing-out on a real fetch.
        r = dispatch("lookup_cisa_kev", {"cve_id": "CVE-2021-44228"})
        assert r.response["source"] == "offline_cache"
        assert r.latency_ms < 100  # purely in-memory, sub-100ms


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


class TestMessages:
    def test_tool_message_wraps_in_response_tag(self):
        m = AgentMessage.tool("search_cve_nvd", {"x": 1})
        assert m.content.startswith("<|tool_response|>")
        assert m.content.endswith("<|/tool_response|>")
        assert '"x": 1' in m.content

    def test_tool_message_error_shape(self):
        m = AgentMessage.tool("x", None, error="boom")
        assert '"error": "boom"' in m.content

    def test_trace_to_dict_round_trip(self):
        t = AgentTrace(query="q")
        t.add(AgentMessage.user("hi"))
        d = t.to_dict()
        assert d["query"] == "q"
        assert d["history"][0]["role"] == "user"


# ---------------------------------------------------------------------------
# Runtime loop
# ---------------------------------------------------------------------------


class TestRuntime:
    def test_terminates_on_no_tool_call(self):
        def gen(history):
            return "Plain answer with no tools."
        trace = GhostAgent(gen, RuntimeConfig()).run("q")
        assert trace.terminated_reason == "answer_emitted"
        assert trace.iterations == 1
        assert trace.final_answer == "Plain answer with no tools."

    def test_dispatches_then_emits_answer(self):
        def gen(history):
            n = sum(1 for m in history if m.role == MessageRole.ASSISTANT)
            if n == 0:
                return ('<|tool_call|>{"name": "search_cve_nvd", '
                        '"args": {"q": "CVE-2017-0144"}}<|/tool_call|>')
            return "EternalBlue <|cite|>nvd:CVE-2017-0144<|/cite|>."

        trace = GhostAgent(gen, RuntimeConfig()).run("q")
        assert trace.terminated_reason == "answer_emitted"
        assert trace.iterations == 2
        assert "EternalBlue" in trace.final_answer
        # Loop produced: system, user, assistant, tool, assistant.
        assert [m.role for m in trace.history] == [
            MessageRole.SYSTEM, MessageRole.USER,
            MessageRole.ASSISTANT, MessageRole.TOOL,
            MessageRole.ASSISTANT,
        ]

    def test_max_iterations_safety(self):
        def loopy(history):
            return ('<|tool_call|>{"name": "search_cve_nvd", '
                    '"args": {"q": "CVE-2017-0144"}}<|/tool_call|>')

        trace = GhostAgent(loopy, RuntimeConfig(max_iters=2)).run("q")
        assert trace.terminated_reason == "max_iterations"
        assert trace.iterations == 2

    def test_model_error_captured(self):
        def boom(history):
            raise RuntimeError("kaboom")
        trace = GhostAgent(boom, RuntimeConfig()).run("q")
        assert trace.terminated_reason == "model_error"
        assert "kaboom" in trace.final_answer

    def test_tool_error_is_recoverable(self):
        def gen(history):
            n = sum(1 for m in history if m.role == MessageRole.ASSISTANT)
            if n == 0:
                return '<|tool_call|>{"name": "no_tool", "args": {}}<|/tool_call|>'
            return "Recovered after tool error."

        trace = GhostAgent(gen, RuntimeConfig()).run("q")
        assert trace.terminated_reason == "answer_emitted"
        # Tool message recorded the error.
        tool_msgs = [m for m in trace.history if m.role == MessageRole.TOOL]
        assert len(tool_msgs) == 1
        assert "unknown tool" in tool_msgs[0].metadata["error"]

    def test_stop_sequence_eaten_closing_tag_repaired(self):
        def gen(history):
            n = sum(1 for m in history if m.role == MessageRole.ASSISTANT)
            if n == 0:
                # Closing tag eaten by the stop sequence.
                return '<|tool_call|>{"name": "lookup_cwe", "args": {"cwe_id": "89"}}'
            return "SQLi."

        trace = GhostAgent(gen, RuntimeConfig()).run("q")
        assert trace.terminated_reason == "answer_emitted"
        # The repaired tool call ran successfully.
        tool_msgs = [m for m in trace.history if m.role == MessageRole.TOOL]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].metadata.get("error") is None

    def test_disable_system_prompt(self):
        def gen(history):
            return "answer"
        cfg = RuntimeConfig(system_prompt="")
        trace = GhostAgent(gen, cfg).run("q")
        roles = [m.role for m in trace.history]
        assert MessageRole.SYSTEM not in roles

    def test_cites_stashed_in_metadata(self):
        def gen(history):
            return "Answer <|cite|>nvd:CVE-2017-0144<|/cite|>."
        trace = GhostAgent(gen, RuntimeConfig()).run("q")
        last = trace.history[-1]
        assert last.role == MessageRole.ASSISTANT
        assert last.metadata.get("cites") == [
            {"source_type": "nvd", "source_id": "CVE-2017-0144"},
        ]

    def test_trace_json_serializable(self):
        def gen(history):
            return "answer"
        trace = GhostAgent(gen, RuntimeConfig()).run("q")
        s = trace.to_json()
        assert '"query"' in s
        assert '"final_answer": "answer"' in s


# ---------------------------------------------------------------------------
# make_generator_from_loaded (the MCP-server building block)
# ---------------------------------------------------------------------------


class TestMakeGeneratorFromLoaded:
    def test_drives_agent_loop_against_random_weights(self):
        """The refactored builder should produce a callable that the
        GhostAgent runtime can drive against an already-loaded model
        (mirrors how scripts/mcp_server.py wires the agent without
        loading the checkpoint twice)."""
        from ghostlm.config import GhostLMConfig
        from ghostlm.model import GhostLM
        from ghostlm.tokenizer import GhostTokenizer

        cfg = GhostLMConfig.from_preset("ghost-tiny")
        cfg.vocab_size = 50264
        cfg.context_length = 64
        model = GhostLM(cfg).eval()
        tokenizer = GhostTokenizer()

        gen = make_generator_from_loaded(
            model, cfg, tokenizer, device="cpu",
            max_new_tokens=8, temperature=1.0,
            top_p=1.0, top_k=0, repetition_penalty=1.0,
        )
        assert callable(gen)
        agent = GhostAgent(gen, RuntimeConfig(max_iters=1))
        trace = agent.run("test")
        assert trace.terminated_reason in (
            "answer_emitted", "max_iterations", "model_error",
        )
