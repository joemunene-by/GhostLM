"""Tests for the GhostAgent SFT pipeline: prep + eval scoring.

Covers:
  - scripts/prep_tool_use_sft.py: parse_trace, trace_to_chat_record,
    hash_for_split, and the end-to-end CLI on a synthetic input.
  - scripts/eval_agent.py: trace_to_full_text, score_record, wilson_ci.

Both scripts are pure-logic over JSONL, so the tests run without any
checkpoint or GPU. The end-to-end CLI test invokes the prep script as
a subprocess against a tiny synthetic corpus to catch wiring issues.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Force agent backends offline for deterministic eval-scoring tests.
os.environ["GHOST_AGENT_OFFLINE"] = "1"

from scripts.prep_tool_use_sft import (  # noqa: E402
    hash_for_split,
    parse_trace,
    trace_to_chat_record,
)
from scripts.eval_agent import (  # noqa: E402
    score_record,
    trace_to_full_text,
    wilson_ci,
)
from ghostlm.agent import (  # noqa: E402
    AgentMessage,
    AgentTrace,
    GhostAgent,
    MessageRole,
    RuntimeConfig,
)


# ---------------------------------------------------------------------------
# parse_trace
# ---------------------------------------------------------------------------


_GOOD_TRACE = (
    "USER: What is CVE-2017-0144?\n"
    'ASSISTANT: <|tool_call|>{"name": "search_cve_nvd", '
    '"args": {"q": "CVE-2017-0144"}}<|/tool_call|>\n'
    'TOOL: <|tool_response|>{"cve": "CVE-2017-0144", '
    '"description": "SMB RCE"}<|/tool_response|>\n'
    "ASSISTANT: CVE-2017-0144 is EternalBlue.\n"
)


class TestParseTrace:
    def test_happy_path(self):
        out = parse_trace(_GOOD_TRACE)
        assert out is not None
        assert out["user"] == "What is CVE-2017-0144?"
        assert out["tool_call"].startswith("<|tool_call|>")
        assert out["tool_call"].endswith("<|/tool_call|>")
        assert out["tool_response"].startswith("<|tool_response|>")
        assert out["answer"] == "CVE-2017-0144 is EternalBlue."

    def test_missing_role_returns_none(self):
        bad = "USER: hi\nASSISTANT: hello\n"  # missing TOOL + final ASST
        assert parse_trace(bad) is None

    def test_wrong_first_role_returns_none(self):
        bad = "ASSISTANT: oops\n" + _GOOD_TRACE
        assert parse_trace(bad) is None

    def test_empty_string_returns_none(self):
        assert parse_trace("") is None

    def test_non_string_input(self):
        assert parse_trace(None) is None
        assert parse_trace(123) is None

    def test_strips_outer_whitespace(self):
        out = parse_trace("\n\n" + _GOOD_TRACE + "\n\n")
        assert out is not None
        assert out["user"] == "What is CVE-2017-0144?"


# ---------------------------------------------------------------------------
# trace_to_chat_record
# ---------------------------------------------------------------------------


class TestTraceToChatRecord:
    def test_produces_four_turns_alternating_roles(self):
        parsed = parse_trace(_GOOD_TRACE)
        rec = trace_to_chat_record(parsed, {"source": "synth_tool_use",
                                              "seed_id": "CVE-2017-0144"})
        assert len(rec["turns"]) == 4
        roles = [t["role"] for t in rec["turns"]]
        assert roles == ["user", "assistant", "user", "assistant"]

    def test_tool_call_in_assistant_one(self):
        parsed = parse_trace(_GOOD_TRACE)
        rec = trace_to_chat_record(parsed, {})
        assert "<|tool_call|>" in rec["turns"][1]["content"]

    def test_tool_response_in_user_two(self):
        parsed = parse_trace(_GOOD_TRACE)
        rec = trace_to_chat_record(parsed, {})
        assert "<|tool_response|>" in rec["turns"][2]["content"]

    def test_answer_in_assistant_two(self):
        parsed = parse_trace(_GOOD_TRACE)
        rec = trace_to_chat_record(parsed, {})
        assert rec["turns"][3]["content"] == "CVE-2017-0144 is EternalBlue."

    def test_metadata_preserved(self):
        parsed = parse_trace(_GOOD_TRACE)
        rec = trace_to_chat_record(parsed, {
            "source": "synth_tool_use",
            "seed_source": "search_cve_nvd",
            "seed_id": "CVE-2017-0144",
        })
        assert rec["source"] == "synth_tool_use"
        assert rec["seed_id"] == "CVE-2017-0144"


# ---------------------------------------------------------------------------
# hash_for_split
# ---------------------------------------------------------------------------


class TestHashForSplit:
    def test_deterministic(self):
        rec = {"source": "x", "seed_id": "y",
                "turns": [{"content": "hello"}]}
        assert hash_for_split(rec) == hash_for_split(rec)

    def test_different_records_differ(self):
        a = {"source": "x", "seed_id": "y",
              "turns": [{"content": "hello"}]}
        b = {"source": "x", "seed_id": "z",
              "turns": [{"content": "hello"}]}
        assert hash_for_split(a) != hash_for_split(b)


# ---------------------------------------------------------------------------
# Prep CLI end-to-end
# ---------------------------------------------------------------------------


class TestPrepCLI:
    def test_writes_train_and_val_files(self, tmp_path):
        in_tu = tmp_path / "synth_tool_use.jsonl"
        in_pv = tmp_path / "synth_tool_use_provenance.jsonl"
        in_tu.write_text(json.dumps({
            "id": "x", "source": "synth_tool_use", "teacher": "templated",
            "seed_source": "search_cve_nvd", "seed_id": "CVE-1",
            "text": _GOOD_TRACE,
        }) + "\n")
        in_pv.write_text(json.dumps({
            "id": "y", "source": "synth_tool_use_provenance",
            "teacher": "templated", "seed_source": "lookup_cwe",
            "seed_id": "CWE-89", "text": _GOOD_TRACE,
        }) + "\n")
        out_train = tmp_path / "train.jsonl"
        out_val = tmp_path / "val.jsonl"
        result = subprocess.run(
            [sys.executable, "scripts/prep_tool_use_sft.py",
             "--in-tool-use", str(in_tu),
             "--in-provenance", str(in_pv),
             "--out-train", str(out_train),
             "--out-val", str(out_val)],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert out_train.exists()
        assert out_val.exists()
        # At least one record in train (val may be empty at this scale).
        train_lines = out_train.read_text().strip().split("\n")
        assert len([ln for ln in train_lines if ln]) >= 1


# ---------------------------------------------------------------------------
# trace_to_full_text + score_record
# ---------------------------------------------------------------------------


class TestEvalScoring:
    def _build_trace(self, *contents) -> AgentTrace:
        t = AgentTrace(query="q")
        roles = [MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT,
                 MessageRole.TOOL, MessageRole.ASSISTANT]
        for content, role in zip(contents, roles):
            t.add(AgentMessage(role=role, content=content))
        return t

    def test_full_text_concats_assistant_and_tool_only(self):
        """Scoring must exclude SYSTEM and USER so substrings present
        in the eval prompt don't count as model output."""
        t = self._build_trace("sys", "user-prompt", "asst1",
                               "tool-resp", "asst2")
        full = trace_to_full_text(t)
        assert "sys" not in full
        assert "user-prompt" not in full
        assert "asst1" in full
        assert "tool-resp" in full
        assert "asst2" in full

    def test_all_required_substrings_present(self):
        t = self._build_trace("sys", "Q", "<|tool_call|>{}<|/tool_call|>",
                               "<|tool_response|>{}<|/tool_response|>",
                               "Answer <|cite|>nvd:CVE-X<|/cite|>")
        score = score_record(t, ["<|tool_call|>", "<|cite|>", "CVE-X"])
        assert score["all_present"] is True
        assert score["fraction"] == 1.0
        assert score["n_hit"] == 3

    def test_partial_match(self):
        # Substring "Q" appears only in the user prompt, which is now
        # excluded from scoring. Substring "tool calls" appears in the
        # ASSISTANT turn and SHOULD count.
        t = self._build_trace("sys", "Q", "has tool calls in asst",
                               "no resp", "no cites either")
        score = score_record(t, ["<|tool_call|>", "tool calls", "Q"])
        # Only "tool calls" appears in scored content.
        assert score["n_hit"] == 1
        assert score["fraction"] == pytest.approx(1 / 3)
        assert score["all_present"] is False

    def test_user_substring_does_not_count(self):
        """Substrings present only in the USER turn (the eval prompt)
        must not be credited."""
        t = self._build_trace("sys", "CVE-2017-0144 in user prompt",
                               "asst1 says nothing",
                               "tool resp says nothing",
                               "asst2 says nothing")
        score = score_record(t, ["CVE-2017-0144"])
        assert score["n_hit"] == 0
        assert score["all_present"] is False

    def test_zero_match(self):
        t = self._build_trace("sys", "Q", "abc", "def", "ghi")
        score = score_record(t, ["<|tool_call|>", "<|cite|>"])
        assert score["all_present"] is False
        assert score["n_hit"] == 0


# ---------------------------------------------------------------------------
# wilson_ci
# ---------------------------------------------------------------------------


class TestWilsonCI:
    def test_zero_n_returns_zero_zero(self):
        assert wilson_ci(0, 0) == (0.0, 0.0)

    def test_full_pass_high_upper_bound(self):
        lo, hi = wilson_ci(15, 15)
        assert hi == pytest.approx(1.0, abs=0.001)
        assert lo > 0.7

    def test_zero_pass_low_lower_bound(self):
        lo, hi = wilson_ci(0, 15)
        assert lo == pytest.approx(0.0, abs=0.001)
        assert hi < 0.3

    def test_half_centred(self):
        lo, hi = wilson_ci(50, 100)
        assert lo < 0.5 < hi
        # Symmetric-ish around 0.5 for n=100.
        assert abs((lo + hi) / 2 - 0.5) < 0.01


# ---------------------------------------------------------------------------
# Stub-generator end-to-end (proves the eval CLI logic with a known
# good output, no checkpoint required)
# ---------------------------------------------------------------------------


class TestStubGeneratorEval:
    def test_well_formed_trace_passes_strict_provenance(self):
        """A stub generator that emits a perfect bet-1+9 trace should
        score 100% strict-pass on a provenance-style eval prompt."""
        def perfect_gen(history):
            n = sum(1 for m in history if m.role == MessageRole.ASSISTANT)
            if n == 0:
                return ('<|tool_call|>{"name": "search_cve_nvd", '
                        '"args": {"q": "CVE-2017-0144"}}<|/tool_call|>')
            return ('CVE-2017-0144 is EternalBlue '
                    '<|cite|>nvd:CVE-2017-0144#description<|/cite|>.')

        agent = GhostAgent(perfect_gen, RuntimeConfig(max_iters=4))
        trace = agent.run("What is CVE-2017-0144?")
        required = ["<|tool_call|>", "<|/tool_call|>",
                    "<|cite|>", "<|/cite|>", "CVE-2017-0144"]
        score = score_record(trace, required)
        assert score["all_present"] is True
        assert score["fraction"] == 1.0
