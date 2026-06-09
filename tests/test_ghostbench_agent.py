"""Tests for the GhostBench agent runner.

Covers:
  - AgentTrace.to_scored_text default + opt-in flags.
  - trace_to_prediction shape matches GhostBench Prediction dataclass.
  - end-to-end: stub-generator agent run produces valid predictions
    that GhostBench's Bench.score consumes without complaint.
  - --baseline subprocess invocation against a real eval set.

Runs entirely without checkpoints by using the stub-generator pattern
that scripts/eval_agent.py and tests/test_agent.py also use.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

os.environ["GHOST_AGENT_OFFLINE"] = "1"

from ghostbench.bench import Bench, Prediction  # noqa: E402
from ghostbench.parsers import DEFAULT_PARSERS  # noqa: E402
from ghostlm.agent import (  # noqa: E402
    AgentMessage,
    AgentTrace,
    GhostAgent,
    MessageRole,
    RuntimeConfig,
)
from scripts.ghostbench_agent_run import trace_to_prediction  # noqa: E402


# ---------------------------------------------------------------------------
# AgentTrace.to_scored_text
# ---------------------------------------------------------------------------


class TestToScoredText:
    def _build(self, *contents) -> AgentTrace:
        roles = [MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT,
                 MessageRole.TOOL, MessageRole.ASSISTANT]
        t = AgentTrace(query="q")
        for content, role in zip(contents, roles):
            t.add(AgentMessage(role=role, content=content))
        return t

    def test_default_excludes_user_and_system(self):
        t = self._build("S", "U", "A1", "T", "A2")
        out = t.to_scored_text()
        assert "S" not in out
        assert "U" not in out
        assert "A1" in out
        assert "T" in out
        assert "A2" in out

    def test_include_user(self):
        t = self._build("S", "U", "A1", "T", "A2")
        assert "U" in t.to_scored_text(include_user=True)
        assert "S" not in t.to_scored_text(include_user=True)

    def test_include_system(self):
        t = self._build("S", "U", "A1", "T", "A2")
        assert "S" in t.to_scored_text(include_system=True)
        assert "U" not in t.to_scored_text(include_system=True)

    def test_include_all(self):
        t = self._build("S", "U", "A1", "T", "A2")
        full = t.to_scored_text(include_user=True, include_system=True)
        for c in ("S", "U", "A1", "T", "A2"):
            assert c in full


# ---------------------------------------------------------------------------
# trace_to_prediction shape
# ---------------------------------------------------------------------------


class TestTraceToPrediction:
    def _trace(self, *contents) -> AgentTrace:
        roles = [MessageRole.SYSTEM, MessageRole.USER, MessageRole.ASSISTANT,
                 MessageRole.TOOL, MessageRole.ASSISTANT]
        t = AgentTrace(query="q")
        for content, role in zip(contents, roles):
            t.add(AgentMessage(role=role, content=content))
        return t

    def test_propagates_eval_tags(self):
        trace = self._trace("S", "What is X?", "asst1", "tool", "asst2")
        eval_rec = {
            "format": "provenance",
            "prompt": "What is X?",
            "required_substrings": ["A1", "A2"],
            "required_fields": [{"path": "$.x", "value": "y"}],
            "seed_id": "x-1",
        }
        pred = trace_to_prediction(trace, eval_rec)
        assert pred["format"] == "provenance"
        assert pred["prompt"] == "What is X?"
        assert pred["required_substrings"] == ["A1", "A2"]
        assert pred["required_fields"] == [{"path": "$.x", "value": "y"}]
        assert pred["seed_id"] == "x-1"

    def test_predicted_artifact_excludes_user(self):
        trace = self._trace("S", "U-prompt", "asst1", "tool-resp", "asst2")
        pred = trace_to_prediction(trace, {"prompt": "U-prompt"})
        assert "U-prompt" not in pred["predicted_artifact"]
        assert "asst1" in pred["predicted_artifact"]
        assert "tool-resp" in pred["predicted_artifact"]

    def test_loadable_into_prediction_dataclass(self):
        trace = self._trace("S", "U", "A", "T", "A2")
        pred_dict = trace_to_prediction(trace, {
            "format": "provenance", "prompt": "U",
            "required_substrings": ["x"],
        })
        pred = Prediction.from_dict(pred_dict)
        assert pred.fmt == "provenance"
        assert pred.prompt == "U"


# ---------------------------------------------------------------------------
# End-to-end: stub agent + Bench.score
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_perfect_agent_passes_provenance_bench(self, tmp_path):
        """A stub generator that emits perfect bet-1 + bet-9 traces
        should yield Predictions that GhostBench's Bench.score for
        bet9_provenance recognises as passing."""
        # Tiny eval JSONL.
        eval_path = tmp_path / "provenance_eval.jsonl"
        eval_path.write_text(json.dumps({
            "format": "provenance",
            "prompt": "What is CVE-2017-0144?",
            "required_substrings": ["<|tool_call|>", "<|/tool_call|>",
                                      "<|cite|>", "<|/cite|>",
                                      "CVE-2017-0144"],
        }) + "\n")

        bench = Bench.from_jsonl(
            name="bet9_provenance",
            description="provenance",
            path=eval_path,
            parsers=DEFAULT_PARSERS,
        )

        def perfect_gen(history):
            n = sum(1 for m in history if m.role == MessageRole.ASSISTANT)
            if n == 0:
                return ('<|tool_call|>{"name": "search_cve_nvd", '
                        '"args": {"q": "CVE-2017-0144"}}<|/tool_call|>')
            return ('CVE-2017-0144 is EternalBlue '
                    '<|cite|>nvd:CVE-2017-0144<|/cite|>.')

        agent = GhostAgent(perfect_gen, RuntimeConfig(max_iters=4))
        trace = agent.run("What is CVE-2017-0144?")
        eval_rec = {
            "format": "provenance",
            "prompt": "What is CVE-2017-0144?",
            "required_substrings": ["<|tool_call|>", "<|/tool_call|>",
                                      "<|cite|>", "<|/cite|>",
                                      "CVE-2017-0144"],
        }
        pred = Prediction.from_dict(trace_to_prediction(trace, eval_rec))
        report = bench.score([pred], run_name="test_agent")
        assert report.n == 1
        # All required substrings are in the trace's scored text
        # (assistant + tool messages), so substrings tier passes.
        sub_passes = sum(1 for s in report.scores if s.tier_pass("substrings"))
        assert sub_passes == 1


# ---------------------------------------------------------------------------
# CLI subprocess: agent vs baseline run on the real provenance eval
# ---------------------------------------------------------------------------


class TestCLI:
    def test_runs_against_real_eval_dir(self, tmp_path):
        """Smoke test: random ghost-tiny weights, single bench, low
        token budget, writes a well-formed predictions JSONL."""
        out_dir = tmp_path / "agent_out"
        result = subprocess.run(
            [sys.executable, "scripts/ghostbench_agent_run.py",
             "--eval-dir", "data/raw",
             "--predictions-dir", str(out_dir),
             "--run-name", "test_smoke",
             "--only", "bet9_provenance",
             "--max-iters", "1",
             "--max-new-tokens", "8",
             "--offline"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        out_file = out_dir / "bet9_provenance.jsonl"
        assert out_file.exists()
        lines = [ln for ln in out_file.read_text().split("\n") if ln]
        assert len(lines) == 15  # provenance eval has 15 prompts
        # Every line is a well-formed Prediction JSON.
        for ln in lines:
            d = json.loads(ln)
            assert "format" in d
            assert "prompt" in d
            assert "predicted_artifact" in d
            assert "required_substrings" in d

    def test_baseline_flag_runs(self, tmp_path):
        out_dir = tmp_path / "baseline_out"
        result = subprocess.run(
            [sys.executable, "scripts/ghostbench_agent_run.py",
             "--eval-dir", "data/raw",
             "--predictions-dir", str(out_dir),
             "--run-name", "test_baseline",
             "--only", "bet9_provenance",
             "--baseline",
             "--max-new-tokens", "8",
             "--offline"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert "max_iters=1" in result.stdout
        assert (out_dir / "bet9_provenance.jsonl").exists()
