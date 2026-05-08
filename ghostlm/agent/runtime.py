"""Agent loop runtime for GhostAgent.

This module provides:

  - ``RuntimeConfig``: tuning knobs (max iters, generation params,
    system prompt, stop sequences).
  - ``GhostAgent``: the model-agnostic loop. Takes a generator
    function, a tools registry, and a config; provides
    ``run(query) -> AgentTrace``.

The loop is the thing that turns a trained checkpoint into a
deployed SOC assistant: take a question, decide which tool to call,
execute it, feed the result back, decide again, and eventually emit a
cite-tagged final answer.

Loop semantics:

  iter 1:  generator sees [system, user(query)]
           emits text -> may contain <|tool_call|>{json}<|/tool_call|>
           if no tool calls: terminate, reason="answer_emitted"
           if tool calls: dispatch each, append TOOL messages
  iter 2+: generator sees [system, user, assistant, tool, ...]
           ... repeat ...

The loop terminates on:

  answer_emitted    Model produced a message with no tool calls.
                    ``plain_text`` (tool-call blocks stripped) is the
                    final answer.
  max_iterations    Hit the configured ceiling. The last assistant
                    message's plain text becomes the final answer.
  model_error       Generator raised. Final answer is a structured
                    error string; loop aborts.

Tool errors are NOT fatal. They get wrapped into the TOOL message
content as ``{"error": "..."}`` and fed back to the model. The bet 1
SFT data trained the model on this pattern, so a well-trained
checkpoint can recover (e.g. retry with corrected args, or fall back
to a different tool). Today's v0.9 chat checkpoint won't do this
gracefully, but the loop is structurally ready.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .messages import AgentMessage, AgentTrace, MessageRole
from .parser import ParsedOutput, parse_agent_output
from .tools import TOOLS_REGISTRY, Tool, dispatch


# A Generator is any callable that takes the current message history
# and returns the next assistant text. The Generator owns: rendering
# messages into the model's chat template, applying stop sequences,
# running sampling, decoding tokens. The runtime is decoupled from
# any specific HF/transformers wiring so you can drop in a remote
# API, a local llama.cpp server, or a unit-test stub equally.
Generator = Callable[[List[AgentMessage]], str]


_DEFAULT_SYSTEM_PROMPT = (
    "You are GhostAgent, a cybersecurity-focused assistant. "
    "You have access to tools for looking up CVEs, MITRE ATT&CK "
    "techniques, CWE entries, and corpus passages.\n\n"
    "When a user asks a question that needs factual lookup, emit a "
    "tool call in this exact format:\n\n"
    "<|tool_call|>{\"name\": \"<TOOL>\", \"args\": {...}}<|/tool_call|>\n\n"
    "Then wait for the <|tool_response|>...<|/tool_response|> reply. "
    "When you have enough information, write a final answer with "
    "inline <|cite|>type:id<|/cite|> tags pointing to each source."
)


@dataclass
class RuntimeConfig:
    """Tuning knobs for the agent loop.

    max_iters         Hard ceiling on model -> tool round-trips.
                      Bounds runtime + cost when the model loops.
    max_new_tokens    Generation budget hint per assistant message.
                      The generator may or may not honour this.
    temperature       Sampling temperature hint.
    top_p             Nucleus sampling threshold hint.
    system_prompt     System message prepended to history. Empty
                      string disables the system message entirely.
    stop_sequences    Suggested stop tokens for the generator. The
                      runtime re-attaches ``<|/tool_call|>`` to the
                      raw text if it was eaten by the stop, so the
                      parser sees a well-formed block.
    """
    max_iters: int = 8
    max_new_tokens: int = 512
    temperature: float = 0.6
    top_p: float = 0.9
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT
    stop_sequences: List[str] = field(default_factory=lambda: [
        "<|/tool_call|>",
    ])


class GhostAgent:
    """Tool-using agent loop.

    The runtime is generator-agnostic: pass any callable
    ``(history) -> str`` and the loop will exercise it correctly.
    See ``ghostlm.agent.runner`` for the HF-checkpoint wiring used
    by the CLI.

    Example:

        from ghostlm.agent import GhostAgent, RuntimeConfig

        def gen(history):
            # render history into a prompt, call the model, decode
            return "<|tool_call|>{\\"name\\": \\"search_cve_nvd\\", "
                   "\\"args\\": {\\"q\\": \\"CVE-2017-0144\\"}}<|/tool_call|>"

        agent = GhostAgent(gen, RuntimeConfig(max_iters=4))
        trace = agent.run("What is EternalBlue?")
        print(trace.final_answer)
        print(trace.to_json(indent=2))
    """

    def __init__(
        self,
        generator: Generator,
        config: Optional[RuntimeConfig] = None,
        tools: Optional[Dict[str, Tool]] = None,
    ):
        self.generator = generator
        self.config = config or RuntimeConfig()
        self.tools = tools if tools is not None else TOOLS_REGISTRY

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------

    def run(self, query: str) -> AgentTrace:
        """Run the agent on a single query. Returns a complete trace."""
        t0 = time.time()
        trace = AgentTrace(query=query)

        if self.config.system_prompt:
            trace.add(AgentMessage(
                role=MessageRole.SYSTEM,
                content=self.config.system_prompt,
            ))
        trace.add(AgentMessage.user(query))

        for it in range(self.config.max_iters):
            trace.iterations = it + 1

            try:
                raw_text = self.generator(trace.history)
            except Exception as e:  # noqa: BLE001
                trace.final_answer = (
                    f"<<model_error: {type(e).__name__}: {e}>>"
                )
                trace.terminated_reason = "model_error"
                trace.total_latency_ms = int((time.time() - t0) * 1000)
                return trace

            raw_text = self._reattach_stop_if_needed(raw_text)
            parsed = parse_agent_output(raw_text)

            # Record the assistant turn.
            asst_msg = AgentMessage.assistant(
                text=raw_text,
                tool_calls=[c.to_dict() for c in parsed.tool_calls] or None,
            )
            if parsed.parse_warnings:
                asst_msg.metadata["parse_warnings"] = parsed.parse_warnings
            if parsed.cites:
                asst_msg.metadata["cites"] = [c.to_dict() for c in parsed.cites]
            trace.add(asst_msg)
            trace.total_tokens_emitted += _approx_tokens(raw_text)

            # No tool calls => the spoken text is the final answer.
            if not parsed.tool_calls:
                trace.final_answer = parsed.plain_text
                trace.terminated_reason = "answer_emitted"
                trace.total_latency_ms = int((time.time() - t0) * 1000)
                return trace

            # Dispatch every tool call in order.
            for call in parsed.tool_calls:
                result = dispatch(call.name, call.args, self.tools)
                trace.add(AgentMessage.tool(
                    tool_name=call.name,
                    response=result.response,
                    error=result.error,
                    latency_ms=result.latency_ms,
                ))

        # Hit the ceiling. Pull final_answer from the last assistant
        # message (its plain prose around the tool calls).
        trace.final_answer = self._last_assistant_plain(trace) or ""
        trace.terminated_reason = "max_iterations"
        trace.total_latency_ms = int((time.time() - t0) * 1000)
        return trace

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reattach_stop_if_needed(self, raw_text: str) -> str:
        """If the generator's stop sequences ate ``<|/tool_call|>`` but
        an opener is still present, re-append it so the parser sees a
        well-formed block."""
        if "<|tool_call|>" not in raw_text:
            return raw_text
        for stop in self.config.stop_sequences:
            if stop == "<|/tool_call|>" and not raw_text.rstrip().endswith(stop):
                return raw_text.rstrip() + stop
        return raw_text

    @staticmethod
    def _last_assistant_plain(trace: AgentTrace) -> str:
        for m in reversed(trace.history):
            if m.role == MessageRole.ASSISTANT:
                return parse_agent_output(m.content).plain_text
        return ""


def _approx_tokens(text: str) -> int:
    """~4 chars per token. Cost proxy for trace.total_tokens_emitted."""
    return max(1, len(text) // 4)
