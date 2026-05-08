"""GhostAgent: a tool-using agent runtime for GhostLM checkpoints.

This module is the production-shape wrapping around a trained
GhostLM checkpoint. It does what a real LLM-powered SOC assistant
needs: take a question, decide which tool to call, execute it,
feed the result back, decide again, and eventually emit a
cite-tagged final answer.

The runtime is intentionally model-agnostic. ``GhostAgent`` takes a
generator function, a tokenizer, and a tool registry, and runs the
loop against any LM that emits the bet 1 four-message trace format
plus the bet 9 ``<|cite|>`` tags. v0.9 chat runs in this loop today
(it produces poor tool calls because it wasn't trained for them,
but the loop terminates correctly via the max-iterations safety).
When ghost-base trains on the synth-tool-use SFT data, the same
runtime drives a model that actually does this well.

Public API:

    from ghostlm.agent import (
        GhostAgent,             # the loop runtime
        Tool, TOOLS_REGISTRY,   # tool registry + 4 default tools
        AgentMessage,           # one message in the conversation
        AgentTrace,             # the full back-and-forth + final answer
        parse_agent_output,     # output parser (tool_call + cite tags)
    )

CLI: ``python -m ghostlm.agent --query "What is CVE-2017-0144?"``
runs the agent against the v0.9 chat checkpoint (or any provided
checkpoint), prints the trace, and exits.
"""

from __future__ import annotations

from .messages import AgentMessage, AgentTrace, MessageRole
from .parser import (
    ParsedOutput, ToolCall, parse_agent_output, parse_cite_tags,
    parse_tool_calls,
)
from .runtime import GhostAgent, RuntimeConfig
from .tools import TOOLS_REGISTRY, Tool, ToolResult

__all__ = [
    "GhostAgent",
    "RuntimeConfig",
    "Tool",
    "ToolResult",
    "TOOLS_REGISTRY",
    "AgentMessage",
    "AgentTrace",
    "MessageRole",
    "ParsedOutput",
    "ToolCall",
    "parse_agent_output",
    "parse_cite_tags",
    "parse_tool_calls",
]

__version__ = "0.1.0"
