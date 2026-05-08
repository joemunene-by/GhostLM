"""Conversation primitives for GhostAgent.

The agent runtime keeps the conversation as an ordered list of
``AgentMessage``. Each message has a role (user / assistant / tool /
system), content (string), and optional metadata (which tool produced
this, latency in ms, error flag for tool failures).

The trace is the full conversation plus the final answer plus
termination metadata (did we hit max iterations, did the model emit
a no-tool-call answer, did a tool error abort the run).

These data classes are JSON-serialisable so traces can be logged for
later analysis (the GhostBench paired-comparison machinery can score
two agents on the same query by comparing their traces).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class MessageRole(str, Enum):
    """Roles in a GhostAgent conversation, mirroring the OpenAI shape
    so traces can be exported as standard chat-completion JSON."""
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"


@dataclass
class AgentMessage:
    """One message in the agent's conversation.

    The ``content`` is always the raw text. For ASSISTANT messages
    that contain ``<|tool_call|>`` blocks, the content includes the
    tool-call wire format; the parser extracts the structured calls
    separately. For TOOL messages the content is the JSON-stringified
    tool response wrapped in ``<|tool_response|>...<|/tool_response|>``
    so the model sees the bet-1-format the synth corpus trained on.
    """
    role: MessageRole
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))

    @classmethod
    def user(cls, text: str) -> "AgentMessage":
        return cls(role=MessageRole.USER, content=text)

    @classmethod
    def assistant(cls, text: str,
                   tool_calls: Optional[List[Dict]] = None) -> "AgentMessage":
        meta = {"tool_calls": tool_calls} if tool_calls else {}
        return cls(role=MessageRole.ASSISTANT, content=text, metadata=meta)

    @classmethod
    def tool(cls, tool_name: str, response: Any,
              error: Optional[str] = None,
              latency_ms: Optional[int] = None) -> "AgentMessage":
        """Build a TOOL message wrapping the response in the bet-1
        on-wire format."""
        body = json.dumps(response, ensure_ascii=False) if not error \
               else json.dumps({"error": error}, ensure_ascii=False)
        wrapped = f"<|tool_response|>{body}<|/tool_response|>"
        meta: Dict[str, Any] = {"tool_name": tool_name}
        if error:
            meta["error"] = error
        if latency_ms is not None:
            meta["latency_ms"] = latency_ms
        return cls(role=MessageRole.TOOL, content=wrapped, metadata=meta)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp_ms": self.timestamp_ms,
        }


@dataclass
class AgentTrace:
    """The full trace of an agent run.

    Fields:
      query                    The original user query.
      history                  Ordered list of every message exchanged.
      final_answer             The assistant's last message content
                               (the no-tool-call answer that triggered
                               loop termination); empty string if
                               terminated without a clean answer.
      terminated_reason        Why the loop stopped: "answer_emitted",
                               "max_iterations", "tool_error_fatal",
                               "model_error".
      iterations               How many model -> tool round-trips ran.
      total_tokens_emitted     Sum of new tokens across all assistant
                               messages (rough cost proxy).
      total_latency_ms         Wall-clock from query to final answer.
    """
    query: str
    history: List[AgentMessage] = field(default_factory=list)
    final_answer: str = ""
    terminated_reason: str = ""
    iterations: int = 0
    total_tokens_emitted: int = 0
    total_latency_ms: int = 0

    def add(self, msg: AgentMessage) -> None:
        self.history.append(msg)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "history": [m.to_dict() for m in self.history],
            "final_answer": self.final_answer,
            "terminated_reason": self.terminated_reason,
            "iterations": self.iterations,
            "total_tokens_emitted": self.total_tokens_emitted,
            "total_latency_ms": self.total_latency_ms,
        }

    def to_json(self, **kwargs) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)
