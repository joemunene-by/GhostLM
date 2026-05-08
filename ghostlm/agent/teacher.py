"""OpenAI-compatible teacher generator for GhostAgent distillation.

Wraps any OpenAI-compatible chat-completions endpoint (the official
OpenAI API, Ollama, vLLM, TGI, Together, Groq, xAI, anything that
speaks the same wire format) into a Generator callable that the
GhostAgent runtime can drive. This is what lets us distill bet-1 +
bet-9 traces from a stronger teacher: the teacher runs through our
own loop, dispatches our tools, sees real responses, and produces
fresh varied traces that the SFT pipeline can then learn from.

Usage:

    from ghostlm.agent import GhostAgent, RuntimeConfig
    from ghostlm.agent.teacher import OpenAICompatGenerator

    teacher = OpenAICompatGenerator(
        base_url="http://localhost:11434/v1",  # local Ollama
        api_key="ollama",
        model="qwen2.5:14b",
    )
    agent = GhostAgent(teacher, RuntimeConfig(max_iters=4))
    trace = agent.run("What is CVE-2017-0144?")
    # trace is a high-quality, real-teacher-generated bet-1+9 trace.

The Generator contract is ``(history) -> str``. The OpenAI client
maps GhostAgent's MessageRole to OpenAI's role strings (USER ->
"user", ASSISTANT -> "assistant", TOOL -> "user" with the existing
<|tool_response|>...</|tool_response|> wrapping inline, SYSTEM ->
"system"). The teacher's response is returned verbatim; if it
contains <|tool_call|> blocks the GhostAgent runtime parses and
dispatches them just like for any other generator.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .messages import AgentMessage, MessageRole


class OpenAICompatGenerator:
    """Generator that proxies to an OpenAI-compatible HTTP endpoint.

    Constructed once per teacher; reused across many ``agent.run()``
    calls. Each call to the generator makes one HTTP request to the
    teacher and returns the assistant message content as a string.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "anything",
        model: str = "gpt-4o-mini",
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_tokens: int = 512,
        timeout: float = 60.0,
        extra_headers: Optional[Dict[str, str]] = None,
        client: Optional[Any] = None,
    ):
        """
        Args:
          base_url        OpenAI-compatible endpoint, ending in /v1
                          (e.g. http://localhost:11434/v1 for Ollama,
                          https://api.openai.com/v1 for OpenAI).
          api_key         Bearer credential. Local servers usually
                          accept any non-empty value.
          model           Model identifier the teacher exposes.
          temperature     Sampling temperature.
          top_p           Nucleus threshold.
          max_tokens      Per-request generation budget.
          timeout         HTTP timeout in seconds.
          extra_headers   Additional headers (e.g. Anthropic API
                          key when proxying through a translator).
          client          Optional httpx.Client for testing. When
                          None, the generator constructs its own.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_headers = dict(extra_headers or {})
        self._client = client  # lazy if None

    def __call__(self, history: List[AgentMessage]) -> str:
        """Generator entry point: history -> next assistant text."""
        client = self._get_client()
        body = self._build_request(history)
        url = f"{self.base_url}/chat/completions"
        resp = client.post(
            url, json=body, timeout=self.timeout,
            headers=self._headers(),
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"teacher returned HTTP {resp.status_code}: "
                f"{resp.text[:300]}"
            )
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(
                f"unexpected teacher response shape: {e!r} "
                f"body={json.dumps(data)[:300]}"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import httpx
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "OpenAICompatGenerator requires httpx. "
                "Install with `pip install httpx`."
            ) from e
        self._client = httpx.Client()
        return self._client

    def _headers(self) -> Dict[str, str]:
        h = {"content-type": "application/json"}
        if self.api_key:
            h["authorization"] = f"Bearer {self.api_key}"
        h.update(self.extra_headers)
        return h

    def _build_request(self, history: List[AgentMessage]) -> Dict[str, Any]:
        """Translate AgentMessage history into an OpenAI request body."""
        messages: List[Dict[str, Any]] = []
        for m in history:
            role = self._map_role(m.role)
            messages.append({"role": role, "content": m.content})
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            # We deliberately do NOT pass tools here. The teacher is
            # expected to emit <|tool_call|> in plain text, which the
            # GhostAgent runtime parses. If the teacher tries to use
            # OpenAI's structured tool-calling instead, we would not
            # see those calls in the response content and the loop
            # would terminate early.
        }

    @staticmethod
    def _map_role(role: MessageRole) -> str:
        """Map GhostAgent roles into OpenAI roles. TOOL becomes USER
        because the bet-1 wire format already wraps tool responses
        in <|tool_response|>...<|/tool_response|>; the teacher sees
        that as user-supplied context."""
        return {
            MessageRole.SYSTEM: "system",
            MessageRole.USER: "user",
            MessageRole.ASSISTANT: "assistant",
            MessageRole.TOOL: "user",
        }[role]
