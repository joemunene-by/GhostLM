"""Streaming chat client for any OpenAI-compatible endpoint (Ollama, the GhostLM server, vLLM, hosted APIs)."""

from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass, field
from typing import Callable

import httpx


class Cancelled(Exception):
    pass


class BackendError(Exception):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class Reply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


class Backend:
    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 900.0,
                 context: int = 16_384):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.timeout = httpx.Timeout(timeout, connect=10.0)
        self.context = context
        self.ollama = self._ollama_root()

    def _ollama_root(self) -> str | None:
        """Ollama's OpenAI endpoint cannot set the context length, so its native API is used when present."""
        root = self.base_url.removesuffix("/v1")
        try:
            r = httpx.get(f"{root}/api/version", timeout=2.0)
            return root if r.status_code == 200 and "version" in r.json() else None
        except (httpx.HTTPError, ValueError):
            return None

    def models(self) -> list[str]:
        try:
            r = httpx.get(f"{self.base_url}/models", headers=self.headers, timeout=5.0)
            r.raise_for_status()
            return [m["id"] for m in r.json().get("data", [])]
        except (httpx.HTTPError, ValueError, KeyError):
            return []

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             on_text: Callable[[str], None] = lambda s: None,
             on_thinking: Callable[[str], None] = lambda s: None,
             cancel: threading.Event | None = None) -> Reply:
        """Stream one reply. The request runs on a worker so a cancel is honoured even mid-read."""
        payload = {"model": self.model, "messages": messages, "stream": True,
                   "stream_options": {"include_usage": True}}
        if tools:
            payload["tools"] = tools
        events: queue.Queue = queue.Queue()
        stop = threading.Event()
        threading.Thread(target=self._stream, args=(payload, events, stop), daemon=True).start()
        try:
            while True:
                if cancel is not None and cancel.is_set():
                    raise Cancelled
                try:
                    kind, value = events.get(timeout=0.05)
                except queue.Empty:
                    continue
                if kind == "text":
                    on_text(value)
                elif kind == "thinking":
                    on_thinking(value)
                elif kind == "error":
                    raise value
                else:
                    return value
        finally:
            # Tells the worker to drop the connection, which is what makes the server stop generating.
            stop.set()

    def _stream(self, payload: dict, events: queue.Queue, stop: threading.Event) -> None:
        reply, calls = Reply(), {}
        url = f"{self.base_url}/chat/completions"
        if self.ollama:
            url = f"{self.ollama}/api/chat"
            payload = {"model": payload["model"], "messages": _ollama_messages(payload["messages"]), "stream": True,
                       "tools": payload.get("tools") or [], "options": {"num_ctx": self.context}}
        try:
            with httpx.stream("POST", url, json=payload, headers=self.headers, timeout=self.timeout) as r:
                if r.status_code >= 400:
                    raise BackendError(f"{r.status_code} from {self.base_url}: {r.read().decode(errors='replace')[:400]}")
                for line in r.iter_lines():
                    if stop.is_set():
                        return
                    data = line[5:].strip() if line.startswith("data:") else line.strip() if self.ollama else ""
                    if not data:
                        continue
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if self.ollama:
                        if chunk.get("error"):
                            raise BackendError(str(chunk["error"]))
                        delta = chunk.get("message") or {}
                        for tc in delta.get("tool_calls") or []:
                            fn = tc.get("function") or {}
                            calls[len(calls)] = {"id": "", "name": fn.get("name", ""),
                                                 "arguments": json.dumps(fn.get("arguments") or {})}
                        if chunk.get("done"):
                            reply.usage = {"total_tokens": chunk.get("prompt_eval_count", 0) + chunk.get("eval_count", 0)}
                        deltas = [dict(delta, tool_calls=None, reasoning=delta.get("thinking"))]
                    else:
                        if chunk.get("usage"):
                            reply.usage = chunk["usage"]
                        deltas = [choice.get("delta") or {} for choice in chunk.get("choices", [])]
                    for delta in deltas:
                        thinking = delta.get("reasoning") or delta.get("reasoning_content")
                        if thinking:
                            events.put(("thinking", thinking))
                        if delta.get("content"):
                            reply.text += delta["content"]
                            events.put(("text", delta["content"]))
                        for tc in delta.get("tool_calls") or []:
                            slot = calls.setdefault(tc.get("index", len(calls)), {"id": "", "name": "", "arguments": ""})
                            fn = tc.get("function") or {}
                            slot["id"] = tc.get("id") or slot["id"]
                            slot["name"] += fn.get("name") or ""
                            slot["arguments"] += fn.get("arguments") or ""
        except httpx.ConnectError as e:
            events.put(("error", BackendError(f"cannot reach {self.base_url} ({e}). Is the model server running?")))
            return
        except (httpx.HTTPError, ValueError) as e:
            events.put(("error", BackendError(f"{type(e).__name__}: {e}")))
            return
        except BackendError as e:
            events.put(("error", e))
            return
        reply.tool_calls = [ToolCall(c["id"] or f"call_{i}", c["name"], c["arguments"] or "{}")
                            for i, c in sorted(calls.items())]
        events.put(("done", reply))


def _ollama_messages(messages: list[dict]) -> list[dict]:
    """Ollama's native format: tool arguments are objects, and tool results name their tool."""
    names, out = {}, []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                names[c["id"]] = c["function"]["name"]
                try:
                    args = json.loads(c["function"]["arguments"] or "{}")
                except ValueError:
                    args = {}
                calls.append({"function": {"name": c["function"]["name"], "arguments": args}})
            m["tool_calls"] = calls
        if m["role"] == "tool":
            m = {"role": "tool", "content": m["content"], "tool_name": names.get(m.get("tool_call_id"), "")}
        out.append(m)
    return out
