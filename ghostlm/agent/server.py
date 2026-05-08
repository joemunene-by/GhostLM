"""HTTP server exposing GhostAgent over multi-vendor compatible APIs.

The server speaks the request/response shapes of the four major LLM
provider APIs, so any client that already targets one of them can
point at this URL unchanged. The agent loop runs server-side: tool
calls happen behind the API and the model's final cite-tagged answer
comes back in whatever shape the caller's SDK expects.

Endpoint families:

  POST /v1/chat/completions
       OpenAI Chat Completions API (the de facto standard; many
       providers like Mistral, xAI, vLLM, and TGI re-implement this
       shape). Returns an OpenAI-compatible chat.completion object
       with optional tool_calls. Supports stream=true via SSE.

  POST /v1/messages
       Anthropic Messages API. Accepts Anthropic's content-block
       request format (system as separate field, content as either
       string or list of typed blocks) and returns the matching
       response shape with tool_use blocks for any tools the loop
       dispatched.

  POST /v1beta/models/{model}:generateContent
       Google Gemini API. Accepts Gemini's contents/parts/role
       request format and returns candidates+content with
       finishReason and usageMetadata.

  POST /api/chat
  POST /api/generate
  GET  /api/tags
       Ollama API. Accepts Ollama's simpler messages-or-prompt
       request format and returns a flat response.

Native:

  POST /v1/agent/run     Returns the full AgentTrace (every message,
                         every tool call, every cite tag) as JSON.
  GET  /v1/models        OpenAI-compat model list.
  GET  /healthz          Readiness probe.

The server is a factory: ``create_app(generator, config, model_name)``
returns a FastAPI app wired around the supplied generator, which makes
test injection trivial. The CLI in this module wires a real GhostLM
checkpoint into the factory; tests inject stub generators directly.

CLI:

    python -m ghostlm.agent.server \\
        --checkpoint runs/v09chat/best.pt --port 8000

    curl -s http://localhost:8000/v1/chat/completions \\
        -H 'content-type: application/json' \\
        -d '{"model": "ghostlm",
             "messages": [{"role": "user",
                            "content": "What is CVE-2017-0144?"}]}'
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

# FastAPI + pydantic are optional runtime deps. Import lazily but at
# module level (so Pydantic v2 can resolve ForwardRefs against fully
# defined classes; defining them inside a closure breaks).
try:
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import HTMLResponse, StreamingResponse
    from pydantic import BaseModel, Field
    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FASTAPI_AVAILABLE = False
    BaseModel = object  # type: ignore
    Field = lambda *a, **k: None  # type: ignore

from .web_ui import INDEX_HTML

from .messages import AgentMessage, AgentTrace, MessageRole
from .runtime import GhostAgent, RuntimeConfig
from .tools import TOOLS_REGISTRY, Tool


def _now_iso() -> str:
    """RFC3339 UTC timestamp for Ollama-compatible responses."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ---------------------------------------------------------------------------
# Request models (defined at module level so Pydantic can resolve them).
# ---------------------------------------------------------------------------


if _FASTAPI_AVAILABLE:

    class ChatMessage(BaseModel):
        role: str
        content: Any  # str or list of content blocks
        tool_calls: Optional[List[Dict[str, Any]]] = None
        tool_call_id: Optional[str] = None
        name: Optional[str] = None

    class ChatCompletionsRequest(BaseModel):
        model: str = "ghostlm"
        messages: List[ChatMessage]
        max_tokens: Optional[int] = None
        temperature: Optional[float] = None
        top_p: Optional[float] = None
        stream: bool = False
        tools: Optional[List[Dict[str, Any]]] = None
        user: Optional[str] = None

    class AgentRunRequest(BaseModel):
        query: str = Field(..., min_length=1, max_length=8000)
        max_iters: Optional[int] = None
        include_trace: bool = True

    class AnthropicMessagesRequest(BaseModel):
        model: str = "ghostlm"
        max_tokens: int = 1024
        system: Optional[Any] = None
        messages: List[Dict[str, Any]]
        temperature: Optional[float] = None
        top_p: Optional[float] = None
        stop_sequences: Optional[List[str]] = None
        stream: bool = False
        tools: Optional[List[Dict[str, Any]]] = None
        tool_choice: Optional[Dict[str, Any]] = None
        metadata: Optional[Dict[str, Any]] = None

    class GeminiRequest(BaseModel):
        contents: List[Dict[str, Any]]
        tools: Optional[List[Dict[str, Any]]] = None
        generationConfig: Optional[Dict[str, Any]] = None
        systemInstruction: Optional[Dict[str, Any]] = None

    class OllamaChatRequest(BaseModel):
        model: str = "ghostlm"
        messages: List[Dict[str, Any]]
        stream: bool = False
        options: Optional[Dict[str, Any]] = None
        keep_alive: Optional[Any] = None

    class OllamaGenerateRequest(BaseModel):
        model: str = "ghostlm"
        prompt: str
        stream: bool = False
        system: Optional[str] = None
        options: Optional[Dict[str, Any]] = None
        raw: bool = False


# ---------------------------------------------------------------------------
# Helpers (vendor-specific shape conversions)
# ---------------------------------------------------------------------------


def _last_user_query_openai(messages: List[Any]) -> str:
    """OpenAI: latest user message's string content."""
    for m in reversed(messages):
        role = m.role if hasattr(m, "role") else m.get("role")
        if role != "user":
            continue
        content = m.content if hasattr(m, "content") else m.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
    raise ValueError("no user message in request")


def _anthropic_extract_query(messages: List[Dict[str, Any]]) -> str:
    """Anthropic: latest user message, content blocks flattened."""
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts: List[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    tc = block.get("content")
                    if isinstance(tc, str):
                        texts.append(f"[tool_result] {tc}")
                    elif isinstance(tc, list):
                        for sub in tc:
                            if isinstance(sub, dict):
                                texts.append(sub.get("text", ""))
            if texts:
                return "\n".join(t for t in texts if t)
    raise ValueError("no user message in Anthropic request")


def _gemini_extract_query(contents: List[Dict[str, Any]]) -> str:
    """Gemini: latest user-role content, parts concatenated."""
    for c in reversed(contents):
        role = c.get("role", "user")
        if role not in ("user", "USER"):
            continue
        parts = c.get("parts", [])
        texts = [p.get("text") for p in parts
                 if isinstance(p, dict) and p.get("text")]
        if texts:
            return "\n".join(texts)
    raise ValueError("no user content in Gemini request")


def _ollama_extract_query_chat(messages: List[Dict[str, Any]]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content", "")
            if isinstance(c, str):
                return c
    raise ValueError("no user message in Ollama chat request")


def _trace_to_openai_message(trace: AgentTrace) -> Dict[str, Any]:
    """Convert an AgentTrace into an OpenAI assistant-message dict."""
    msg: Dict[str, Any] = {
        "role": "assistant",
        "content": trace.final_answer,
    }
    tool_calls: List[Dict[str, Any]] = []
    for hist in trace.history:
        if hist.role != MessageRole.ASSISTANT:
            continue
        for tc in hist.metadata.get("tool_calls", []) or []:
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:16]}",
                "type": "function",
                "function": {
                    "name": tc.get("name", ""),
                    "arguments": json.dumps(
                        tc.get("args", {}), ensure_ascii=False,
                    ),
                },
            })
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _trace_to_anthropic_content(trace: AgentTrace) -> List[Dict[str, Any]]:
    """Anthropic response content: list of typed blocks (tool_use + text)."""
    blocks: List[Dict[str, Any]] = []
    for hist in trace.history:
        if hist.role != MessageRole.ASSISTANT:
            continue
        for tc in hist.metadata.get("tool_calls", []) or []:
            blocks.append({
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:16]}",
                "name": tc.get("name", ""),
                "input": tc.get("args", {}),
            })
    if trace.final_answer:
        blocks.append({"type": "text", "text": trace.final_answer})
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    return blocks


def _openai_finish_reason(trace: AgentTrace) -> str:
    return {
        "answer_emitted": "stop",
        "max_iterations": "length",
        "model_error": "stop",
    }.get(trace.terminated_reason, "stop")


def _anthropic_stop_reason(trace: AgentTrace) -> str:
    return {
        "answer_emitted": "end_turn",
        "max_iterations": "max_tokens",
        "model_error": "end_turn",
    }.get(trace.terminated_reason, "end_turn")


def _gemini_finish_reason(trace: AgentTrace) -> str:
    return {
        "answer_emitted": "STOP",
        "max_iterations": "MAX_TOKENS",
        "model_error": "OTHER",
    }.get(trace.terminated_reason, "STOP")


def _ollama_done_reason(trace: AgentTrace) -> str:
    return {
        "answer_emitted": "stop",
        "max_iterations": "length",
        "model_error": "stop",
    }.get(trace.terminated_reason, "stop")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_app(
    generator: Callable[[List[AgentMessage]], str],
    config: Optional[RuntimeConfig] = None,
    model_name: str = "ghostlm",
    tools: Optional[Dict[str, Tool]] = None,
):
    """Build a FastAPI app wrapping a GhostAgent loop."""
    if not _FASTAPI_AVAILABLE:
        raise RuntimeError(
            "ghostlm.agent.server requires fastapi + pydantic. "
            "Install with `pip install fastapi uvicorn pydantic`."
        )

    cfg = config or RuntimeConfig()
    tool_reg = tools if tools is not None else TOOLS_REGISTRY
    agent = GhostAgent(generator, cfg, tool_reg)

    app = FastAPI(
        title="GhostAgent HTTP API",
        version="0.1.0",
        description="Multi-vendor compatible HTTP endpoints "
                    "(OpenAI, Anthropic, Gemini, Ollama) for the "
                    "GhostLM tool-using agent runtime.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], allow_credentials=False,
        allow_methods=["*"], allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Health + introspection
    # ------------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        """Static demo UI: minimal HTML chat that hits /v1/agent/run."""
        return INDEX_HTML

    @app.get("/healthz")
    def healthz() -> Dict[str, Any]:
        return {"status": "ok", "model": model_name,
                "max_iters": cfg.max_iters,
                "tools": sorted(tool_reg.keys())}

    @app.get("/v1/models")
    def list_models() -> Dict[str, Any]:
        return {
            "object": "list",
            "data": [{
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ghostlm",
            }],
        }

    @app.get("/api/tags")
    def ollama_tags() -> Dict[str, Any]:
        return {
            "models": [{
                "name": model_name,
                "modified_at": _now_iso(),
                "size": 0,
                "digest": "ghostlm",
                "details": {
                    "format": "ghostlm",
                    "family": "ghostlm",
                    "parameter_size": "81M",
                    "quantization_level": "fp32",
                },
            }],
        }

    # ------------------------------------------------------------------
    # Native /v1/agent/run
    # ------------------------------------------------------------------

    @app.post("/v1/agent/run")
    def agent_run(req: AgentRunRequest = Body(...)) -> Dict[str, Any]:
        if req.max_iters is not None and req.max_iters != cfg.max_iters:
            local_cfg = RuntimeConfig(
                max_iters=req.max_iters,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature, top_p=cfg.top_p,
                system_prompt=cfg.system_prompt,
                stop_sequences=list(cfg.stop_sequences),
            )
            local_agent = GhostAgent(generator, local_cfg, tool_reg)
            trace = local_agent.run(req.query)
        else:
            trace = agent.run(req.query)
        out: Dict[str, Any] = {
            "model": model_name,
            "final_answer": trace.final_answer,
            "terminated_reason": trace.terminated_reason,
            "iterations": trace.iterations,
            "total_latency_ms": trace.total_latency_ms,
        }
        if req.include_trace:
            out["trace"] = trace.to_dict()
        return out

    # ------------------------------------------------------------------
    # OpenAI /v1/chat/completions
    # ------------------------------------------------------------------

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionsRequest = Body(...)):
        try:
            query = _last_user_query_openai(req.messages)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if req.stream:
            return StreamingResponse(
                _stream_openai_chunks(query, agent, model_name),
                media_type="text/event-stream",
            )
        trace = agent.run(query)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        return {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or model_name,
            "choices": [{
                "index": 0,
                "message": _trace_to_openai_message(trace),
                "finish_reason": _openai_finish_reason(trace),
            }],
            "usage": {
                "prompt_tokens": max(1, len(query) // 4),
                "completion_tokens": trace.total_tokens_emitted,
                "total_tokens": (max(1, len(query) // 4)
                                  + trace.total_tokens_emitted),
            },
        }

    # ------------------------------------------------------------------
    # Anthropic /v1/messages
    # ------------------------------------------------------------------

    @app.post("/v1/messages")
    def anthropic_messages(req: AnthropicMessagesRequest = Body(...)):
        try:
            query = _anthropic_extract_query(req.messages)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        trace = agent.run(query)
        return {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "model": req.model or model_name,
            "content": _trace_to_anthropic_content(trace),
            "stop_reason": _anthropic_stop_reason(trace),
            "stop_sequence": None,
            "usage": {
                "input_tokens": max(1, len(query) // 4),
                "output_tokens": trace.total_tokens_emitted,
            },
        }

    # ------------------------------------------------------------------
    # Google Gemini /v1beta/models/{model}:generateContent
    # ------------------------------------------------------------------

    @app.post("/v1beta/models/{model}:generateContent")
    def gemini_generate_content(model: str,
                                  req: GeminiRequest = Body(...)):
        try:
            query = _gemini_extract_query(req.contents)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        trace = agent.run(query)
        return {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": trace.final_answer}],
                },
                "finishReason": _gemini_finish_reason(trace),
                "index": 0,
            }],
            "usageMetadata": {
                "promptTokenCount": max(1, len(query) // 4),
                "candidatesTokenCount": trace.total_tokens_emitted,
                "totalTokenCount": (max(1, len(query) // 4)
                                     + trace.total_tokens_emitted),
            },
            "modelVersion": model or model_name,
        }

    # ------------------------------------------------------------------
    # Ollama /api/chat + /api/generate
    # ------------------------------------------------------------------

    @app.post("/api/chat")
    def ollama_chat(req: OllamaChatRequest = Body(...)):
        try:
            query = _ollama_extract_query_chat(req.messages)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        trace = agent.run(query)
        return {
            "model": req.model or model_name,
            "created_at": _now_iso(),
            "message": {"role": "assistant",
                         "content": trace.final_answer},
            "done": True,
            "done_reason": _ollama_done_reason(trace),
            "total_duration": trace.total_latency_ms * 1_000_000,
            "eval_count": trace.total_tokens_emitted,
            "prompt_eval_count": max(1, len(query) // 4),
        }

    @app.post("/api/generate")
    def ollama_generate(req: OllamaGenerateRequest = Body(...)):
        trace = agent.run(req.prompt)
        return {
            "model": req.model or model_name,
            "created_at": _now_iso(),
            "response": trace.final_answer,
            "done": True,
            "done_reason": _ollama_done_reason(trace),
            "total_duration": trace.total_latency_ms * 1_000_000,
            "eval_count": trace.total_tokens_emitted,
            "prompt_eval_count": max(1, len(req.prompt) // 4),
        }

    return app


# ---------------------------------------------------------------------------
# OpenAI streaming helper (module-level so app handlers stay terse)
# ---------------------------------------------------------------------------


def _stream_openai_chunks(query: str, agent, model_name: str):
    """Iteration-level SSE stream. Emits one delta per assistant
    message, then a closing chunk with the final answer + DONE."""
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def _wrap(delta: Dict[str, Any], finish: Optional[str] = None):
        payload = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": delta,
                          "finish_reason": finish}],
        }
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    yield _wrap({"role": "assistant"})
    trace = agent.run(query)
    for hist in trace.history:
        if hist.role == MessageRole.ASSISTANT:
            yield _wrap({"content": hist.content + "\n"})
    yield _wrap({"content": trace.final_answer},
                 finish=_openai_finish_reason(trace))
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# CLI: python -m ghostlm.agent.server
# ---------------------------------------------------------------------------


def cli_main() -> int:
    import argparse
    import os
    import sys

    p = argparse.ArgumentParser(prog="python -m ghostlm.agent.server")
    p.add_argument("--checkpoint", default=None,
                    help="GhostLM .pt checkpoint. Omit for random "
                         "ghost-tiny smoke (output will be noise).")
    p.add_argument("--device", default="auto")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model-name", default="ghostlm",
                    help="Identifier surfaced in vendor responses.")
    p.add_argument("--max-iters", type=int, default=6)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--repetition-penalty", type=float, default=1.15)
    p.add_argument("--offline", action="store_true",
                    help="Force tool backends to use offline caches.")
    args = p.parse_args()

    if args.offline:
        os.environ["GHOST_AGENT_OFFLINE"] = "1"

    try:
        import uvicorn
    except ImportError:
        print("pip install fastapi uvicorn pydantic", file=sys.stderr)
        return 1

    from .runner import make_generator

    generator, is_random = make_generator(
        args.checkpoint, args.device,
        args.max_new_tokens, args.temperature,
        args.top_p, args.top_k, args.repetition_penalty,
    )
    if is_random:
        print("[note] random ghost-tiny weights (output is noise).")
    cfg = RuntimeConfig(
        max_iters=args.max_iters,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    app = create_app(generator, cfg, model_name=args.model_name)
    print(f"[ghostagent-server] {args.host}:{args.port} "
          f"model={args.model_name}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(cli_main())
