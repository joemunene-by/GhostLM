"""GhostLM web app: chat with retrieval, citations and live training status.

    python -m ghostlm.app --port 8091

Serves on all interfaces for the home network / Twingate; it has no login,
so do not expose it to the public internet.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from ghostlm.app.engine import Engine

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).with_name("static")
DEFAULT_CHAT_DIR = Path.home() / "Library/Application Support/GhostLM/chats"


class ChatRequest(BaseModel):
    message: str
    chat_id: Optional[str] = None
    model: str = "ghost-small-gen"
    use_retrieval: bool = True


def create_app(engine: Optional[Engine] = None, chat_dir: Path = DEFAULT_CHAT_DIR,
               training_url: str = "http://127.0.0.1:8090/api/status") -> FastAPI:
    engine = engine or Engine(ROOT)
    chat_dir.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="GhostLM")

    def chat_path(chat_id: str) -> Path:
        if not chat_id.replace("-", "").isalnum():
            raise HTTPException(400, "bad chat id")
        return chat_dir / f"{chat_id}.json"

    def load_chat(chat_id: str) -> dict:
        p = chat_path(chat_id)
        if not p.exists():
            raise HTTPException(404, "chat not found")
        return json.loads(p.read_text())

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/models")
    def models():
        return engine.model_list()

    @app.get("/api/chats")
    def chats():
        items = []
        for p in sorted(chat_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            c = json.loads(p.read_text())
            items.append({"id": c["id"], "title": c["title"], "updated": c["updated"]})
        return items

    @app.get("/api/chats/{chat_id}")
    def get_chat(chat_id: str):
        return load_chat(chat_id)

    @app.delete("/api/chats/{chat_id}")
    def delete_chat(chat_id: str):
        chat_path(chat_id).unlink(missing_ok=True)
        return {"ok": True}

    @app.get("/api/training")
    def training():
        try:
            with urllib.request.urlopen(training_url, timeout=2) as r:
                return json.loads(r.read())
        except OSError:
            return {"state": "offline"}

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        if not req.message.strip():
            raise HTTPException(400, "empty message")
        if req.model not in engine.models:
            raise HTTPException(400, "unknown model")
        chat = load_chat(req.chat_id) if req.chat_id else {
            "id": uuid.uuid4().hex, "title": req.message.strip()[:60], "turns": [], "created": time.time()}
        history = [{"user": t["user"], "assistant": t["assistant"]} for t in chat["turns"]]

        def events():
            turn = {"user": req.message.strip(), "assistant": "", "sources": [], "tools": [], "model": req.model}
            yield _sse({"type": "chat", "id": chat["id"], "title": chat["title"]})
            try:
                for ev in engine.stream_answer(turn["user"], req.model, history, req.use_retrieval):
                    if ev["type"] == "sources":
                        turn["sources"] = ev["passages"]
                    elif ev["type"] == "tool":
                        turn["tools"].append(ev)
                    elif ev["type"] == "facts":
                        turn["facts"] = ev["facts"]
                    elif ev["type"] == "notice":
                        turn["notice"] = ev["text"]
                    elif ev["type"] == "done":
                        turn["assistant"] = ev["answer"]
                    yield _sse(ev)
            except (MemoryError, ValueError) as e:
                turn["assistant"] = f"[{e}]"
                yield _sse({"type": "error", "message": str(e)})
            chat["turns"].append(turn)
            chat["updated"] = time.time()
            chat_path(chat["id"]).write_text(json.dumps(chat))

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    return app


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def main() -> None:
    p = argparse.ArgumentParser(description="GhostLM web app")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8091)
    args = p.parse_args()
    import uvicorn
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
