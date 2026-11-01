"""One chat completion against an OpenAI-compatible HTTP endpoint, for the teacher scripts.

Groq's free tier gives each model its own 200k tokens a day, so ``model`` may be a
comma-separated chain: a daily-limit 429 moves the call to the next model, and only
when every model is out do we sleep for the Retry-After the server asks for.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENDPOINTS = {"groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY")}
DEFAULT_MODELS = {"groq": "openai/gpt-oss-120b,qwen/qwen3.8-27b,openai/gpt-oss-20b"}
_local = threading.local()


def last_model() -> str:
    return getattr(_local, "model", "")


def load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def ask_http(backend: str, model: str, prompt: str, timeout: int = 300, max_tokens: int = 6000) -> str:
    load_env()
    url, key_name = ENDPOINTS[backend]
    key = os.environ.get(key_name)
    if not key:
        raise SystemExit(f"{key_name} is not set (put it in {ROOT / '.env'})")
    chain = [m.strip() for m in model.split(",") if m.strip()]
    deadline = time.time() + max(timeout, 3 * 3600)
    exhausted: dict = {}
    while time.time() < deadline:
        live = [m for m in chain if exhausted.get(m, 0) < time.time()]
        if not live:
            wait = min(exhausted.values()) - time.time()
            print(f"{time.strftime('%H:%M:%S')} all models at their daily limit, waiting {wait:.0f}s",
                  file=sys.stderr, flush=True)
            time.sleep(max(wait, 1))
            continue
        name = live[0]
        body = json.dumps({"model": name, "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": max_tokens, "temperature": 0.8}).encode()
        req = urllib.request.Request(url, data=body, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json",
            "User-Agent": "ghostlm-teacher/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=min(timeout, 180)) as resp:
                data = json.load(resp)
            _local.model = name
            return data["choices"][0]["message"].get("content") or ""
        except urllib.error.HTTPError as e:
            if e.code != 429:
                print(f"{time.strftime('%H:%M:%S')} HTTP {e.code} from {name}: {e.read()[:200]!r}",
                      file=sys.stderr, flush=True)
                return ""
            wait = min(float(e.headers.get("Retry-After") or 20), 3600)
            detail = json.loads(e.read() or b"{}").get("error", {}).get("message", "")
            if "per day" in detail:
                exhausted[name] = time.time() + wait
                print(f"{time.strftime('%H:%M:%S')} {name} daily limit, next model (retry in {wait:.0f}s)",
                      file=sys.stderr, flush=True)
            else:
                time.sleep(wait)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError):
            return ""
    return ""
