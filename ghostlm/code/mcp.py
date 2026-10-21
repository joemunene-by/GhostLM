"""A small MCP (stdio) client so Ghost Code can use external tool servers such as REA."""

from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

from ghostlm.code.tools import Result

PROTOCOL = "2025-06-18"
START_TIMEOUT = 30
CALL_TIMEOUT = 300
MAX_OUTPUT = 12_000
PATH_ARGS = ("path", "input_path")

# REA ships 207 tools and megabytes of schema, far past what a local 8B model can hold. These are its
# static inspectors that need no Hopper, Ghidra or IDA, so they work on a plain Mac.
REA = {
    "command": ["rea", "--mcp"],
    "read_only": True,
    "opener": "open_binary",
    "tools": ["open_binary", "inspect_signature", "inspect_macho", "inspect_plist", "inspect_electron_fuses",
              "inspect_go_binary", "inspect_rust_binary", "inspect_pe_resources", "inspect_with_objdump",
              "inspect_android_package"],
    "hint": ("rea_* tools reverse engineer apps and binaries statically, without running them. Call "
             "rea_open_binary on an app or executable first, then rea_inspect_signature (signing, team, "
             "entitlements, hardened runtime) or rea_inspect_macho (imports, linked libraries). The other rea "
             "tools take a path directly: inspect_electron_fuses shows whether an Electron app can be hijacked "
             "with ELECTRON_RUN_AS_NODE or the inspector, and inspect_plist reads an app's Info.plist."),
}


class MCPError(Exception):
    pass


def _short(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = cut.rfind(". ")
    return cut[:end + 1] if end > limit // 3 else cut.rstrip() + "..."


def _slim(schema: dict) -> dict:
    """Drop the parts of a JSON schema a model does not need, keeping types, enums and short descriptions."""
    if not isinstance(schema, dict):
        return schema
    out = {}
    for k, v in schema.items():
        if k in ("$schema", "examples", "pattern", "additionalProperties", "minProperties"):
            continue
        if k == "description":
            out[k] = _short(v, 160)
        elif k == "properties":
            out[k] = {name: _slim(p) for name, p in v.items()}
        elif isinstance(v, dict):
            out[k] = _slim(v)
        elif isinstance(v, list):
            out[k] = [_slim(i) if isinstance(i, dict) else i for i in v]
        else:
            out[k] = v
    return out


@dataclass
class Server:
    name: str
    command: list[str]
    allow: list[str] | None = None
    read_only: bool = False
    hint: str = ""
    opener: str = ""
    tools: list[dict] = field(default_factory=list)
    error: str = ""

    def __post_init__(self):
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.next_id = 0

    def start(self) -> None:
        try:
            self.proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
            self.request("initialize", {"protocolVersion": PROTOCOL, "capabilities": {},
                                        "clientInfo": {"name": "ghost-code", "version": "1"}}, START_TIMEOUT)
            self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            listed, cursor = [], None
            while True:
                page = self.request("tools/list", {"cursor": cursor} if cursor else {}, START_TIMEOUT)
                listed += page.get("tools", [])
                cursor = page.get("nextCursor")
                if not cursor:
                    break
            self.tools = [t for t in listed if self.allow is None or t["name"] in self.allow]
        except (OSError, MCPError) as e:
            self.error = str(e)
            self.close()

    def send(self, message: dict) -> None:
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def request(self, method: str, params: dict, timeout: float, cancel: threading.Event | None = None) -> dict:
        if not self.proc or self.proc.poll() is not None:
            raise MCPError(f"{self.name} is not running")
        with self.lock:
            self.next_id += 1
            wanted = self.next_id
            self.send({"jsonrpc": "2.0", "id": wanted, "method": method, "params": params})
            box: dict = {}
            reader = threading.Thread(target=self._read_until, args=(wanted, box), daemon=True)
            reader.start()
            waited = 0.0
            while reader.is_alive() and waited < timeout:
                reader.join(0.2)
                waited += 0.2
                if cancel is not None and cancel.is_set():
                    self.close()
                    raise MCPError("interrupted")
            if reader.is_alive():
                self.close()
                raise MCPError(f"{self.name} did not answer {method} within {int(timeout)}s")
        if "error" in box:
            raise MCPError(box["error"])
        return box.get("result", {})

    def _read_until(self, wanted: int, box: dict) -> None:
        for line in self.proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            if "method" in msg and "id" in msg:
                # Server-initiated requests: answer ping, refuse the rest so the server is not left waiting.
                reply = {"result": {}} if msg["method"] == "ping" else {
                    "error": {"code": -32601, "message": "not supported by Ghost Code"}}
                self.send({"jsonrpc": "2.0", "id": msg["id"], **reply})
            elif msg.get("id") == wanted:
                if "error" in msg:
                    box["error"] = msg["error"].get("message", str(msg["error"]))
                else:
                    box["result"] = msg.get("result", {})
                return
        box["error"] = f"{self.name} exited"

    def call(self, tool: str, args: dict, cancel: threading.Event | None = None) -> Result:
        try:
            res = self.request("tools/call", {"name": tool, "arguments": args}, CALL_TIMEOUT, cancel)
        except MCPError as e:
            if self.proc is None:
                self.start()
            return Result(str(e), "Failed", ok=False)
        text = "\n".join(c.get("text", "") for c in res.get("content", []) if c.get("type") == "text")
        if not text and "structuredContent" in res:
            text = json.dumps(res["structuredContent"])
        failed = bool(res.get("isError"))
        summary = "Failed" if failed else f"{len(text):,} characters"
        if failed and (m := re.search(r'"message":"([^"]+)"', text)):
            summary = _short(m.group(1), 80)
        if len(text) > MAX_OUTPUT:
            text = text[:MAX_OUTPUT] + f"\n[truncated, {len(text) - MAX_OUTPUT:,} more characters]"
        return Result(text, summary, ok=not failed)

    def close(self) -> None:
        if self.proc:
            try:
                self.proc.kill()
                self.proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
        self.proc = None


def configured(cfg: dict) -> dict:
    """Servers from the config's "mcp" block; REA is added by default when it is installed."""
    servers = cfg.get("mcp")
    if servers is None:
        servers = {"rea": REA} if shutil.which("rea") else {}
    return servers


class Registry:
    def __init__(self, servers: dict):
        self.servers = [Server(name, list(s["command"]), s.get("tools"), bool(s.get("read_only")), s.get("hint", ""),
                               s.get("opener", "")) for name, s in servers.items()]
        self.ready = threading.Event()
        threading.Thread(target=self._start, daemon=True).start()
        atexit.register(self.close)

    def _start(self) -> None:
        for s in self.servers:
            s.start()
        self.ready.set()

    def hints(self) -> str:
        return "\n".join(s.hint for s in self.servers if s.hint)

    def _tools(self):
        self.ready.wait(START_TIMEOUT * 2)
        for s in self.servers:
            for t in s.tools:
                yield f"{s.name}_{t['name']}", s, t

    def specs(self) -> list[dict]:
        return [{"type": "function", "function": {
            "name": name, "description": _short(t.get("description", ""), 300),
            "parameters": _slim(t.get("inputSchema") or {"type": "object", "properties": {}})}}
            for name, _, t in self._tools()]

    def find(self, name: str):
        for full, s, t in self._tools():
            if full == name:
                return s, t
        return None

    def owns(self, name: str) -> bool:
        return any(name.startswith(s.name + "_") for s in self.servers) and self.find(name) is not None

    def read_only(self, name: str) -> bool:
        found = self.find(name)
        return bool(found and found[0].read_only)

    def call(self, name: str, args: dict, cwd: Path, cancel: threading.Event | None = None) -> Result:
        found = self.find(name)
        if not found:
            return Result(f"Unknown tool: {name}", "Unknown tool", ok=False)
        server, tool = found
        args = dict(args)
        for key in PATH_ARGS:
            if isinstance(args.get(key), str):
                # REA wants absolute paths and rejects symlinks such as Homebrew's bin/ entries.
                p = Path(os.path.expanduser(args[key]))
                args[key] = str((p if p.is_absolute() else cwd / p).resolve())
        accepted = (tool.get("inputSchema") or {}).get("properties", {})
        if server.opener and "path" in args and "path" not in accepted:
            # Small models pass the target to tools that act on the opened one; open it for them.
            opened = server.call(server.opener, {"path": args.pop("path")}, cancel)
            if not opened.ok:
                return opened
        return server.call(tool["name"], args, cancel)

    def status(self) -> list[str]:
        self.ready.wait(START_TIMEOUT * 2)
        lines = []
        for s in self.servers:
            if s.error:
                lines.append(f"{s.name}: failed to start ({s.error})")
            else:
                lines.append(f"{s.name}: {len(s.tools)} tools · " + ", ".join(t["name"] for t in s.tools))
        return lines or ["No MCP servers configured"]

    def close(self) -> None:
        for s in self.servers:
            s.close()
