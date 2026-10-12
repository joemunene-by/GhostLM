"""The agent loop: send the conversation, run the tools the model asks for, repeat."""

from __future__ import annotations

import datetime
import json
import platform
import subprocess
import threading
from pathlib import Path
from typing import Protocol

from ghostlm.code import tools
from ghostlm.code.backend import Backend, Cancelled, Reply

MODES = ["default", "accept edits", "plan"]
MAX_STEPS = 50
INSTRUCTION_FILES = ("GHOST.md", "AGENTS.md", "CLAUDE.md")

SYSTEM = """You are Ghost Code, a security-minded agent working in the user's terminal. You help them \
secure their own machine and their code, and you handle everyday coding and shell work with the same care.

Securing the machine: gather evidence with security_check before you conclude anything, and quote the \
evidence you are relying on. Rank findings by real risk to this user, not by checklist order, and say \
when something is fine. For each problem give the exact command that fixes it and what it changes. \
Never change system settings, kill processes, or remove files without the user choosing that fix. \
Use secret_scan for leaked credentials and never print a secret's full value. Use lookup_cve when a \
CVE id comes up instead of recalling details from memory.

Working on code: read before you edit, make the smallest change that solves the problem, match the \
surrounding style, and run the tests or linter afterwards when there is an obvious way to. Paths are \
relative to the working directory.

Ghost Code is part of GhostLM, an open-source language model project built from scratch by Joe Munene \
(github.com/joemunene-by/GhostLM). You are the model the user chose to run behind it, not GhostLM's own weights.

You work defensively and only on systems the user owns or is authorised to test. Decline to attack, \
scan or break into anything else.

Be direct. Say what you found and did in a few sentences, not a report, unless a report was asked for. \
If a command failed or you could not check something, say so plainly. Never claim a result you did not \
observe.
"""

PLAN_NOTE = ("Plan mode is on, so nothing was changed. Investigate with the read-only tools and "
             "reply with a step by step plan for the user to approve.")


class UI(Protocol):
    def on_text(self, chunk: str) -> None: ...
    def on_thinking(self, chunk: str) -> None: ...
    def start_reply(self) -> None: ...
    def end_reply(self, reply: Reply) -> None: ...
    def tool_call(self, label: str) -> None: ...
    def tool_result(self, result: tools.Result) -> None: ...
    def ask(self, title: str, detail: str, options: list[str]) -> int: ...
    def notice(self, text: str) -> None: ...


def environment(cwd: Path) -> str:
    lines = [f"Working directory: {cwd}", f"Platform: {platform.system()} {platform.machine()}",
             f"Date: {datetime.date.today().isoformat()}"]
    try:
        branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, capture_output=True,
                                text=True, timeout=5)
        if branch.returncode == 0:
            lines.append(f"Git branch: {branch.stdout.strip()}")
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "\n".join(lines)


def project_instructions(cwd: Path) -> str:
    for name in INSTRUCTION_FILES:
        f = cwd / name
        if f.is_file():
            return f"\nProject instructions from {name}:\n{f.read_text()[:12_000]}"
    return ""


def bash_rule(command: str) -> str:
    """Session allow-rules for bash match on the first two words, e.g. 'git status'."""
    return " ".join(command.split()[:2])


class Agent:
    def __init__(self, backend: Backend, ui: UI, cwd: Path, mode: str = "default"):
        self.backend, self.ui, self.cwd, self.mode = backend, ui, cwd.resolve(), mode
        self.allowed: set[str] = set()
        self.allow_all = False
        self.cancel = threading.Event()
        self.tokens = 0
        self.messages: list[dict] = []
        self.reset()

    def reset(self) -> None:
        self.messages = [{"role": "system",
                          "content": SYSTEM + "\n" + environment(self.cwd) + project_instructions(self.cwd)}]

    def cycle_mode(self) -> str:
        self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
        return self.mode

    def permitted(self, name: str, args: dict) -> tuple[bool, str]:
        """Decide whether a call may run, asking the user when the mode requires it."""
        if name in tools.READ_ONLY or self.allow_all:
            return True, ""
        if self.mode == "plan":
            return False, PLAN_NOTE
        if name in tools.EDITS:
            path = tools.resolve(args.get("path", ""), self.cwd)
            in_tree = tools.inside(path, self.cwd)
            if in_tree and (self.mode == "accept edits" or "edits" in self.allowed):
                return True, ""
            verb = "Create" if name == "write_file" and not path.exists() else "Edit"
            choice = self.ui.ask(f"{verb} file", tools.preview(name, args, self.cwd), [
                "Yes", "Yes, and allow all edits in this folder for this session", "No"])
            if choice == 1 and in_tree:
                self.allowed.add("edits")
            return choice in (0, 1), "The user declined this edit."
        command = args.get("command", "")
        rule = "bash:" + bash_rule(command)
        if name == "bash" and rule in self.allowed:
            return True, ""
        choice = self.ui.ask("Bash command" if name == "bash" else name, command or json.dumps(args), [
            "Yes", f"Yes, and don't ask again for `{bash_rule(command)}` this session", "No"])
        if choice == 1 and name == "bash":
            self.allowed.add(rule)
        return choice in (0, 1), "The user declined this command."

    def run_turn(self, text: str) -> str:
        """Run one user turn to completion. Returns the final assistant text."""
        self.cancel.clear()
        self.messages.append({"role": "user", "content": text})
        final = ""
        for _ in range(MAX_STEPS):
            self.ui.start_reply()
            try:
                reply = self.backend.chat(self.messages, tools.SPECS, self.ui.on_text, self.ui.on_thinking,
                                          self.cancel)
            except Cancelled:
                self.ui.end_reply(Reply())
                self.messages.append({"role": "assistant", "content": "[interrupted by the user]"})
                self.ui.notice("Interrupted")
                return final
            except BaseException:
                self.ui.end_reply(Reply())
                raise
            self.ui.end_reply(reply)
            self.tokens = reply.usage.get("total_tokens", self.tokens)
            message = {"role": "assistant", "content": reply.text}
            if reply.tool_calls:
                message["tool_calls"] = [{"id": c.id, "type": "function",
                                          "function": {"name": c.name, "arguments": c.arguments}}
                                         for c in reply.tool_calls]
            self.messages.append(message)
            final = reply.text or final
            if not reply.tool_calls:
                return final
            for call in reply.tool_calls:
                self.messages.append({"role": "tool", "tool_call_id": call.id, "content": self._execute(call)})
                if self.cancel.is_set():
                    self.ui.notice("Interrupted")
                    return final
        self.ui.notice(f"Stopped after {MAX_STEPS} steps. Say 'continue' to keep going.")
        return final

    def _execute(self, call) -> str:
        try:
            args = json.loads(call.arguments or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except ValueError as e:
            self.ui.tool_call(f"{call.name}(?)")
            result = tools.Result(f"Could not parse the tool arguments: {e}", "Invalid arguments", ok=False)
            self.ui.tool_result(result)
            return result.output
        self.ui.tool_call(tools.label(call.name, args, self.cwd))
        result = tools.precheck(call.name, args, self.cwd)
        if result is None:
            ok, reason = self.permitted(call.name, args)
            result = tools.run(call.name, args, self.cwd, self.cancel) if ok else tools.Result(reason, "Not run", ok=False)
        self.ui.tool_result(result)
        return result.output

    def repair(self) -> None:
        """After an interrupt, answer any tool calls left without a result so the history stays valid."""
        answered = {m.get("tool_call_id") for m in self.messages if m["role"] == "tool"}
        for m in list(self.messages):
            for call in m.get("tool_calls") or []:
                if call["id"] not in answered:
                    self.messages.append({"role": "tool", "tool_call_id": call["id"],
                                          "content": "[interrupted before this ran]"})

    def compact(self) -> str:
        """Replace the conversation with a model-written summary to free context."""
        ask = self.messages + [{"role": "user", "content":
                                "Summarise this session so far for your own future reference: the goal, what "
                                "was done, files touched, decisions, and what is left. Be specific and brief."}]
        summary = self.backend.chat(ask, None).text
        self.reset()
        self.messages.append({"role": "user", "content": f"Summary of the session so far:\n{summary}"})
        self.messages.append({"role": "assistant", "content": "Understood. Continuing from that summary."})
        return summary
