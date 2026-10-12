"""`ghost`: the interactive prompt, slash commands and the non-interactive --print mode."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shutil
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

from rich.text import Text

from ghostlm.code import __version__, tools
from ghostlm.code.agent import MODES, Agent
from ghostlm.code.backend import Backend, BackendError
from ghostlm.code.ui import GHOST_ROWS, PlainUI, Status, TerminalUI, idle_stage, intro_stage

HOME = Path(os.environ.get("GHOST_HOME", Path.home() / ".ghostcode"))
DEFAULT_URL = "http://localhost:11434/v1"
COMMANDS = {
    "/help": "show commands and shortcuts",
    "/clear": "start a fresh conversation",
    "/compact": "summarise the conversation to free context",
    "/model": "show or switch the model, e.g. /model qwen3:8b",
    "/mode": "show or set the permission mode",
    "/status": "model, endpoint, mode and context size",
    "/audit": "security audit of this machine (read-only)",
    "/scan": "scan this folder for secrets and risky code",
    "/init": "write a GHOST.md describing this project",
    "/exit": "quit",
}
SHORTCUTS = [("enter", "send"), ("\\ + enter, or esc then enter", "new line"), ("shift+tab", "cycle permission mode"),
             ("esc", "interrupt the model"), ("! command", "run a shell command yourself"),
             ("@ path", "mention a file"), ("ctrl+c twice", "quit")]
AUDIT_PROMPT = ("Audit the security of this machine. Run security_check with area 'all', then follow up on "
                "anything that needs a closer look. Report a table of findings (area, status, severity, evidence), "
                "worst first, including what is already fine. Then list the fixes worth making, each with the exact "
                "command and what it changes. Change nothing yet: ask me which fixes to apply.")
SCAN_PROMPT = ("Scan the code in this folder for security problems. Work through these steps: 1) run secret_scan. "
               "2) list the files with glob and read every source and config file. 3) look for shell injection, "
               "eval or exec on input, SQL built from strings, disabled TLS verification, debug mode left on, and "
               "world-writable permissions. 4) check dependency manifests for old pinned versions. "
               "This is about the code, not the machine, so do not run security_check. "
               "Report findings worst first with file and line. Change nothing yet: ask me which to fix.")
PLACEHOLDER = 'Try "/audit" or "is anything listening on my network?"'
INIT_PROMPT = ("Explore this repository and write a GHOST.md at its root: what the project is, how to "
               "build, test and lint it, the layout, and any conventions an agent working here should follow. "
               "Keep it under 60 lines.")

AGENT_COMMANDS = {"/init": INIT_PROMPT, "/audit": AUDIT_PROMPT, "/scan": SCAN_PROMPT}


def load_config() -> dict:
    try:
        return json.loads((HOME / "config.json").read_text())
    except (OSError, ValueError):
        return {}


def pick_model(available: list[str]) -> str | None:
    for hint in ("coder", "qwen", "llama"):
        for name in available:
            if hint in name.lower():
                return name
    return available[0] if available else None


def session_file(cwd: Path) -> Path:
    return HOME / "sessions" / (hashlib.sha1(str(cwd).encode()).hexdigest()[:12] + ".json")


def save_session(agent: Agent) -> None:
    f = session_file(agent.cwd)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"cwd": str(agent.cwd), "messages": agent.messages}))


def short_path(path: Path) -> str:
    home = str(Path.home())
    return "~" + str(path)[len(home):] if str(path).startswith(home) else str(path)


def build_backend(args) -> Backend:
    cfg = load_config()
    url = args.base_url or os.environ.get("GHOST_BASE_URL") or cfg.get("base_url") or DEFAULT_URL
    key = os.environ.get("GHOST_API_KEY") or cfg.get("api_key")
    backend = Backend(url, "", key, context=int(os.environ.get("GHOST_CONTEXT") or cfg.get("context") or 16_384))
    model = args.model or os.environ.get("GHOST_MODEL") or cfg.get("model") or pick_model(backend.models())
    if not model:
        sys.exit(f"ghost: no model found at {url}.\n"
                 "Start one (for example `ollama serve`, or `python -m ghostlm.agent.server --checkpoint ...`)\n"
                 "or point at another endpoint with --base-url and --model.")
    backend.model = model
    return backend


def rich_line(fragments: list[tuple[str, str]]) -> Text:
    """A row of prompt fragments as rich text, for printing the header into the transcript."""
    named = {"class:title": "bold", "class:dim": "#8a8a8a"}
    line = Text()
    for style, text in fragments:
        colours = dict(part.split(":") for part in style.split() if part[:3] in ("fg:", "bg:"))
        line.append(text, style=named.get(style) or " on ".join(
            v for v in (colours.get("fg"), colours.get("bg")) if v) or None)
    return line


class Repl:
    """One long-lived prompt. Input goes to a worker thread, so the box stays put while the agent runs."""

    def __init__(self, agent: Agent, ui: TerminalUI):
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import Completer, Completion
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.styles import Style

        self.agent, self.ui = agent, ui
        self.started, self.fresh, self.last_ctrl_c, self.hint = time.time(), True, 0.0, ""
        self.idle_since = time.time() + 0.9
        self.inbox: queue.Queue = queue.Queue()
        self.waiting: list[str] = []
        self.busy = False
        HOME.mkdir(parents=True, exist_ok=True)
        repl = self

        class GhostCompleter(Completer):
            def get_completions(self, document, complete_event):
                text = document.text_before_cursor
                if text.startswith("/") and " " not in text:
                    for name, about in COMMANDS.items():
                        if name.startswith(text):
                            yield Completion(name, start_position=-len(text), display_meta=about)
                    return
                word = text.split()[-1] if text and not text[-1].isspace() else ""
                if not word.startswith("@"):
                    return
                partial = word[1:]
                base = (repl.agent.cwd / partial).parent if "/" in partial else repl.agent.cwd
                try:
                    entries = sorted(base.iterdir())
                except OSError:
                    return
                stem = partial.rsplit("/", 1)[-1]
                for p in entries:
                    if p.name.startswith(stem) and p.name not in tools.SKIP_DIRS and not p.name.startswith("."):
                        rel = os.path.relpath(p, repl.agent.cwd) + ("/" if p.is_dir() else "")
                        yield Completion("@" + rel, start_position=-len(word))

        kb = KeyBindings()
        asking = Condition(lambda: repl.ui.question is not None)
        # A question only takes keys while the input is empty, so typing a message can never answer it by accident.
        choosing = asking & Condition(lambda: not repl.session.default_buffer.text)

        @kb.add("enter")
        def _(event):
            buf, state = event.current_buffer, event.current_buffer.complete_state
            if repl.ui.question and not buf.text:
                repl.ui.question.answer(repl.ui.question.selected)
            elif state and state.current_completion:
                buf.apply_completion(state.current_completion)
            elif state and state.completions and buf.text.startswith("/") and buf.text not in COMMANDS:
                buf.apply_completion(state.completions[0])
            elif buf.text.endswith("\\"):
                buf.delete_before_cursor(1)
                buf.insert_text("\n")
            elif buf.text.strip():
                text = buf.text.strip()
                buf.reset(append_to_history=True)
                repl.submit(text)

        @kb.add("escape", "enter")
        @kb.add("c-j")
        def _(event):
            event.current_buffer.insert_text("\n")

        @kb.add("up", filter=choosing)
        @kb.add("down", filter=choosing)
        def _(event):
            q = repl.ui.question
            q.selected = (q.selected + (1 if event.key_sequence[0].key == "down" else -1)) % len(q.options)

        for digit in "123":
            @kb.add(digit, filter=choosing)
            def _(event, digit=digit):
                if int(digit) <= len(repl.ui.question.options):
                    repl.ui.question.answer(int(digit) - 1)

        @kb.add("escape", eager=True, filter=asking)
        def _(event):
            repl.ui.question.answer(len(repl.ui.question.options) - 1)

        @kb.add("escape", eager=True, filter=Condition(lambda: repl.busy) & ~asking)
        def _(event):
            repl.agent.cancel.set()

        @kb.add("s-tab")
        def _(event):
            repl.agent.cycle_mode()

        @kb.add("c-c")
        def _(event):
            buf = event.current_buffer
            if repl.ui.question:
                repl.ui.question.answer(len(repl.ui.question.options) - 1)
            elif repl.busy:
                repl.agent.cancel.set()
            elif buf.text:
                buf.reset()
            elif time.time() - repl.last_ctrl_c < 1.5:
                event.app.exit(exception=EOFError)
            else:
                repl.last_ctrl_c, repl.hint = time.time(), "press ctrl+c again to exit"

        self.session = PromptSession(
            history=FileHistory(str(HOME / "history")), completer=GhostCompleter(), key_bindings=kb,
            multiline=True, complete_while_typing=True, prompt_continuation="  ",
            reserve_space_for_menu=0,
            style=Style.from_dict({"bottom-toolbar": "noreverse", "rule": "#4a4a4a", "dim": "#8a8a8a",
                                   "placeholder": "#5c5c5c", "title": "bold"}))
        ui.on_change = self.redraw

    def redraw(self) -> None:
        app = self.session.app
        if app.is_running and app.loop:
            app.loop.call_soon_threadsafe(app.invalidate)

    def _rule(self) -> tuple[str, str]:
        return ("class:rule", "─" * shutil.get_terminal_size().columns + "\n")

    def _header(self) -> list[list[tuple[str, str]]]:
        size, host = shutil.get_terminal_size(), urlparse(self.agent.backend.base_url).netloc
        where = short_path(self.agent.cwd)
        return [[("class:title", " Ghost Code"), ("class:dim", f" v{__version__}")],
                [("class:dim", f" {self.agent.backend.model} · {host}")],
                [("class:dim", " " + (where if len(where) < size.columns - 4 else "…" + where[4 - size.columns:]))]]

    def _stage_height(self, used: int) -> int:
        """Rows for the ghost: all the free space above the prompt, and never fewer than it needs."""
        from prompt_toolkit.application import get_app
        from prompt_toolkit.renderer import HeightIsUnknownError
        app = get_app()
        try:
            above = app.renderer.rows_above_layout
        except HeightIsUnknownError:
            return GHOST_ROWS
        typed = app.current_buffer.document.line_count
        return max(GHOST_ROWS, shutil.get_terminal_size().lines - above - used - typed - 3)

    def _message(self):
        """Everything above the input line: the welcome header, then the ghost's stage."""
        top = []
        if self.fresh:
            for line in self._header():
                top += [*line, ("", "\n")]
        queued = [("class:dim", f"  queued: {text[:80]}\n") for text in self.waiting]
        width = shutil.get_terminal_size().columns
        height = self._stage_height(len(self._header()) * self.fresh + len(queued))
        alive = time.time() - self.started
        if self.ui.question:
            live = self.ui.question.fragments(height)
        elif self.ui.status:
            live = self.ui.status.fragments(width, height)
        elif self.fresh and alive < 0.9:
            live = intro_stage(height, int(alive * 16))
        else:
            live = idle_stage(width, height, int((time.time() - self.idle_since) * 8))
        return [*top, *live, *queued, self._rule(), ("", "❯ ")]

    def _toolbar(self):
        if self.hint and time.time() - self.last_ctrl_c > 1.5:
            self.hint = ""
        mode = self.agent.mode
        left = f"  ▸▸ {mode} on" if mode != "default" else "  ? /help for commands"
        tail = self.hint or "shift+tab to cycle modes · @ for files · ! for shell"
        return [self._rule(), ("bold" if mode != "default" else "class:dim", left), ("class:dim", f" · {tail}")]

    def submit(self, text: str) -> None:
        """Called on the UI thread when the user presses enter."""
        if self.busy:
            self.waiting.append(text)
        self.inbox.put(text)

    def _work(self) -> None:
        while True:
            line = self.inbox.get()
            if line is None:
                return
            self.busy = True
            if line in self.waiting:
                self.waiting.remove(line)
            if self.fresh:
                for row in self._header():
                    self.ui.console.print(rich_line(row))
                self.fresh = False
            self.ui.echo(line)
            keep_going = True
            quick = line.startswith("/") and line.split()[0] not in (*AGENT_COMMANDS, "/compact")
            self.ui.status = None if quick else Status()
            try:
                if line.startswith("/"):
                    keep_going = self.command(line)
                elif line.startswith("!"):
                    self.shell(line[1:].strip())
                else:
                    self.turn(line)
            except Exception as e:  # a crash in one turn must not take the prompt down with it
                self.agent.repair()
                self.ui.error(f"{type(e).__name__}: {e}")
            self.ui.status, self.busy, self.idle_since = None, False, time.time()
            self.redraw()
            if not keep_going:
                app = self.session.app
                app.loop.call_soon_threadsafe(lambda: app.exit(exception=EOFError))
                return

    def command(self, line: str) -> bool:
        """Handle a slash command. Returns False when the session should end."""
        name, _, arg = line.partition(" ")
        arg, agent, say = arg.strip(), self.agent, self.ui.notice
        if name in ("/exit", "/quit"):
            return False
        if name == "/help":
            for cmd, about in COMMANDS.items():
                say(f"{cmd:<10} {about}")
            for keys, about in SHORTCUTS:
                say(f"{keys:<30} {about}")
        elif name == "/clear":
            agent.reset()
            say("Conversation cleared")
        elif name == "/compact":
            before = len(agent.messages)
            agent.compact()
            say(f"Compacted {before} messages into a summary")
        elif name == "/model":
            if arg:
                agent.backend.model = arg
            say(f"Model: {agent.backend.model}")
            if not arg and (names := agent.backend.models()):
                say("Available: " + ", ".join(names))
        elif name == "/mode":
            if arg in MODES:
                agent.mode = arg
            elif arg:
                say(f"Unknown mode. Choose from: {', '.join(MODES)}")
            say(f"Mode: {agent.mode}")
        elif name == "/status":
            say(f"Model: {agent.backend.model} at {agent.backend.base_url}")
            say(f"Mode: {agent.mode} · {len(agent.messages)} messages · about {agent.tokens:,} tokens in context")
            say(f"Working directory: {agent.cwd}")
        elif name in AGENT_COMMANDS:
            self.turn(AGENT_COMMANDS[name])
        else:
            say(f"Unknown command {name}. Try /help")
        return True

    def shell(self, command: str) -> None:
        self.ui.tool_call(f"Bash({command})")
        self.agent.cancel.clear()
        result = tools.bash({"command": command}, self.agent.cwd, self.agent.cancel)
        self.ui.tool_result(result)
        self.agent.messages.append({"role": "user", "content": f"I ran `{command}` myself. Output:\n{result.output}"})

    def turn(self, text: str) -> None:
        try:
            self.agent.run_turn(text)
        except BackendError as e:
            self.agent.repair()
            self.ui.error(str(e))
        save_session(self.agent)

    def run(self, first: str | None = None) -> None:
        from prompt_toolkit.formatted_text import FormattedText
        from prompt_toolkit.patch_stdout import patch_stdout

        worker = threading.Thread(target=self._work, daemon=True)
        worker.start()
        if first:
            self.submit(first)
        try:
            with patch_stdout(raw=True):
                self.session.prompt(
                    lambda: FormattedText(self._message()), bottom_toolbar=lambda: FormattedText(self._toolbar()),
                    placeholder=lambda: FormattedText([("class:placeholder", PLACEHOLDER if self.fresh else "")]),
                    refresh_interval=0.125)
        except (EOFError, KeyboardInterrupt):
            pass
        self.agent.cancel.set()
        if self.ui.question:
            self.ui.question.answer(len(self.ui.question.options) - 1)
        self.inbox.put(None)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="ghost", description="Ghost Code: a security and coding agent in your terminal.")
    p.add_argument("prompt", nargs="*", help="Start with this prompt.")
    p.add_argument("-p", "--print", action="store_true", help="Answer the prompt and exit, without the interactive UI.")
    p.add_argument("-c", "--continue", dest="resume", action="store_true", help="Resume the last session in this folder.")
    p.add_argument("--model", help="Model name at the endpoint.")
    p.add_argument("--base-url", help=f"OpenAI-compatible endpoint, default {DEFAULT_URL}")
    p.add_argument("--mode", choices=MODES, default="default", help="Starting permission mode.")
    p.add_argument("--allow-all", action="store_true", help="Run every tool without asking. Use with care.")
    p.add_argument("--verbose", action="store_true", help="With --print: show tool calls on stderr.")
    p.add_argument("--version", action="version", version=f"Ghost Code {__version__}")
    args = p.parse_args(argv)
    prompt = " ".join(args.prompt) or None
    if args.print and not prompt:
        prompt = sys.stdin.read().strip() if not sys.stdin.isatty() else None
        if not prompt:
            p.error("--print needs a prompt")

    backend = build_backend(args)
    cwd = Path.cwd()
    if args.print:
        agent = Agent(backend, PlainUI(args.verbose), cwd, args.mode)
        agent.allow_all = args.allow_all
        try:
            print(agent.run_turn(prompt))
        except BackendError as e:
            sys.exit(f"ghost: {e}")
        return

    agent = Agent(backend, None, cwd, args.mode)
    agent.allow_all = args.allow_all

    agent.ui = ui = TerminalUI()
    if args.resume:
        try:
            agent.messages = json.loads(session_file(agent.cwd).read_text())["messages"]
            ui.notice(f"Resumed {len(agent.messages)} messages from the last session here")
        except (OSError, ValueError, KeyError):
            ui.notice("No earlier session in this folder")
    Repl(agent, ui).run(prompt)


if __name__ == "__main__":
    main()
