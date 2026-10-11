"""Terminal rendering: transcript, live status line, permission prompts. Monochrome by design."""

from __future__ import annotations

import os
import random
import select
import sys
import threading
import time
from typing import Callable

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.style import Style
from rich.syntax import ANSISyntaxTheme
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from ghostlm.code import tools
from ghostlm.code.backend import Reply

try:
    import termios
    import tty
except ImportError:  # Windows: no raw key reading, prompts fall back to typed answers
    termios = tty = None

THEME = Theme({
    "markdown.code": "bold #f5f5f4 on #262626", "markdown.code_block": "#d4d4d4",
    "markdown.link": "underline", "markdown.link_url": "dim underline",
    "markdown.h1": "bold underline", "markdown.h2": "bold", "markdown.h3": "bold", "markdown.h4": "bold",
    "markdown.item.bullet": "#a8a8a6", "markdown.item.number": "#a8a8a6",
    "markdown.block_quote": "dim", "markdown.hr": "dim",
})
from pygments.token import Comment, Keyword, Name, Number, Operator, String, Token  # noqa: E402

CODE = ANSISyntaxTheme({
    Token: Style(color="#d4d4d4"), Comment: Style(color="#7a7a7a", italic=True), Keyword: Style(color="#f5f5f4", bold=True),
    String: Style(color="#a8a8a6"), Number: Style(color="#a8a8a6"), Operator: Style(color="#8a8a8a"),
    Name.Function: Style(color="#f5f5f4"), Name.Class: Style(color="#f5f5f4", bold=True),
})
BULLET = "●"
VERBS = ["Haunting", "Drifting", "Phasing", "Conjuring", "Materialising", "Whispering", "Lurking", "Hovering",
         "Manifesting", "Flickering", "Rattling chains", "Walking through walls", "Tokenising", "Attending",
         "Sampling", "Decoding", "Unspooling", "Shimmering"]
WAVE = ["#3d3d3d", "#6b6b6a", "#a8a8a6", "#f5f5f4", "#a8a8a6", "#6b6b6a"]


def read_key(timeout: float | None = None) -> str | None:
    """One keypress from a terminal already in cbreak mode."""
    fd = sys.stdin.fileno()
    if timeout is not None and not select.select([fd], [], [], timeout)[0]:
        return None
    data = os.read(fd, 32)
    return {b"\x1b": "esc", b"\x1b[A": "up", b"\x1b[B": "down", b"\r": "enter", b"\n": "enter",
            b"\x03": "ctrl-c"}.get(data, data.decode(errors="ignore"))


class cbreak:
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.saved)


class EscWatcher:
    """Calls back when Esc is pressed while the model is working."""

    def __init__(self, callback: Callable[[], None]):
        self.callback, self._stop, self._thread, self._mode = callback, threading.Event(), None, None

    def start(self) -> None:
        if termios is None or not sys.stdin.isatty():
            return
        self._stop.clear()
        self._mode = cbreak()
        self._mode.__enter__()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._mode.__exit__()
        self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            if read_key(0.1) == "esc":
                self.callback()


class Status:
    """The live line under the transcript while a reply streams in."""

    def __init__(self):
        self.started, self.verb, self.text, self.thinking = time.time(), random.choice(VERBS), "", False

    def __rich__(self):
        tick = int((time.time() - self.started) * 8)
        line = Text()
        for i in range(3):
            line.append("■", style=WAVE[(tick - i) % len(WAVE)])
        elapsed = int(time.time() - self.started)
        line.append(f" {'Thinking' if self.thinking and not self.text else self.verb}…", style="bold")
        tokens = len(self.text) // 4
        detail = f"{elapsed}s" + (f" · ↓ {tokens:,} tokens" if tokens else "") + " · esc to interrupt"
        line.append(f" ({detail})", style="dim")
        tail = [ln for ln in self.text.splitlines() if ln.strip()][-6:]
        if not tail:
            return line
        return Group(Text("\n".join(f"  {ln[:160]}" for ln in tail), style="dim"), Text(""), line)


class TerminalUI:
    def __init__(self, on_interrupt: Callable[[], None] = lambda: None):
        self.console = Console(theme=THEME, highlight=False)
        self.watcher = EscWatcher(on_interrupt)
        self.live: Live | None = None
        self.status: Status | None = None

    def start_reply(self) -> None:
        self.status = Status()
        if self.console.is_terminal:
            self.console.print()
            self.live = Live(self.status, console=self.console, refresh_per_second=8, transient=True)
            self.live.start()
            self.watcher.start()

    def on_text(self, chunk: str) -> None:
        if self.status:
            self.status.text += chunk

    def on_thinking(self, chunk: str) -> None:
        if self.status:
            self.status.thinking = True

    def end_reply(self, reply: Reply) -> None:
        self.watcher.stop()
        if self.live:
            self.live.stop()
            self.live = None
            # The blank line printed before the live region; reclaim it so spacing stays even.
            self.console.file.write("\x1b[1A\x1b[2K")
        self.status = None
        if reply.text.strip():
            self.say(reply.text.strip())

    def say(self, text: str) -> None:
        grid = Table.grid(padding=(0, 1))
        grid.add_column(width=1)
        grid.add_column()
        grid.add_row(Text(BULLET, style="#f5f5f4"), Markdown(text, code_theme=CODE))
        self.console.print()
        self.console.print(grid)

    def tool_call(self, label: str) -> None:
        name, _, rest = label.partition("(")
        self.console.print()
        self.console.print(Text.assemble((BULLET + " ", "#a8a8a6"), (name, "bold"), ("(" + rest, "")),
                           overflow="ellipsis", no_wrap=True)

    def tool_result(self, result: tools.Result) -> None:
        lines = result.summary.splitlines() or [""]
        shown = lines[:4]
        if len(lines) > 4:
            shown.append(f"… +{len(lines) - 4} lines")
        for i, line in enumerate(shown):
            self.console.print(Text(("  ⎿  " if i == 0 else "     ") + line, style="dim"),
                               overflow="ellipsis", no_wrap=True)
        for line in result.diff:
            if line.startswith("+"):
                style = "#f5f5f4 on #2e2e2e"
            elif line.startswith("-"):
                style = "#8a8a8a on #171717"
            else:
                style = "dim"
            self.console.print(Text("       " + line, style=style), overflow="ellipsis", no_wrap=True)

    def notice(self, text: str) -> None:
        self.console.print(Text("  ⎿  " + text, style="dim"))

    def error(self, text: str) -> None:
        self.console.print()
        self.console.print(Text.assemble((BULLET + " ", "bold"), (text, "bold")))

    def ask(self, title: str, detail: str, options: list[str]) -> int:
        """Numbered choice. Returns the option index; the last option is always the refusal."""
        refuse = len(options) - 1
        self.console.print()
        self.console.print(Panel(Text(detail), title=title, title_align="left", border_style="#6b6b6a",
                                 padding=(0, 1)))
        if termios is None or not sys.stdin.isatty():
            return refuse
        selected = 0

        def menu():
            t = Text(" Do you want to proceed?\n")
            for i, opt in enumerate(options):
                t.append(f" {'❯' if i == selected else ' '} {i + 1}. {opt}\n", style="bold" if i == selected else "")
            t.append(" enter to confirm · esc to refuse", style="dim")
            return t

        with cbreak(), Live(menu(), console=self.console, auto_refresh=False, transient=True) as live:
            while True:
                key = read_key()
                if key in ("esc", "ctrl-c"):
                    selected = refuse
                    break
                if key == "enter":
                    break
                if key == "up":
                    selected = (selected - 1) % len(options)
                elif key == "down":
                    selected = (selected + 1) % len(options)
                elif key and key.isdigit() and 1 <= int(key) <= len(options):
                    selected = int(key) - 1
                    break
                live.update(menu(), refresh=True)
        if selected:
            self.console.print(Text(f"  ⎿  {options[selected]}", style="dim"))
        return selected


class PlainUI:
    """For --print: no live region, no prompts. Anything that needs approval is refused."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def on_text(self, chunk: str) -> None: ...
    def on_thinking(self, chunk: str) -> None: ...
    def start_reply(self) -> None: ...
    def end_reply(self, reply: Reply) -> None: ...

    def tool_call(self, label: str) -> None:
        if self.verbose:
            print(f"{BULLET} {label}", file=sys.stderr)

    def tool_result(self, result: tools.Result) -> None:
        if self.verbose:
            print(f"  ⎿  {result.summary.splitlines()[0] if result.summary else ''}", file=sys.stderr)

    def ask(self, title: str, detail: str, options: list[str]) -> int:
        return len(options) - 1

    def notice(self, text: str) -> None:
        print(text, file=sys.stderr)
