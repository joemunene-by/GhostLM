"""Terminal rendering: the transcript, plus the status and question state the prompt draws. Monochrome."""

from __future__ import annotations

import random
import sys
import threading
import time
from typing import Callable

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.style import Style
from rich.syntax import ANSISyntaxTheme
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from ghostlm.code import mascot, tools
from ghostlm.code.backend import Reply

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
GHOST_ROWS = mascot.ROWS // 2
VERBS = ["Haunting", "Drifting", "Phasing", "Conjuring", "Materialising", "Whispering", "Lurking", "Hovering",
         "Manifesting", "Flickering", "Rattling chains", "Walking through walls", "Tokenising", "Attending",
         "Sampling", "Decoding", "Unspooling", "Shimmering"]
LABELS = {"read": "Reading", "search": "Searching", "write": "Writing", "run": "Running", "guard": "Inspecting",
          "talk": "Answering", "ask": "Waiting for you"}
TOOL_ACTIVITY = {"Read": "read", "List": "read", "Glob": "read", "Search": "search", "SecretScan": "search",
                 "Write": "write", "Edit": "write", "Bash": "run", "Check": "guard", "CVE": "guard"}
WAVE = ["#3d3d3d", "#6b6b6a", "#a8a8a6", "#f5f5f4", "#a8a8a6", "#6b6b6a"]


def stage(height: int, x: int, y: int, ghost: list[list[tuple[str, str]]],
          lines: list[list[tuple[str, str]]]) -> list[tuple[str, str]]:
    """The strip above the prompt: the ghost at (x, y) with text lines to its right, row for row."""
    out = []
    for row in range(height):
        i = row - y
        if 0 <= i < len(ghost):
            out += [("", " " * (x + 1)), *ghost[i], ("", "   "), *(lines[i] if i < len(lines) else [])]
        out.append(("", "\n"))
    return out


def idle_stage(width: int, height: int, tick: int) -> list[tuple[str, str]]:
    """Nothing to do: the ghost wanders the free space, looks around and naps."""
    x, y, pose, caption = mascot.idle(tick, max(0, width - mascot.COLS - 8), height - GHOST_ROWS)
    return stage(height, x, y, mascot.render(mascot.grid(**pose)), [[("class:dim", line)] for line in caption])


def intro_stage(height: int, step: int) -> list[tuple[str, str]]:
    frames = mascot.intro_frames()
    return stage(height, 0, height - GHOST_ROWS, frames[min(step, len(frames) - 1)], [])


class Status:
    """What the agent is doing during a turn, drawn above the prompt with the ghost acting it out."""

    HOLD = 0.9

    def __init__(self):
        self.started, self.text, self.verb = time.time(), "", random.choice(VERBS)
        self.activity, self.since, self.queued = "think", time.time(), None

    def set(self, activity: str, now: bool = False) -> None:
        """Switch activity, letting the current one show for a moment first so quick tools are still seen."""
        if activity == self.activity:
            self.queued = None
        elif now or time.time() - self.since >= self.HOLD:
            self.activity, self.since, self.queued = activity, time.time(), None
        else:
            self.queued = activity

    def fragments(self, width: int, height: int) -> list[tuple[str, str]]:
        if self.queued and time.time() - self.since >= self.HOLD:
            self.activity, self.since, self.queued = self.queued, time.time(), None
        tick = int((time.time() - self.started) * 8)
        label = self.verb if self.activity == "think" else LABELS[self.activity]
        tokens = len(self.text) // 4
        detail = f"{int(time.time() - self.started)}s" + (f" · ↓ {tokens:,} tokens" if tokens else "")
        head = [*[(f"fg:{WAVE[(tick - i) % len(WAVE)]}", "■") for i in range(3)], ("bold", f" {label}…"),
                ("class:dim", f" ({detail} · esc to interrupt)")]
        tail = [ln for ln in self.text.splitlines() if ln.strip()][-3:] if self.activity == "talk" else []
        lines = [[], head, [], *[[("class:dim", ln[:max(10, width - 18)])] for ln in tail]]
        return stage(height, 0, height - GHOST_ROWS, mascot.working_frame(self.activity, tick), lines)


class Question:
    """A permission prompt waiting for the user. The prompt's key bindings answer it."""

    def __init__(self, options: list[str]):
        self.options, self.selected, self.answered, self.asked = options, 0, threading.Event(), time.time()

    def fragments(self, height: int) -> list[tuple[str, str]]:
        lines = [[("", "Do you want to proceed?")]]
        for i, option in enumerate(self.options):
            chosen = i == self.selected
            lines.append([("bold" if chosen else "", f"{'❯' if chosen else ' '} {i + 1}. {option}")])
        lines.append([("class:dim", "enter to confirm · esc to refuse")])
        tick = int((time.time() - self.asked) * 8)
        return stage(height, 0, height - GHOST_ROWS, mascot.working_frame("ask", tick), lines)

    def answer(self, index: int) -> None:
        self.selected = index
        self.answered.set()


class TerminalUI:
    """Prints the transcript. The live parts (status, questions) are state the prompt draws from."""

    def __init__(self, on_change: Callable[[], None] = lambda: None):
        self.console = Console(theme=THEME, highlight=False, force_terminal=True, color_system="truecolor")
        self.on_change = on_change
        self.status: Status | None = None
        self.question: Question | None = None

    def start_reply(self) -> None:
        if self.status:
            self.status.text = ""
            self.status.set("think")

    def on_text(self, chunk: str) -> None:
        if self.status:
            self.status.text += chunk
            self.status.set("talk")

    def on_thinking(self, chunk: str) -> None:
        if self.status:
            self.status.set("think")

    def end_reply(self, reply: Reply) -> None:
        if reply.text.strip():
            self.say(reply.text.strip())

    def say(self, text: str) -> None:
        grid = Table.grid(padding=(0, 1))
        grid.add_column(width=1)
        grid.add_column()
        grid.add_row(Text(BULLET, style="#f5f5f4"), Markdown(text, code_theme=CODE))
        self.console.print()
        self.console.print(grid)

    def echo(self, text: str) -> None:
        self.console.print()
        self.console.print(Text.assemble(("❯ ", "#8a8a8a"), text))

    def tool_call(self, label: str) -> None:
        name, _, rest = label.partition("(")
        if self.status:
            self.status.set(TOOL_ACTIVITY.get(name, "think"))
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
        """Numbered choice, answered from the prompt. The last option is always the refusal."""
        self.console.print()
        self.console.print(Panel(Text(detail), title=title, title_align="left", border_style="#6b6b6a",
                                 padding=(0, 1)))
        question = self.question = Question(options)
        self.on_change()
        question.answered.wait()
        self.question = None
        self.on_change()
        if question.selected:
            self.console.print(Text(f"  ⎿  {options[question.selected]}", style="dim"))
        return question.selected


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
