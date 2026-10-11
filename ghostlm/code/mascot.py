"""The token ghost from the README banner, drawn with half-block characters."""

from __future__ import annotations

import random

COLS, ROWS = 11, 12
SHADES = {1.0: "#f5f5f4", 0.7: "#a8a8a6", 0.42: "#6b6b6a", 0.2: "#3d3d3d"}
DOME = {0: range(3, 8), 1: range(1, 10)}
EYE_ROWS, EYE_COLS = (4, 5), (2, 3, 7, 8)
FADE = {9: 0.7, 10: 0.42, 11: 0.2}


def grid(phase: int = 0, blink: bool = False, glance: int = 0, reveal: set | None = None) -> list[list[float]]:
    """Brightness per cell, 0 for empty. glance shifts the eyes a column left (-1) or right (1)."""
    shift = phase % 4
    cells = []
    for row in range(ROWS):
        line = []
        for col in range(COLS):
            on = col in DOME.get(row, range(COLS))
            # The hem ends in feet, then scatters into loose tokens; each phase slides it sideways.
            if row == 10:
                on = (col - shift) % 4 != 3
            elif row == 11:
                on = (col - shift) % 4 == 1
            if row in EYE_ROWS and (col - glance) in EYE_COLS and not blink:
                on = False
            if reveal is not None and (col, row) not in reveal:
                on = False
            line.append(FADE.get(row, 1.0) if on else 0.0)
        cells.append(line)
    return cells


def render(cells: list[list[float]]) -> list[list[tuple[str, str]]]:
    """Pack two pixel rows into each text row as (style, char) fragments."""
    lines = []
    for top_row in range(0, ROWS, 2):
        line = []
        for col in range(COLS):
            top, bottom = cells[top_row][col], cells[top_row + 1][col]
            if top and bottom:
                line.append((f"fg:{SHADES[top]} bg:{SHADES[bottom]}", "▀"))
            elif top:
                line.append((f"fg:{SHADES[top]}", "▀"))
            elif bottom:
                line.append((f"fg:{SHADES[bottom]}", "▄"))
            else:
                line.append(("", " "))
        lines.append(line)
    return lines


def frame(tick: int) -> list[list[tuple[str, str]]]:
    """Idle animation: the hem drifts, the eyes blink, and now and then the ghost glances sideways."""
    beat = tick % 60
    glance = -1 if 30 <= beat < 34 else 1 if 36 <= beat < 40 else 0
    return render(grid(phase=tick // 3, blink=beat in (14, 52), glance=glance))


def intro_frames(steps: int = 14, seed: int = 7) -> list[list[list[tuple[str, str]]]]:
    """The ghost assembling itself token by token."""
    order = [(c, r) for r in range(ROWS) for c in range(COLS)]
    random.Random(seed).shuffle(order)
    frames = []
    for i in range(1, steps + 1):
        frames.append(render(grid(reveal=set(order[: len(order) * i // steps]))))
    return frames


def to_ansi(lines: list[list[tuple[str, str]]]) -> list[str]:
    def code(style: str) -> str:
        out = []
        for part in style.split():
            kind, hexa = part.split(":#")
            r, g, b = (int(hexa[i:i + 2], 16) for i in (0, 2, 4))
            out.append(f"{38 if kind == 'fg' else 48};2;{r};{g};{b}")
        return f"\x1b[{';'.join(out)}m" if out else ""

    return ["".join(f"{code(s)}{ch}\x1b[0m" if s else ch for s, ch in line) for line in lines]
