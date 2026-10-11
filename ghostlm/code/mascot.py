"""The token ghost from the README banner, drawn with half-block characters."""

from __future__ import annotations

import random

COLS, ROWS = 9, 10
SHADES = {1.0: "#f5f5f4", 0.7: "#a8a8a6", 0.42: "#6b6b6a", 0.2: "#3d3d3d"}
EYES = {(3, 3), (3, 4), (5, 3), (5, 4)}
# The hem thins out over the last three rows; each phase shifts which tokens have dropped away.
HEM = [
    {8: {1, 4, 7}, 9: {0, 2, 3, 5, 6, 8}},
    {8: {2, 5, 8}, 9: {0, 1, 3, 4, 6, 7}},
    {8: {0, 3, 6}, 9: {1, 2, 4, 5, 7, 8}},
]
FADE = {7: 0.7, 8: 0.42, 9: 0.2}


def _in_body(col: int, row: int) -> bool:
    if row == 0:
        return 3 <= col <= 5
    if row in (1, 2):
        return 1 <= col <= 7
    return True


def grid(phase: int = 0, blink: bool = False, reveal: set | None = None) -> list[list[float]]:
    """Brightness per cell, 0 for empty."""
    gaps = HEM[phase % len(HEM)]
    cells = []
    for row in range(ROWS):
        line = []
        for col in range(COLS):
            on = _in_body(col, row) and col not in gaps.get(row, ())
            if (col, row) in EYES and not blink:
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
    """Idle animation: the hem drifts every few ticks and the eyes blink now and then."""
    return render(grid(phase=tick // 3, blink=tick % 23 == 22))


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
