"""The token ghost from the README banner, drawn with half-block characters."""

from __future__ import annotations

import math
import random

COLS, ROWS = 11, 12
SHADES = {1.0: "#f5f5f4", 0.7: "#a8a8a6", 0.42: "#6b6b6a", 0.2: "#3d3d3d"}
DOME = {0: range(3, 8), 1: range(1, 10)}
EYE_ROWS, EYE_COLS = (4, 5), (2, 3, 7, 8)
FADE = {9: 0.7, 10: 0.42, 11: 0.2}


def grid(phase: int = 0, blink: bool = False, glance: int = 0, reveal: set | None = None, look: int = 0,
         squint: bool = False, mouth: int = 0, lift: int = 0) -> list[list[float]]:
    """Brightness per cell, 0 for empty.

    glance moves the eyes a column left (-1) or right (1), look a row up (-1) or down (1). squint
    halves the eyes, mouth opens a gap 1 or 3 cells wide, and lift drops the body a row to bob.
    """
    shift = phase % 4
    eye_rows = [r + look for r in EYE_ROWS][1 if squint else 0:]
    mouth_cols = {1: (5,), 3: (4, 5, 6)}.get(mouth, ())
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
            if row in eye_rows and (col - glance) in EYE_COLS and not blink:
                on = False
            if row == 7 and col in mouth_cols:
                on = False
            if reveal is not None and (col, row) not in reveal:
                on = False
            line.append(FADE.get(row, 1.0) if on else 0.0)
        cells.append(line)
    if lift:
        cells = [[0.0] * COLS] + cells[:-1]
    return cells


def pose(activity: str, tick: int) -> dict:
    """What the ghost is doing at this tick (8 per second) for each kind of work."""
    hem = tick // 4
    if activity == "read":
        return {"phase": hem, "look": 1, "glance": (-1, -1, 0, 0, 1, 1)[(tick // 2) % 6]}
    if activity == "search":
        return {"phase": hem, "squint": True, "glance": (-1, 1)[(tick // 3) % 2]}
    if activity == "write":
        return {"phase": hem, "look": 1, "lift": (tick // 3) % 2, "mouth": 1 if tick % 16 < 3 else 0}
    if activity == "run":
        return {"phase": tick, "lift": (tick // 2) % 2}
    if activity == "guard":
        return {"phase": hem, "squint": True, "glance": (-1, 0, 1, 0)[(tick // 6) % 4]}
    if activity == "talk":
        return {"phase": hem, "mouth": (1, 3, 1, 0)[(tick // 2) % 4], "blink": tick % 40 == 39}
    if activity == "ask":
        return {"phase": hem, "blink": tick % 16 == 15, "lift": (tick // 8) % 2}
    # Thinking runs longest, so it rotates through four moods of about two and a half seconds each.
    mood, beat = (tick // 20) % 4, tick % 20
    if mood == 0:
        return {"phase": hem, "look": -1, "glance": -1}
    if mood == 1:
        return {"phase": hem, "blink": True, "lift": (beat // 5) % 2}
    if mood == 2:
        return {"phase": hem, "look": -1, "glance": 1}
    return {"phase": tick // 2, "glance": (-1, 0, 1, 0)[(beat // 3) % 4], "lift": (beat // 3) % 2}


def working_frame(activity: str, tick: int) -> list[list[tuple[str, str]]]:
    return render(grid(**pose(activity, tick)))


IDLE_LOOP = 320


def idle(tick: int, span_x: int, span_y: int) -> tuple[int, int, dict, list[str]]:
    """Where the idle ghost is and what it is doing: (x, y, pose, caption lines beside it).

    A forty second loop: drift across the free space, look around, drift back, settle in the corner
    for a nap, then wake up. span_x and span_y are how far it can travel from the top left.
    """
    t, hem = tick % IDLE_LOOP, tick // 4
    height = math.sin(tick / 23) * 0.5 + 0.5
    if t < 88:
        return round(span_x * t / 88), round(span_y * height), {"phase": tick // 2, "glance": 1, "lift": (t // 6) % 2}, []
    if t < 120:
        beat = (t - 88) // 8
        return span_x, round(span_y * height), {"phase": hem, "glance": (-1, 0, 1, 0)[beat], "look": (0, -1, -1, 0)[beat]}, []
    if t < 208:
        x = round(span_x * (208 - t) / 88)
        return x, round(span_y * height), {"phase": tick // 2, "glance": -1, "lift": (t // 6) % 2}, []
    if t < 232:
        settle = (t - 208) / 24
        return 0, round(span_y * (height + (1 - height) * settle)), {"phase": hem, "look": 1}, []
    if t < 296:
        snore = ["", "z", "z z", "z z Z"][((t - 232) // 6) % 4]
        return 0, span_y, {"phase": hem, "blink": True, "lift": ((t - 232) // 10) % 2}, ["", snore]
    yawn = (3, 3, 1, 0)[min(3, (t - 296) // 6)]
    return 0, span_y, {"phase": hem, "mouth": yawn, "blink": (t - 296) % 8 == 7}, []


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
