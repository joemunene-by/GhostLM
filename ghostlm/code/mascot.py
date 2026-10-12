"""The token ghost from the README banner, drawn with half-block characters and animated smoothly.

The ghost is a continuous shape sampled several times per pixel, so eyes, mouth, hem and the whole
body can sit between pixels. Motion is eased between keyframes instead of jumping a cell at a time.
"""

from __future__ import annotations

import math

COLS, ROWS = 11, 12
CANVAS_COLS, CANVAS_ROWS = 12, 7
PAPER = (245, 245, 244)
# Partly covered edge pixels are blended toward this, a typical dark terminal background.
BACKDROP = (20, 20, 20)
SUB = (1 / 8, 3 / 8, 5 / 8, 7 / 8)
DOME = {0: range(3, 8), 1: range(1, 10)}
EYE_ROWS, EYE_COLS = (4, 5), (2, 3, 7, 8)
FADE = {9: 0.7, 10: 0.42, 11: 0.2}


def grid() -> list[list[float]]:
    """The ghost at rest on whole pixels, brightness per cell. Used for the README banner."""
    cells = []
    for row in range(ROWS):
        line = []
        for col in range(COLS):
            on = col in DOME.get(row, range(COLS))
            if row == 10:
                on = col % 4 != 3
            elif row == 11:
                on = col % 4 == 1
            if row in EYE_ROWS and col in EYE_COLS:
                on = False
            line.append(FADE.get(row, 1.0) if on else 0.0)
        cells.append(line)
    return cells


def _noise(col: int, row: int) -> float:
    return ((col * 73 + row * 151 + col * row * 31) % 97) / 97


def _brightness(x: float, y: float, p: dict) -> float:
    """Brightness of the ghost at a point, in its own pixel space (11 wide, 12 tall)."""
    y -= p.get("bob", 0.0)
    if not (0 <= x < COLS and 0 <= y < ROWS):
        return 0.0
    if y < 3.2 and ((x - 5.5) / 5.5) ** 2 + ((y - 3.2) / 3.2) ** 2 > 1:
        return 0.0
    if p.get("reveal", 1.0) < 1.0 and _noise(int(x), int(y)) > p["reveal"]:
        return 0.0
    # The hem moves a whole token at a time; tokens are discrete, so they should not smear.
    slide = (x - int(p.get("hem", 0.0))) % 4
    if y >= 11:
        return 0.2 if 1 <= slide < 2 else 0.0
    if y >= 10:
        return 0.42 if slide < 3 else 0.0
    eye_left, eye_bottom = 2 + p.get("dx", 0.0), 6 + p.get("dy", 0.0)
    if eye_bottom - 2 * p.get("eye", 1.0) <= y < eye_bottom and (
            eye_left <= x < eye_left + 2 or eye_left + 5 <= x < eye_left + 7):
        return 0.0
    mouth = p.get("mouth", 0.0)
    if mouth > 0.05 and abs(x - 5.5) < mouth / 2 and 7.1 <= y < 7.1 + min(1.0, 0.5 + mouth / 4):
        return 0.0
    return 0.7 if y >= 9 else 1.0


def _colour(level: float) -> str:
    return "#" + "".join(f"{round(b + (c - b) * level):02x}" for b, c in zip(BACKDROP, PAPER))


def _level(col: int, row: int, p: dict, fx: float, fy: float) -> float:
    """Brightness of one pixel. Coverage is sharpened so edges stay crisp instead of smearing to grey."""
    hits = [b for b in (_brightness(col + sx - fx, row + sy - fy, p) for sx in SUB for sy in SUB) if b]
    if not hits:
        return 0.0
    cover = ease((len(hits) / len(SUB) ** 2 - 0.2) / 0.6)
    # Faint fringe pixels read as a dark halo on a dark terminal, so thin coverage is dropped outright.
    return sum(hits) / len(hits) * cover if cover >= 0.35 else 0.0


def draw(p: dict | None = None, fx: float = 0.0, fy: float = 0.0) -> list[list[tuple[str, str]]]:
    """Render the ghost into a 12 by 7 character canvas, offset by a fraction of a pixel (fx, fy)."""
    p = p or {}
    levels = [[_level(col, row, p, fx, fy) for col in range(CANVAS_COLS)] for row in range(CANVAS_ROWS * 2)]
    lines = []
    for top_row in range(0, CANVAS_ROWS * 2, 2):
        line = []
        for col in range(CANVAS_COLS):
            top, bottom = levels[top_row][col], levels[top_row + 1][col]
            top, bottom = (top if top > 0.07 else 0), (bottom if bottom > 0.07 else 0)
            if top and bottom:
                line.append((f"fg:{_colour(top)} bg:{_colour(bottom)}", "▀"))
            elif top:
                line.append((f"fg:{_colour(top)}", "▀"))
            elif bottom:
                line.append((f"fg:{_colour(bottom)}", "▄"))
            else:
                line.append(("", " "))
        lines.append(line)
    return lines


def ease(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * (3 - 2 * u)


def track(t: float, keys: list[tuple[float, float]]) -> float:
    """Value at time t on a looping keyframe track, eased between keys. The last key sets the loop length."""
    t %= keys[-1][0]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            return v0 + (v1 - v0) * ease((t - t0) / (t1 - t0))
    return keys[-1][1]


def wave(t: float, period: float) -> float:
    """0 to 1 and back, smoothly, once per period."""
    return 0.5 - 0.5 * math.cos(2 * math.pi * t / period)


def blink(t: float, every: float = 3.7) -> float:
    """Eye openness for a natural blink: shut and open again within a quarter second."""
    return track(t, [(0, 1), (every - 0.25, 1), (every - 0.13, 0), (every, 1)])


def pose(activity: str, t: float) -> dict:
    """What the ghost is doing t seconds into each kind of work."""
    if activity == "read":
        return {"hem": t * 0.8, "dy": 0.8, "dx": track(t, [(0, -1), (1.3, 1), (1.6, -1)]),
                "eye": blink(t, 4.1)}
    if activity == "search":
        return {"hem": t * 1.2, "eye": 0.5, "dx": track(t, [(0, -1), (0.45, 1), (0.9, -1)]),
                "dy": track(t, [(0, 0), (1.5, 0), (1.8, 1), (3.3, 1), (3.6, -1), (5.1, -1), (5.4, 0)])}
    if activity == "write":
        return {"hem": t * 1.2, "dy": 0.9, "dx": track(t, [(0, -0.6), (0.7, 0.6), (0.9, -0.6)]),
                "bob": wave(t, 0.9), "mouth": track(t, [(0, 0), (2.6, 0), (2.9, 1.2), (3.3, 0)]), "eye": blink(t)}
    if activity == "run":
        return {"hem": t * 9, "bob": 0.9 * wave(t, 0.42), "dx": 0.6, "eye": 0.85}
    if activity == "guard":
        return {"hem": t * 0.6, "eye": track(t, [(0, 0.5), (5.2, 0.5), (5.5, 1), (6.2, 1), (6.5, 0.5)]),
                "dx": track(t, [(0, -1), (1.6, -1), (3.2, 1), (4.8, 1), (6.5, -1)])}
    if activity == "talk":
        chatter = 1.4 + 1.1 * math.sin(t * 11.0) * (0.6 + 0.4 * math.sin(t * 2.3))
        return {"hem": t * 0.9, "mouth": max(0.0, chatter), "bob": 0.6 * wave(t, 1.3), "eye": blink(t, 3.1),
                "dx": 0.4 * math.sin(t * 0.7)}
    if activity == "ask":
        return {"hem": t * 0.5, "bob": 0.7 * wave(t, 2.2), "eye": blink(t, 2.4),
                "dy": track(t, [(0, 0), (3.0, 0), (3.3, -0.5), (4.2, -0.5), (4.5, 0)])}
    # Thinking runs longest, so it moves through four moods: gaze up left, shut-eyed pondering,
    # gaze up right, then a short pace from side to side.
    return {
        "hem": t * 0.9,
        "dx": track(t, [(0, 0), (0.5, -1), (2.2, -1), (2.7, 0), (5.0, 0), (5.5, 1), (7.2, 1), (7.7, 0),
                        (8.1, -1), (8.7, 1), (9.3, -1), (9.8, 0), (10, 0)]),
        "dy": track(t, [(0, 0), (0.5, -1), (2.2, -1), (2.7, 0), (5.0, 0), (5.5, -1), (7.2, -1), (7.7, 0), (10, 0)]),
        "eye": min(blink(t, 4.3), track(t, [(0, 1), (2.8, 1), (3.1, 0), (4.6, 0), (4.9, 1), (10, 1)])),
        "bob": (0.5 + 0.4 * track(t, [(0, 0), (2.8, 0), (3.2, 1), (4.6, 1), (5.0, 0), (10, 0)])) * wave(t, 1.9),
    }


def working_frame(activity: str, t: float) -> list[list[tuple[str, str]]]:
    return draw(pose(activity, t))


IDLE_LOOP = 40.0


def idle(t: float, span_x: float, span_y: float) -> tuple[float, float, dict, list[str]]:
    """Where the idle ghost is and what it is doing: (x, y, pose, caption lines beside it).

    A forty second loop: drift across the free space, look around, drift back, settle in the corner
    for a nap, then wake up. x is in columns and y in rows from the top left, both fractional.
    """
    t %= IDLE_LOOP
    across = track(t, [(0, 0), (10, 1), (14, 1), (24, 0), (IDLE_LOOP, 0)])
    float_y = track(t, [(0, 1), (3, 0.25), (7, 0.7), (12, 0.2), (17, 0.75), (22, 0.3), (27, 1), (IDLE_LOOP, 1)])
    p = {"hem": t * 1.4, "bob": 0.6 * wave(t, 2.4), "eye": blink(t)}
    caption: list[str] = []
    if t < 10:
        p["dx"] = track(t, [(0, 0), (0.6, 1), (9.2, 1), (10, 0)])
    elif t < 14:
        p["dx"] = track(t - 10, [(0, 0), (0.8, -1), (1.8, 1), (2.8, 0), (4, 0)])
        p["dy"] = track(t - 10, [(0, 0), (0.8, -0.8), (2.8, -0.8), (3.6, 0), (4, 0)])
    elif t < 24:
        p["dx"] = track(t - 14, [(0, 0), (0.6, -1), (9.2, -1), (10, 0)])
    elif t < 27:
        p["dy"] = track(t - 24, [(0, 0), (1.5, 0.8), (3, 0.8)])
    elif t < 36:
        asleep = track(t - 27, [(0, 0), (1.2, 1), (9, 1)])
        p.update(eye=1 - asleep, dy=0.8, bob=0.9 * wave(t, 3.4), hem=27 * 1.4 + (t - 27) * 0.3)
        caption = ["", ["", "z", "z z", "z z Z"][int((t - 28.2) / 0.9) % 4] if t > 28.2 else ""]
    else:
        p.update(eye=track(t - 36, [(0, 0), (0.6, 0.4), (1.2, 0.3), (2.2, 1), (4, 1)]),
                 mouth=track(t - 36, [(0, 0), (0.8, 3), (1.8, 3), (2.6, 0), (4, 0)]),
                 dy=track(t - 36, [(0, 0.8), (2.4, 0), (4, 0)]))
    return span_x * across, span_y * float_y, p, caption


def intro(t: float) -> dict:
    """The ghost assembling itself token by token over the first second."""
    return {"reveal": ease(t / 0.9), "eye": ease((t - 0.6) / 0.4)}


def to_ansi(lines: list[list[tuple[str, str]]]) -> list[str]:
    def code(style: str) -> str:
        out = []
        for part in style.split():
            kind, hexa = part.split(":#")
            r, g, b = (int(hexa[i:i + 2], 16) for i in (0, 2, 4))
            out.append(f"{38 if kind == 'fg' else 48};2;{r};{g};{b}")
        return f"\x1b[{';'.join(out)}m" if out else ""

    return ["".join(f"{code(s)}{ch}\x1b[0m" if s else ch for s, ch in line) for line in lines]
