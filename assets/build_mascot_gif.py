"""Renders assets/ghost_mascot.gif: the Ghost Code mascot in each of its activities.

Run: python assets/build_mascot_gif.py   (needs agg, https://github.com/asciinema/agg)
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ghostlm.code import mascot  # noqa: E402

OUT = Path(__file__).parent / "ghost_mascot.gif"
CELLS = [("think", "Thinking"), ("read", "Reading"), ("search", "Searching"),
         ("write", "Writing"), ("run", "Running a command"), ("guard", "Inspecting security"),
         ("talk", "Answering"), ("ask", "Waiting for you"), ("idle", "Idle")]
# 50 frames a second is the ceiling for a GIF: its frame delay is in hundredths of a second, and
# browsers slow anything under two of them right down.
CELL_W, PER_ROW, FPS, SECONDS = 26, 3, 50, 20
DIM, BOLD, RESET = "\x1b[38;2;138;138;138m", "\x1b[1m", "\x1b[0m"


def cell(activity: str, label: str, t: float) -> list[str]:
    caption = []
    if activity == "idle":
        # Twice speed and no travel, so the whole loop (drift, look around, nap, yawn) fits the reel.
        _, _, pose, caption = mascot.idle(t * 2, 0, 0)
    else:
        pose = mascot.pose(activity, t)
    rows = []
    for i, line in enumerate(mascot.to_ansi(mascot.draw(pose))):
        note = caption[i] if i < len(caption) else ""
        rows.append("  " + line + f" {DIM}{note}{RESET}" + " " * (CELL_W - mascot.CANVAS_COLS - 3 - len(note)))
    return [*rows, f"  {BOLD}{label}{RESET}" + " " * (CELL_W - 2 - len(label)), "", ""]


def frame(t: float) -> str:
    lines = ["", f"  {BOLD}Ghost Code{RESET} {DIM}· the ghost acts out what the agent is doing{RESET}", "", ""]
    for start in range(0, len(CELLS), PER_ROW):
        cells = [cell(a, label, t) for a, label in CELLS[start:start + PER_ROW]]
        lines += ["".join(parts) for parts in zip(*cells)]
    return "\x1b[H" + "\r\n".join(line + "\x1b[K" for line in lines)


def main() -> None:
    height = len(frame(0).split("\r\n"))
    events = [json.dumps({"version": 2, "width": CELL_W * PER_ROW + 2, "height": height}),
              json.dumps([0.0, "o", "\x1b[?25l\x1b[2J"])]
    events += [json.dumps([round(i / FPS, 3), "o", frame(i / FPS)]) for i in range(SECONDS * FPS)]
    with tempfile.NamedTemporaryFile("w", suffix=".cast", delete=False) as f:
        f.write("\n".join(events) + "\n")
    theme = "0a0a0b,f5f5f4," + ",".join(["0a0a0b"] + ["d0d0d0"] * 6 + ["f5f5f4"])
    subprocess.run(["agg", "--theme", theme, "--font-size", "16", "--fps-cap", str(FPS), "--idle-time-limit", "1",
                    "--last-frame-duration", "0.02", f.name, str(OUT)], check=True, stdout=subprocess.DEVNULL)
    Path(f.name).unlink()


if __name__ == "__main__":
    main()
