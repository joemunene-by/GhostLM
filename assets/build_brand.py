"""Draws the GhostLM app icon and README banner. Run: python assets/build_brand.py

Writes the SVG sources here; needs rsvg-convert and iconutil for the PNG and .icns exports.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ghostlm.code import mascot  # noqa: E402

OUT = Path(__file__).parent
INK, PAPER = "#0a0a0b", "#f5f5f4"


def ghost(cx, top, r, bottom, feet=3):
    w = 2 * r / feet
    d = f"M{cx - r},{bottom} L{cx - r},{top + r} A{r},{r} 0 0 1 {cx + r},{top + r} L{cx + r},{bottom} "
    return d + "".join(f"a{w / 2},{w / 2} 0 0 1 {-w},0 " for _ in range(feet)) + "Z"


def icon():
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1024 1024">
<defs><linearGradient id="tile" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#1d1d20"/><stop offset="1" stop-color="#050506"/></linearGradient></defs>
<rect x="100" y="100" width="824" height="824" rx="186" fill="url(#tile)"/>
<path d="{ghost(512, 235, 225, 715)}" fill="{PAPER}"/>
<polyline points="388,420 468,480 388,540" fill="none" stroke="{INK}" stroke-width="40" stroke-linecap="round" stroke-linejoin="round"/>
<rect x="520" y="512" width="120" height="38" rx="19" fill="{INK}"/>
<rect x="101.5" y="101.5" width="821" height="821" rx="184.5" fill="none" stroke="#ffffff" stroke-opacity="0.12" stroke-width="3"/>
</svg>
'''


def token_ghost(x0, top, cell, gap):
    """The mascot grid from ghostlm/code/mascot.py, one rounded square per token."""
    out = []
    for row, line in enumerate(mascot.grid()):
        for col, level in enumerate(line):
            if level:
                x, y = x0 + col * (cell + gap), top + row * (cell + gap)
                out.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="{cell * 0.22:.1f}" '
                           f'fill="{PAPER}" fill-opacity="{level}"/>')
    return "\n".join(out)


def banner():
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 400">
<rect width="1280" height="400" rx="28" fill="{INK}"/>
{token_ghost(130, 34, 22, 6)}
<text x="500" y="218" font-family="Menlo, monospace" font-weight="bold" font-size="112" letter-spacing="-3" fill="{PAPER}">GhostLM</text>
<text x="506" y="278" font-family="Menlo, monospace" font-size="27" fill="{PAPER}" fill-opacity="0.55">a language model built from scratch</text>
</svg>
'''


def export():
    (OUT / "ghostlm_icon.svg").write_text(icon())
    (OUT / "ghostlm_banner.svg").write_text(banner())
    subprocess.run(["rsvg-convert", "-w", "2560", OUT / "ghostlm_banner.svg", "-o", OUT / "ghostlm_banner.png"], check=True)
    for size in (128, 256, 512, 1024):
        subprocess.run(["rsvg-convert", "-w", str(size), OUT / "ghostlm_icon.svg", "-o", OUT / f"ghostlm_icon_{size}.png"], check=True)
    with tempfile.TemporaryDirectory() as tmp:
        iconset = Path(tmp) / "GhostLM.iconset"
        iconset.mkdir()
        for size in (16, 32, 128, 256, 512):
            for scale in (1, 2):
                name = f"icon_{size}x{size}{'@2x' if scale == 2 else ''}.png"
                subprocess.run(["rsvg-convert", "-w", str(size * scale), OUT / "ghostlm_icon.svg", "-o", iconset / name], check=True)
        subprocess.run(["iconutil", "-c", "icns", iconset, "-o", OUT / "GhostLM.icns"], check=True)


if __name__ == "__main__":
    export()
