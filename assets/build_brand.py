"""Draws the GhostLM app icon and README banner. Run: python assets/build_brand.py

Writes the SVG sources here; needs rsvg-convert and iconutil for the PNG and .icns exports.
"""
import subprocess
import tempfile
from pathlib import Path

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
    """Ghost drawn as a 9-wide grid of tokens that thins out toward the hem."""
    step, cols = cell + gap, 9
    r, cx = 4.5 * step - gap / 2, x0 + (9 * step - gap) / 2
    eyes = {(3, 3), (3, 4), (5, 3), (5, 4)}
    drop = {(1, 8), (4, 8), (7, 8), (0, 9), (2, 9), (3, 9), (5, 9), (6, 9), (8, 9)}
    fade = {7: 0.7, 8: 0.42, 9: 0.2}
    out = []
    for row in range(10):
        for col in range(cols):
            mx, my = x0 + col * step + cell / 2, top + row * step + cell / 2
            in_dome = my >= top + r or (mx - cx) ** 2 + (my - top - r) ** 2 <= r * r
            if in_dome and (col, row) not in eyes | drop:
                out.append(f'<rect x="{mx - cell / 2:.1f}" y="{my - cell / 2:.1f}" width="{cell}" height="{cell}" '
                           f'rx="{cell * 0.22:.1f}" fill="{PAPER}" fill-opacity="{fade.get(row, 1)}"/>')
    return "\n".join(out)


def banner():
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 400">
<rect width="1280" height="400" rx="28" fill="{INK}"/>
{token_ghost(150, 62, 23, 6)}
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
