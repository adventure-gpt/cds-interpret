"""Draw the thoughts overlay's panel outside the game, into an SVG: the same drawing code the overlay runs in RLBot,
with a renderer that writes SVG shapes instead of sending them to the game (1920 x 1080 screen, RLBot's 10 x 20
pixel font approximated by a monospace one).

    python scripts/render_panel.py                     # assets/overlay-panel.svg and assets/overlay-panel-team.svg
    python scripts/render_panel.py out.svg [--team]    # one of them, somewhere else
"""

from __future__ import annotations

import html
import sys
from pathlib import Path
from types import SimpleNamespace

from cds_interpret.overlay.panel import draw_thoughts

W, H = 1920, 1080


class SvgRenderer:
    can_render = True

    def __init__(self) -> None:
        self.items: list[str] = []
        self.box = [W, H, 0, 0]

    def _grow(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self.box = [min(self.box[0], x0), min(self.box[1], y0), max(self.box[2], x1), max(self.box[3], y1)]

    def begin_rendering(self, *a: object) -> None:
        pass

    def end_rendering(self) -> None:
        pass

    def draw_rect_2d(self, x: float, y: float, w: float, h: float, color: tuple) -> None:
        r, g, b, a = color
        self.items.append(f'<rect x="{x * W:.1f}" y="{y * H:.1f}" width="{w * W:.1f}" height="{h * H:.1f}" '
                          f'fill="rgb({r},{g},{b})" fill-opacity="{a / 255:.2f}"/>')
        self._grow(x * W, y * H, (x + w) * W, (y + h) * H)

    def draw_string_2d(self, text: str, x: float, y: float, scale: float, fg: tuple, bg: tuple, align: int) -> None:
        r, g, b, _ = fg
        anchor = "end" if align == 2 else "start"
        self.items.append(f'<text x="{x * W:.1f}" y="{y * H + 16 * scale:.1f}" font-size="{17 * scale:.1f}" '
                          f'text-anchor="{anchor}" fill="rgb({r},{g},{b})">{html.escape(text)}</text>')

    def __getattr__(self, name: str):                 # the 3D calls: not part of the panel
        return lambda *a, **k: None


RESULT = {"value": 0.55,
          "action": {"words": "throttle, steer right, boost", "p": 0.81},
          "features": [{"name": "lined up for a shot", "act": 2.1, "on": True},
                       {"name": "ball on our side", "act": 0.0, "on": False},
                       {"name": "low on boost", "act": 0.0, "on": False}],
          "why": [{"name": "lined up for a shot", "push": 0.92}, {"name": "ball on our side", "push": 0.31},
                  {"name": "low on boost", "push": -0.74}],
          "looking": {"own car": 0.46, "ball": 0.38, "opponents": 0.16},
          "probes": {"Threat on goal": 0.08, "First to the ball": 0.74, "Time to the ball": 0.9},
          "looks": {"Threat on goal": {"unit": "%", "color": [240, 90, 80]},
                    "First to the ball": {"unit": "%", "color": [90, 220, 120]},
                    "Time to the ball": {"unit": "s"}}}
TEAM = {**RESULT, "looking": {"own car": 0.41, "ball": 0.33, "opponents": 0.14, "team-mates": 0.12}}


def main(out: str, team: bool = False) -> None:
    flat = SimpleNamespace(Color=lambda r, g, b, a=255: (r, g, b, a),
                           Vector3=lambda x=0.0, y=0.0, z=0.0: SimpleNamespace(x=x, y=y, z=z),
                           CarAnchor=lambda i, v: None, TextHAlign=SimpleNamespace(Left=0, Center=1, Right=2))
    r = SvgRenderer()
    draw_thoughts(r, flat, TEAM if team else RESULT, title="DEMO BOT'S MIND", value_scale=1.0, frame=0, state={},
                  subtitle="car 2 of 3 - on camera" if team else None, available={"why", "looking"})
    x0, y0, x1, y1 = r.box
    pad = 14
    vb = f"{x0 - pad:.0f} {y0 - pad:.0f} {x1 - x0 + 2 * pad:.0f} {y1 - y0 + 2 * pad:.0f}"
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb}" width="{(x1 - x0 + 2 * pad) * 1.4:.0f}" '
           f'role="img" aria-label="The thoughts overlay\'s panel">'
           '<defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#3b5a7a"/>'
           '<stop offset="1" stop-color="#1d2b3a"/></linearGradient></defs>'
           f'<rect x="{x0 - pad:.0f}" y="{y0 - pad:.0f}" width="{x1 - x0 + 2 * pad:.0f}" height="{y1 - y0 + 2 * pad:.0f}" '
           'fill="url(#bg)" rx="10"/>'
           '<g font-family="Consolas, Menlo, \'DejaVu Sans Mono\', monospace">' + "".join(r.items) + "</g></svg>")
    Path(out).write_text(svg, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        main(args[0], team="--team" in sys.argv)
    else:
        main("assets/overlay-panel.svg")
        main("assets/overlay-panel-team.svg", team=True)
