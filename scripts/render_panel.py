"""Draw the thoughts overlay outside the game, into SVG pictures for the docs: the same drawing code the overlay runs
in RLBot, with renderers that write SVG shapes instead of sending them to the game - the panel (1920 x 1080 screen,
RLBot's 10 x 20 pixel font approximated by a monospace one), and a 3v3 moment seen from above with the 3D markers
(each car's landing ring joined to its car, RLBot's cross, the word above the panel's car).

    python scripts/render_panel.py                     # the three pictures in assets/
    python scripts/render_panel.py out.svg [--team]    # one panel picture, somewhere else
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


# --------------------------------------------------------------------------------------- a team moment from above
FS = 0.055                              # picture pixels per unreal unit
FX, FY, GOAL_W, GOAL_D = 4096.0, 5120.0, 893.0, 880.0
OX, OY = (FX + 300) * FS, (FY + GOAL_D + 300) * FS
BLUE, ORANGE = (58, 141, 222), (242, 140, 40)
CARS = {0: (-2300.0, -3500.0, 0), 1: (-900.0, 2700.0, 1), 2: (500.0, -900.0, 0), 3: (2000.0, 3600.0, 1),
        4: (2500.0, -2600.0, 0), 5: (100.0, 4700.0, 1)}
BALL, BALL_LANDS = (800.0, 700.0), (1150.0, 1750.0)
GUESSES = {0: (1750.0, 2350.0), 2: (1000.0, 1700.0), 4: (1650.0, 1250.0)}   # each car's own network's landing guess
CAPTION = ("Blue rings: where each car's own network thinks the ball",
           "will come down, each joined to its car.",
           "White cross: where RLBot's ball prediction says it will.")
MONO = "Consolas, Menlo, 'DejaVu Sans Mono', monospace"
SANS = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def _P(x: float, y: float) -> tuple[float, float]:
    return OX + x * FS, OY - y * FS


class SvgFieldRenderer:
    """The overlay's 3D calls drawn from above (heights left out); its 2D panel calls are left out too."""

    can_render = True

    def __init__(self) -> None:
        self.items: list[str] = []

    @staticmethod
    def _xy(v: object) -> tuple[float, float]:
        if isinstance(v, tuple) and v[0] == "car":
            return _P(*CARS[v[1]][:2])
        return _P(v.x, v.y)  # type: ignore[attr-defined]

    def draw_polyline_3d(self, points: list, color: tuple) -> None:
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (self._xy(p) for p in points))
        self.items.append(f'<polyline points="{pts}" fill="none" stroke="rgb{color[:3]}" stroke-width="1.6"/>')

    def draw_line_3d(self, a: object, b: object, color: tuple) -> None:
        (x0, y0), (x1, y1) = self._xy(a), self._xy(b)
        self.items.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="rgb{color[:3]}" '
                          'stroke-width="1.6"/>')

    def draw_string_3d(self, text: str, anchor: object, scale: float, fg: tuple, bg: tuple, align: int) -> None:
        x, y = self._xy(anchor)
        w = 7.4 * len(text) + 10
        self.items.append(f'<rect x="{x - w / 2:.1f}" y="{y - 38:.1f}" width="{w:.1f}" height="19" rx="3" '
                          'fill="rgb(0,0,0)" fill-opacity="0.63"/>'
                          f'<text x="{x:.1f}" y="{y - 24:.1f}" font-size="12" text-anchor="middle" '
                          f'fill="rgb{fg[:3]}">{html.escape(text)}</text>')

    def __getattr__(self, name: str):                 # (the panel's 2D calls, begin / end)
        return lambda *a, **k: None


def field(out: str) -> None:
    """A 3v3 moment from above: the panel follows car 2 (on camera), every car of the bot has its ring."""
    from cds_interpret.overlay.panel import Panel
    flat = SimpleNamespace(Color=lambda r, g, b, a=255: (r, g, b, a),
                           Vector3=lambda x=0.0, y=0.0, z=0.0: SimpleNamespace(x=x, y=y, z=z),
                           CarAnchor=lambda i, v: ("car", i), TextHAlign=SimpleNamespace(Left=0, Center=1, Right=2))
    r = SvgFieldRenderer()
    Panel().draw(r, flat, RESULT, car_index=2, landing_true=(*BALL_LANDS, 93.0), state={}, frame=0,
                 marks=list(GUESSES.items()), available={"why", "looking"}, team=0)
    w, h = 2 * (FX + 300) * FS, 2 * (FY + GOAL_D + 300) * FS + 64
    line = "#5b7a99"
    x0, y0 = _P(-FX, FY)
    x1, y1 = _P(FX, -FY)
    gx0, gy_top = _P(-GOAL_W, FY + GOAL_D)
    gx1, gy_bot = _P(GOAL_W, -FY - GOAL_D)
    cx, cy = _P(0.0, 0.0)
    parts = [f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" height="{y1 - y0:.1f}" rx="40" fill="#22374b" '
             f'stroke="{line}" stroke-width="2"/>',
             f'<rect x="{gx0:.1f}" y="{gy_top:.1f}" width="{gx1 - gx0:.1f}" height="{y0 - gy_top:.1f}" fill="none" '
             f'stroke="rgb{ORANGE}" stroke-width="2"/>',
             f'<rect x="{gx0:.1f}" y="{y1:.1f}" width="{gx1 - gx0:.1f}" height="{gy_bot - y1:.1f}" fill="none" '
             f'stroke="rgb{BLUE}" stroke-width="2"/>',
             f'<line x1="{x0:.1f}" y1="{cy:.1f}" x2="{x1:.1f}" y2="{cy:.1f}" stroke="{line}"/>',
             f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{1000 * FS:.1f}" fill="none" stroke="{line}"/>']
    (bx, by), (lx, ly) = _P(*BALL), _P(*BALL_LANDS)
    parts.append(f'<line x1="{bx:.1f}" y1="{by:.1f}" x2="{lx:.1f}" y2="{ly:.1f}" stroke="#e1e0d9" stroke-width="1.2" '
                 'stroke-dasharray="3 3"/>')
    ours = sorted(c for c, v in CARS.items() if v[2] == 0)
    for i, (x, y, team) in CARS.items():
        px, py = _P(x, y)
        col = BLUE if team == 0 else ORANGE
        parts.append(f'<rect x="{px - 7:.1f}" y="{py - 11:.1f}" width="14" height="22" rx="3" fill="rgb{col}"/>')
        if team == 0:
            tag = f"car {ours.index(i) + 1}" + (" (on camera)" if i == 2 else "")
            parts.append(f'<text x="{px + 12:.1f}" y="{py + 5:.1f}" font-size="13" fill="#e1e0d9" '
                         f'font-family="{SANS}">{tag}</text>')
    parts.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="{93 * FS * 1.6:.1f}" fill="#f4f4f0"/>')
    caption = "".join(f'<text x="14" y="{h - 50 + 17 * k:.0f}" font-size="13" fill="#c3c2b7" font-family="{SANS}">'
                      f'{html.escape(t)}</text>' for k, t in enumerate(CAPTION))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" role="img" '
           'aria-label="A 3v3 moment from above, with a landing ring for each of the bot&#x27;s three cars">'
           f'<rect width="{w:.0f}" height="{h:.0f}" fill="#1d2b3a" rx="10"/>' + "".join(parts)
           + f'<g font-family="{MONO}">' + "".join(r.items) + "</g>" + caption + "</svg>")
    Path(out).write_text(svg, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        main(args[0], team="--team" in sys.argv)
    else:
        main("assets/overlay-panel.svg")
        main("assets/overlay-panel-team.svg", team=True)
        field("assets/overlay-team-field.svg")
