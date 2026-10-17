"""Top-down drawings of game moments (any bot's: they read Frames), as SVG, and short animated clips for HTML pages.

    svg = moment_svg(frames, t, me, landing=(x, y), title="...")    # one moment, seen from car `me`'s side
    data = clip_data(frames, t0, t1, me)                             # a clip's frames as plain lists (JSON-able)
    CLIP_JS                                                          # a small player: <canvas data-clip='...'>

The field is drawn in `me`'s team frame: it attacks up the page. Blue / orange are the two teams (the game's own
colours); our car has a ring; lines from cars and ball show 0.5 s of their present velocity; a cross marks a point
passed in (e.g. the predicted landing).
"""
from __future__ import annotations

import html
import json
from collections.abc import Sequence

import numpy as np

from ..frames import Frames, canon, team_sign

FX, FY, GOAL_W, GOAL_D, CORNER = 4096.0, 5120.0, 892.755, 880.0, 1152.0
BLUE, ORANGE = "#2a78d6", "#eb6834"


def _field_path(s: float, ox: float, oy: float) -> str:
    """The field outline (corners cut at 45 degrees) and both goal boxes, in pixels."""
    def P(x, y):
        return f"{ox + x * s:.1f},{oy - y * s:.1f}"
    pts = [(-FX + CORNER, FY), (-GOAL_W, FY), (-GOAL_W, FY + GOAL_D), (GOAL_W, FY + GOAL_D), (GOAL_W, FY),
           (FX - CORNER, FY), (FX, FY - CORNER), (FX, -FY + CORNER), (FX - CORNER, -FY), (GOAL_W, -FY),
           (GOAL_W, -FY - GOAL_D), (-GOAL_W, -FY - GOAL_D), (-GOAL_W, -FY), (-FX + CORNER, -FY), (-FX, -FY + CORNER),
           (-FX, FY - CORNER)]
    return "M" + " L".join(P(*p) for p in pts) + " Z"


def moment_svg(fr: Frames, t: int, me: int, scale: float = 0.04, landing=None, title: str = "",
               caption: str = "") -> str:
    team = float(fr.car_team[t, me])
    sg = float(team_sign(np.array([team]))[0])
    w = int(2 * FX * scale + 24)
    h = int(2 * (FY + GOAL_D) * scale + 24 + (18 if title else 0) + (16 if caption else 0))
    ox, oy = w / 2, (18 if title else 0) + 12 + (FY + GOAL_D) * scale
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
           f'aria-label="{html.escape(title or "game moment")}">',
           '<style>svg{--surface:#fcfcfb;--line:#c3c2b7;--ink:#0b0b0b;--ink2:#52514e;--ball:#52514e}'
           '@media (prefers-color-scheme: dark){svg{--surface:#1a1a19;--line:#383835;--ink:#fff;--ink2:#c3c2b7;--ball:#e1e0d9}}'
           'text{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:11px;fill:var(--ink2)}</style>',
           f'<rect width="{w}" height="{h}" fill="var(--surface)"/>']
    if title:
        out.append(f'<text x="8" y="14" style="fill:var(--ink);font-weight:600">{html.escape(title)}</text>')
    out.append(f'<path d="{_field_path(scale, ox, oy)}" fill="none" stroke="var(--line)" stroke-width="1.5"/>')
    out.append(f'<line x1="{ox - FX * scale:.1f}" x2="{ox + FX * scale:.1f}" y1="{oy:.1f}" y2="{oy:.1f}" stroke="var(--line)"/>')
    out.append(f'<circle cx="{ox}" cy="{oy}" r="{1000 * scale:.1f}" fill="none" stroke="var(--line)"/>')

    def P(v):
        return ox + v[0] * scale, oy - v[1] * scale
    pos = canon(fr.car_pos[t], sg)
    vel = canon(fr.car_vel[t], sg)
    fwd = canon(fr.car_fwd[t], sg)
    for j in range(fr.n_cars):
        if fr.car_present[t, j] < 0.5:
            continue
        col = BLUE if fr.car_team[t, j] < 0.5 else ORANGE
        x, y = P(pos[j])
        f = fwd[j, :2] / max(np.linalg.norm(fwd[j, :2]), 1e-6)
        r = 120 * scale * 1.6
        tip = (x + f[0] * r * 1.4, y - f[1] * r * 1.4)
        lft = (x - f[0] * r * 0.8 - f[1] * r * 0.8, y + f[1] * r * 0.8 - f[0] * r * 0.8)
        rgt = (x - f[0] * r * 0.8 + f[1] * r * 0.8, y + f[1] * r * 0.8 + f[0] * r * 0.8)
        vx, vy = P(pos[j] + vel[j] * 0.5)
        out.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{vx:.1f}" y2="{vy:.1f}" stroke="{col}" stroke-width="1.5" '
                   f'stroke-opacity="0.6"/>')
        demoed = fr.car_demoed[t, j] > 0.5
        out.append(f'<polygon points="{tip[0]:.1f},{tip[1]:.1f} {lft[0]:.1f},{lft[1]:.1f} {rgt[0]:.1f},{rgt[1]:.1f}" '
                   f'fill="{col}" fill-opacity="{0.25 if demoed else 1}" stroke="var(--surface)" stroke-width="1.5">'
                   f'<title>{"us" if j == me else ("mate" if fr.car_team[t, j] == team else "opponent")}: z {pos[j, 2]:.0f}, '
                   f'{np.linalg.norm(vel[j]):.0f} uu/s, boost {fr.car_boost[t, j]:.0f}</title></polygon>')
        if j == me:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r * 1.9:.1f}" fill="none" stroke="var(--ink)" stroke-width="1.2"/>')
        if fr.car_pos[t, j, 2] > 300:
            out.append(f'<text x="{x + r * 1.6:.1f}" y="{y - r:.1f}">{fr.car_pos[t, j, 2]:.0f}</text>')
    b = canon(fr.ball_pos[t], sg)
    bv = canon(fr.ball_vel[t], sg)
    bx, by = P(b)
    ex, ey = P(b + bv * 0.5)
    out.append(f'<line x1="{bx:.1f}" y1="{by:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="var(--ball)" stroke-width="1.5"/>')
    out.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="{max(92.75 * scale * 1.6, 3):.1f}" fill="var(--ball)" '
               f'stroke="var(--surface)" stroke-width="1.5"><title>ball: z {b[2]:.0f}, {np.linalg.norm(bv):.0f} uu/s</title></circle>')
    if b[2] > 200:
        out.append(f'<text x="{bx + 6:.1f}" y="{by + 12:.1f}">{b[2]:.0f}</text>')
    if landing is not None and np.all(np.isfinite(landing)):
        lx, ly = ox + landing[0] * scale, oy - landing[1] * scale
        out.append(f'<path d="M{lx - 4:.1f},{ly - 4:.1f} L{lx + 4:.1f},{ly + 4:.1f} M{lx - 4:.1f},{ly + 4:.1f} '
                   f'L{lx + 4:.1f},{ly - 4:.1f}" stroke="var(--ink)" stroke-width="1.5"/>')
    if caption:
        out.append(f'<text x="8" y="{h - 5}">{html.escape(caption)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def _mix(c1: str, c2: str, t: float) -> str:
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{int(round(x + (y - x) * t)):02x}" for x, y in zip(a, b))


def diverging(v: float, vmax: float) -> str:
    """Blue (negative) - neutral grey - red (positive), linear in RGB between the poles and the midpoint."""
    t = max(-1.0, min(1.0, v / max(vmax, 1e-12)))
    return _mix("#f0efec", "#c0392b", t) if t >= 0 else _mix("#f0efec", "#1c5cab", -t)


def field_heatmap(values: np.ndarray, xs: np.ndarray, ys: np.ndarray, title: str, vmax: float | None = None,
                  marks: Sequence[tuple[float, float, str]] = (), scale: float = 0.035, caption: str = "") -> str:
    """A value per grid point over the field (team frame: attacking up): values [len(ys), len(xs)] at centres xs, ys
    (uu), a diverging blue-grey-red scale around 0, marks (x, y, label) drawn on top (e.g. where the cars are)."""
    vmax = float(np.nanmax(np.abs(values))) if vmax is None else vmax
    w = int(2 * FX * scale + 24)
    h = int(2 * (FY + GOAL_D) * scale + 24 + 18 + 40 + (16 if caption else 0))
    ox, oy = w / 2, 18 + 12 + (FY + GOAL_D) * scale
    dx = (xs[1] - xs[0]) * scale if len(xs) > 1 else 10
    dy = (ys[1] - ys[0]) * scale if len(ys) > 1 else 10
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
           f'aria-label="{html.escape(title)}">',
           '<style>svg{--surface:#fcfcfb;--line:#52514e;--ink:#0b0b0b;--ink2:#52514e}'
           '@media (prefers-color-scheme: dark){svg{--surface:#1a1a19;--line:#c3c2b7;--ink:#fff;--ink2:#c3c2b7}}'
           'text{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:11px;fill:var(--ink2)}</style>',
           f'<rect width="{w}" height="{h}" fill="var(--surface)"/>',
           f'<text x="8" y="14" style="fill:var(--ink);font-weight:600">{html.escape(title)}</text>']
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            v = values[i, j]
            if not np.isfinite(v):
                continue
            out.append(f'<rect x="{ox + x * scale - dx / 2:.1f}" y="{oy - y * scale - dy / 2:.1f}" width="{dx + 0.4:.1f}" '
                       f'height="{dy + 0.4:.1f}" fill="{diverging(v, vmax)}"><title>({x:.0f}, {y:.0f}): {v:+.3f}</title></rect>')
    out.append(f'<path d="{_field_path(scale, ox, oy)}" fill="none" stroke="var(--line)" stroke-width="1.5"/>')
    out.append(f'<line x1="{ox - FX * scale:.1f}" x2="{ox + FX * scale:.1f}" y1="{oy:.1f}" y2="{oy:.1f}" '
               f'stroke="var(--line)" stroke-opacity="0.5"/>')
    for x, y, lab in marks:
        px, py = ox + x * scale, oy - y * scale
        out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="var(--ink)" stroke="var(--surface)" stroke-width="2"/>')
        out.append(f'<text x="{px + 8:.1f}" y="{py + 4:.1f}" style="fill:var(--ink)">{html.escape(lab)}</text>')
    ly = h - 30 - (16 if caption else 0)
    for k in range(-5, 6):
        out.append(f'<rect x="{w / 2 - 66 + (k + 5) * 12:.1f}" y="{ly}" width="12" height="8" fill="{diverging(k / 5 * vmax, vmax)}"/>')
    out.append(f'<text x="{w / 2 - 72:.1f}" y="{ly + 8}" text-anchor="end">{-vmax:+.2f}</text>')
    out.append(f'<text x="{w / 2 + 72:.1f}" y="{ly + 8}">{vmax:+.2f}</text>')
    if caption:
        out.append(f'<text x="8" y="{h - 6}">{html.escape(caption)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def clip_data(fr: Frames, t0: int, t1: int, me: int, mark: int | None = None, step_seconds: float = 8 / 120) -> dict:
    """Frames t0..t1 (inclusive) of one game as lists in me's team frame, rounded to whole uu: for CLIP_JS.
    mark: the row to outline (the moment itself); step_seconds: the time between rows."""
    t1 = min(t1, len(fr) - 1)
    t0 = max(t0, 0)
    team = float(fr.car_team[t0, me])
    sg = float(team_sign(np.array([team]))[0])
    frames = []
    for t in range(t0, t1 + 1):
        pos = canon(fr.car_pos[t], sg)
        fwd = canon(fr.car_fwd[t], sg)
        b = canon(fr.ball_pos[t], sg)
        cars = []
        for j in range(fr.n_cars):
            if fr.car_present[t, j] < 0.5:
                continue
            ang = float(np.arctan2(fwd[j, 1], fwd[j, 0]))
            cars.append([int(pos[j, 0]), int(pos[j, 1]), int(pos[j, 2]), round(ang, 2), int(fr.car_team[t, j] != team),
                         int(j == me), int(fr.car_demoed[t, j] > 0.5)])
        frames.append({"b": [int(b[0]), int(b[1]), int(b[2])], "c": cars})
    return {"frames": frames, "mark": None if mark is None else int(mark - t0), "dt": float(step_seconds)}


CLIP_JS = r"""
function drawClip(cv){
  const d = JSON.parse(cv.dataset.clip), ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height, s = Math.min(W / 8400, H / 12200), ox = W / 2, oy = H / 2;
  const dark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  const line = dark ? '#383835' : '#c3c2b7', ball = dark ? '#e1e0d9' : '#52514e', ink = dark ? '#fff' : '#0b0b0b';
  const pts = [[-2944,5120],[-893,5120],[-893,6000],[893,6000],[893,5120],[2944,5120],[4096,3968],[4096,-3968],
               [2944,-5120],[893,-5120],[893,-6000],[-893,-6000],[-893,-5120],[-2944,-5120],[-4096,-3968],[-4096,3968]];
  let k = 0;
  function frame(){
    const f = d.frames[k];
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = line; ctx.lineWidth = 1.5; ctx.beginPath();
    pts.forEach((p, i) => { const x = ox + p[0] * s, y = oy - p[1] * s; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.closePath(); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(ox - 4096 * s, oy); ctx.lineTo(ox + 4096 * s, oy); ctx.stroke();
    for (const c of f.c){
      const x = ox + c[0] * s, y = oy - c[1] * s, r = 190 * s;
      ctx.save(); ctx.translate(x, y); ctx.rotate(-c[3]);
      ctx.globalAlpha = c[6] ? 0.25 : 1;
      ctx.fillStyle = c[4] ? '#eb6834' : '#2a78d6';
      ctx.beginPath(); ctx.moveTo(r * 1.4, 0); ctx.lineTo(-r * 0.8, r * 0.8); ctx.lineTo(-r * 0.8, -r * 0.8); ctx.closePath(); ctx.fill();
      ctx.restore();
      if (c[5]){ ctx.strokeStyle = ink; ctx.lineWidth = 1.2; ctx.beginPath(); ctx.arc(x, y, r * 1.9, 0, 6.283); ctx.stroke(); }
    }
    const bx = ox + f.b[0] * s, by = oy - f.b[1] * s;
    ctx.fillStyle = ball; ctx.beginPath(); ctx.arc(bx, by, Math.max(150 * s, 3) * (1 + f.b[2] / 2000), 0, 6.283); ctx.fill();
    if (d.mark !== null && Math.abs(k - d.mark) < 3){ ctx.strokeStyle = ink; ctx.lineWidth = 2; ctx.strokeRect(1, 1, W - 2, H - 2); }
    k = (k + 1) % d.frames.length;
  }
  frame();
  cv._timer = setInterval(frame, d.dt * 1000);
}
document.querySelectorAll('canvas[data-clip]').forEach(drawClip);
"""


def clip_canvas(data: dict, w: int = 168, h: int = 244, label: str = "") -> str:
    return (f'<canvas width="{w}" height="{h}" data-clip=\'{html.escape(json.dumps(data, separators=(",", ":")), quote=True)}\' '
            f'aria-label="{html.escape(label or "animated clip of the moment")}"></canvas>')
