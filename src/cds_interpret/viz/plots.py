"""Dependency-free SVG charts for the toolkit's reports (no plotting library needed).

    svg = heatmap(rows, cols, values, title=..., vmin=0.5, vmax=1.0, fmt="{:.2f}")
    svg = lines({"name": (xs, ys), ...}, title=..., xlabel=..., ylabel=..., vlines={20.0: "demos appear"})
    svg = bars(labels, values, title=...)
    save(svg, "figure.svg")

Every chart carries its own light / dark colours (a <style> with prefers-color-scheme), hairline grids, 2 px lines,
8 px markers with a surface ring, a legend for two or more series, text in ink colours (never the series colour), and
a <title> tooltip on every mark. Categorical colours are taken in a fixed order (validated palette: blue, orange,
aqua, yellow, magenta, green); magnitudes use one blue ramp.
"""
from __future__ import annotations

import html
import math

import numpy as np

SERIES_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SERIES_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300"]
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
        "#184f95", "#104281", "#0d366b"]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _style() -> str:
    light = "".join(f"--s{i}:{c};" for i, c in enumerate(SERIES_LIGHT))
    dark = "".join(f"--s{i}:{c};" for i, c in enumerate(SERIES_DARK))
    return ("<style>"
            f"svg{{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;{light}}}"
            f"@media (prefers-color-scheme: dark){{svg{{--surface:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--muted:#898781;"
            f"--grid:#2c2c2a;--axis:#383835;{dark}}}}}"
            f"text{{font-family:{FONT};fill:var(--ink2);font-size:12px}} .t{{fill:var(--ink);font-size:14px;font-weight:600}}"
            ".m{fill:var(--muted);font-size:11px} .g{stroke:var(--grid);stroke-width:1} .ax{stroke:var(--axis);stroke-width:1}"
            "</style>")


def _open(w: int, h: int, title: str) -> list[str]:
    return [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
            f'aria-label="{html.escape(title)}">', _style(),
            f'<rect width="{w}" height="{h}" fill="var(--surface)"/>',
            f'<text class="t" x="16" y="24">{html.escape(title)}</text>']


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        return [lo]
    raw = (hi - lo) / max(n - 1, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    start = math.ceil(lo / step - 1e-9) * step
    out, v = [], start
    while v <= hi + 1e-9 * step:
        out.append(round(v, 10))
        v += step
    return out


def _fmt_tick(v: float) -> str:
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:g}"


def _ramp(t: float) -> str:
    t = min(max(t, 0.0), 1.0)
    return RAMP[int(round(t * (len(RAMP) - 1)))]


def _ink_on(hex_color: str) -> str:
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#0b0b0b" if lum > 0.55 else "#ffffff"


def heatmap(rows: list[str], cols: list[str], values, title: str, vmin: float = 0.0, vmax: float = 1.0,
            fmt: str = "{:.2f}", cell: int = 52, row_w: int = 150, note: str = "") -> str:
    """values [len(rows), len(cols)] (NaN = empty cell) on one blue ramp from vmin (lightest) to vmax (darkest), the
    number printed in each cell."""
    vals = np.asarray(values, float)
    w = row_w + cell * len(cols) + 60
    h = 70 + 22 * len(rows) + 40 + (18 if note else 0)
    ch = 22
    out = _open(w, h, title)
    top = 64
    for j, c in enumerate(cols):
        x = row_w + cell * j + cell / 2
        out.append(f'<text class="m" x="{x:.1f}" y="{top - 8}" text-anchor="middle">{html.escape(c)}</text>')
    for i, r in enumerate(rows):
        y = top + ch * i
        out.append(f'<text x="{row_w - 8}" y="{y + 15}" text-anchor="end">{html.escape(r)}</text>')
        for j, _ in enumerate(cols):
            v = vals[i, j]
            x = row_w + cell * j
            if not np.isfinite(v):
                continue
            t = (v - vmin) / max(vmax - vmin, 1e-12)
            col = _ramp(t)
            out.append(f'<rect x="{x + 1}" y="{y + 1}" width="{cell - 2}" height="{ch - 2}" rx="2" fill="{col}">'
                       f'<title>{html.escape(r)} / {html.escape(cols[j])}: {fmt.format(v)}</title></rect>')
            out.append(f'<text x="{x + cell / 2}" y="{y + 15}" text-anchor="middle" style="fill:{_ink_on(col)};font-size:11px">'
                       f'{fmt.format(v)}</text>')
    # scale legend
    ly = top + ch * len(rows) + 16
    for k in range(len(RAMP)):
        out.append(f'<rect x="{row_w + 12 * k}" y="{ly}" width="12" height="8" fill="{RAMP[k]}"/>')
    out.append(f'<text class="m" x="{row_w - 6}" y="{ly + 8}" text-anchor="end">{fmt.format(vmin)}</text>')
    out.append(f'<text class="m" x="{row_w + 12 * len(RAMP) + 6}" y="{ly + 8}">{fmt.format(vmax)}</text>')
    if note:
        out.append(f'<text class="m" x="16" y="{ly + 30}">{html.escape(note)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def lines(series: dict[str, tuple], title: str, xlabel: str = "", ylabel: str = "", w: int = 720, h: int = 360,
          ylim: tuple | None = None, vlines: dict | None = None, bands: dict | None = None, markers: bool = True,
          note: str = "") -> str:
    """series name -> (xs, ys) or (xs, ys, lo, hi) (a shaded interval). One y axis. vlines {x: label}: thin vertical
    markers (e.g. an event in training)."""
    right = (50 + 7 * max(len(n) for n in series)) if len(series) > 1 else 24
    left, top, bottom = 64, 48, 52 + (16 if note else 0)
    pw, ph = w - left - right, h - top - bottom
    xs_all = np.concatenate([np.asarray(s[0], float) for s in series.values()])
    ys_all = np.concatenate([np.asarray(s[1], float) for s in series.values()]
                            + [np.asarray(s[k], float) for s in series.values() if len(s) == 4 for k in (2, 3)])
    ys_all = ys_all[np.isfinite(ys_all)]
    x0, x1 = float(np.nanmin(xs_all)), float(np.nanmax(xs_all))
    y0, y1 = ylim if ylim else (float(ys_all.min()), float(ys_all.max()))
    if y1 <= y0:
        y1 = y0 + 1.0
    if x1 <= x0:
        x1 = x0 + 1.0
    pad = 0.05 * (y1 - y0) if not ylim else 0.0
    y0, y1 = y0 - pad, y1 + pad

    def X(v):
        return left + (v - x0) / (x1 - x0) * pw

    def Y(v):
        return top + (1 - (v - y0) / (y1 - y0)) * ph
    out = _open(w, h, title)
    for t in _nice_ticks(y0, y1):
        out.append(f'<line class="g" x1="{left}" x2="{left + pw}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>')
        out.append(f'<text class="m" x="{left - 6}" y="{Y(t) + 4:.1f}" text-anchor="end">{_fmt_tick(t)}</text>')
    for t in _nice_ticks(x0, x1, 7):
        out.append(f'<text class="m" x="{X(t):.1f}" y="{top + ph + 16}" text-anchor="middle">{_fmt_tick(t)}</text>')
    out.append(f'<line class="ax" x1="{left}" x2="{left + pw}" y1="{top + ph}" y2="{top + ph}"/>')
    for xv, lab in (vlines or {}).items():
        out.append(f'<line class="ax" x1="{X(xv):.1f}" x2="{X(xv):.1f}" y1="{top}" y2="{top + ph}"/>')
        out.append(f'<text class="m" x="{X(xv) + 4:.1f}" y="{top + 10}">{html.escape(lab)}</text>')
    if xlabel:
        out.append(f'<text class="m" x="{left + pw / 2}" y="{h - 14 - (16 if note else 0)}" text-anchor="middle">{html.escape(xlabel)}</text>')
    if ylabel:
        out.append(f'<text class="m" x="14" y="{top + ph / 2}" text-anchor="middle" '
                   f'transform="rotate(-90 14 {top + ph / 2})">{html.escape(ylabel)}</text>')
    for i, (name, s) in enumerate(series.items()):
        col = f"var(--s{i % len(SERIES_LIGHT)})"
        xs, ys = np.asarray(s[0], float), np.asarray(s[1], float)
        ok = np.isfinite(ys)
        if len(s) == 4:
            lo, hi = np.asarray(s[2], float), np.asarray(s[3], float)
            pts = [f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(xs[ok], hi[ok])] + \
                  [f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(xs[ok][::-1], lo[ok][::-1])]
            out.append(f'<polygon points="{" ".join(pts)}" fill="{col}" fill-opacity="0.1"/>')
        line = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(xs[ok], ys[ok]))
        out.append(f'<polyline points="{line}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round" '
                   f'stroke-linecap="round"/>')
        if markers:
            for a, b in zip(xs[ok], ys[ok]):
                out.append(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="4" fill="{col}" stroke="var(--surface)" '
                           f'stroke-width="2"><title>{html.escape(name)}: {a:g}, {b:.3g}</title></circle>')
        if len(series) > 1:
            ly = top + 8 + 20 * i
            out.append(f'<line x1="{left + pw + 16}" x2="{left + pw + 34}" y1="{ly}" y2="{ly}" stroke="{col}" stroke-width="2"/>')
            out.append(f'<text x="{left + pw + 40}" y="{ly + 4}">{html.escape(name)}</text>')
    if note:
        out.append(f'<text class="m" x="16" y="{h - 10}">{html.escape(note)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def bars(labels: list[str], values, title: str, w: int = 720, fmt: str = "{:.2f}", lo=None, hi=None,
         label_w: int = 220, note: str = "") -> str:
    """Horizontal bars, one series (one colour), a value at each bar's tip, optional intervals [lo, hi]."""
    vals = np.asarray(values, float)
    bh, gap = 18, 8
    h = 48 + (bh + gap) * len(labels) + 30 + (16 if note else 0)
    pw = w - label_w - 80
    vmax = float(np.nanmax(np.concatenate([vals, np.asarray(hi, float) if hi is not None else vals]))) or 1.0
    vmin = min(0.0, float(np.nanmin(vals)))
    out = _open(w, h, title)

    def X(v):
        return label_w + (v - vmin) / (vmax - vmin) * pw
    for i, (lab, v) in enumerate(zip(labels, vals)):
        y = 44 + i * (bh + gap)
        out.append(f'<text x="{label_w - 8}" y="{y + 13}" text-anchor="end">{html.escape(lab)}</text>')
        x_a, x_b = X(min(0.0, v)), X(max(0.0, v))
        out.append(f'<rect x="{x_a:.1f}" y="{y}" width="{max(x_b - x_a, 1):.1f}" height="{bh}" rx="4" fill="var(--s0)">'
                   f'<title>{html.escape(lab)}: {fmt.format(v)}</title></rect>')
        if lo is not None and hi is not None:
            out.append(f'<line x1="{X(lo[i]):.1f}" x2="{X(hi[i]):.1f}" y1="{y + bh / 2}" y2="{y + bh / 2}" '
                       f'stroke="var(--ink)" stroke-width="1.5"/>')
        tip = X(hi[i]) if hi is not None else x_b
        out.append(f'<text x="{tip + 6:.1f}" y="{y + 13}">{fmt.format(v)}</text>')
    out.append(f'<line class="ax" x1="{X(0):.1f}" x2="{X(0):.1f}" y1="40" y2="{44 + len(labels) * (bh + gap)}"/>')
    if note:
        out.append(f'<text class="m" x="16" y="{h - 10}">{html.escape(note)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def save(svg: str, path) -> None:
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(svg, encoding="utf-8")
