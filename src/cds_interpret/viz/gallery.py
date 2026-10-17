"""A feature gallery: one card per feature - its name (a guess, said so), how often it fires, what it relates to,
its top moments drawn from above, a short animated clip of the strongest one, and how it grew over training.

    html = gallery_html("My bot's features", intro_html, [feature_card(...), ...])

Self-contained (inline SVG, a small canvas player from field.CLIP_JS), light and dark, any bot's features.
"""
from __future__ import annotations

import html
from collections.abc import Sequence

from .field import CLIP_JS, clip_canvas

CSS = """
:root{--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--hot:#2a78d6}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--card:#1a1a19;--ink:#fff;--ink2:#c3c2b7;
--muted:#898781;--line:#2c2c2a;--hot:#3987e5}}
:root[data-theme="dark"]{--bg:#0d0d0d;--card:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--line:#2c2c2a;--hot:#3987e5}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:24px;margin:0 0 6px} .intro{color:var(--ink2);max-width:780px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px;margin-top:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.card h2{font-size:16px;margin:0} .sub{color:var(--muted);font-size:12px;margin:2px 0 8px}
.guess{display:inline-block;font-size:11px;color:var(--ink2);border:1px solid var(--line);border-radius:999px;padding:0 8px;margin-left:6px}
.stats{display:flex;gap:14px;margin:6px 0 8px;font-size:12px;color:var(--ink2)} .stats b{color:var(--ink);font-size:15px}
table{border-collapse:collapse;width:100%;font-size:12px;margin:6px 0} td{padding:2px 4px;border-top:1px solid var(--line)}
td.n{text-align:right;font-variant-numeric:tabular-nums}
.moments{display:flex;gap:6px;overflow-x:auto;padding-bottom:4px} .moments svg{flex:none}
.desc{font-size:11px;color:var(--muted);margin-top:4px} canvas{border-radius:6px;background:var(--card)}
.row{display:flex;gap:10px;align-items:flex-start} .growth svg{max-width:100%;height:auto}
"""


def feature_card(fid: str, name: str, freq: float, relations: list[tuple[str, str]], moment_svgs: list[str],
                 clip: dict | None = None, growth_svg: str | None = None, note: str = "", descriptions: Sequence[str] = (),
                 guess: bool = True) -> str:
    rel_rows = "".join(f"<tr><td>{html.escape(k)}</td><td class='n'>{html.escape(v)}</td></tr>" for k, v in relations)
    clip_html = clip_canvas(clip, label=f"{name}: its strongest moment, animated") if clip else ""
    descs = "".join(f"<div class='desc'>{i + 1}. {html.escape(d)}</div>" for i, d in enumerate(descriptions))
    return (f"<section class='card' id='f{html.escape(fid)}'><h2>{html.escape(name)}"
            f"{'<span class=guess>name is a guess</span>' if guess else ''}</h2>"
            f"<div class='sub'>feature {html.escape(fid)}{(' - ' + html.escape(note)) if note else ''}</div>"
            f"<div class='stats'><span><b>{freq * 100:.2f}%</b> of moments</span></div>"
            f"<div class='row'>{clip_html}<table>{rel_rows}</table></div>"
            f"<div class='moments'>{''.join(moment_svgs)}</div>{descs}"
            + (f"<div class='growth'>{growth_svg}</div>" if growth_svg else "") + "</section>")


def gallery_html(title: str, intro_html: str, cards: list[str]) -> str:
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{CSS}</style></head>"
            f"<body><main><h1>{html.escape(title)}</h1><div class='intro'>{intro_html}</div>"
            f"<div class='grid'>{''.join(cards)}</div></main><script>{CLIP_JS}</script></body></html>")
