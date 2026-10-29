"""The study report: one self-contained HTML page (inline SVG, light and dark, no internet needed).

    study.report("report.html")                    # usually through the Study

The page is made of sections, each a function ``(study) -> html`` (empty to leave it out). Add your own - a new
technique's results, say - with the decorator; they appear in ``order`` among the built-in ones (overview 0, probes
10, steadiness 20, features 30, edits 40, attribution 45, attention heads 47, message channel 48, extra 50, notes 90)::

    from cds_interpret.report import section

    @section("My technique", order=35)
    def _(study):
        return "<p>...</p>"
"""

from __future__ import annotations

import html
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import analysis
from ._version import __version__
from .labels import describe_label
from .probes import headline
from .viz import field, gallery, plots


@dataclass
class Section:
    title: str
    order: float
    render: Callable[[Any], str]


SECTIONS: list[Section] = []


def section(title: str, order: float = 60.0) -> Callable[[Callable[[Any], str]], Callable[[Any], str]]:
    """Register a report section: ``fn(study) -> html`` (return "" to leave it out of a page)."""
    def deco(fn: Callable[[Any], str]) -> Callable[[Any], str]:
        SECTIONS[:] = [s for s in SECTIONS if s.title != title]
        SECTIONS.append(Section(title, order, fn))
        return fn
    return deco


CSS = gallery.CSS + """
header{margin-bottom:8px} header .sub{color:var(--ink2);font-size:14px}
section.block{margin:28px 0} section.block>h2{font-size:19px;margin:0 0 6px}
.lead{color:var(--ink2);max-width:820px;margin:0 0 12px}
.figs{display:flex;flex-wrap:wrap;gap:16px;align-items:flex-start} .figs svg{max-width:100%;height:auto}
table.data{border-collapse:collapse;font-size:13px;margin:8px 0;width:auto;min-width:320px}
table.data th{text-align:left;font-weight:600;color:var(--ink2);border-bottom:1px solid var(--line);padding:4px 8px}
table.data td{padding:3px 8px;border-top:1px solid var(--line)} table.data td.n{text-align:right}
.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:1px 10px;margin:2px 4px 2px 0;
font-size:12px;color:var(--ink2)} .ok{color:#1baf7a} .warn{color:#c98500} .bad{color:#d95926}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:2px 16px;font-size:13px} .kv b{color:var(--ink2);font-weight:500}
.scroll{overflow-x:auto}
"""


def _f(v: Any, fmt: str = "{:.3f}") -> str:
    try:
        return fmt.format(v) if np.isfinite(v) else "-"
    except (TypeError, ValueError):
        return "-"


# ---------------------------------------------------------------------------------------------------- the sections
@section("Overview", order=0)
def _overview(study: Any) -> str:
    m = study.moments
    meta = m.meta
    src = meta.get("source", "unknown")
    rows = [("policy", html.escape(repr(study.policy))), ("moments", f"{len(m):,} from {m.n_games} games "
            f"({len(np.unique(study.games[study.test]))} held out for scoring)"),
            ("recorded", html.escape(str(src)) + (f" against {html.escape(str(meta['opponent']))}"
                                                  if meta.get("opponent") else "")),
            ("layers studied", ", ".join(f"<code>{html.escape(x)}</code>" for x in study.layers)),
            ("labels", f"{len(study.labels)}")]
    if src == "synthetic":
        rows.append(("note", "synthetic moments are random states, not play: good for trying the toolkit, not for "
                             "findings about a bot"))
    kv = "".join(f"<b>{k}</b><span>{v}</span>" for k, v in rows)
    return f"<div class='kv'>{kv}</div>"


@section("What it knows", order=10)
def _probes(study: Any) -> str:
    pr = study.probe_results
    if pr is None:
        return ""
    layers = pr.layers
    cols = [*layers, "input", "random"]
    out = ["<p class='lead'>A probe is a linear readout of a layer, fitted on some games and scored on games it never "
           "saw: AUROC for a yes / no concept (0.5 = chance), R² for a number (0 = no better than the average). "
           "<b>input</b> is the same probe on the network's raw input and <b>random</b> on the same layer of a network "
           "with random weights: a concept they read as well is not something the network worked out.</p>"]
    learned = pr.learned()
    if learned:
        out.append("<p>Read clearly better than both baselines: "
                   + "".join(f"<span class='pill' title='{html.escape(describe_label(k))}'>{html.escape(k)}</span>"
                             for k in learned) + "</p>")
    figs = []
    for kind, metric, vmin in (("bin", "AUROC", 0.5), ("reg", "R²", 0.0)):
        labs = [k for k in pr.labels if study.kinds.get(k) == kind and pr.best(k) is not None]
        if not labs:
            continue
        vals = np.full((len(labs), len(cols)), np.nan)
        for i, k in enumerate(labs):
            for j, layer in enumerate(layers):
                vals[i, j] = pr.score(layer, k)
            vals[i, -2] = pr.score("input", k)
            best = pr.best(k)
            vals[i, -1] = pr.score(f"random:{best[0]}", k) if best else np.nan
        figs.append(plots.heatmap(labs, cols, vals, f"{'Yes / no' if kind == 'bin' else 'Number'} concepts: held-out "
                                  f"{metric}", vmin=vmin, vmax=1.0, note="random = random weights, at the label's best layer"))
    out.append(f"<div class='figs scroll'>{''.join(figs)}</div>")
    return "".join(out)


@section("How steady each readout is", order=20)
def _steady(study: Any) -> str:
    pr = study.probe_results
    if pr is None:
        return ""
    rows = []
    for k in pr.labels:
        b = pr.best(k)
        if b is None:
            continue
        r = pr.by_layer[b[0]][k]
        rows.append((k, b[0], headline(r), r.get("wnorm"), r.get("noise_rel"), r.get("jitter"), r.get("jitter_label")))
    if not rows:
        return ""
    body = []
    for k, layer, s, wn, nr, j, jl in rows:
        cls = "ok" if np.isfinite(nr) and nr < 0.1 else "warn" if np.isfinite(nr) and nr < 0.3 else "bad"
        spread = (r.get("noise", np.nan) / nr) if np.isfinite(nr) and nr > 0 else np.nan
        moves = j is not None and jl is not None and np.isfinite(j) and np.isfinite(jl) and np.isfinite(spread)
        ratio = j / jl if moves and jl > 0.02 * spread else np.nan        # (a label that hardly moves: no ratio)
        body.append(f"<tr><td>{html.escape(k)}</td><td><code>{html.escape(layer)}</code></td><td class='n'>{_f(s)}</td>"
                    f"<td class='n'>{_f(wn, '{:.2f}')}</td><td class='n {cls}'>{_f(nr, '{:.3f}')}</td>"
                    f"<td class='n'>{_f(ratio, '{:.2f}')}</td></tr>")
    return ("<p class='lead'>A probe that scores well can still be useless on screen if it leans on units that barely "
            "move. <b>noise</b>: how far the readout moves when every unit of the layer gets a little noise, over its "
            "own spread (under 0.1 is steady). <b>jitter</b>: how far it moves between consecutive moments against "
            "how far the concept itself does (near 1 is honest; far above 1 is noise).</p>"
            "<div class='scroll'><table class='data'><tr><th>label</th><th>best layer</th><th>score</th>"
            "<th>weight norm</th><th>noise</th><th>jitter / label's</th></tr>" + "".join(body) + "</table></div>")


def _distinct(m: Any, idx: np.ndarray, k: int, min_dist: float = 300.0) -> list[int]:
    """The first k moments of idx that differ: no two with the bot's car and the ball both within min_dist uu."""
    out: list[int] = []
    for t in idx:
        car, ball = m.frames.car_pos[t, m.me[t]], m.frames.ball_pos[t]
        if all(np.linalg.norm(car - m.frames.car_pos[u, m.me[u]]) > min_dist
               or np.linalg.norm(ball - m.frames.ball_pos[u]) > min_dist for u in out):
            out.append(int(t))
        if len(out) >= k:
            break
    return out


def feature_cards(study: Any, fs: Any, features: Any = None, n: int = 8, exclude: Any = None,
                  notes: dict[int, str] | None = None, titles: dict[int, str] | None = None) -> list[str]:
    """A gallery card per feature (default: the set's top n): its name or suggested name, how often it fires, what it
    goes with, its three strongest distinct moments drawn from above and in words, and a clip when there are tracks.
    exclude: a [N] mask of moments never used as top moments (kickoffs, say)."""
    cards = []
    m = study.moments
    tracks = m.tracks
    for j in (fs.top(n) if features is None else [int(f) for f in features]):
        act = fs.activation(j)
        if exclude is not None:
            act = np.where(np.asarray(exclude, bool), 0.0, act)
        rel = sorted(fs.relations.get(j, {}).items(), key=lambda kv: -analysis.relation_strength(kv[1]))[:5]
        relations = []
        for k, r in rel:
            if "auroc" in r:
                relations.append((k, f"AUROC {r['auroc']:.2f}, lift {r['lift']:.1f}x"))
            else:
                relations.append((k, f"r {r['r']:+.2f}"))
        top = _distinct(m, analysis.top_moments(act, k=40, groups=m.game, steps=m.step, min_gap=8), 3)
        svgs, descs = [], []
        for t in top:
            svgs.append(field.moment_svg(m.frames, int(t), int(m.me[t]), scale=0.018,
                                         title=f"{act[t]:.2f}"))
            descs.append(analysis.describe(m.frames, int(m.me[t]), int(t)))
        clip = None
        if tracks is not None and len(top) and int(m.game[top[0]]) in tracks._span:
            t0 = int(top[0])
            win, first = tracks.window(int(m.game[t0]), int(m.step[t0]) - 20, int(m.step[t0]) + 10)
            if len(win) > 2:
                clip = field.clip_data(win, 0, len(win) - 1, int(m.me[t0]), int(m.step[t0]) - first, tracks.step_seconds)
        named = j in fs.names
        title = fs.name(j) if named or not titles or j not in titles else f"{titles[j]}?"
        cards.append(gallery.feature_card(str(j), title, float(fs.freq[j]), relations, svgs, clip,
                                          note=(notes or {}).get(j, ""), descriptions=descs, guess=not named))
    return cards


def feature_gallery_html(study: Any, layer: str, features: Any = None, n: int = 24, exclude: Any = None,
                         title: str | None = None) -> str:
    """A page of feature cards for one layer, each with its suggested name and any warnings - the page to look at
    while you name features (Study.feature_gallery writes it)."""
    from .names import suggest
    fs = study.feature_sets[layer]
    listed = suggest(fs, None if features is None else [int(f) for f in features], study.labels, top=n)
    feats = [s.feature for s in listed]                 # (the same list, names and order as study.suggest_names)
    sugg = {s.feature: s for s in listed}

    def note(s: Any) -> str:
        z = s.sizes
        size = (f" When it fires: {z['start']:.2f} to {z['highest']:.2f}, typically {z['typical']:.2f}."
                if z and z["highest"] > 0 else "")
        return f"suggested: {s.name} ({s.detail}).{size}" + "".join(f" - {w}" for w in s.warnings)
    notes = {j: note(s) for j, s in sugg.items()}
    intro = ("<p>Each card is one feature of <code>" + html.escape(layer) + "</code>: the name it has (or the one the "
             "labels suggest), how often it fires, the labels it lines up with, and its three strongest moments, drawn "
             "from above (your car has a ring; it attacks up the page). Confirm or rename them in your names file.</p>")
    return gallery.gallery_html(title or f"Features of {layer}", intro,
                                feature_cards(study, fs, feats, exclude=exclude, notes=notes,
                                              titles={j: s.name for j, s in sugg.items()}))


@section("Features", order=30)
def _features(study: Any) -> str:
    if not study.feature_sets:
        return ""
    out = ["<p class='lead'>A sparse autoencoder rewrites a layer as a few active features out of many. A feature's "
           "name here is a guess from the labels it goes with - check its strongest moments before you trust it.</p>"]
    for layer, fs in study.feature_sets.items():
        e = fs.effect.get("reconstruction", {})
        mean = fs.effect.get("layer's mean", {})
        out.append(f"<h3><code>{html.escape(layer)}</code>: {fs.sae.m} features, {fs.sae.k} active per moment</h3>"
                   f"<p>Held-out variance unexplained: <b>{fs.fvu:.1%}</b>. The policy run on the reconstruction: "
                   f"KL {_f(e.get('kl'), '{:.4f}')}, chosen action changed on {_f(e.get('argmax_changed'), '{:.1%}')} "
                   f"of moments (with the layer replaced by its mean: KL {_f(mean.get('kl'), '{:.4f}')}, "
                   f"{_f(mean.get('argmax_changed'), '{:.1%}')}).</p>")
        out.append(f"<div class='grid'>{''.join(feature_cards(study, fs))}</div>")
    return "".join(out)


@section("Feature checks", order=35)
def _checks(study: Any) -> str:
    checks = getattr(study, "feature_checks", {})
    if not checks:
        return ""
    rows = []
    for (layer, j), c in checks.items():
        for scale, r in c.results.items():
            shifts = ", ".join(f"{k} {v:+.2f}" for k, v in r.get("top_shifts", []))
            rows.append(f"<tr><td><code>{html.escape(layer)}</code></td><td>{html.escape(c.name)} ({j})</td>"
                        f"<td>{'off' if scale == 0 else f'x{scale:g}'}</td><td class='n'>{c.n}</td>"
                        f"<td class='n'>{_f(r['changed'], '{:.1%}')}</td><td class='n'>{_f(r['kl'], '{:.3f}')}</td>"
                        f"<td>{html.escape(shifts) or '-'}</td></tr>")
    return ("<p class='lead'>Each feature turned off (and turned up) on the held-out moments where it fires: how often "
            "the bot's chosen action changes, and how its controls move on average. A feature the bot relies on changes "
            "what it does; a name that is right says how.</p><div class='scroll'><table class='data'><tr><th>layer</th>"
            "<th>feature</th><th>edit</th><th>moments</th><th>action changed</th><th>KL</th><th>controls</th></tr>"
            + "".join(rows) + "</table></div>")


@section("Edits", order=40)
def _edits(study: Any) -> str:
    if not study.edit_results:
        return ""
    rows = []
    for name, e in study.edit_results.items():
        if "kl" in e:
            rows.append(f"<tr><td>{html.escape(name)}</td><td class='n'>{_f(e['kl'], '{:.4f}')}</td>"
                        f"<td class='n'>{_f(e['argmax_changed'], '{:.1%}')}</td><td class='n'>{_f(e['tv'], '{:.3f}')}</td></tr>")
        else:
            rows.append(f"<tr><td>{html.escape(name)}</td><td class='n'>-</td>"
                        f"<td class='n'>{_f(e.get('changed'), '{:.1%}')}</td>"
                        f"<td class='n'>{_f(e.get('control_change'), '{:.3f}')}</td></tr>")
    return ("<p class='lead'>Each edit changes what one layer computes and measures how far the policy's choices move "
            "on the held-out moments. A concept a layer encodes is not always one the policy uses: an edit that "
            "moves the policy is evidence it does - confirm it in play, over enough games for an interval.</p>"
            "<div class='scroll'><table class='data'><tr><th>edit</th><th>KL</th><th>action changed</th>"
            "<th>total variation</th></tr>" + "".join(rows) + "</table></div>")


@section("What decisions rest on", order=45)
def _attribution(study: Any) -> str:
    res = getattr(study, "attribution_results", None)
    if not res:
        return ""
    names = res["groups"]
    head = "".join(f"<th>{html.escape(g)}</th>" for g in names)
    rows = []
    for sit, sh in res["shares"].items():
        rows.append(f"<tr><td>{html.escape(sit)}</td>" + "".join(f"<td class='n'>{_f(sh[g], '{:.2f}')}</td>"
                                                                 for g in names) + "</tr>")
    fa = res.get("faithfulness", {})
    return ("<p class='lead'>Each entity of the game state is kept or swapped for the same entity of another recorded "
            "moment, every combination is run, and exact Shapley values split how far the policy moves: the share "
            "each entity carries. Interactions can make a share negative.</p>"
            f"<div class='scroll'><table class='data'><tr><th>moments</th>{head}</tr>{''.join(rows)}</table></div>"
            f"<p>How faithful the ranking is (1 = the best possible order of removal, 0 = chance): Shapley "
            f"<b>{_f(fa.get('Shapley'), '{:.2f}')}</b>, one entity swapped at a time "
            f"{_f(fa.get('one swapped at a time'), '{:.2f}')}, random {_f(fa.get('random'), '{:.2f}')} "
            f"({res['n']} held-out moments, {res['donors']} donors each).</p>")


@section("What each attention head looks at", order=47)
def _attention(study: Any) -> str:
    maps = getattr(study, "attention_results", None)
    if maps is None:
        return ""
    from .attention import figures
    figs = figures(maps)
    who = (f"the row of the <b>{html.escape(maps.query)}</b> token" if maps.query else
           "the average row of every token that asks (every query)")
    out = [f"<p class='lead'>An attention head lets each token of the network ask the others what they hold, and "
           f"weighs their answers: these are those weights, per head, recomputed from each attention layer's own "
           f"projections (softmax of queries times keys) on {len(maps.focus):,} recorded moments. What a head "
           f"<i>looks at</i> is {who}. A head can look somewhere without the decision resting on it: test what matters "
           "with edits.</p>",
           "<ul class='lead'>" + "".join(f"<li>{html.escape(t)}</li>" for t in maps.describe().values()) + "</ul>",
           f"<div class='figs scroll'>{figs['focus']}{figs['heads']}</div>"]
    if "shifts" in figs:
        out.append("<p class='lead'>How each head's attention moves with the situation: the share it gives each token "
                   "where the situation holds against where it doesn't (every situation but kickoff compared within "
                   "open play). <b>Points</b> are percentage points of the head's attention. A change is printed when "
                   "its 95% interval, from resampling whole games, leaves 0 out.</p>")
        out.append(f"<div class='figs scroll'>{figs['shifts']}</div>")
        details = [v for k, v in figs.items() if k.startswith("shift: ")]
        if details:
            out.append(f"<div class='figs scroll'>{''.join(details)}</div>")
    rows = []
    for r in maps.table():
        rows.append(f"<tr><td>{html.escape(r['head'])}</td><td><code>{html.escape(maps.paths.get(r['layer'], ''))}"
                    f"</code></td><td class='n'>{_f(r['spread'], '{:.2f}')}</td>"
                    f"<td class='n'>{_f(r['entropy'], '{:.2f}')}</td><td class='n'>{_f(r['selfish'], '{:.0%}')}</td></tr>")
    out.append("<div class='scroll'><table class='data'><tr><th>head</th><th>module</th><th>spread (0 one token, 1 "
               "even)</th><th>entropy (nats)</th><th>on itself</th></tr>" + "".join(rows) + "</table></div>")
    skipped = {k: r["skipped"] for k, r in maps.shifts.items() if "skipped" in r and r["n_in"] and r["n_out"]}
    if skipped:
        out.append("<p class='lead'>Not compared: " + "; ".join(f"{html.escape(k)} ({html.escape(v)})"
                                                                for k, v in skipped.items()) + ".</p>")
    return "".join(out)


@section("Message channel", order=48)
def _messages(study: Any) -> str:
    rep = getattr(study, "message_results", None)
    if rep is None:
        return ""
    figs = rep.figures()
    out = ["<p class='lead'>Each car's network also sends one of "
           f"{rep.usage['symbols']} symbols every decision, and its team-mates read it at their next one. Three "
           "questions: which symbols get sent, what they say about the situation (mutual information with each label, "
           "in bits, against the same label shifted in time), and whether the team-mates' play depends on them (their "
           "action distribution with the real symbol against every other symbol and against silence).</p>",
           "<ul class='lead'>" + "".join(f"<li><b>{html.escape(k.capitalize())}.</b> {html.escape(v)}</li>"
                                         for k, v in rep.verdicts.items()) + "</ul>",
           f"<div class='figs scroll'>{''.join(figs.values())}</div>"]
    rows = []
    for s_, ph in rep.phrases("sent").items():
        rows.append(f"<tr><td class='n'>{s_}</td><td class='n'>{_f(rep.usage['share_sent'][s_], '{:.1%}')}</td>"
                    f"<td class='n'>{_f(rep.usage['mean_prob'][s_], '{:.1%}')}</td><td>{html.escape(ph)}</td></tr>")
    if rows:
        out.append("<div class='scroll'><table class='data'><tr><th>symbol</th><th>sent</th><th>mean probability</th>"
                   "<th>when (as sent)</th></tr>" + "".join(rows) + "</table></div>")
    return "".join(out)


@section("More", order=50)
def _extra(study: Any) -> str:
    return "".join(f"<h3>{html.escape(k)}</h3>{v}" for k, v in study.extra_sections.items())


@section("Reading this page well", order=90)
def _notes(study: Any) -> str:
    return ("<ul class='lead'><li>Scores are on games the probes never saw; moments of one game are alike, so a "
            "split by moment would flatter them.</li><li>Always compare with the baselines: the raw input and random "
            "weights.</li><li>A feature's name is a guess until its top moments agree with it.</li><li>An edit that "
            "moves the policy is a claim about the network: test it in whole games, with an interval "
            "(intervene.bootstrap_share).</li></ul>")


def build(study: Any, title: str) -> str:
    """The whole page for a study."""
    parts = []
    for s in sorted(SECTIONS, key=lambda s: s.order):
        body = s.render(study)
        if body:
            parts.append(f"<section class='block'><h2>{html.escape(s.title)}</h2>{body}</section>")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{CSS}</style>"
            f"</head><body><main><header><h1>{html.escape(title)}</h1><div class='sub'>made with cds-interpret "
            f"{__version__}</div></header>{''.join(parts)}</main><script>{field.CLIP_JS}</script></body></html>")
