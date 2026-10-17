"""The thoughts panel: what the overlay draws in the game, and how you design your own.

Out of the box the panel shows, top to bottom: a title; **Outlook** (a bar from the middle: green to the right when
the bot expects to score, red to the left when it is in trouble); **Next move** (the action it leans toward, in words,
and how sure it is); **Noticing** (the named features firing now, each held on screen a moment so nothing flickers);
then every probe of the pack (a chance as a bar under its name, a time in seconds). In 3D it draws where the bot's own
layer thinks the ball will land (a bright blue ring with a crosshair, joined to the car) and where RLBot's prediction
says it will (a white cross).

Everything about it can be changed in a small TOML file, without touching this code - which rows appear and in what
order, their names, kinds, colours, the firing thresholds and hold time, the position and width, the smoothing, the
title and the markers - and Python hooks can draw rows or markers of their own. ``cds-interpret panel`` writes a
commented example of every option; docs/howto/customize-the-panel.md explains each one.

    panel = PanelConfig.load("panel.toml")               # or PanelConfig() for the default
    with_thoughts(MyBot, pack="thoughts.npz", panel=panel)

Two rules hold whatever the config says: every string is cut to the panel's width, and bars sit under their labels.
"""

from __future__ import annotations

import difflib
import importlib
import math
import os
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from ..errors import CDSInterpretError

PAD = 0.006
LINE_H = 0.026
BAR_H = 0.009
BAR_ROW = 0.016                         # a bar's line under its label
CHAR_W = 0.0053                         # one character of RLBot's font at scale 1, as a fraction of the width
FPS = 120

ROW_KINDS = ("outlook", "action", "noticing", "probes", "bar", "percent", "seconds", "value", "centred", "text",
             "spacer", "custom")

DEFAULT_COLORS: dict[str, tuple[int, ...]] = {
    "background": (0, 0, 0, 175), "track": (70, 70, 70, 220), "title": (255, 255, 255), "label": (215, 215, 215),
    "value": (255, 255, 255), "good": (90, 220, 120), "bad": (240, 90, 80), "notice": (255, 196, 64),
    "notice_held": (200, 150, 60), "bar": (0, 170, 255), "divider": (255, 255, 255),
}


# ------------------------------------------------------------------------------------------------------ the config
@dataclass
class RowSpec:
    """One row of the panel (one ``[[rows]]`` table of the TOML file)."""

    kind: str
    label: str | None = None             # the text on the left (default: the source's name)
    source: str | None = None            # what it shows: "value", "probe:<name>" or "feature:<name>"
    color: tuple[int, ...] | None = None  # its bar's (or text's) colour
    scale: float | None = None           # the value that fills a bar (outlook / centred / feature bars)
    format: str | None = None            # for "value" rows: a Python format, e.g. "{:.1f} uu"
    text: str | None = None              # for "text" rows (may hold {value})
    lines: int = 2                       # "action": how many lines the move's words may take
    max: int = 3                         # "noticing": how many features at most
    empty: str = "nothing unusual"       # "noticing": what to say when nothing fires
    features: list[str] | None = None    # "noticing": which features may appear (default: all the pack's)
    words: tuple[str, str, str] = ("in trouble", "even", "good")   # "outlook" / "centred": below, around, above
    threshold: float = 0.15              # "outlook" / "centred": how far from the middle the words change
    height: float = 0.5                  # "spacer": in lines
    hook: str | None = None              # "custom": "module:function" or a name given to Panel(hooks=...)

    def __post_init__(self) -> None:
        if self.kind not in ROW_KINDS:
            close = difflib.get_close_matches(self.kind, ROW_KINDS, n=1)
            hint = f" Did you mean {close[0]!r}?" if close else ""
            raise CDSInterpretError(f"unknown row kind {self.kind!r}.{hint} The kinds: {', '.join(ROW_KINDS)}")
        if self.source is not None and not (self.source == "value" or self.source.startswith(("probe:", "feature:"))):
            raise CDSInterpretError(f"row source {self.source!r}: use \"value\", \"probe:<name>\" or \"feature:<name>\"")
        if self.kind in ("bar", "percent", "seconds", "value", "centred") and self.source is None:
            raise CDSInterpretError(f"a {self.kind!r} row needs a source: source = \"probe:<name>\" (a probe's name in "
                                    "your pack), \"feature:<name>\" or \"value\"")
        if self.kind == "custom" and not self.hook:
            raise CDSInterpretError("a \"custom\" row needs a hook = \"module:function\" (or a name passed in hooks=)")
        if self.color is not None:
            self.color = _rgba(self.color, f"row {self.label or self.kind!r} color")
        self.words = tuple(self.words)  # type: ignore[assignment]


@dataclass
class FeatureStyle:
    """How one named feature is shown (a ``[features."name"]`` table)."""

    label: str | None = None             # the name shown (default: its name in the pack)
    threshold: float | None = None       # fires above this activation (default: the pack's threshold)
    show: bool = True                    # False: never listed, never above the car
    color: tuple[int, ...] | None = None


@dataclass
class Markers:
    """The 3D markers (the ``[markers]`` table)."""

    landing: bool = True                 # the bot's own landing guess: a ring with a crosshair
    landing_color: tuple[int, ...] = (0, 170, 255)
    landing_radius: float = 150.0        # uu
    landing_line: bool = True            # a line from the car to it
    landing_smooth: float = 0.25         # seconds of smoothing (0: every reading as it comes)
    prediction: bool = True              # RLBot's prediction of the landing: a white cross
    prediction_color: tuple[int, ...] = (255, 255, 255)
    car_label: bool = True               # the first firing feature, floating above the car
    car_label_color: tuple[int, ...] | None = None   # default: colors.notice
    hook: str | None = None              # "module:function" drawing markers of your own


@dataclass
class PanelConfig:
    """The whole panel (see the module's notes; every field has a TOML key of the same name)."""

    title: str | None = None             # None: the pack's title (make_pack(title=...)), "THOUGHTS" by default
    x: float = 0.012                     # top-left corner, as fractions of the screen
    y: float = 0.30
    width: float = 0.17                  # the panel's width, as a fraction of the screen's
    text_scale: float = 1.0
    title_scale: float = 1.2
    smoothing: float = 0.35              # numbers move this share of the way to each new reading per draw (1: none)
    hold: float = 0.6                    # seconds a feature stays listed after it stops firing
    draw_every: int = 8                  # frames between draws (8: 15 a second)
    value_scale: float | None = None     # the outlook value that fills its bar (default: the pack's value_scale)
    colors: dict[str, tuple[int, ...]] = field(default_factory=lambda: dict(DEFAULT_COLORS))
    markers: Markers = field(default_factory=Markers)
    features: dict[str, FeatureStyle] = field(default_factory=dict)
    rows: list[RowSpec] = field(default_factory=lambda: [RowSpec("outlook", "Outlook"), RowSpec("action", "Next move"),
                                                         RowSpec("noticing", "Noticing"), RowSpec("probes")])

    def __post_init__(self) -> None:
        if not 0.02 <= self.width <= 1.0:
            raise CDSInterpretError(f"width = {self.width}: it's a share of the screen's width, from 0.02 to 1 "
                                    "(0.15-0.25 suits most screens)")
        if not 0.0 < self.smoothing <= 1.0:
            raise CDSInterpretError(f"smoothing {self.smoothing}: use a share in (0, 1] (1 = no smoothing)")
        self.colors = {**DEFAULT_COLORS, **{k: _rgba(v, f"colors.{k}") for k, v in self.colors.items()}}

    # ---- files
    @classmethod
    def from_dict(cls, d: dict[str, Any], where: str = "panel config") -> PanelConfig:
        known_top = {"panel", "colors", "markers", "features", "rows"}
        _check_keys(d, known_top, where)
        kw: dict[str, Any] = {}
        panel_keys = {f.name for f in fields(cls)} - {"colors", "markers", "features", "rows"}
        p = d.get("panel", {})
        _check_keys(p, panel_keys, f"{where} [panel]")
        _check_types(cls, p, f"{where} [panel]")
        kw.update(p)
        if "colors" in d:
            _check_keys(d["colors"], set(DEFAULT_COLORS), f"{where} [colors]")
            kw["colors"] = d["colors"]
        if "markers" in d:
            _check_keys(d["markers"], {f.name for f in fields(Markers)}, f"{where} [markers]")
            _check_types(Markers, d["markers"], f"{where} [markers]")
            m = dict(d["markers"])
            for k in ("landing_color", "prediction_color", "car_label_color"):
                if m.get(k) is not None:
                    m[k] = _rgba(m[k], f"markers.{k}")
            kw["markers"] = Markers(**m)
        if "features" in d:
            feats = {}
            for name, spec in d["features"].items():
                _check_keys(spec, {f.name for f in fields(FeatureStyle)}, f"{where} [features.\"{name}\"]")
                _check_types(FeatureStyle, spec, f"{where} [features.\"{name}\"]")
                spec = dict(spec)
                if spec.get("color") is not None:
                    spec["color"] = _rgba(spec["color"], f"features.\"{name}\".color")
                feats[name] = FeatureStyle(**spec)
            kw["features"] = feats
        if "rows" in d:
            rows = []
            for i, r in enumerate(d["rows"]):
                at = f"{where} [[rows]] #{i + 1}" + (f" ({r['kind']})" if isinstance(r, dict) and "kind" in r else "")
                for k in (r if isinstance(r, dict) else {}):
                    if k in panel_keys and k not in {f.name for f in fields(RowSpec)}:
                        raise CDSInterpretError(
                            f"{at}: {k!r} is a [panel] setting, not a row's. In TOML every line after a [[rows]] "
                            "heading belongs to that row, so put [panel] settings above the first [[rows]].")
                _check_keys(r, {f.name for f in fields(RowSpec)}, at)
                if "kind" not in r:
                    raise CDSInterpretError(f"{at} has no kind (one of {', '.join(ROW_KINDS)})")
                _check_types(RowSpec, r, at)
                try:
                    rows.append(RowSpec(**r))
                except CDSInterpretError as e:
                    raise CDSInterpretError(f"{at}: {e}") from None
            kw["rows"] = rows
        try:
            return cls(**kw)
        except (TypeError, CDSInterpretError) as e:
            raise CDSInterpretError(f"{where}: {e}") from None

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> PanelConfig:
        """A panel from a TOML file (see ``cds-interpret panel`` for an example of every option)."""
        p = Path(path)
        if not p.exists():
            raise CDSInterpretError(f"no panel config at {p} (write an example with: cds-interpret panel {p.name})")
        try:
            d = tomllib.loads(p.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as e:
            hint = (" A [table] may appear only once in a file: put these lines in the one that's already there "
                    "(the file from `cds-interpret panel` has every table, most lines commented out with #)."
                    if "twice" in str(e) else "")
            raise CDSInterpretError(f"{p} is not valid TOML: {e}.{hint}") from None
        return cls.from_dict(d, where=str(p.name))


def _rgba(c: Any, what: str) -> tuple[int, ...]:
    """A colour from [r, g, b], [r, g, b, a] (0-255) or "#rrggbb" / "#rrggbbaa"."""
    want = "a colour: [r, g, b] or [r, g, b, a] with numbers 0-255, or \"#rrggbb\""
    if isinstance(c, str):
        h = c.strip().lstrip("#")
        if len(h) in (6, 8) and all(ch in "0123456789abcdefABCDEF" for ch in h):
            return tuple(int(h[i:i + 2], 16) for i in range(0, len(h), 2))
        raise CDSInterpretError(f"{what} should be {want}, not {c!r}")
    try:
        vals = tuple(int(v) for v in c)
    except (TypeError, ValueError):
        raise CDSInterpretError(f"{what} should be {want}, not {c!r}") from None
    if len(vals) not in (3, 4) or not all(0 <= v <= 255 for v in vals):
        raise CDSInterpretError(f"{what} should be {want}, not {list(c)}")
    return vals


def _check_types(cls: type, d: Any, where: str) -> None:
    """Numbers where numbers go, words where words go - so a mistake shows when the file loads, not mid-match."""
    if not isinstance(d, dict):
        return
    kinds = {f.name: str(f.type) for f in fields(cls)}
    for k, v in d.items():
        t = kinds.get(k, "")
        if any(w in t for w in ("tuple", "list", "dict", "Markers", "RowSpec", "FeatureStyle")):
            continue
        if "float" in t:
            ok, want = isinstance(v, (int, float)) and not isinstance(v, bool), "a number"
        elif "bool" in t:
            ok, want = isinstance(v, bool), "true or false"
        elif "int" in t:
            ok, want = isinstance(v, int) and not isinstance(v, bool), "a whole number"
        elif "str" in t:
            ok, want = isinstance(v, str), "text in quotes"
        else:
            continue
        if not ok:
            hint = " - without the quotes" if isinstance(v, str) else ""
            raise CDSInterpretError(f"{where}: {k} = {v!r} should be {want}{hint}")


def _check_keys(d: Any, known: set[str], where: str) -> None:
    if not isinstance(d, dict):
        raise CDSInterpretError(f"{where} should be a table, not {type(d).__name__}")
    for k in d:
        if k not in known:
            close = difflib.get_close_matches(k, sorted(known), n=1)
            hint = f" Did you mean {close[0]!r}?" if close else ""
            raise CDSInterpretError(f"{where}: unknown key {k!r}.{hint} Known keys: {', '.join(sorted(known))}")


def _resolve(hook: str | Callable | None, hooks: dict[str, Callable]) -> Callable | None:
    if hook is None or callable(hook):
        return hook
    if hook in hooks:
        return hooks[hook]
    mod, _, attr = hook.partition(":")
    if not attr:
        raise CDSInterpretError(f"hook {hook!r}: give \"module:function\", or pass the function in hooks={{...}}")
    try:
        return getattr(importlib.import_module(mod), attr)
    except (ImportError, AttributeError) as e:
        raise CDSInterpretError(f"hook {hook!r} could not be loaded: {e}") from None


def check_panel(config: PanelConfig | str | os.PathLike[str] | None, pack: Any) -> list[str]:
    """What in a panel config doesn't match a pack - each of these would otherwise just not show: rows whose probe or
    feature the pack doesn't have, [features."..."] styles and noticing lists naming features it doesn't have, an
    outlook row of your own without an outlook. pack: a pack, its .npz file or a Thoughts. [] when all is well."""
    from .pack import Thoughts, load_pack
    cfg = PanelConfig() if config is None else config if isinstance(config, PanelConfig) else PanelConfig.load(config)
    pk = pack.pack if isinstance(pack, Thoughts) else pack if isinstance(pack, dict) else load_pack(pack)
    probes = [str(q["name"]) for q in pk.get("probes", [])]
    feats = [str(f["name"]) for f in pk.get("features", [])]

    def near(name: str, names: list[str], what: str) -> str:
        close = [n for n in names if n.lower() == name.lower()] or difflib.get_close_matches(name, names, n=1)
        hint = f" Did you mean {close[0]!r}?" if close else ""
        return f"{hint} The pack's {what}: {', '.join(repr(n) for n in names) or 'none'}."
    out = []
    custom_rows = cfg.rows != PanelConfig().rows        # (the default panel hides a missing outlook by design)
    for i, r in enumerate(cfg.rows):
        at = f"row #{i + 1} ({r.kind})"
        if r.source and r.source.startswith("probe:") and r.source[6:] not in probes:
            out.append(f"{at} shows probe {r.source[6:]!r}, which the pack doesn't have, so the row won't appear."
                       + near(r.source[6:], probes, "probes"))
        if r.source and r.source.startswith("feature:") and r.source[8:] not in feats:
            out.append(f"{at} shows feature {r.source[8:]!r}, which the pack doesn't have, so the row won't appear."
                       + near(r.source[8:], feats, "features"))
        for n in (r.features or []) if r.kind == "noticing" else []:
            if n not in feats:
                out.append(f"{at} lists feature {n!r}, which the pack doesn't have." + near(n, feats, "features"))
        if r.kind == "outlook" and (r.source or "value") == "value" and pk.get("outlook") is None and custom_rows:
            out.append(f"{at}: the pack has no outlook (there were no goals in its games to learn one from), so the "
                       "row stays hidden.")
    for n in cfg.features:
        if n not in feats:
            out.append(f"[features.\"{n}\"] styles a feature the pack doesn't have, so the style is never used."
                       + near(n, feats, "features"))
    return out


# --------------------------------------------------------------------------------------------------- drawing
def _fit(text: str, width: float, scale: float = 1.0) -> str:
    n = max(4, int(width / (CHAR_W * scale)))
    return text if len(text) <= n else text[:n - 1] + "~"


def _wrap(words: str, width: float, lines: int = 2, scale: float = 1.0) -> list[str]:
    """The action's words over up to ``lines`` lines of the panel's width, split after commas."""
    n = max(4, int(width / (CHAR_W * scale)))
    out, cur = [], ""
    for part in [w.strip() for w in words.split(",") if w.strip()]:
        nxt = f"{cur}, {part}" if cur else part
        if len(nxt) <= n or not cur:
            cur = nxt
        else:
            out.append(cur + ",")
            cur = part
    out.append(cur)
    return [_fit(t, width, scale) for t in out[:max(1, lines)]]


def smooth_point(prev: Any, new: Any, dt: float, tau: float = 0.25) -> tuple[float, ...] | None:
    """A drawn point smoothed over time: ``new`` (x, y) reached from ``prev`` by an exponential moving average with
    time constant ``tau`` seconds, ``dt`` seconds after the last draw. prev None, tau <= 0 or new None: ``new``."""
    if new is None or prev is None or tau <= 0:
        return None if new is None else tuple(float(v) for v in new)
    a = 1.0 - math.exp(-max(float(dt), 0.0) / float(tau))
    return tuple(float(p) + a * (float(q) - float(p)) for p, q in zip(prev, new))


class RowDrawer:
    """What a custom row hook draws with: ``line(left, right)``, ``bar(fraction)``, ``gap(lines)``, ``color(r, g, b)``.
    Strings are cut to the panel's width; a bar always sits under the line before it."""

    def __init__(self, panel: Panel, flat: Any, x: float, y: float):
        self._p, self.flat, self.x = panel, flat, x
        self.cy = y
        self.ops: list[tuple] = []
        cfg = panel.config
        self.width = cfg.width
        self.inner = cfg.width - 2 * PAD
        self.scale = cfg.text_scale

    def color(self, r: int, g: int, b: int, a: int = 255) -> Any:
        return self.flat.Color(r, g, b, a)

    def named(self, name: str) -> Any:
        """A colour of the config by name ("label", "value", "good", "bad", "notice", ...)."""
        return self.color(*self._p.config.colors[name])

    def _text(self, s: str, col: Any, scale: float, right: bool, indent: float, width: float) -> None:
        self.ops.append(("s", _fit(s, width, scale), self.x + (self.width - PAD if right else PAD + indent), self.cy,
                         scale, col, self.flat.TextHAlign.Right if right else self.flat.TextHAlign.Left))

    def line(self, left: str = "", right: str | None = None, left_color: Any = None, right_color: Any = None,
             indent: float = 0.0, split: float = 0.7, scale: float | None = None) -> None:
        """One line: ``left`` at the left edge, ``right`` (if any) at the right edge, then move down a line."""
        sc = self.scale if scale is None else scale
        lw = self.inner * (split if right is not None else 1.0) - indent
        if left:
            self._text(left, left_color or self.named("label"), sc, False, indent, lw)
        if right is not None:
            self._text(right, right_color or self.named("value"), sc, True, 0.0, self.inner * (1 - split))
        self.cy += LINE_H * sc

    def bar(self, frac: float, color: Any = None, centred: bool = False) -> None:
        """A bar under the line above: from the left (fraction 0-1), or from the middle (-1 to 1) when centred."""
        track = self.named("track")
        col = color or self.named("bar")
        top = self.cy + 0.002
        self.ops.append(("r", self.x + PAD, top, self.inner, BAR_H, track))
        if centred:
            f = max(-1.0, min(1.0, frac))
            mid = self.x + PAD + self.inner / 2
            self.ops.append(("r", min(mid, mid + f * self.inner / 2), top, abs(f) * self.inner / 2, BAR_H, col))
            self.ops.append(("r", mid - 0.0008, self.cy - 0.001, 0.0016, BAR_H + 0.006, self.named("divider")))
        else:
            self.ops.append(("r", self.x + PAD, top, self.inner * max(0.0, min(1.0, frac)), BAR_H, col))
        self.cy += BAR_ROW

    def gap(self, lines: float = 0.2) -> None:
        self.cy += LINE_H * lines


class Panel:
    """A panel config with its hooks, ready to draw results. Keep one per bot; ``state`` (a dict you keep between
    draws) holds the smoothing and the features held on screen."""

    def __init__(self, config: PanelConfig | str | os.PathLike[str] | None = None, hooks: dict[str, Callable] | None = None):
        if config is None:
            config = PanelConfig()
        elif not isinstance(config, PanelConfig):
            config = PanelConfig.load(config)
        self.config = config
        self.hooks = dict(hooks or {})
        self._row_hooks = {i: _resolve(r.hook, self.hooks) for i, r in enumerate(config.rows) if r.kind == "custom"}
        self._marker_hook = _resolve(config.markers.hook, self.hooks)

    # ---- values
    def _smooth(self, state: dict[str, Any] | None, key: str, v: float) -> float:
        if state is None:
            return v
        ema = state.setdefault("ema", {})
        old = ema.get(key)
        ema[key] = v if old is None else old + self.config.smoothing * (v - old)
        return float(ema[key])

    def _source(self, th: dict[str, Any], source: str) -> tuple[float | None, str]:
        """(value, unit) of a row's source in a result."""
        if source == "value":
            return th.get("value"), ""
        kind, _, name = source.partition(":")
        if kind == "probe":
            v = th.get("probes", {}).get(name)
            return v, th.get("looks", {}).get(name, {}).get("unit", "")
        for f in th.get("features", []):
            if f["name"] == name:
                return f["act"], ""
        return None, ""

    def _features_on(self, th: dict[str, Any]) -> list[dict[str, Any]]:
        out = []
        for f in th.get("features", []):
            style = self.config.features.get(f["name"], FeatureStyle())
            on = f["on"] if style.threshold is None else f["act"] > style.threshold
            out.append({**f, "on": bool(on), "show": style.show, "label": style.label or f["name"],
                        "color": style.color})
        return out

    # ---- the panel
    def draw(self, renderer: Any, flat: Any, th: dict[str, Any], car_index: int | None = None, landing_true: Any = None,
             landing_think: Any = None, state: dict[str, Any] | None = None, frame: int | None = None,
             value_scale: float | None = None, team: int = 0) -> None:
        """Draw one result (see the module's notes). landing_true: world (x, y, z) from RLBot's prediction;
        landing_think: world (x, y) from the bot's own layer; value_scale: the outlook value that fills its bar when
        the config does not say."""
        cfg = self.config
        d = RowDrawer(self, flat, cfg.x, cfg.y)
        d.line("THOUGHTS" if cfg.title is None else cfg.title, left_color=d.named("title"), scale=cfg.title_scale * cfg.text_scale)
        d.gap(0.05)                                     # (the title takes 1.25 lines in all)
        feats = self._features_on(th)
        placed = {r.source.split(":", 1)[1] for r in cfg.rows if r.source and r.source.startswith("probe:")}
        vscale = cfg.value_scale if cfg.value_scale is not None else (value_scale or 1.0)
        for i, row in enumerate(cfg.rows):
            self._row(d, flat, th, row, i, feats, placed, state, frame, vscale)
        d.cy += 0.002
        renderer.begin_rendering("cds-interpret-thoughts")
        renderer.draw_rect_2d(cfg.x, cfg.y - 0.006, cfg.width, d.cy - cfg.y + 0.006, d.color(*cfg.colors["background"]))
        for op in d.ops:
            if op[0] == "r":
                renderer.draw_rect_2d(op[1], op[2], op[3], op[4], op[5])
            else:
                renderer.draw_string_2d(op[1], op[2], op[3], op[4], op[5], d.color(0, 0, 0, 0), op[6])
        self._markers(renderer, flat, th, feats, car_index, landing_true, landing_think, team)
        renderer.end_rendering()

    def _row(self, d: RowDrawer, flat: Any, th: dict[str, Any], row: RowSpec, i: int, feats: list[dict[str, Any]],
             placed: set[str], state: dict[str, Any] | None, frame: int | None, vscale: float) -> None:
        cfg = self.config
        k = row.kind
        if k == "spacer":
            d.gap(row.height)
        elif k == "text":
            d.gap(0.2)
            d.line(row.text or "", left_color=d.color(*row.color) if row.color else None)
        elif k in ("outlook", "centred"):
            v, _ = self._source(th, row.source or "value")
            if v is None:
                return
            scale = row.scale or (vscale if k == "outlook" else 1.0)
            pos = max(-1.0, min(1.0, self._smooth(state, f"row{i}", float(v)) / max(scale, 1e-9)))
            lo, mid, hi = row.words
            if pos > row.threshold:
                word, col = hi, d.named("good")
            elif pos < -row.threshold:
                word, col = lo, d.named("bad")
            else:
                word, col = mid, d.named("label")
            d.line(row.label or ("Outlook" if k == "outlook" else (row.source or "")), word, right_color=col, split=0.5)
            bar_col = d.color(*row.color) if row.color else (d.named("good") if pos >= 0 else d.named("bad"))
            d.bar(pos, bar_col, centred=True)
        elif k == "action":
            a = th.get("action")
            if not a:
                return
            d.gap(0.2)
            d.line(row.label or "Next move", f"{a['p'] * 100:.0f}%")
            for text in _wrap(a["words"], d.inner - 0.008, row.lines, cfg.text_scale):
                d.line(text, left_color=d.named("value"), indent=0.008)
        elif k == "noticing":
            seen = state.setdefault("seen", {}) if state is not None else {}
            hold = int(round(cfg.hold * FPS))
            allowed = set(row.features) if row.features else None
            shown = []
            for f in feats:
                if not f["show"] or (allowed is not None and f["name"] not in allowed):
                    continue
                if f["on"] and frame is not None:
                    seen[f["name"]] = frame
                recent = frame is not None and f["name"] in seen and frame - seen[f["name"]] < hold
                if f["on"] or recent:
                    shown.append(f)
            d.gap(0.2)
            d.line(row.label or "Noticing")
            if shown:
                for f in shown[:row.max]:
                    base = f["color"] or cfg.colors["notice"]
                    col = base if f["on"] else (f["color"] or cfg.colors["notice_held"])
                    d.line(f"> {f['label']}", left_color=d.color(*col), indent=0.004)
            else:
                d.line(row.empty, indent=0.008)
        elif k == "probes":
            names = [n for n in th.get("probes", {}) if n not in placed]
            if names:
                d.gap(0.2)
            for n in names:
                unit = th.get("looks", {}).get(n, {}).get("unit", "")
                color = th.get("looks", {}).get(n, {}).get("color")
                kind = "bar" if unit == "%" else "seconds" if unit == "s" else "value"
                self._row(d, flat, th, RowSpec(kind, n, f"probe:{n}", tuple(color) if color else None), i, feats,
                          placed, state, frame, vscale)
        elif k in ("bar", "percent", "seconds", "value"):
            v, unit = self._source(th, row.source or "")
            if v is None:
                return
            name = row.source.split(":", 1)[-1] if row.source else ""
            v = self._smooth(state, row.source or f"row{i}", float(v))
            label = row.label or name
            if k == "seconds":
                d.line(label, f"{max(v, 0.0):.1f} s", split=0.72)
            elif k == "value":
                fmt = row.format or ("{:.2f}" if not unit or unit in ("%", "s") else "{:.1f} " + unit)
                d.line(label, fmt.format(v), split=0.70)
            else:
                frac = v / row.scale if row.scale else v
                frac = max(0.0, min(1.0, frac))
                d.line(label, f"{frac * 100:.0f}%", split=0.75)
                if k == "bar":
                    d.bar(frac, d.color(*row.color) if row.color else None)
        elif k == "custom":
            hook = self._row_hooks.get(i)
            if hook is not None:
                hook(d, th, state if state is not None else {})

    def _markers(self, renderer: Any, flat: Any, th: dict[str, Any], feats: list[dict[str, Any]], car_index: int | None,
                 landing_true: Any, landing_think: Any, team: int) -> None:
        m = self.config.markers
        on = [f for f in feats if f["on"] and f["show"]]
        if m.car_label and car_index is not None and on:
            col = m.car_label_color or on[0]["color"] or self.config.colors["notice"]
            renderer.draw_string_3d(on[0]["label"], flat.CarAnchor(car_index, flat.Vector3(0, 0, 120)), 1.2,
                                    flat.Color(*col), flat.Color(0, 0, 0, 160), flat.TextHAlign.Center)
        if m.prediction and landing_true is not None:
            p = flat.Vector3(*[float(c) for c in landing_true])
            col = flat.Color(*m.prediction_color)
            renderer.draw_line_3d(flat.Vector3(p.x - 80, p.y, 20), flat.Vector3(p.x + 80, p.y, 20), col)
            renderer.draw_line_3d(flat.Vector3(p.x, p.y - 80, 20), flat.Vector3(p.x, p.y + 80, 20), col)
        if m.landing and landing_think is not None:
            col = flat.Color(*m.landing_color)
            lx, ly, z, r = float(landing_think[0]), float(landing_think[1]), 20.0, m.landing_radius
            for rad in (r - 10.0, r):           # two close rings for a bold line
                ring = [flat.Vector3(lx + rad * math.cos(2 * math.pi * k / 32), ly + rad * math.sin(2 * math.pi * k / 32), z)
                        for k in range(33)]
                renderer.draw_polyline_3d(ring, col)
            renderer.draw_line_3d(flat.Vector3(lx - r, ly, z), flat.Vector3(lx + r, ly, z), col)
            renderer.draw_line_3d(flat.Vector3(lx, ly - r, z), flat.Vector3(lx, ly + r, z), col)
            if m.landing_line and car_index is not None:
                renderer.draw_line_3d(flat.CarAnchor(car_index, flat.Vector3(0, 0, 0)), flat.Vector3(lx, ly, z), col)
        if self._marker_hook is not None:
            self._marker_hook(renderer, flat, th, {"car_index": car_index, "team": team, "landing_true": landing_true,
                                                   "landing_think": landing_think})


def draw_thoughts(renderer: Any, flat: Any, th: dict[str, Any], car_index: int | None = None, landing_true: Any = None,
                  landing_think: Any = None, title: str | None = None, x: float | None = None, y: float | None = None,
                  value_scale: float = 1.0, state: dict[str, Any] | None = None, frame: int | None = None,
                  config: PanelConfig | None = None) -> None:
    """Draw one result with the default panel (or ``config``); title, x and y override the config's."""
    cfg = config or PanelConfig()
    if title is not None or x is not None or y is not None:
        from dataclasses import replace
        cfg = replace(cfg, title=cfg.title if title is None else title, x=cfg.x if x is None else x,
                      y=cfg.y if y is None else y)
    Panel(cfg).draw(renderer, flat, th, car_index, landing_true, landing_think, state, frame, value_scale)
