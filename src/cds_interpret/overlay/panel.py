"""The thoughts panel: what the overlay draws in the game, and how you design your own.

Out of the box the panel shows, top to bottom: a title (and in 2v2 / 3v3 a line saying whose thoughts these are);
**Outlook** (a bar from the middle: green to the right when the bot expects to score, red to the left when it is in
trouble); **Next move** (the action it leans toward, in words, and how sure it is); **Why this move** (the named
features pushing that move hardest, and the strongest one pushing against it); **Looking at** (how much the move rests
on its own car, the ball, the opponents and the team-mates, as one coloured bar); **Noticing** (the named features
firing now, each held on screen a moment so nothing flickers); then every probe of the pack (a chance as a bar under
its name, a time in seconds). In 3D it draws where each car's own network thinks the ball will land (a bright blue
ring with a crosshair, joined to its car) and where RLBot's prediction says it will (a white cross).

"Why this move" and "Looking at" appear when the overlay can compute them (the pack has what they need and the
overlay was given your bot's policy: docs/howto/why-and-looking.md); without, the panel is the one above without them.

Everything about it can be changed in a small TOML file, without touching this code - which rows appear and in what
order, their names, kinds, colours, the firing thresholds and hold time, the position and width, the smoothing, the
title, whose thoughts it follows in team modes, and the markers - and Python hooks can draw rows or markers of their
own. ``cds-interpret panel`` writes a commented example of every option; docs/howto/customize-the-panel.md explains
each one.

    panel = PanelConfig.load("panel.toml")               # or PanelConfig() for the default
    with_thoughts(MyBot, pack="thoughts.npz", panel=panel)

Three rules hold whatever the config says: every string is cut to the panel's width, bars sit under their labels,
and (unless ``keep_size = false``) every row keeps its full height, so the panel never changes size and no row ever
moves because the one above it got shorter or longer.
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
from .team import DEFAULT_FOLLOW, check_follow, check_landing_for, check_subtitle

PAD = 0.006
LINE_H = 0.026
BAR_H = 0.009
BAR_ROW = 0.016                         # a bar's line under its label
CHAR_W = 0.0053                         # one character of RLBot's font at scale 1, as a fraction of the width
FPS = 120
GROUP = "cds-interpret-thoughts"        # the render group the panel and its markers are drawn in

ROW_KINDS = ("outlook", "action", "why", "looking", "noticing", "probes", "bar", "percent", "seconds", "value",
             "centred", "text", "spacer", "custom")
ENTITY_COLORS = {"own car": "own_car", "ball": "ball", "opponents": "opponents", "team-mates": "team_mates"}

DEFAULT_COLORS: dict[str, tuple[int, ...]] = {
    "background": (0, 0, 0, 175), "track": (70, 70, 70, 220), "title": (255, 255, 255), "label": (215, 215, 215),
    "value": (255, 255, 255), "good": (90, 220, 120), "bad": (240, 90, 80), "notice": (255, 196, 64),
    "notice_held": (200, 150, 60), "bar": (0, 170, 255), "divider": (255, 255, 255),
    "push": (255, 255, 255), "push_strong": (255, 196, 64), "push_against": (240, 90, 80),
    "own_car": (80, 180, 255), "ball": (255, 196, 64), "opponents": (240, 90, 80), "team_mates": (170, 130, 255),
    "team_blue": (80, 160, 255), "team_orange": (255, 150, 40),
}
# per-kind defaults of the row keys several kinds share
_MAX = {"noticing": 3, "why": 2}
_EMPTY = {"noticing": "nothing unusual", "why": "no named idea"}
_WORDS = {"outlook": ("in trouble", "even", "good"), "centred": ("in trouble", "even", "good"),
          "looking": ("car", "ball", "opp", "mates")}


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
    max: int | None = None               # "noticing" / "why": how many features at most (3 / 2)
    empty: str | None = None             # "noticing" / "why": what to say when there are none
    features: list[str] | None = None    # "noticing" / "why": which features may appear (default: all the pack's)
    words: tuple[str, ...] | None = None  # "outlook" / "centred": below, around, above; "looking": the parts' words
    threshold: float = 0.15              # "outlook" / "centred": how far from the middle the words change
    height: float = 0.5                  # "spacer": in lines
    hook: str | None = None              # "custom": "module:function" or a name given to Panel(hooks=...)
    min_push: float = 0.15               # "why": the smallest push on the move worth naming (log-probability)
    strong: float = 0.6                  # "why": a push this big is highlighted (and an "against" one is shown)
    against: bool = True                 # "why": also show the strongest feature pushing against the move
    every: int = 4                       # "looking": computed every Nth reading (the worker's heaviest part)
    donors: int = 3                      # "looking": recorded moments each reading is averaged over
    legend: bool = True                  # "looking": the line of words and percentages under the bar

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
        if self.max is None:
            self.max = _MAX.get(self.kind, 3)
        if self.empty is None:
            self.empty = _EMPTY.get(self.kind, "nothing unusual")
        self.words = tuple(self.words) if self.words is not None else _WORDS.get(self.kind, _WORDS["outlook"])
        want = 4 if self.kind == "looking" else 3 if self.kind in ("outlook", "centred") else None
        if want is not None and len(self.words) != want:
            what = ("the words for own car, ball, opponents and team-mates, e.g. [\"car\", \"ball\", \"opp\", "
                    "\"mates\"]" if want == 4 else "three words: below, around and above the middle, e.g. "
                    "[\"in trouble\", \"even\", \"good\"]")
            raise CDSInterpretError(f"a {self.kind!r} row's words should be {what}, not {list(self.words)}")
        for k, lo in (("lines", 1), ("max", 1), ("every", 1), ("donors", 1)):
            if getattr(self, k) < lo:
                tip = " (to turn the row off, leave it out of [[rows]])" if k == "every" else ""
                raise CDSInterpretError(f"a {self.kind!r} row's {k} = {getattr(self, k)}: use {lo} or more{tip}")
        for k in ("min_push", "strong"):
            if getattr(self, k) < 0:
                raise CDSInterpretError(f"a {self.kind!r} row's {k} = {getattr(self, k)}: it's a size (log-"
                                        "probability), 0 or more")


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

    landing: bool = True                 # the bot's own landing guess: a ring with a crosshair (false: none at all)
    landing_for: str | list[int] = "all"  # whose: "all" your cars, "panel", "camera", "none", or car numbers [1, 3]
    landing_color: tuple[int, ...] = (0, 170, 255)
    landing_radius: float = 150.0        # uu
    landing_line: bool = True            # a line from each car to its ring
    landing_smooth: float = 0.25         # seconds of smoothing (0: every reading as it comes)
    prediction: bool = True              # RLBot's prediction of the landing: a white cross
    prediction_color: tuple[int, ...] = (255, 255, 255)
    car_label: bool = True               # a word above the panel's car
    car_label_from: str = "why"          # "why": the feature driving its move (else the first firing); "noticing"
    car_label_color: tuple[int, ...] | None = None   # default: the car's team colour (colors.team_blue / team_orange)
    hook: str | None = None              # "module:function" drawing markers of your own

    def __post_init__(self) -> None:
        self.landing_for = check_landing_for(self.landing_for, "[markers] landing_for")
        if self.car_label_from not in ("why", "noticing"):
            raise CDSInterpretError(f"[markers] car_label_from = {self.car_label_from!r}: use \"why\" (the feature "
                                    "driving the move, else the first one firing) or \"noticing\" (the first firing)")

    @property
    def landing_cars(self) -> str | list[int]:
        """``landing_for``, or "none" when ``landing`` is off."""
        return self.landing_for if self.landing else "none"


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
    keep_size: bool = True               # every row keeps its full height: the panel never changes size
    follow: list[str | int] = field(default_factory=lambda: list(DEFAULT_FOLLOW))   # team modes: whose thoughts
    subtitle: str = "car {car} of {cars} - {rule}"   # team modes: the line under the title ("" for none)
    colors: dict[str, tuple[int, ...]] = field(default_factory=lambda: dict(DEFAULT_COLORS))
    markers: Markers = field(default_factory=Markers)
    features: dict[str, FeatureStyle] = field(default_factory=dict)
    rows: list[RowSpec] = field(default_factory=lambda: [RowSpec("outlook", "Outlook"), RowSpec("action", "Next move"),
                                                         RowSpec("why", "Why this move"),
                                                         RowSpec("looking", "Looking at"),
                                                         RowSpec("noticing", "Noticing"), RowSpec("probes")])

    def __post_init__(self) -> None:
        if not 0.02 <= self.width <= 1.0:
            raise CDSInterpretError(f"width = {self.width}: it's a share of the screen's width, from 0.02 to 1 "
                                    "(0.15-0.25 suits most screens)")
        if not 0.0 < self.smoothing <= 1.0:
            raise CDSInterpretError(f"smoothing {self.smoothing}: use a share in (0, 1] (1 = no smoothing)")
        self.colors = {**DEFAULT_COLORS, **{k: _rgba(v, f"colors.{k}") for k, v in self.colors.items()}}
        self.follow = check_follow(self.follow, "[panel] follow")
        if not isinstance(self.subtitle, str):
            raise CDSInterpretError(f"[panel] subtitle = {self.subtitle!r} should be text in quotes (\"\" for none)")
        self.subtitle = check_subtitle(self.subtitle, "[panel] subtitle")

    def rows_of(self, kind: str) -> list[RowSpec]:
        """The rows of one kind."""
        return [r for r in self.rows if r.kind == kind]

    @property
    def wants(self) -> dict[str, Any]:
        """What the panel asks the worker for: "why" (a why row, or a car label from it), "looking" (a looking row),
        "every" (its rate) and "donors"."""
        look = self.rows_of("looking")
        why = bool(self.rows_of("why")) or (self.markers.car_label and self.markers.car_label_from == "why")
        return {"why": why, "looking": bool(look), "every": min((r.every for r in look), default=4),
                "donors": max((r.donors for r in look), default=3)}

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
            try:
                kw["markers"] = Markers(**m)
            except CDSInterpretError as e:
                raise CDSInterpretError(f"{where}: {e}") from None
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
    feature the pack doesn't have, [features."..."] styles and noticing / why lists naming features it doesn't have,
    an outlook row of your own without an outlook, "why this move" and "looking at" rows of your own without what
    they need in the pack. pack: a pack, its .npz file or a Thoughts. [] when all is well."""
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
    custom_rows = cfg.rows != PanelConfig().rows        # (the default panel hides what a pack lacks by design)
    for i, r in enumerate(cfg.rows):
        at = f"row #{i + 1} ({r.kind})"
        if r.source and r.source.startswith("probe:") and r.source[6:] not in probes:
            out.append(f"{at} shows probe {r.source[6:]!r}, which the pack doesn't have, so the row won't appear."
                       + near(r.source[6:], probes, "probes"))
        if r.source and r.source.startswith("feature:") and r.source[8:] not in feats:
            out.append(f"{at} shows feature {r.source[8:]!r}, which the pack doesn't have, so the row won't appear."
                       + near(r.source[8:], feats, "features"))
        for n in (r.features or []) if r.kind in ("noticing", "why") else []:
            if n not in feats:
                out.append(f"{at} lists feature {n!r}, which the pack doesn't have." + near(n, feats, "features"))
        if r.kind == "outlook" and (r.source or "value") == "value" and pk.get("outlook") is None and custom_rows:
            out.append(f"{at}: the pack has no outlook (there were no goals in its games to learn one from), so the "
                       "row stays hidden.")
        if r.kind == "why" and custom_rows:
            fs = pk.get("features") or []
            if not fs:
                out.append(f"{at}: the pack has no named features, so there is nothing to say why with and the row "
                           "stays hidden.")
            elif not all(f.get("d") is not None for f in fs):
                out.append(f"{at} needs the features' decoder directions, which this pack doesn't have, so the row "
                           "stays hidden. Add them with overlay.add_decoders(pack, sae), the dictionary the features "
                           "came from (packs made with make_pack have them).")
        if r.kind == "looking" and custom_rows and not pk.get("donors"):
            out.append(f"{at} needs recorded moments in the pack to compare with, which it doesn't have, so the row "
                       "stays hidden. Add them with overlay.add_donors(pack, moments) (Study.make_pack adds them).")
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
    """What a custom row hook draws with: ``line(left, right)``, ``bar(fraction)``, ``spans([(text, colour)])``,
    ``blank(lines)``, ``gap(lines)``, ``color(r, g, b)``. Strings are cut to the panel's width; a bar always sits
    under the line before it."""

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

    def spans(self, parts: list[tuple[str, Any]], indent: float = 0.0, scale: float | None = None,
              sep: str = "  ") -> None:
        """One line of pieces of text, each in its own colour, left to right (what does not fit is left out), then
        move down a line."""
        sc = self.scale if scale is None else scale
        room = int((self.inner - indent) / (CHAR_W * sc))
        used = 0
        for k, (text, col) in enumerate(parts):
            at = used + (len(sep) if k else 0)
            if at + len(text) > room:
                break
            self.ops.append(("s", text, self.x + PAD + indent + at * CHAR_W * sc, self.cy, sc, col,
                             self.flat.TextHAlign.Left))
            used = at + len(text)
        self.cy += LINE_H * sc

    def blank(self, lines: float = 1.0, scale: float | None = None) -> None:
        """The room of ``lines`` lines of text, left empty (what keeps a row's height when it has less to say)."""
        self.cy += LINE_H * (self.scale if scale is None else scale) * max(0.0, lines)

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

    def stacked(self, parts: list[tuple[float, Any]]) -> None:
        """A bar under the line above split into coloured pieces, left to right: [(fraction, colour)] (fractions of
        the whole bar; they may add up to less than 1, the rest is the empty track)."""
        top = self.cy + 0.002
        self.ops.append(("r", self.x + PAD, top, self.inner, BAR_H, self.named("track")))
        x0 = self.x + PAD
        for frac, col in parts:
            w = self.inner * max(0.0, min(1.0, float(frac)))
            if w > 0:
                self.ops.append(("r", x0, top, min(w, self.x + PAD + self.inner - x0), BAR_H, col))
                x0 += w
        self.cy += BAR_ROW

    def gap(self, lines: float = 0.2) -> None:
        self.cy += LINE_H * lines


def _readings(available: Any) -> set[str]:
    """{"why", "looking"} from what the overlay can compute (a set, a list or the worker's {"why": bool, ...})."""
    if available is None:
        return set()
    if isinstance(available, dict):
        return {k for k, v in available.items() if v}
    return {str(a) for a in available}


class Panel:
    """A panel config with its hooks, ready to draw results. Keep one per bot; ``state`` (a dict you keep between
    draws, one per car) holds the smoothing and the features held on screen."""

    def __init__(self, config: PanelConfig | str | os.PathLike[str] | None = None, hooks: dict[str, Callable] | None = None):
        if config is None:
            config = PanelConfig()
        elif not isinstance(config, PanelConfig):
            config = PanelConfig.load(config)
        self.config = config
        self.hooks = dict(hooks or {})
        self._row_hooks = {i: _resolve(r.hook, self.hooks) for i, r in enumerate(config.rows) if r.kind == "custom"}
        self._marker_hook = _resolve(config.markers.hook, self.hooks)
        self._seen: set[str] = set()          # the readings results have carried so far (their rows keep a place)

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

    def _style(self, name: str) -> FeatureStyle:
        return self.config.features.get(name, FeatureStyle())

    def _features_on(self, th: dict[str, Any]) -> list[dict[str, Any]]:
        out = []
        for f in th.get("features", []):
            style = self._style(f["name"])
            on = f["on"] if style.threshold is None else f["act"] > style.threshold
            out.append({**f, "on": bool(on), "show": style.show, "label": style.label or f["name"],
                        "color": style.color})
        return out

    def _why(self, th: dict[str, Any], state: dict[str, Any] | None) -> list[dict[str, Any]] | None:
        """This result's "why this move" (or, until the next one comes, the last one shown for this car)."""
        why = th.get("why")
        if why is not None:
            self._seen.add("why")
            if state is not None:
                state["why_last"] = why
            return why
        return state.get("why_last") if state is not None else None

    def _driver(self, why: list[dict[str, Any]] | None, row: RowSpec | None) -> str | None:
        """The label of the feature pushing the move hardest (at least the row's min_push), or None."""
        if not why:
            return None
        lo = row.min_push if row is not None else RowSpec("why").min_push
        best = None
        for w in why:
            style = self._style(w["name"])
            if style.show and w["push"] >= lo and (best is None or w["push"] > best["push"]):
                best = w
        return None if best is None else (self._style(best["name"]).label or best["name"])

    # ---- the panel
    def draw(self, renderer: Any, flat: Any, th: dict[str, Any] | None, car_index: int | None = None,
             landing_true: Any = None, landing_think: Any = None, state: dict[str, Any] | None = None,
             frame: int | None = None, value_scale: float | None = None, team: int | None = None,
             marks: list[tuple[int | None, Any]] | None = None, subtitle: str | None = None,
             available: Any = None) -> None:
        """Draw one result (see the module's notes) and the markers.

        th: what the worker read for the panel's car (None: no panel, only ``marks``). car_index: the panel car's
        player index. landing_true: world (x, y, z) from RLBot's prediction (a white cross). landing_think: world
        (x, y) of the panel car's own landing guess - or ``marks``, [(car index, (x, y))], one ring per car. state: a
        dict kept between draws for this car. value_scale: the outlook value that fills its bar when the config does
        not say. team: the panel car's team (0 blue, 1 orange: the word above it takes the team's colour). subtitle: the line under the title (whose thoughts these are). available: the readings the overlay
        can compute ({"why", "looking"}): with ``keep_size``, their rows keep their place before their first
        reading."""
        cfg = self.config
        if marks is None:
            marks = [(car_index, landing_think)] if landing_think is not None else []
        renderer.begin_rendering(GROUP)
        driver = None
        if th is not None:
            ready = _readings(available) | self._seen
            d = RowDrawer(self, flat, cfg.x, cfg.y)
            d.line("THOUGHTS" if cfg.title is None else cfg.title, left_color=d.named("title"),
                   scale=cfg.title_scale * cfg.text_scale)
            d.gap(0.05)                                 # (the title takes 1.25 lines in all)
            if subtitle:
                d.line(subtitle, left_color=d.named("label"), scale=0.9 * cfg.text_scale)
            feats = self._features_on(th)
            placed = {r.source.split(":", 1)[1] for r in cfg.rows if r.source and r.source.startswith("probe:")}
            vscale = cfg.value_scale if cfg.value_scale is not None else (value_scale or 1.0)
            ctx: dict[str, Any] = {"ready": ready}
            for i, row in enumerate(cfg.rows):
                self._row(d, flat, th, row, i, feats, placed, state, frame, vscale, ctx)
            d.cy += 0.002
            renderer.draw_rect_2d(cfg.x, cfg.y - 0.006, cfg.width, d.cy - cfg.y + 0.006,
                                  d.color(*cfg.colors["background"]))
            for op in d.ops:
                if op[0] == "r":
                    renderer.draw_rect_2d(op[1], op[2], op[3], op[4], op[5])
                else:
                    renderer.draw_string_2d(op[1], op[2], op[3], op[4], op[5], d.color(0, 0, 0, 0), op[6])
            driver = ctx.get("driver")
            if driver is None and cfg.markers.car_label_from == "why" and "why" not in ctx:
                driver = self._driver(self._why(th, state), None)
            self._car_label(renderer, flat, feats, car_index, driver, team)
        self._markers(renderer, flat, th, car_index, landing_true, marks, team)
        renderer.end_rendering()

    def _row(self, d: RowDrawer, flat: Any, th: dict[str, Any], row: RowSpec, i: int, feats: list[dict[str, Any]],
             placed: set[str], state: dict[str, Any] | None, frame: int | None, vscale: float,
             ctx: dict[str, Any] | None = None) -> None:
        cfg = self.config
        keep = cfg.keep_size
        ctx = {} if ctx is None else ctx
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
            lo, mid, hi = row.words or _WORDS["outlook"]
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
            lines = _wrap(a["words"], d.inner - 0.008, row.lines, cfg.text_scale)
            for text in lines:
                d.line(text, left_color=d.named("value"), indent=0.008)
            if keep:                                    # (a one-line move keeps the room of `lines` lines)
                d.blank(row.lines - len(lines))
        elif k == "why":
            self._why_row(d, th, row, state, frame, ctx, keep)
        elif k == "looking":
            self._looking_row(d, th, row, state, ctx, keep)
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
            n_max = int(row.max or 3)
            d.gap(0.2)
            d.line(row.label or "Noticing")
            if shown:
                for f in shown[:n_max]:
                    base = f["color"] or cfg.colors["notice"]
                    col = base if f["on"] else (f["color"] or cfg.colors["notice_held"])
                    d.line(f"> {f['label']}", left_color=d.color(*col), indent=0.004)
            else:
                d.line(row.empty or "", indent=0.008)
            if keep:
                d.blank(n_max - max(1, len(shown[:n_max])))
        elif k == "probes":
            names = [n for n in th.get("probes", {}) if n not in placed]
            if names:
                d.gap(0.2)
            for n in names:
                unit = th.get("looks", {}).get(n, {}).get("unit", "")
                color = th.get("looks", {}).get(n, {}).get("color")
                kind = "bar" if unit == "%" else "seconds" if unit == "s" else "value"
                self._row(d, flat, th, RowSpec(kind, n, f"probe:{n}", tuple(color) if color else None), i, feats,
                          placed, state, frame, vscale, ctx)
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

    def _why_row(self, d: RowDrawer, th: dict[str, Any], row: RowSpec, state: dict[str, Any] | None,
                 frame: int | None, ctx: dict[str, Any], keep: bool) -> None:
        """The named features pushing the move hardest (held a moment after they drop below min_push, like
        "noticing"), and the strongest one pushing against it."""
        cfg = self.config
        why = self._why(th, state)
        ctx["why"] = True
        if why is None and not (keep and "why" in ctx["ready"]):
            return
        n_max = int(row.max or 2)
        d.gap(0.2)
        d.line(row.label or "Why this move")
        if why is None:                                 # (its place kept until the first reading comes)
            d.blank(n_max + (1 if row.against else 0))
            return
        held = state.setdefault("why_seen", {}) if state is not None else {}
        hold = int(round(cfg.hold * FPS))
        allowed = set(row.features) if row.features else None
        toward, against = [], None
        for w in why:
            name, push = str(w["name"]), float(w["push"])
            style = self._style(name)
            if not style.show or (allowed is not None and name not in allowed):
                continue
            sm = self._smooth(state, "why:" + name, push)
            if push >= row.min_push and frame is not None:
                held[name] = frame
            recent = frame is not None and name in held and frame - held[name] < hold
            if (push >= row.min_push or (recent and push > 0)) and sm > 0:   # (held only while it still pushes for)
                toward.append((sm, style.label or name))
            if push < 0 and (against is None or push < against[0]):
                against = (push, style.label or name)
        toward.sort(key=lambda t: -t[0])
        if against is not None and against[0] <= -row.strong:            # (never listed both ways in one draw)
            toward = [t for t in toward if t[1] != against[1]]
        if toward:
            ctx["driver"] = toward[0][1]
            for sm, label in toward[:n_max]:
                d.line(f"+ {label}", left_color=d.named("push_strong" if sm >= row.strong else "push"), indent=0.004)
        else:
            d.line(row.empty or "", indent=0.008)
        if keep:
            d.blank(n_max - max(1, len(toward[:n_max])))
        if row.against:
            if against is not None and against[0] <= -row.strong:
                d.line(f"- {against[1]}", left_color=d.named("push_against"), indent=0.004)
            elif keep:
                d.blank(1)

    def _looking_row(self, d: RowDrawer, th: dict[str, Any], row: RowSpec, state: dict[str, Any] | None,
                     ctx: dict[str, Any], keep: bool) -> None:
        """One bar split by what the move rests on (own car, ball, opponents, team-mates), with a legend."""
        look = th.get("looking")
        if look is not None:
            self._seen.add("looking")
            if state is not None:
                state["looking_last"] = look
        elif state is not None:
            look = state.get("looking_last")            # (computed a few times a second: the last one stays up)
        if not look and not (keep and "looking" in ctx["ready"]):
            return
        words = dict(zip(ENTITY_COLORS, row.words or _WORDS["looking"]))
        parts = []
        for ent, ckey in ENTITY_COLORS.items():
            if look and ent in look:
                parts.append((words[ent], d.named(ckey), max(0.0, self._smooth(state, "look:" + ent,
                                                                                float(look[ent])))))
        tot = sum(v for _, _, v in parts) or 1.0
        d.gap(0.2)
        d.line(row.label or "Looking at")
        d.stacked([(v / tot, col) for _, col, v in parts])
        if row.legend:
            sc = 0.9 * self.config.text_scale
            if parts:
                d.spans([(f"{w} {round(100 * v / tot):d}", col) for w, col, v in parts], scale=sc)
            else:
                d.blank(1, scale=sc)

    def _car_label(self, renderer: Any, flat: Any, feats: list[dict[str, Any]], car_index: int | None,
                   driver: str | None, team: int | None = None) -> None:
        """The word above the panel's car: in markers.car_label_color when set, else in the feature's own colour
        when its style gives one, else in the car's team colour (colors.team_blue / team_orange)."""
        m = self.config.markers
        if not m.car_label or car_index is None:
            return
        on = [f for f in feats if f["on"] and f["show"]]
        text = driver if m.car_label_from == "why" and driver else (on[0]["label"] if on else None)
        if not text:
            return
        own = next((f["color"] for f in feats if f["label"] == text and f["color"]), None)
        side = {0: "team_blue", 1: "team_orange"}.get(-1 if team is None else int(team), "notice")
        col = m.car_label_color or own or self.config.colors[side]
        renderer.draw_string_3d(text, flat.CarAnchor(car_index, flat.Vector3(0, 0, 120)), 1.2, flat.Color(*col),
                                flat.Color(0, 0, 0, 160), flat.TextHAlign.Center)

    def _markers(self, renderer: Any, flat: Any, th: dict[str, Any] | None, car_index: int | None, landing_true: Any,
                 marks: list[tuple[int | None, Any]], team: int | None) -> None:
        m = self.config.markers
        if m.prediction and landing_true is not None:
            p = flat.Vector3(*[float(c) for c in landing_true])
            col = flat.Color(*m.prediction_color)
            renderer.draw_line_3d(flat.Vector3(p.x - 80, p.y, 20), flat.Vector3(p.x + 80, p.y, 20), col)
            renderer.draw_line_3d(flat.Vector3(p.x, p.y - 80, 20), flat.Vector3(p.x, p.y + 80, 20), col)
        if m.landing:
            col = flat.Color(*m.landing_color)
            for ci, xy in marks:
                if xy is None:
                    continue
                lx, ly, z, r = float(xy[0]), float(xy[1]), 20.0, m.landing_radius
                for rad in (r - 10.0, r):           # two close rings for a bold line
                    ring = [flat.Vector3(lx + rad * math.cos(2 * math.pi * k / 32),
                                         ly + rad * math.sin(2 * math.pi * k / 32), z) for k in range(33)]
                    renderer.draw_polyline_3d(ring, col)
                renderer.draw_line_3d(flat.Vector3(lx - r, ly, z), flat.Vector3(lx + r, ly, z), col)
                renderer.draw_line_3d(flat.Vector3(lx, ly - r, z), flat.Vector3(lx, ly + r, z), col)
                if m.landing_line and ci is not None:
                    renderer.draw_line_3d(flat.CarAnchor(ci, flat.Vector3(0, 0, 0)), flat.Vector3(lx, ly, z), col)
        if self._marker_hook is not None and th is not None:
            own = next((xy for ci, xy in marks if ci == car_index), None)
            self._marker_hook(renderer, flat, th, {"car_index": car_index, "team": team, "landing_true": landing_true,
                                                   "landing_think": own, "marks": list(marks)})


def draw_thoughts(renderer: Any, flat: Any, th: dict[str, Any] | None, car_index: int | None = None,
                  landing_true: Any = None, landing_think: Any = None, title: str | None = None, x: float | None = None,
                  y: float | None = None, value_scale: float = 1.0, state: dict[str, Any] | None = None,
                  frame: int | None = None, config: PanelConfig | None = None,
                  marks: list[tuple[int | None, Any]] | None = None, subtitle: str | None = None,
                  available: Any = None, team: int | None = None) -> None:
    """Draw one result with the default panel (or ``config``); title, x and y override the config's. marks: a landing
    ring for each of several cars, [(car index, (x, y))], in place of landing_think's one; subtitle: a line under the
    title; available: the readings the overlay can compute (see Panel.draw); team: the car's team (0 blue, 1 orange),
    for the colour of the word above it."""
    cfg = config or PanelConfig()
    if title is not None or x is not None or y is not None:
        from dataclasses import replace
        cfg = replace(cfg, title=cfg.title if title is None else title, x=cfg.x if x is None else x,
                      y=cfg.y if y is None else y)
    Panel(cfg).draw(renderer, flat, th, car_index, landing_true, landing_think, state, frame, value_scale, team,
                    marks=marks, subtitle=subtitle, available=available)
