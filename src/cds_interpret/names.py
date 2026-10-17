"""Naming features: suggestions from the labels, and a names file you confirm them in.

A sparse autoencoder hands you features as numbers - feature 12, feature 40. Naming them is a short loop:

1. **Suggest**: ``suggest(feature_set)`` proposes a name for each feature from the game-state label it lines up with
   best (AUROC for yes / no labels, correlation for numbers), with warnings when something looks off - like a
   feature that fires mostly at kickoffs.
2. **Look**: ``study.feature_gallery(layer, "features.html")`` draws each feature's strongest moments from above.
3. **Write it down**: ``write_names("names.toml", suggestions, layer)`` starts a names file; you edit it - confirm a
   name, write a better one, or reject it.
4. **Check**: ``study.check_feature(layer, j)`` turns the feature off and up and says what the bot does differently.
5. **Use**: ``study.make_pack(layer, names="names.toml")`` puts the confirmed names on the live panel.

The names file is plain TOML::

    layer = "body.3"
    dictionary = "3f9a0c51d2e7"     # which dictionary the numbers belong to (written for you; see below)

    [features.12]
    name = "lined up for a shot"
    status = "confirmed"        # "suggested" (not looked at yet), "confirmed" or "rejected"
    threshold = 0.3             # optional: when it counts as firing on the live panel
    note = "fires as it squares up to the ball, both halves of the field"

Feature numbers belong to one dictionary: train it again and feature 12 is something else. So the file keeps the
dictionary's code (``FeatureSet.fingerprint``), and the toolkit refuses to put its names on another dictionary. Keep
the dictionary you name with ``feats.save(path)`` and bring it back with ``study.load_features(path)``.

    [features.40]
    name = "opponent about to win the ball?"
    status = "suggested"

A feature may also be written in one line: ``features.12 = "lined up for a shot"`` (confirmed).
"""

from __future__ import annotations

import difflib
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .errors import CDSInterpretError
from .labels import LABEL_KIND, short_label

STATUSES = ("suggested", "confirmed", "rejected")


@dataclass
class Suggestion:
    """A proposed name for one feature, with what it is based on."""

    feature: int
    name: str
    label: str | None                    # the label it lines up with best
    strength: float                      # 0 = no relation, 1 = perfect (|2 AUROC - 1| or |r|)
    detail: str                          # "AUROC 0.93, 6.1x as likely when it fires" / "r +0.71"
    freq: float                          # the share of moments it fires on
    warnings: list[str] = field(default_factory=list)
    sizes: dict[str, float] | None = None  # how big it is when it fires: "start", "typical", "highest"
    dictionary: str | None = None        # the dictionary's code (FeatureSet.fingerprint)

    def __str__(self) -> str:
        warn = "".join(f"\n      ! {w}" for w in self.warnings)
        return f"feature {self.feature:5d}  {self.name:28s}  ({self.detail}; fires on {self.freq:.1%}){warn}"


NUMBER_NAMES: dict[str, tuple[str, str]] = {
    "ball_dist": ("far from the ball", "close to the ball"),
    "land_dist": ("far from where the ball lands", "near where the ball lands"),
    "land_t": ("ball landing later", "ball about to land"),
    "land_y": ("ball landing in their half", "ball landing in our half"),
    "ttb_me": ("a long way from the ball", "nearly at the ball"),
    "ttb_opp": ("opponent far from the ball", "opponent nearly at the ball"),
    "ttb_mate": ("team-mate far from the ball", "team-mate nearly at the ball"),
    "ttb_adv": ("winning the race to the ball", "losing the race to the ball"),
    "boost": ("lots of boost", "low on boost"),
    "boost_adv": ("more boost than them", "less boost than them"),
    "height": ("high up", "near the ground"),
    "ball_height": ("ball high up", "ball near the ground"),
    "speed": ("going fast", "going slowly"),
    "ball_vy": ("ball heading their way", "ball heading our way"),
    "threat_t": ("threat still a while off", "threat about to arrive"),
    "demo_ttc": ("demo still a while off", "demo about to happen"),
    "outlook": ("expects to score", "expects to concede"),
}
"""What a feature that goes with a number label is called: (when the number is high, when it's low)."""


def _name_for(label: str, sign: str, rel: dict[str, float]) -> str:
    base = short_label(label)
    if LABEL_KIND.get(label) == "bin" or "auroc" in rel:
        return base if sign == "+" else f"not {base}"
    if label in NUMBER_NAMES:
        return NUMBER_NAMES[label][0 if sign == "+" else 1]
    return f"high {base}" if sign == "+" else f"low {base}"


def suggest(fs: Any, features: Any = None, labels: dict[str, np.ndarray] | None = None, top: int = 24,
            min_strength: float = 0.2) -> list[Suggestion]:
    """A suggested name for each feature of a FeatureSet (``features``: which ones). By default: ``top`` of its
    features most worth a look, as many different suggestions as there are before any repeats (so ten "kickoff"
    features don't crowd out everything else). ``labels`` (the study's labels) lets the suggestions warn about
    kickoffs and features that fire almost always."""
    explicit = features is not None
    feats = list(features) if explicit else fs.top(4 * top)
    code = getattr(fs, "fingerprint", None)
    out = []
    kick = None if labels is None or "kickoff" not in labels else np.asarray(labels["kickoff"]) > 0.5
    for j in feats:
        j = int(j)
        rel = fs.relations.get(j, {})
        guess = fs.guess(j, 1)
        warnings: list[str] = []
        if not guess or guess[0][1] < min_strength:
            name, label, strength, detail = f"feature {j} (no clear match)", None, 0.0, "no label lines up with it"
            warnings.append("no label lines up with it: look at its moments - it may be something worth a label")
        else:
            label, strength, sign = guess[0]
            r = rel[label]
            name = _name_for(label, sign, r)
            if "auroc" in r and sign == "+":
                detail = f"AUROC {r['auroc']:.2f}, {r.get('lift', float('nan')):.1f}x as likely when it fires"
            elif "auroc" in r:                          # it fires when the fact is NOT true: say it that way round
                base, p_on = r.get("base", float("nan")), r.get("p_given_on", float("nan"))
                lift_not = (1 - p_on) / (1 - base) if base < 1 else float("nan")
                detail = f"AUROC {1 - r['auroc']:.2f} for '{name}', {lift_not:.1f}x as likely when it fires"
            else:
                detail = f"correlation {r['r']:+.2f}"
        freq = float(fs.freq[j])
        if freq > 0.5:
            warnings.append(f"fires on {freq:.0%} of moments: more a background than an event")
        if kick is not None and fs.codes is not None and label != "kickoff":
            act = fs.activation(j)
            on = act > 0
            if on.any():
                share = float(kick[on].mean())
                if share >= 0.3:
                    warnings.append(f"{share:.0%} of its firings are kickoffs: is it about '{name}', or about "
                                    "kickoffs? Check its moments without them")
        sizes = fs.sizes(j) if fs.codes is not None and hasattr(fs, "sizes") else None
        out.append(Suggestion(j, name, label, float(strength), detail, freq, warnings, sizes, code))
    if not explicit:                                # each different suggestion once first, then the repeats
        names_seen: set[str] = set()
        first: list[Suggestion] = []
        rest: list[Suggestion] = []
        for sg in out:
            (rest if sg.name in names_seen else first).append(sg)
            names_seen.add(sg.name)
        out = (first + rest)[:top]
    seen: dict[str, int] = {}
    for sg in out:                                  # two features with the same suggestion get "#2", "#3"
        seen[sg.name] = seen.get(sg.name, 0) + 1
        if seen[sg.name] > 1:
            sg.name = f"{sg.name} #{seen[sg.name]}"
    return out


# ---------------------------------------------------------------------------------------------- the names file
@dataclass
class FeatureName:
    name: str
    status: str = "confirmed"
    threshold: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise CDSInterpretError(f"status {self.status!r}: use one of {', '.join(STATUSES)}")


@dataclass
class Names:
    """A names file in memory: the layer, the dictionary's code, and {feature index: FeatureName}."""

    layer: str | None
    features: dict[int, FeatureName] = field(default_factory=dict)
    dictionary: str | None = None        # FeatureSet.fingerprint of the dictionary the numbers belong to

    def confirmed(self) -> dict[str, int]:
        """{name: feature} of the confirmed names (two features may not share a name: the panel shows names)."""
        out: dict[str, int] = {}
        for j, f in self.features.items():
            if f.status != "confirmed":
                continue
            if f.name in out:
                raise CDSInterpretError(f"Features {out[f.name]} and {j} are both called {f.name!r}. Give them "
                                        "different names (or reject one) - the panel tells them apart by name.")
            out[f.name] = j
        return out

    def thresholds(self) -> dict[str, float]:
        return {f.name: f.threshold for f in self.features.values() if f.threshold is not None
                and f.status == "confirmed"}


def _q(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def write_names(path: str | os.PathLike[str], suggestions: list[Suggestion], layer: str,
                overwrite: bool = False) -> Path:
    """Start a names file from suggestions: every name "suggested", with what it is based on as a comment. Edit it:
    set ``status = "confirmed"`` on the names you agree with (rename them freely), ``"rejected"`` on the others."""
    p = Path(path)
    if p.exists() and not overwrite:
        raise CDSInterpretError(f"{p} already exists - your edits would be lost. Pass overwrite=True, or use "
                                "update_names() to add new features to it")
    codes = {s.dictionary for s in suggestions if s.dictionary}
    if len(codes) > 1:
        raise CDSInterpretError("these suggestions come from different dictionaries: write one names file per "
                                "dictionary")
    lines = ["# What each feature means. Set status = \"confirmed\" on a name once its moments agree; rename freely;",
             "# \"rejected\" hides a feature. The live panel shows confirmed names only.", f"layer = {_q(layer)}"]
    if codes:
        lines.append(f"dictionary = {_q(codes.pop())}   # the numbers below belong to this dictionary only: "
                     "keep it with feats.save(...)")
    lines.append("")
    for s in suggestions:
        lines.append(f"[features.{s.feature}]")
        lines.append(f"name = {_q(s.name)}")
        lines.append('status = "suggested"')
        lines.append(f"# {s.detail}; fires on {s.freq:.1%} of moments")
        if s.sizes and s.sizes["highest"] > 0:
            z = s.sizes
            lines.append(f"# when it fires: {z['start']:.2f} to {z['highest']:.2f}, typically {z['typical']:.2f} "
                         f"(the panel counts it from {z['start']:.2f} unless you set a threshold)")
        lines.extend(f"# ! {w}" for w in s.warnings)
        lines.append("")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def read_names(path: str | os.PathLike[str]) -> Names:
    """A names file (see the module's notes)."""
    p = Path(path)
    if not p.exists():
        raise CDSInterpretError(f"no names file at {p}: start one with write_names(...) (docs/howto/name-features.md)")
    try:
        d = tomllib.loads(p.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise CDSInterpretError(f"{p} is not valid TOML: {e}") from None
    for key in d:
        if key not in ("layer", "dictionary", "features"):
            close = difflib.get_close_matches(key, ["layer", "dictionary", "features"], n=1)
            hint = f" Did you mean {close[0]!r}?" if close else ""
            raise CDSInterpretError(f"{p}: unknown setting {key!r}.{hint} A names file has layer, dictionary and "
                                    "[features.N] tables (a key under a [features.N] heading belongs to that feature)")
    out = Names(d.get("layer"), dictionary=d.get("dictionary"))
    for key, spec in (d.get("features") or {}).items():
        try:
            j = int(key)
        except ValueError:
            raise CDSInterpretError(f"{p}: [features.{key}] - a feature is named by its number, e.g. [features.12]") \
                from None
        if isinstance(spec, str):
            out.features[j] = FeatureName(spec)
            continue
        unknown = set(spec) - {"name", "status", "threshold", "note"}
        if unknown:
            raise CDSInterpretError(f"{p}: [features.{key}] has unknown keys {sorted(unknown)} "
                                    "(use name, status, threshold, note)")
        if "name" not in spec:
            raise CDSInterpretError(f"{p}: [features.{key}] has no name")
        out.features[j] = FeatureName(str(spec["name"]), str(spec.get("status", "confirmed")),
                                      None if spec.get("threshold") is None else float(spec["threshold"]),
                                      str(spec.get("note", "")))
    return out


def update_names(path: str | os.PathLike[str], suggestions: list[Suggestion], layer: str) -> Path:
    """Add suggestions for features the names file does not have yet (the file's own entries are kept as they are)."""
    p = Path(path)
    if not p.exists():
        return write_names(p, suggestions, layer)
    have = read_names(p)
    codes = {s.dictionary for s in suggestions if s.dictionary}
    if have.dictionary and codes and codes != {have.dictionary}:
        raise CDSInterpretError(f"{p} is for another dictionary (code {have.dictionary}; these suggestions: "
                                f"{', '.join(sorted(codes))}): its feature numbers mean other features. Start a new "
                                "names file for this dictionary, or load the one you named (study.load_features).")
    new = [s for s in suggestions if s.feature not in have.features]
    if not new:
        return p
    text = p.read_text(encoding="utf-8").rstrip() + "\n\n"
    tmp = p.with_suffix(".new.toml")
    write_names(tmp, new, layer, overwrite=True)
    written = tmp.read_text(encoding="utf-8")
    body = written[written.index("[features."):]                    # without the header lines
    tmp.unlink()
    p.write_text(text + body, encoding="utf-8")
    return p
