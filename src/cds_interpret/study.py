"""A Study: your policy, its moments, and every technique of the toolkit on them, in a few calls.

    import cds_interpret as ci

    study = ci.Study(policy, moments)             # labels, the network's inputs, a split by game
    probes = study.probes()                       # what each layer knows - against the raw input and random weights
    print(probes.summary())
    feats = study.features("body.3")              # a sparse autoencoder of one layer, its features related to labels
    print(feats.summary())
    study.edits("body.3", {"no threat": study.probe_edit("body.3", "threat", 0.0)})   # does the policy use it?
    study.attribution()                           # what each decision rests on: ball, own car, opponents, team-mates
    study.report("report.html")                   # everything above on one page

Everything is scored on games the probes and dictionaries never saw (``test_frac`` of the games).
"""

from __future__ import annotations

import difflib
import os
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import analysis, intervene
from . import probes as P
from . import sae as S
from .errors import CDSInterpretError, ShapeError
from .labels import LABEL_KIND, short_label
from .moments import Moments
from .policy import Policy

Edit = Callable[[torch.Tensor], torch.Tensor]


def _check_label(label: Any, have: Any) -> None:
    """A friendly error for a label these moments don't have."""
    import difflib
    if label in have:
        return
    close = difflib.get_close_matches(str(label), list(have), n=3)
    hint = f" Did you mean {', '.join(repr(c) for c in close)}?" if close else ""
    raise CDSInterpretError(f"There's no label {label!r} in these moments.{hint} docs/labels.md lists them all.")


# --------------------------------------------------------------------------------------------------------- probes
@dataclass
class ProbeResults:
    """Held-out probe results: ``by_layer[layer][label]`` (and ``baselines["input"]``, ``baselines["random:<layer>"]``)
    is a dict with ``r2`` / ``r`` (numbers) or ``auroc`` / ``ap`` / ``base`` (yes / no), the steadiness numbers
    (``wnorm``, ``noise``, ``noise_rel``, ``jitter``, ``jitter_label``) and the fitted ``probe``."""

    by_layer: dict[str, dict[str, dict[str, Any]]]
    baselines: dict[str, dict[str, dict[str, Any]]]
    labels: list[str]
    skipped: dict[str, str] = field(default_factory=dict)      # label -> why it has no probe

    @property
    def layers(self) -> list[str]:
        return list(self.by_layer)

    def score(self, layer: str, label: str) -> float:
        """The headline held-out score (R² or AUROC); NaN when the label could not be probed there."""
        src = self.by_layer.get(layer) or self.baselines.get(layer) or {}
        r = src.get(label)
        return float("nan") if r is None else float(P.headline(r))

    def probe(self, layer: str, label: str) -> P.LinearProbe:
        """The fitted probe of a label at a layer."""
        if layer not in self.by_layer:
            raise CDSInterpretError(f"Layer {layer!r} wasn't probed. Probed layers: {', '.join(self.by_layer)}")
        if label not in self.by_layer[layer]:
            if label in self.skipped:
                raise CDSInterpretError(f"There's no probe for {label!r}: {self.skipped[label]}. Record more (or "
                                        "longer) games where it happens, or ask about another label.")
            _check_label(label, dict.fromkeys(self.labels))
            raise CDSInterpretError(f"There's no probe for {label!r} at layer {layer!r} (too few examples there).")
        return self.by_layer[layer][label]["probe"]

    def best(self, label: str) -> tuple[str, str] | None:
        """(layer, label) of the layer that reads ``label`` best, or None."""
        sc = [(self.score(layer, label), layer) for layer in self.by_layer]
        sc = [t for t in sc if np.isfinite(t[0])]
        return (max(sc)[1], label) if sc else None

    def learned(self, margin: float = 0.05) -> list[str]:
        """Labels some layer reads at least ``margin`` better than both the raw input and random weights do: what
        the network itself has worked out."""
        out = []
        for k in self.labels:
            b = self.best(k)
            if b is None:
                continue
            s = self.score(b[0], k)
            floor = 0.5 if "auroc" in self.by_layer[b[0]][k] else 0.0          # chance
            base = [v for v in (self.score("input", k), self.score(f"random:{b[0]}", k)) if np.isfinite(v)]
            if base and s - max(base) >= margin and s - floor >= 2 * margin:
                out.append(k)
        return out

    def summary(self, top: int | None = None, margin: float = 0.05) -> str:
        """A table: for each concept, the layer that knows it best and how well, beside the raw input and a network
        with random weights - first the concepts the network worked out for itself (``learned``), then the rest."""
        learned = set(self.learned(margin))
        rows = []
        for k in self.labels:
            b = self.best(k)
            if b is None:
                continue
            layer = b[0]
            r = self.by_layer[layer][k]
            rows.append((k, layer, self.score(layer, k), self.score("input", k), self.score(f"random:{layer}", k),
                         "AUROC" if "auroc" in r else "R2", r.get("noise_rel", float("nan"))))

        def gain(t: tuple) -> float:                 # the best score's lead over the baselines and over chance
            floor = 0.5 if t[5] == "AUROC" else 0.0
            base = max([floor] + [v for v in (t[3], t[4]) if np.isfinite(v)])
            return t[2] - base
        rows.sort(key=lambda t: (t[0] not in learned, -gain(t)))
        shown = rows[:top] if top else rows

        def f(v: float) -> str:
            return f"{v:5.2f}" if np.isfinite(v) else "    -"

        def line(t: tuple) -> str:
            k, layer, sc, i, rn, metric, nr = t
            steady = "-" if not np.isfinite(nr) else ("yes" if nr < 0.1 else "so-so" if nr < 0.3 else "no")
            name = f"{k} ({short_label(k)})" if short_label(k) != k else k
            return f"  {name[:34]:34s} {layer[:10]:10s} {f(sc)}  {f(i)}  {f(rn):>6s}  {metric:7s} {steady}"
        lines = ["What each layer knows, scored on games the probes never saw. For a yes / no fact the score is an",
                 "AUROC (0.50 is a guess, 1.00 perfect); for a number it's R2 (0.00 is a guess, 1.00 perfect).",
                 "\"input\" and \"random\" ask the same question of the bot's raw input and of a copy of the network",
                 "with random weights. A layer shows something it worked out only where it beats both.",
                 "\"steady\": does the answer hold still when the layer's numbers wobble a little? \"yes\" is fine to",
                 "show live, \"so-so\" may flicker, \"no\" will.",
                 "",
                 f"  {'concept':34s} {'layer':10s} {'score':>5s}  {'input':>5s}  {'random':>6s}  {'measure':7s} steady",
                 "  " + "-" * 82]
        top_learned = [t for t in shown if t[0] in learned]
        others = [t for t in shown if t[0] not in learned]
        if top_learned:
            lines.append(f"  Worked out by the network (at least {margin:.2f} better than both):")
            lines.extend(line(t) for t in top_learned)
        else:
            lines.append(f"  Nothing beats both baselines by {margin:.2f} or more - what you'd expect from an untrained "
                         "network.")
        if others:
            lines.append("  Readable from the raw input or random weights too:" if top_learned else
                         "  What the layers read (no better than the raw input or random weights):")
            lines.extend(line(t) for t in others)
        if self.skipped and not top:
            lines.append("")
            lines.append("Not scored:")
            lines.extend(f"  {k}: {why}" for k, why in self.skipped.items())
        elif self.skipped:
            lines.append(f"  ... and {len(self.skipped)} labels not scored (too few examples): summary() lists why")
        return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------- features
@dataclass
class FeatureSet:
    """A sparse autoencoder of one layer and what its features go with."""

    layer: str
    sae: S.TopKSAE
    fvu: float                                   # held-out fraction of the layer's variance it leaves unexplained
    effect: dict[str, dict[str, float]]          # the policy run on the reconstruction / on the layer's mean
    freq: np.ndarray                             # [m] share of moments each feature fires on
    relations: dict[int, dict[str, dict[str, float]]] = field(default_factory=dict)
    names: dict[int, str] = field(default_factory=dict)          # names you gave (name_feature)
    codes: tuple[torch.Tensor, torch.Tensor] | None = None       # top-k (indices, values) of every moment

    def activation(self, j: int) -> np.ndarray:
        """Feature j's activation at every moment [N]."""
        if self.codes is None:
            raise CDSInterpretError("this FeatureSet has no codes")
        return S.dense_column(self.codes[0], self.codes[1], int(j)).numpy()

    def guess(self, j: int, top: int = 3) -> list[tuple[str, float, str]]:
        """The labels feature j goes with most: [(label, strength 0..1, "+" / "-")] - a starting point for a name,
        not a finding: check its top moments."""
        return analysis.guess_name(self.relations.get(int(j), {}), top=top)

    def name(self, j: int) -> str:
        """Its name: the one you gave, else a guess marked with a '?'."""
        if int(j) in self.names:
            return self.names[int(j)]
        from .names import _name_for
        g = self.guess(j, 1)
        if not g:
            return f"feature {j}"
        return f"{_name_for(g[0][0], g[0][2], self.relations.get(int(j), {}).get(g[0][0], {}))}?"

    def name_feature(self, j: int, name: str) -> None:
        """Give feature j a name (after looking at its top moments)."""
        self.names[int(j)] = name

    def top(self, n: int = 12, min_freq: float = 0.002, max_freq: float = 0.5) -> list[int]:
        """The n features most worth a look: firing on min_freq..max_freq of moments, ranked by how strongly they go
        with some label."""
        cand = []
        for j in range(len(self.freq)):
            if not min_freq <= self.freq[j] <= max_freq:
                continue
            g = self.guess(j, 1)
            cand.append((g[0][1] if g else 0.0, j))
        cand.sort(reverse=True)
        return [j for _, j in cand[:n]]

    def detectors(self, min_auroc: float = 0.8, min_lift: float = 3.0, min_freq: float = 0.002,
                  top: int = 8, min_ap: float = 0.1, min_ap_lift: float = 4.0) -> list[tuple[int, str, float, float]]:
        """Features that detect a yes / no label clearly: [(feature, label, AUROC, lift)], each feature once (at its
        best label), the best first - the easiest features to name, and good candidates for the live overlay. A
        feature counts when its precision is at least max(min_ap, min_ap_lift x the label's base rate) - what decides
        for a sparse feature, whose AUROC stays low even when it detects well - or its AUROC is at least min_auroc
        with a lift of min_lift (analysis.detects)."""
        best: dict[int, tuple[float, str, float, float]] = {}
        for j, rel in self.relations.items():
            if self.freq[j] < min_freq:
                continue
            for k, r in rel.items():
                if not analysis.detects(r, min_auroc, min_lift, min_ap, min_ap_lift):
                    continue
                s = analysis.relation_strength(r)
                if s > best.get(j, (-1.0, "", 0.0, 0.0))[0]:
                    best[j] = (s, k, r["auroc"], r.get("lift", float("nan")))
        out = sorted(best.items(), key=lambda t: -t[1][0])
        return [(j, k, a, lift) for j, (_, k, a, lift) in out[:top]]

    def apply_names(self, names: Any) -> FeatureSet:
        """Use the confirmed names of a names file (a path or names.Names) for this set's features."""
        from .names import Names, read_names
        nm = names if isinstance(names, Names) else read_names(names)
        self.check_names(nm, names if not isinstance(names, Names) else "the names file")
        for j, f in nm.features.items():
            if f.status == "confirmed":
                self.names[int(j)] = f.name
        return self

    @property
    def fingerprint(self) -> str:
        """A short code for this exact dictionary (sae.fingerprint): names files keep it."""
        return S.fingerprint(self.sae)

    def check_names(self, nm: Any, where: Any = "the names file") -> None:
        """Stop if a names file was written for another layer, or for another dictionary of this layer (whose
        features have the same numbers but are different features)."""
        if nm.layer is not None and nm.layer != self.layer:
            raise CDSInterpretError(f"{where} is for layer {nm.layer!r}, but these features are of {self.layer!r}.")
        if nm.dictionary and nm.dictionary != self.fingerprint:
            raise CDSInterpretError(
                f"{where} was written for another dictionary of {self.layer} (its code: {nm.dictionary}; this one's: "
                f"{self.fingerprint}). Feature numbers don't carry over: a dictionary trained again has different "
                "features under the same numbers, so these names would land on the wrong features.\n"
                "Load the dictionary you named: study.load_features(\"<file you saved it to>\"). If you didn't save "
                "it, training again with the same seed on the same recording usually gives it back (this check will tell "
                "you); otherwise name this one afresh. (Keep a dictionary for good with feats.save(\"body3.sae\").)")

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Keep this dictionary for good: ``study.load_features(path)`` brings it back with the same feature numbers,
        in a later session or on another computer, so a names file written for it stays true."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        S.save(self.sae, p, layer=self.layer, fingerprint=self.fingerprint)
        return p

    def sizes(self, j: int) -> dict[str, float]:
        """How big feature j is when it fires: "start" (its smallest typical firing value - the live panel's
        automatic threshold), "typical" (the median) and "highest", in the dictionary's units."""
        a = self.activation(j)
        on = a[a > 0]
        if not len(on):
            return {"start": 0.0, "typical": 0.0, "highest": 0.0}
        return {"start": float(np.quantile(on, 0.05)), "typical": float(np.median(on)), "highest": float(on.max())}

    def index(self, feature: int | str) -> int:
        """A feature's number from its number or from a name you gave it (apply_names / name_feature)."""
        if isinstance(feature, str) and not feature.strip().isdigit():
            by_name = {n: j for j, n in self.names.items()}
            if feature in by_name:
                return by_name[feature]
            close = difflib.get_close_matches(feature, list(by_name), n=1)
            hint = f" Did you mean {close[0]!r}?" if close else ""
            known = (f" The names it knows: {', '.join(repr(n) for n in by_name)}." if by_name else
                     " It knows no names yet: give it your names file first, feats.apply_names(\"names.toml\").")
            raise CDSInterpretError(f"{feature!r} isn't a name of a feature of {self.layer}.{hint}{known} "
                                    "(Or give the feature's number, like 97.)")
        j = int(feature)
        if not 0 <= j < self.sae.m:
            raise CDSInterpretError(f"There's no feature {j}: this dictionary of {self.layer} has features 0 to "
                                    f"{self.sae.m - 1}.")
        return j

    def summary(self, top: int = 10) -> str:
        """The dictionary's quality and its most interesting features, in words."""
        e = self.effect.get("reconstruction", {})
        mean = self.effect.get("layer's mean", {})
        para = (f"It keeps {1 - self.fvu:.1%} of the layer's variance on held-out games; with it in place of the layer, "
                f"the bot's chosen action changes on {e.get('argmax_changed', float('nan')):.0%} of moments "
                f"(KL {e.get('kl', float('nan')):.3f}). For scale: losing the layer altogether (its average in its "
                f"place) changes {mean.get('argmax_changed', float('nan')):.0%} (KL {mean.get('kl', float('nan')):.3f}).")
        lines = [f"A sparse autoencoder of {self.layer} found {self.sae.m} features ({self.sae.k} active at a time).",
                 *textwrap.wrap(para, 110), "",
                 "The features most worth a look (AUROC for yes / no facts: 0.5 is chance, 1 perfect; correlation for",
                 "numbers: 0 is none, +1 or -1 perfect):"]
        for j in self.top(top):
            rel = self.relations.get(int(j), {})
            parts = []
            for k, _v, sgn in self.guess(j):
                r = rel.get(k, {})
                if "auroc" in r:
                    au = r["auroc"] if sgn == "+" else 1 - r["auroc"]
                    parts.append(f"{'not ' if sgn == '-' else ''}{short_label(k)} (AUROC {au:.2f})")
                elif "r" in r:
                    parts.append(f"{short_label(k)} (correlation {r['r']:+.2f})")
            what = ", ".join(parts) or "no clear label"
            lines.extend(textwrap.wrap(f"  feature {j:<5d} fires on {self.freq[j]:5.1%} of moments   lines up with: "
                                       f"{what}", 118, subsequent_indent=" " * 10))
        return "\n".join(lines)


@dataclass
class FeatureCheck:
    """What changes when a feature is turned off or up (Study.check_feature)."""

    layer: str
    feature: int
    name: str
    n: int                                        # the held-out moments it fires on
    results: dict[float, dict[str, Any]]          # scale -> {"kl", "changed", "shifts" {control: mean change}, ...}

    def summary(self) -> str:
        lines = [f"feature {self.feature} ({self.name}) of {self.layer}, on the {self.n} held-out moments it fires on:"]
        for scale, r in self.results.items():
            what = "turned off" if scale == 0 else f"turned up x{scale:g}"
            shifts = ", ".join(f"{k} {v:+.2f}" for k, v in r.get("top_shifts", [])) or "no clear change of controls"
            lines.append(f"  {what:14s} the chosen action changes on {r['changed']:.0%} of them (KL {r['kl']:.3f}); "
                         f"on average: {shifts}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------- study
class Study:
    """Your policy and its moments, with the inputs built, the labels computed and a split by game.

    Args:
        policy: a :class:`~cds_interpret.Policy`.
        moments: :class:`~cds_interpret.Moments` (recorded, or demo.synthetic_moments()).
        layers: the module paths to study (default: every hidden layer after its nonlinearity).
        labels: the labels to probe (default: every label the moments have that can be probed).
        test_frac: the share of games held out for scoring.
        log: a function for progress lines (print by default; None for quiet).
    """

    def __init__(self, policy: Policy, moments: Moments, layers: list[str] | None = None,
                 labels: list[str] | None = None, test_frac: float = 0.25, seed: int = 0,
                 log: Callable[[str], Any] | None = print, batch: int = 4096):
        if not isinstance(policy, Policy):
            raise TypeError("Study needs a cds_interpret.Policy: wrap your network with ci.Policy(net, ...)")
        if not isinstance(moments, Moments):
            raise TypeError("Study needs cds_interpret.Moments (recorded, Moments.load(path), or "
                            "ci.demo.synthetic_moments())")
        self.policy, self.moments = policy, moments
        self.log = log or (lambda *_: None)
        self.batch = batch
        if moments.obs is not None:
            self.x = torch.as_tensor(np.asarray(moments.obs, np.float32))
        else:
            self.log(f"Building the network's inputs for {len(moments)} moments ...")
            self.x = policy.observe(moments.frames, moments.me)
        try:
            with torch.no_grad():
                policy.forward(self.x[:2])
        except Exception as e:
            raise ShapeError(f"Your network didn't accept these inputs ({self.x.shape[1]} numbers per moment). PyTorch "
                             f"said: {e}. Usually the moments were recorded with a different bot, or the obs function "
                             "builds a different layout than the network expects.") from e
        self.layers = list(layers) if layers else policy.default_layers(self.x[:8])
        if not self.layers:
            raise CDSInterpretError("No layers to study were found. Pass layers=[...]; policy.layers(x) lists them.")
        for layer in self.layers:
            self._check_layer(layer)
        if not moments.labels:
            moments.with_labels()
        for k in labels or []:
            _check_label(k, moments.labels)
        names = labels or [k for k in moments.labels if k in LABEL_KIND]
        self.labels = {k: np.asarray(moments.labels[k], np.float32) for k in names if k in moments.labels}
        self.kinds = {k: LABEL_KIND.get(k, "reg") for k in self.labels}
        self.games = moments.game
        self.train, self.test = P.group_split(self.games, test_frac, seed)
        if moments.n_games < 8:
            self.log(f"  (only {moments.n_games} games: scores from so few games are rough - 10 or more is "
                     "better)")
        self._acts: dict[str, torch.Tensor] = {}
        self._random: Policy | None = None
        self.probe_results: ProbeResults | None = None
        self.feature_sets: dict[str, FeatureSet] = {}
        self.edit_results: dict[str, dict[str, float]] = {}
        self.feature_checks: dict[tuple[str, int], FeatureCheck] = {}
        self.attribution_results: dict[str, Any] | None = None
        self.extra_sections: dict[str, str] = {}
        self.log(f"Studying {policy.name} on {len(moments):,} moments from {moments.n_games} games "
                 f"({len(np.unique(self.games[self.test]))} games kept aside to score on). "
                 f"Layers: {', '.join(self.layers)}")

    def __repr__(self) -> str:
        return (f"Study({self.policy.name!r}, {len(self.moments)} moments, {self.moments.n_games} games, "
                f"layers={self.layers})")

    def _check_layer(self, layer: Any) -> str:
        """A layer name that names a part of the network, or a friendly error."""
        from .hooks import get_module
        if not isinstance(layer, str):
            raise CDSInterpretError(f"A layer is named by its path in your network, like \"layers.3\" - not "
                                    f"{layer!r}. These are studied: {', '.join(self.layers)}")
        get_module(self.policy.net, layer)
        return layer

    def why_not(self, label: str) -> str | None:
        """Why a label has no probe in this study, in words (None when it can have one)."""
        _check_label(label, self.labels)
        return P.why_not_probed(self.labels[label], self.kinds[label], self.train, self.test)

    # ---- activations
    def activations(self, layer: str) -> torch.Tensor:
        """The layer's output at every moment [N, D] (computed once, then kept)."""
        self._check_layer(layer)
        if layer not in self._acts:
            self._acts[layer] = self.policy.activations(self.x, [layer], batch=self.batch)[layer]
        return self._acts[layer]

    def random_policy(self) -> Policy:
        if self._random is None:
            self._random = self.policy.random_like(seed=12345)
        return self._random

    # ---- probes
    def probes(self, layers: list[str] | None = None, labels: list[str] | None = None, baselines: bool = True,
               scale: str = "global") -> ProbeResults:
        """Ridge probes of every label on every layer, fitted on the training games and scored on the held-out
        ones, with how steady each readout is - and the same probes on the raw input and on a random network of the
        same shape (``baselines``): a concept the input already gives, or random weights already read, is not
        something the network learned."""
        layers = [self._check_layer(la) for la in (layers or self.layers)]
        for k in labels or []:
            _check_label(k, self.labels)
        labs = {k: self.labels[k] for k in (labels or self.labels)}
        kinds = {k: self.kinds[k] for k in labs}
        kw: dict[str, Any] = {"groups": self.games, "keep": True, "scale": scale, "seq": self.moments.seq,
                              "order": self.moments.step}
        by_layer = {}
        for layer in layers:
            self.log(f"  probing layer {layer} ...")
            by_layer[layer] = P.ridge_suite(self.activations(layer), labs, kinds, self.train, self.test, **kw)
        base: dict[str, dict[str, dict[str, Any]]] = {}
        if baselines:
            base["input"] = P.ridge_suite(self.x.reshape(len(self.x), -1), labs, kinds, self.train, self.test, **kw)
            rnd = self.random_policy()
            racts = rnd.activations(self.x, layers, batch=self.batch)
            for layer in layers:
                base[f"random:{layer}"] = P.ridge_suite(racts[layer], labs, kinds, self.train, self.test, **kw)
        skipped = {}
        for k in labs:
            if not any(k in r for r in by_layer.values()):
                skipped[k] = P.why_not_probed(labs[k], kinds[k], self.train, self.test) or "too few examples"
        self.probe_results = ProbeResults(by_layer, base, list(labs), skipped)
        return self.probe_results

    def steadiness(self, layer: str, labels: tuple[str, ...] = ("land_x", "land_y")) -> dict[str, Any]:
        """How steady a readout of several labels read together is (e.g. the landing point x, y: moves are then
        distances), on the held-out games: probes.readout_stability."""
        res = self.probe_results.by_layer[layer] if self.probe_results and layer in self.probe_results.by_layer \
            else P.ridge_suite(self.activations(layer), {k: self.labels[k] for k in labels},
                               {k: self.kinds[k] for k in labels}, self.train, self.test, groups=self.games, keep=True)
        missing = [k for k in labels if k not in res]
        if missing:
            raise CDSInterpretError(f"no probes for {missing} at {layer}")
        w = np.stack([res[k]["probe"].w.numpy() for k in labels], 1)
        b = [res[k]["probe"].b for k in labels]
        y = np.stack([self.labels[k] for k in labels], 1)
        ok = self.test[np.isfinite(y[self.test]).all(1)]
        return P.readout_stability(w, b, self.activations(layer), ok, y=y, seq=self.moments.seq, order=self.moments.step)

    # ---- features
    def features(self, layer: str, m: int | None = None, k: int | None = None, steps: int = 2000, batch: int = 1024,
                 lr: float = 1e-3, seed: int = 0) -> FeatureSet:
        """A top-k sparse autoencoder of one layer, trained on the training games: its held-out variance explained,
        what the policy does on its reconstruction, how often each feature fires and which labels it goes with.
        m: how many features (default 8 per unit of the layer); k: how many may be active at once (default m / 32).
        The same seed on the same moments gives the same dictionary, so the same feature numbers; to keep one for
        good, ``save`` it and bring it back with ``load_features``."""
        h = self.activations(self._check_layer(layer))
        d = h.shape[1]
        m = m or max(64, 8 * d)
        k = k or max(2, min(32, m // 32))
        self.log(f"  training a sparse autoencoder on {layer} ({m} features, {k} active at a time) ...")
        tr = torch.from_numpy(self.train)
        with torch.random.fork_rng(devices=[]):       # its starting weights come from the seed, not from what ran before
            torch.manual_seed(seed)
            sae = S.TopKSAE(d, m, k).init_from(h[tr[:20000]])
            S.train_sae(sae, h.numpy(), rows=self.train, steps=steps, batch=min(batch, len(self.train)), lr=lr,
                        seed=seed)
        return self._feature_set(layer, sae.eval())

    def load_features(self, path: str | os.PathLike[str], layer: str | None = None) -> FeatureSet:
        """A dictionary kept with ``FeatureSet.save``, with everything ``features`` works out about it, on these
        moments - the same feature numbers as when you named them."""
        p = Path(path)
        if not p.exists():
            raise CDSInterpretError(f"There's no saved dictionary at {p}. Save one with feats.save({str(p)!r}) after "
                                    "study.features(...).")
        sae = S.load(p)
        layer = layer or getattr(sae, "meta", {}).get("layer")
        if layer is None:
            raise CDSInterpretError(f"{p} doesn't say which layer it's for: pass layer=\"...\"")
        h = self.activations(self._check_layer(layer))
        if h.shape[1] != sae.d:
            raise ShapeError(f"{p} is a dictionary of a layer {sae.d} wide, but {layer} is {h.shape[1]} wide here. "
                             "Is it for another layer, or another bot?")
        self.log(f"  loaded a sparse autoencoder of {layer} ({sae.m} features, {sae.k} active at a time) from {p}")
        return self._feature_set(layer, sae)

    def _feature_set(self, layer: str, sae: S.TopKSAE) -> FeatureSet:
        h = self.activations(layer)
        tr = torch.from_numpy(self.train)
        te = torch.from_numpy(self.test)
        fvu = S.fvu(sae, h[te])
        effect = self.edits(layer, {"reconstruction": intervene.replace_with(sae),
                                    "layer's mean": intervene.mean_ablate(h[tr].mean(0))}, store=False)
        stats = S.latent_stats(sae, h)
        codes = S.encode_all(sae, h)
        rows_nz, vals_nz, starts = analysis.sparse_columns(codes[0].numpy(), codes[1].numpy(), sae.m)
        relations = {}
        for j in range(sae.m):
            a, b = starts[j], starts[j + 1]
            if b - a < 5:
                continue
            relations[j] = analysis.sparse_relations(rows_nz[a:b], vals_nz[a:b], self.labels, self.kinds, len(h),
                                                     min_pos=30)
        fs = FeatureSet(layer, sae, float(fvu), effect, stats["freq"].numpy(), relations, codes=codes)
        self.feature_sets[layer] = fs
        return fs

    # ---- naming features
    def suggest_names(self, layer: str, top: int = 24, features: Any = None) -> list[Any]:
        """A suggested name for each of the layer's top features (study.features(layer) first), from the labels it
        lines up with, with warnings (a feature that fires mostly at kickoffs, say). See names.suggest."""
        from .names import suggest
        return suggest(self._features(layer), features, self.labels, top=top)

    def feature_gallery(self, layer: str, path: str | os.PathLike[str] = "features.html", features: Any = None,
                        n: int = 24, exclude_kickoffs: bool = False, title: str | None = None) -> Path:
        """A page of the layer's features (default: its top n): each one's suggested name and warnings, the labels it
        lines up with, and its three strongest moments drawn from above. exclude_kickoffs: never pick a kickoff as a
        top moment (for features that fire on kickoffs as well as on what you are after)."""
        from .report import feature_gallery_html
        self._features(layer)
        kick = self.labels.get("kickoff") if exclude_kickoffs else None
        exclude = None if kick is None else np.asarray(kick) > 0.5
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(feature_gallery_html(self, layer, features, n, exclude, title), encoding="utf-8")
        return p

    def check_feature(self, layer: str, feature: int | str, scales: tuple[float, ...] = (0.0, 3.0),
                      min_moments: int = 20, store: bool = True) -> FeatureCheck:
        """Turn a feature (its number, or a name you gave it) off (scale 0) and up (scale 3) on the held-out moments
        where it fires (all moments where it fires if the held-out games have too few) and measure what the bot does
        differently: how often its chosen action changes, the KL divergence, and - with an action table - how each
        control moves on average."""
        from .frames import CONTROLS
        fs = self._features(layer)
        j = fs.index(feature)
        act = fs.activation(j)
        rows = self.test[act[self.test] > 0]
        if len(rows) < min_moments:
            rows = np.nonzero(act > 0)[0]
        if len(rows) == 0:
            raise CDSInterpretError(f"Feature {j} of {layer} never fires on these moments, so there's nothing to "
                                    "check. If you picked its number in another session: a dictionary trained again "
                                    "has different features under the same numbers. Save the one you name "
                                    "(feats.save) and load it back (study.load_features).")
        x = self.x[torch.from_numpy(rows)]
        table = None if self.policy.actions is None else torch.as_tensor(self.policy.actions)
        results: dict[float, dict[str, Any]] = {}
        with torch.no_grad():
            if not self.policy.discrete:
                raise CDSInterpretError("check_feature measures a change of the action distribution: it needs a "
                                        "discrete policy (for a continuous one, use study.edits with feature_edit)")
            lp0 = self.policy.log_probs(x)
            c0 = None if table is None or table.shape[0] != lp0.shape[1] else lp0.exp() @ table
            for s in scales:
                with intervene.editing(self.policy, layer, intervene.ablate_features(fs.sae, [j], s)):
                    lp1 = self.policy.log_probs(x)
                r: dict[str, Any] = {"kl": float((lp0.exp() * (lp0 - lp1)).sum(-1).mean()),
                                     "changed": float((lp0.argmax(-1) != lp1.argmax(-1)).float().mean())}
                if c0 is not None and table is not None:
                    c1 = lp1.exp() @ table
                    d = (c1 - c0).mean(0).numpy()
                    r["shifts"] = {k: float(v) for k, v in zip(CONTROLS, d)}
                    r["top_shifts"] = sorted(((k, float(v)) for k, v in zip(CONTROLS, d) if abs(v) >= 0.02),
                                             key=lambda kv: -abs(kv[1]))[:3]
                results[float(s)] = r
        check = FeatureCheck(layer, j, fs.name(j), len(rows), results)
        if store:
            self.feature_checks[(layer, j)] = check
        return check

    def _features(self, layer: str) -> FeatureSet:
        if layer not in self.feature_sets:
            raise CDSInterpretError(f"There are no features for layer {layer!r} yet. Train them first: "
                                    f"study.features({layer!r})")
        return self.feature_sets[layer]

    # ---- edits
    def probe_edit(self, layer: str, label: str, factor: float = 0.0) -> Edit:
        """An edit that scales the layer's component along a label's probe direction (0 removes it, 2 doubles it)."""
        _check_label(label, self.labels)
        self._check_layer(layer)
        if self.probe_results is not None and layer in self.probe_results.by_layer:
            pr = self.probe_results.probe(layer, label)
        else:
            res = P.ridge_suite(self.activations(layer), {label: self.labels[label]}, {label: self.kinds[label]},
                                self.train, self.test, groups=self.games, keep=True)
            if label not in res:
                why = self.why_not(label) or "too few examples"
                raise CDSInterpretError(f"There's no probe for {label!r} at {layer!r}: {why}.")
            pr = res[label]["probe"]
        return intervene.scale_direction(pr.direction(), factor)

    def feature_edit(self, layer: str, features: list[int | str], scale: float = 0.0) -> Edit:
        """An edit that removes (scale 0) or scales those SAE features of the layer (study.features(layer) first),
        each given by its number or a name you gave it."""
        fs = self._features(layer)
        return intervene.ablate_features(fs.sae, [fs.index(j) for j in features], scale)

    def edits(self, layer: str, edits: dict[str, Edit], store: bool = True) -> dict[str, dict[str, float]]:
        """How far each edit of the layer moves the policy on the held-out moments (intervene.edit_effect): mean KL,
        the share of moments whose chosen action changes, total variation."""
        self._check_layer(layer)
        x = self.x[torch.from_numpy(self.test)]
        res = intervene.edit_effect(self.policy, x, layer, edits, batch=self.batch)
        if store:
            for name, v in res.items():
                self.edit_results[f"{layer}: {name}"] = v
        return res

    # ---- what decisions rest on
    def attribution(self, n: int = 400, donors: int = 4, pool: int = 4000, by_labels: tuple[str, ...] = (
            "kickoff", "demo_chance", "threat", "shot_soon", "touch_soon", "in_air"), seed: int = 0) -> dict[str, Any]:
        """What each decision rests on, by entity (attribution.py): on ``n`` held-out moments, the ball, the bot's own
        car, its boost, the opponents and the team-mates are each kept or swapped for another recorded moment's
        (``donors`` of them, from other games), every combination is run, and exact Shapley values split how far the
        policy moves. Also: the same split for the moments where each label of ``by_labels`` holds, and how faithful
        the Shapley ranking is (1 = the best possible order, 0 = chance). Needs the policy's obs function."""
        from . import attribution as AT
        self.policy.require_obs()
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(self.test, size=min(n, len(self.test)), replace=False))
        others = np.setdiff1d(np.arange(len(self.moments)), rows)
        pool_rows = np.sort(rng.choice(others, size=min(pool, len(others)), replace=False))
        allrows = np.concatenate([rows, pool_rows])
        codec = AT.FramesCodec(self.policy)
        parts = codec.decompose(self.moments.frames.take(allrows), self.moments.me[allrows])
        dsel = AT.pick_donors(AT.role_signature(parts), donors, groups=self.games[allrows], seed=seed,
                              pool=np.arange(len(rows), len(allrows)))[:, :len(rows)]
        groups = codec.groups()
        self.log(f"Attribution: {len(rows)} held-out moments x {donors} donors x {2 ** len(groups)} combinations ...")
        tab = AT.coalition_table(self.policy.log_probs, parts, dsel, groups, codec.compose, batch=256)
        kl = AT.kl_table(tab)
        phi = AT.shapley(-kl[..., None])[..., 0]
        occ = np.stack([kl[:, :, (2 ** len(groups) - 1) & ~(1 << g)] for g in range(len(groups))], 2).mean(0)
        names = list(groups)

        def split(m: np.ndarray) -> dict[str, float]:
            tot = float(kl[:, m, 0].mean())
            return {g: float(phi[m, i].mean() / tot) if tot > 0 else float("nan") for i, g in enumerate(names)}
        res: dict[str, Any] = {"n": int(len(rows)), "donors": donors, "groups": names,
                               "shares": {"all moments": split(np.ones(len(rows), bool))},
                               "kl_all_swapped": float(kl[:, :, 0].mean())}
        for lab in by_labels:
            if lab in self.labels:
                m = np.nan_to_num(self.labels[lab][rows]) > 0.5
                if m.sum() >= 20:
                    res["shares"][lab] = split(m)
        orders = {"Shapley": AT.order_from(phi), "one swapped at a time": AT.order_from(occ),
                  "random": np.stack([rng.permutation(len(names)) for _ in range(len(rows))])}
        res["faithfulness"] = {k: v["score_pooled"] for k, v in AT.faithfulness(kl, orders, average_donors=True).items()}
        self.attribution_results = res
        return res

    # ---- the overlay
    def make_pack(self, layer: str, features: dict[str, int] | None = None, probes: dict[str, Any] | None = None,
                  land_layer: str | None = None, title: str = "THOUGHTS", value_scale: float = 1.0,
                  outlook: bool = True, output_layer: str = "", names: Any = None, decoders: bool = True,
                  donors: int = 32) -> dict[str, Any]:
        """A thoughts pack for the live overlay from this study: named features of ``layer``'s SAE, probes of
        ``layer`` {display name: label or (label, look)}, the landing marker from ``land_layer`` (default: the
        layer whose land_x / land_y readout is steadiest among those that read it well), and an outlook probe
        (the ``outlook`` label) when the moments have it. names: a names file (or names.Names) whose confirmed
        names become the pack's features, with their thresholds (added to ``features``). decoders: keep the
        features' decoder directions (the panel's "why this move"); donors: how many recorded moments to keep for
        the panel's "looking at" (0: none)."""
        from .overlay import pack as OP
        self._check_layer(layer)
        if land_layer is not None:
            self._check_layer(land_layer)
        if self.probe_results is None or layer not in self.probe_results.by_layer:
            self.probes(layers=sorted(set(self.layers) | {layer}), baselines=False)
        pr = self.probe_results
        assert pr is not None
        named = {}
        for disp, spec in (probes or {}).items():
            label, look = (spec, {}) if isinstance(spec, str) else spec
            named[disp] = (pr.probe(layer, label), look)
        fs = self.feature_sets.get(layer)
        name_thresholds: dict[str, float] = {}
        if names is not None:
            from .names import Names, read_names
            nm = names if isinstance(names, Names) else read_names(names)
            if nm.layer is not None and nm.layer != layer:
                raise CDSInterpretError(f"the names file is for layer {nm.layer!r}, not {layer!r}")
            if fs is not None:
                fs.check_names(nm, names if not isinstance(names, Names) else "the names file")
            features = {**nm.confirmed(), **(features or {})}
            name_thresholds = nm.thresholds()
            if not features:
                raise CDSInterpretError("the names file has no confirmed names yet: set status = \"confirmed\" on "
                                        "the ones whose moments agree")
        if features and fs is None:
            raise CDSInterpretError(f"Named features need the layer's features first: study.features({layer!r})")
        if features and fs is not None:
            bad = {n: j for n, j in features.items() if not 0 <= int(j) < fs.sae.m}
            if bad:
                raise CDSInterpretError(f"Features {bad} don't exist: this layer's autoencoder has features "
                                        f"0-{fs.sae.m - 1}.")
        th = S.firing_thresholds(fs.sae, self.activations(layer), list(features.values())) if features and fs else {}
        thresholds = {n: name_thresholds.get(n, th[j]) for n, j in (features or {}).items()}
        if name_thresholds and fs is not None and features:
            with torch.no_grad():
                pre = fs.sae.pre(self.activations(layer)).clamp_min(0)
            for n, t in name_thresholds.items():
                j = int(features[n])
                top = float(pre[:, j].max())
                if t >= top:
                    sz = fs.sizes(j)
                    raise CDSInterpretError(
                        f"The names file gives {n!r} (feature {j}) a threshold of {t:g}, but it never got above "
                        f"{top:.2f} in these games, so it would never light up on the panel. When it fires it's "
                        f"from {sz['start']:.2f} to {sz['highest']:.2f}, typically {sz['typical']:.2f}. Pick a "
                        f"threshold in that range, or leave it out to use {sz['start']:.2f}.")
        outlook_probe = None
        if outlook and "outlook" in pr.by_layer.get(layer, {}):
            if np.nanstd(self.labels["outlook"][self.train]) > 0:
                outlook_probe = pr.probe(layer, "outlook")
            else:
                self.log("  (no outlook bar: there were no goals in the training games to learn it from)")
        pk = OP.make_pack(layer, fs.sae if fs else None, features, thresholds, named, self.policy.actions,
                          self.policy.output if self.policy.output != "controls" else "logits", output_layer, title,
                          value_scale, outlook_probe, decoders=decoders)
        if donors > 0:
            try:
                OP.add_donors(pk, self.moments, n=donors)
            except CDSInterpretError as e:
                self.log(f"  (no moments for 'looking at': {e})")
        moves = all(k in self.labels and np.nanstd(self.labels[k]) > 50.0 for k in ("land_x", "land_y"))
        if land_layer is None and moves:
            cands = [(self.steadiness(lay)["miss_med"], lay) for lay in pr.by_layer
                     if "land_x" in pr.by_layer[lay] and "land_y" in pr.by_layer[lay]]
            land_layer = min(cands)[1] if cands else None
        if land_layer is None:
            self.log("  (no landing marker: the ball's landing point didn't vary enough in these games to learn it)")
        else:
            px, py = pr.probe(land_layer, "land_x"), pr.probe(land_layer, "land_y")
            st = self.steadiness(land_layer)
            OP.add_land(pk, land_layer, np.stack([px.w.numpy(), py.w.numpy()], 1), [px.b, py.b],
                        {"miss_med": st.get("miss_med"), "jitter": st.get("jitter")})
        return pk

    # ---- the report
    def report(self, path: str | os.PathLike[str] = "report.html", title: str | None = None) -> Path:
        """Everything this study has computed, on one self-contained HTML page (light and dark)."""
        from .report import build
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(build(self, title or f"Inside {self.policy.name}"), encoding="utf-8")
        return p
