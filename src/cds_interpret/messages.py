"""What a learned team-mate message channel says, and whether anyone listens.

Some team bots learn to talk. Every decision, each car's network also picks one of K symbols (a message head next
to its action head), and its team-mates read those symbols as inputs at their next decision. Nobody tells the
symbols what to mean: training either finds a use for them or it doesn't. This module answers three questions about
such a channel, from recorded moments:

  1. Usage - which symbols get sent, and how sure each decision is: the share of each symbol, the entropy of each
     decision's message distribution and of all the messages together (bits; log2 K is the most there can be).
  2. Meaning - what the symbols say about the situation: the mutual information between the symbol and each label
     (bits), against the same number with the label shifted in time (a baseline that keeps the label's own rhythm),
     and for every symbol the situations it is sent in more often than usual, as a short phrase ("sent mostly when
     closest to the ball and the ball is in their half").
  3. Listening - what the team-mates do with it, the decisive test: for every receiving car, its action distribution
     with the symbol it really got against each other symbol and against silence (KL, bits), and how often its chosen
     action changes. A channel whose symbols track the game perfectly still means nothing if nobody's play depends on
     it.

    import cds_interpret as ci

    chan = ci.messages.Channel(symbols=8, columns=[range(600, 608), range(650, 658)])   # each team-mate's symbol
    rep = study.messages(logits=message_logits, channel=chan)    # logits: [N, 8] array, or fn(inputs) -> [N, 8]
    print(rep.summary())

or without a Study: ``ci.messages.analyze(logits, labels=..., games=..., x=inputs, forward=policy, channel=chan)``.
Symbols as played are the logits' argmax (``sent=`` for others); the distribution (softmax) is the channel as trained
with sampled symbols. docs/howto/message-channel.md walks through it.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .errors import CDSInterpretError, ShapeError

LOG2 = math.log(2.0)
# the verdict thresholds (see verdicts()): a label's information counts when it beats every shifted copy and is at
# least this many bits above their mean; listening counts by the KL of the receivers' action distribution or how often
# their chosen action changes
MIN_BITS = 0.005
LISTEN_CHANGE, BARELY_CHANGE = 0.05, 0.005     # (set high: a soft policy flips near-ties under the tiniest nudge)
LISTEN_KL, BARELY_KL = 0.02, 0.002

# phrases for a label's level, for the per-symbol sentences ((label, level) -> words); "yes" / "no" for yes / no
# labels, "low" / "high" for the bottom and top quarter of a number
PHRASES: dict[tuple[str, str], str] = {
    ("first_to_ball", "yes"): "it is first to the ball", ("first_to_ball", "no"): "another car is first to the ball",
    ("closest", "yes"): "it is its team's closest car to the ball",
    ("closest", "no"): "a team-mate is closer to the ball",
    ("last_back", "yes"): "it is the last car back", ("last_back", "no"): "a team-mate is further back",
    ("goal_side", "yes"): "it is goal-side of the ball", ("goal_side", "no"): "it is caught up-field of the ball",
    ("kickoff", "yes"): "at a kickoff", ("kickoff", "no"): "in open play",
    ("threat", "yes"): "the ball is heading into its own goal",
    ("on_target", "yes"): "the ball is heading into the opponents' goal",
    ("in_air", "yes"): "it is in the air", ("in_air", "no"): "it is on the ground",
    ("on_wall", "yes"): "it is on a wall",
    ("ball_half", "yes"): "the ball is in the opponents' half", ("ball_half", "no"): "the ball is in its own half",
    ("ours_last", "yes"): "its team touched the ball last", ("ours_last", "no"): "the opponents touched the ball last",
    ("demo_chance", "yes"): "it has a demolition chance",
    ("supersonic", "yes"): "it is supersonic", ("has_flip", "no"): "it has used its flip",
    ("boost", "low"): "its boost is low", ("boost", "high"): "its boost is high",
    ("ball_height", "low"): "the ball is on the ground", ("ball_height", "high"): "the ball is high in the air",
    ("ball_dist", "low"): "it is near the ball", ("ball_dist", "high"): "it is far from the ball",
    ("mate_dist", "low"): "a team-mate is close to it", ("mate_dist", "high"): "it is far from its team-mates",
    ("speed", "low"): "it is slow", ("speed", "high"): "it is fast",
    ("touch_age", "low"): "the ball was just touched", ("touch_age", "high"): "nobody has touched the ball for a while",
    ("ttb_adv", "low"): "the opponents will reach the ball first", ("ttb_adv", "high"): "it will reach the ball first",
    ("ball_vy", "low"): "the ball is coming toward its goal", ("ball_vy", "high"): "the ball is going up-field",
    ("height", "high"): "it is high up",
    ("land_y", "low"): "the ball will come down in its own end", ("land_y", "high"): "the ball will come down in "
    "the opponents' end", ("land_dist", "low"): "the ball will come down near it",
    ("land_dist", "high"): "the ball will come down far from it", ("land_t", "low"): "the ball is on the ground",
    ("land_t", "high"): "the ball is in the air for a while",
    ("ttb_me", "low"): "it can reach the ball soon", ("ttb_me", "high"): "it is far from reaching the ball",
    ("ttb_mate", "low"): "a team-mate can reach the ball soon",
    ("ttb_mate", "high"): "its team-mates are far from the ball",
    ("ttb_opp", "low"): "an opponent can reach the ball soon", ("ttb_opp", "high"): "the opponents are far from "
    "the ball", ("boost_adv", "low"): "it has less boost than the opponents",
    ("boost_adv", "high"): "it has more boost than the opponents",
    ("threat_t", "low"): "the ball is about to go into its own goal",
}


def phrase(label: str, level: str) -> str:
    """Words for a label's level ("ball_half", "yes" -> "the ball is in the opponents' half")."""
    if (label, level) in PHRASES:
        return PHRASES[(label, level)]
    name = label.replace("_", " ")
    return {"yes": name, "no": f"not {name}", "low": f"{name} is low", "high": f"{name} is high"}.get(level,
                                                                                                    f"{name}: {level}")


# ----------------------------------------------------------------------------------------------------- the channel
@dataclass
class Channel:
    """Where each team-mate's symbol sits in the network's input.

    Args:
        symbols: K, the number of symbols.
        columns: one list of K column indices per sender slot (a range or a slice works): the K inputs that carry
            that team-mate's symbol as a one-hot.
        silence: the value of a silent slot's columns (0).
        live: ``fn(x) -> [N, slots] bool`` - which slots carry a team-mate's message at each moment (default: a slot
            whose columns hold a one-hot; an empty slot, an opponent's or a silent one is all zeros).
    """

    symbols: int
    columns: Sequence[Any]
    silence: float = 0.0
    live: Callable[[torch.Tensor], torch.Tensor] | None = None

    def __post_init__(self) -> None:
        cols = []
        for c in self.columns:
            if isinstance(c, slice):
                c = range(c.start or 0, c.stop, c.step or 1)
            idx = [int(i) for i in c]
            if len(idx) != self.symbols:
                raise ShapeError(f"Channel: a sender slot has {len(idx)} columns; there are {self.symbols} symbols. "
                                 "Each slot needs one column per symbol, in symbol order.")
            cols.append(idx)
        if not cols:
            raise CDSInterpretError("Channel: no sender slots. Pass columns=[[...K columns...], ...], one per "
                                    "team-mate slot in the input.")
        self.columns = cols
        self._idx = torch.as_tensor(cols, dtype=torch.long)                      # [slots, K]
        self._dev: dict[str, torch.Tensor] = {}

    @property
    def slots(self) -> int:
        return len(self.columns)

    def _on(self, device: torch.device) -> torch.Tensor:
        """The column indices [slots, K] on a device (kept: no copy a call)."""
        key = str(device)
        if key not in self._dev:
            self._dev[key] = self._idx.to(device)
        return self._dev[key]

    def incoming(self, x: torch.Tensor) -> torch.Tensor:
        """[N, slots] the symbol each slot holds (-1: silent / not a team-mate)."""
        v = x[:, self._on(x.device)]                                              # [N, slots, K]
        sym = v.argmax(-1)
        on = (v - self.silence).abs().sum(-1) > 0.5
        if self.live is not None:
            on = on & self.live(x).to(on.device).bool()
        return torch.where(on, sym, torch.full_like(sym, -1))

    def put(self, x: torch.Tensor, slot: int, symbol: int | None) -> torch.Tensor:
        """A copy of x with slot's symbol replaced (None: silence)."""
        y = x.clone()
        idx = self._on(x.device)[slot]
        y[:, idx] = self.silence
        if symbol is not None:
            y[:, idx[int(symbol)]] = 1.0
        return y


# ------------------------------------------------------------------------------------------------------- the math
def entropy_bits(p: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy in bits along an axis (rows that sum to 1)."""
    p = np.asarray(p, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, -p * np.log2(np.where(p > 0, p, 1.0)), 0.0)
    return t.sum(axis)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, np.float64)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def usage(probs: np.ndarray, sent: np.ndarray) -> dict[str, Any]:
    """Which symbols are sent and how sure each decision is. probs [N, K] (each decision's distribution), sent [N]
    (the symbol played). Entropies in bits."""
    probs = np.asarray(probs, np.float64)
    k = probs.shape[1]
    sent = np.asarray(sent, np.int64)
    share = np.bincount(sent, minlength=k)[:k] / max(len(sent), 1)
    mean_p = probs.mean(0) if len(probs) else np.full(k, np.nan)
    h_dec = float(entropy_bits(probs).mean()) if len(probs) else float("nan")
    h_mean = float(entropy_bits(mean_p))
    return {"symbols": k, "moments": int(len(sent)), "share_sent": share.tolist(), "mean_prob": mean_p.tolist(),
            "entropy_per_decision": h_dec, "entropy_sent": float(entropy_bits(share)), "entropy_mean_prob": h_mean,
            # the sampled symbol's information about the moment it was sent in (H of the average minus the
            # average H); the played (argmax) symbol is a function of the moment, so its information is entropy_sent
            "info_sampled": max(h_mean - h_dec, 0.0), "max_bits": math.log2(k),
            "top_prob": float(probs.max(1).mean()) if len(probs) else float("nan")}


def discretize(values: np.ndarray, kind: str | None = None, bins: int = 4) -> tuple[np.ndarray, list[str]]:
    """A label as codes 0..L-1 and a name per level: yes / no labels as "no" / "yes", numbers in quantile bins
    ("low" the bottom bin, "high" the top one, "q2", ... between), NaN as its own level "n/a"."""
    v = np.asarray(values, np.float64)
    ok = np.isfinite(v)
    vals = np.unique(v[ok])
    if kind == "bin" or (kind is None and len(vals) <= 2 and set(vals.tolist()) <= {0.0, 1.0}):
        codes = (v > 0.5).astype(np.int64)
        names = ["no", "yes"]
    else:
        qs = np.unique(np.quantile(v[ok], np.linspace(0, 1, bins + 1)[1:-1])) if ok.any() else np.array([])
        codes = np.searchsorted(qs, v, side="right").astype(np.int64)
        n = len(qs) + 1
        names = ["low"] + [f"q{i + 1}" for i in range(1, n - 1)] + ["high"] if n > 1 else ["all"]
    if (~ok).any():
        codes = np.where(ok, codes, len(names))
        names = names + ["n/a"]
    return codes, names


def mutual_info(a: np.ndarray, b: np.ndarray) -> float:
    """Plug-in mutual information (bits) between two code arrays."""
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    if not len(a):
        return float("nan")
    na, nb = int(a.max()) + 1, int(b.max()) + 1
    joint = np.bincount(a * nb + b, minlength=na * nb).reshape(na, nb) / len(a)
    pa, pb = joint.sum(1, keepdims=True), joint.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(joint > 0, joint * np.log2(joint / (pa * pb)), 0.0)
    return float(max(t.sum(), 0.0))


def mutual_info_soft(probs: np.ndarray, codes: np.ndarray) -> float:
    """Mutual information (bits) between a symbol drawn from each moment's distribution and a label: H(average
    distribution) - sum over levels of P(level) H(average distribution in that level)."""
    probs, codes = np.asarray(probs, np.float64), np.asarray(codes, np.int64)
    if not len(codes):
        return float("nan")
    n_lev = int(codes.max()) + 1
    cnt = np.bincount(codes, minlength=n_lev).astype(np.float64)
    sums = np.stack([np.bincount(codes, weights=probs[:, j], minlength=n_lev) for j in range(probs.shape[1])], 1)
    keep = cnt > 0
    cond = sums[keep] / cnt[keep, None]
    return float(max(entropy_bits(probs.mean(0)) - (cnt[keep] / len(codes) * entropy_bits(cond)).sum(), 0.0))


def _shift(codes: np.ndarray, order: np.ndarray, groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The codes shifted by a random offset within each group, in time order (circularly): every group keeps its own
    sequence of label values - their rhythm and how long each lasts - but out of step with the symbols."""
    out = codes.copy()
    for idx in np.split(order, np.flatnonzero(np.diff(groups[order])) + 1):
        n = len(idx)
        if n > 2:
            out[idx] = codes[np.roll(idx, int(rng.integers(1, n)))]
    return out


def lifts(weights: np.ndarray, codes: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
    """P(level | symbol) / P(level) for every symbol and level, with the symbols weighted per moment by ``weights``
    [N, K] (a one-hot of what was sent, or each moment's distribution). Returns {"lift": [K, L], "p_given": [K, L],
    "base": [L], "mass": [K]}."""
    w = np.asarray(weights, np.float64)
    n_lev = len(names)
    onehot = np.zeros((len(codes), n_lev))
    onehot[np.arange(len(codes)), codes] = 1.0
    mass = w.sum(0)                                                              # [K]
    p_given = (w.T @ onehot) / np.maximum(mass[:, None], 1e-12)                  # [K, L]
    base = onehot.mean(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = np.where(base[None] > 0, p_given / base[None], np.nan)
    return {"lift": lift, "p_given": p_given, "base": base, "mass": mass}


def meaning(sent: np.ndarray, labels: dict[str, np.ndarray], probs: np.ndarray | None = None,
            kinds: dict[str, str] | None = None, games: np.ndarray | None = None, order: np.ndarray | None = None,
            seq: np.ndarray | None = None, bins: int = 4, shuffles: int = 20, seed: int = 0, min_moments: int = 30,
            min_lift: float = 1.25, min_share: float = 0.1) -> dict[str, Any]:
    """What the symbols say about the situation.

    For each label: the mutual information (bits) between the symbol as sent and the label (in ``bins`` quantile
    bins for numbers), and between a symbol drawn from each decision's distribution and the label (``probs``);
    each against ``shuffles`` copies of the label shifted in time within each sequence (``seq``: one car's moments
    of one game, in ``order``; default one sequence per game, else one in all) - ``excess`` is the information above
    their mean, and it counts (``beats``) when it is above every shifted copy. ``share`` is the excess over the
    label's own entropy (how much of the label the symbol tells).

    For each symbol: its lift for every label level - how much more often the level holds when that symbol is sent
    than overall - and a phrase from the strongest ones (at least ``min_lift`` times as often, in at least
    ``min_share`` of the symbol's moments, from ``min_moments`` moments or more).
    """
    sent = np.asarray(sent, np.int64)
    n = len(sent)
    k = int(probs.shape[1]) if probs is not None else int(sent.max()) + 1
    seq = np.zeros(n, np.int64) if seq is None and games is None else np.asarray(seq if seq is not None else games)
    order = np.lexsort((np.arange(n) if order is None else np.asarray(order), seq))
    rng = np.random.default_rng(seed)
    w_sent = np.zeros((n, k))
    w_sent[np.arange(n), sent] = 1.0
    out: dict[str, Any] = {"labels": {}, "symbols": {}, "levels": {}}
    lift_rows: dict[str, dict[str, Any]] = {}
    for name, vals in labels.items():
        vals = np.asarray(vals)
        if len(vals) != n:
            raise ShapeError(f"label {name!r} has {len(vals)} values for {n} moments")
        codes, levels = discretize(vals, (kinds or {}).get(name), bins)
        if len(np.unique(codes)) < 2:
            continue
        h_lab = float(entropy_bits(np.bincount(codes) / n))
        mi = mutual_info(sent, codes)
        mi_soft = mutual_info_soft(probs, codes) if probs is not None else float("nan")
        base, base_soft = [], []
        for _ in range(shuffles):
            sh = _shift(codes, order, seq, rng)
            base.append(mutual_info(sent, sh))
            if probs is not None:
                base_soft.append(mutual_info_soft(probs, sh))
        base_a = np.asarray(base)
        row = {"levels": levels, "entropy": h_lab, "mi": mi, "baseline": float(base_a.mean()),
               "baseline_max": float(base_a.max()), "excess": mi - float(base_a.mean()),
               "beats": bool(mi > base_a.max()) if shuffles else False}
        row["share"] = row["excess"] / h_lab if h_lab > 0 else float("nan")
        if probs is not None:
            bs = np.asarray(base_soft)
            row.update({"mi_soft": mi_soft, "baseline_soft": float(bs.mean()), "baseline_soft_max": float(bs.max()),
                        "excess_soft": mi_soft - float(bs.mean()), "beats_soft": bool(mi_soft > bs.max())})
        out["labels"][name] = row
        lift_rows[name] = {"codes": codes, "levels": levels}
    out["levels"] = {name: r["levels"] for name, r in lift_rows.items()}
    for which, w in (("sent", w_sent), ("soft", np.asarray(probs, np.float64) if probs is not None else None)):
        if w is None:
            continue
        per_sym: dict[int, dict[str, Any]] = {}
        for name, r in lift_rows.items():
            lf = lifts(w, r["codes"], r["levels"])
            for s in range(k):
                d = per_sym.setdefault(s, {"mass": float(lf["mass"][s]), "share": float(lf["mass"][s] / max(n, 1)),
                                           "levels": []})
                for j, lev in enumerate(r["levels"]):
                    if lev in ("n/a", "all") or (lev.startswith("q")):
                        continue
                    d["levels"].append({"label": name, "level": lev, "lift": float(lf["lift"][s, j]),
                                        "p_given": float(lf["p_given"][s, j]), "base": float(lf["base"][j])})
        for s, d in per_sym.items():
            d["levels"].sort(key=lambda e: -(e["lift"] if np.isfinite(e["lift"]) else -1.0))
            first = _strongest(d["levels"], min_lift, min_share) if d["mass"] >= min_moments else None
            if first is not None:
                # the second part of the phrase: among the moments where the first holds, the level whose lift is
                # largest there (a label that only repeats the first - the ball's landing spot after "the ball is in
                # their half" - has no lift left and is passed over)
                r1 = lift_rows[first["label"]]
                m = r1["codes"] == r1["levels"].index(first["level"])
                cond = []
                for name, r in lift_rows.items():
                    if name == first["label"] or w[m, s].sum() < min_moments:
                        continue
                    lf = lifts(w[m][:, [s]], r["codes"][m], r["levels"])
                    for j, lev in enumerate(r["levels"]):
                        if lev not in ("n/a", "all") and not lev.startswith("q"):
                            cond.append({"label": name, "level": lev, "lift": float(lf["lift"][0, j]),
                                         "p_given": float(lf["p_given"][0, j])})
                cond.sort(key=lambda e: -(e["lift"] if np.isfinite(e["lift"]) else -1.0))
                d["then"] = _strongest(cond, min_lift, min_share)
            d["phrase"] = symbol_phrase(d, min_moments=min_moments, min_lift=min_lift, min_share=min_share)
        out["symbols"][which] = per_sym
    return out


def _strongest(levels: list[dict[str, Any]], min_lift: float, min_share: float) -> dict[str, Any] | None:
    """The first level (of a list sorted by lift) at least min_lift times as frequent as usual and in at least
    min_share of the moments."""
    for e in levels:
        if np.isfinite(e["lift"]) and e["lift"] >= min_lift and e["p_given"] >= min_share:
            return e
    return None


def symbol_phrase(d: dict[str, Any], min_moments: int = 30, min_lift: float = 1.25, min_share: float = 0.1) -> str:
    """A short plain-words phrase for one symbol from its lifts (meaning()'s per-symbol entry): the situation it is
    sent in most more often than usual and, among those moments, the one that adds most."""
    if d["mass"] < min_moments:
        return "hardly ever sent" if d["mass"] > 0 else "never sent"
    first = _strongest(d["levels"], min_lift, min_share)
    if first is None:
        return "sent about as often in every situation (no clear meaning)"
    out = f"sent mostly when {phrase(first['label'], first['level'])} ({first['lift']:.1f}x as often)"
    then = d.get("then")
    if then is not None:
        out += f", especially if {phrase(then['label'], then['level'])} too ({then['lift']:.1f}x)"
    return out


# -------------------------------------------------------------------------------------------------- the listening
def _kl_bits(lp: torch.Tensor, lq: torch.Tensor) -> torch.Tensor:
    """KL(p || q) in bits from log-probabilities [..., A]."""
    return (lp.exp() * (lp - lq)).sum(-1) / LOG2


@torch.no_grad()
def listen(forward: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, channel: Channel, batch: int = 4096,
           references: dict[str, Callable[[torch.Tensor, int], torch.Tensor]] | None = None) -> dict[str, np.ndarray]:
    """The listening test on inputs x [N, D]: for every moment and every live sender slot, the receiver's action
    distribution (``forward(x)`` -> action logits) with the symbol it really got, with each of the K symbols in its
    place and with silence. ``references``: other edits of a slot's inputs to compare with, {name: fn(x, slot) ->
    x'} (e.g. a team-mate's boost changed), each scored the same way.

    Every version of a moment - as it was, each edit - goes through the network in the same call, so the comparison
    holds no rounding of its own (on a card, one input in two calls of different sizes can come out a hair apart, and
    a near-tie between two actions would flip): ``noise`` is the KL of the real symbol put back in place, 0 when that
    holds.

    Returns one row per (moment, live slot): ``moment``, ``slot``, ``real`` (the symbol it got), ``kl`` [K + 1]
    (bits; K symbols, then silence; 0 at the real symbol), ``changed`` [K + 1] (the chosen action differs), ``noise``,
    and per reference ``kl_<name>`` / ``changed_<name>``."""
    k = channel.symbols
    refs = references or {}
    keys = ["moment", "slot", "real", "kl", "changed", "noise"] + [f"{p}_{n}" for n in refs for p in ("kl", "changed")]
    rows: dict[str, list] = {key: [] for key in keys}
    inc_all = torch.cat([channel.incoming(x[a:a + batch]) for a in range(0, len(x), batch)]) if len(x) else \
        torch.zeros(0, channel.slots, dtype=torch.long)
    for s in range(channel.slots):
        live_all = torch.nonzero(inc_all[:, s] >= 0).flatten()
        n_ed = 2 + k + len(refs)                          # as it was, K symbols, silence, the references
        step = max(1, batch // n_ed)
        for a in range(0, len(live_all), step):
            live = live_all[a:a + step]
            xs = x[live.to(x.device)]
            real = inc_all[live, s]
            versions = [xs] + [channel.put(xs, s, sym) for sym in range(k)] + [channel.put(xs, s, None)]
            versions += [fn(xs, s) for fn in refs.values()]
            lp = torch.log_softmax(forward(torch.cat(versions)).float(), -1).view(n_ed, len(live), -1)
            kl = (lp[0].exp()[None] * (lp[0][None] - lp[1:])).sum(-1).clamp_min(0.0) / LOG2      # [n_ed - 1, m]
            changed = lp[1:].argmax(-1) != lp[0].argmax(-1)[None]
            rows["moment"].append(live.cpu())
            rows["slot"].append(torch.full((len(live),), s, dtype=torch.long))
            rows["real"].append(real.cpu())
            rows["kl"].append(kl[:k + 1].T.cpu())
            rows["changed"].append(changed[:k + 1].T.cpu())
            rows["noise"].append(kl[real.to(kl.device).clamp_min(0), torch.arange(len(live), device=kl.device)].cpu())
            for j, name in enumerate(refs):
                rows[f"kl_{name}"].append(kl[k + 1 + j].cpu())
                rows[f"changed_{name}"].append(changed[k + 1 + j].cpu())
    out = {}
    for key, v in rows.items():
        if v:
            out[key] = torch.cat(v).numpy()
        else:
            width = (k + 1,) if key in ("kl", "changed") else ()
            out[key] = np.zeros((0,) + width, np.float32 if key.startswith(("kl", "noise")) else np.int64)
    order = np.argsort(out["moment"], kind="stable")
    return {key: v[order] for key, v in out.items()}


def listening(rows: dict[str, np.ndarray], symbols: int, share_sent: Sequence[float] | None = None) -> dict[str, Any]:
    """listen()'s rows in numbers: over every (receiver, live slot), the mean KL (bits) and the share of decisions
    whose action changes when the real symbol is swapped for another (each other symbol alike; ``swap_natural``
    weighs them by how often each is sent, as a natural wrong message would be) and when it is silenced; the worst
    swap per decision; the [real, other] matrix of mean KL; and the references."""
    k = symbols
    n = len(rows["real"])
    out: dict[str, Any] = {"pairs": int(n)}
    if not n:
        return out
    kl, ch, real = rows["kl"].astype(np.float64), rows["changed"].astype(np.float64), rows["real"].astype(np.int64)
    other = np.ones((n, k), bool)
    other[np.arange(n), real] = False
    out["kl_swap"] = float((kl[:, :k] * other).sum() / other.sum())
    out["changed_swap"] = float((ch[:, :k] * other).sum() / other.sum())
    out["kl_swap_max"] = float(np.where(other, kl[:, :k], 0.0).max(1).mean())
    out["changed_any"] = float((ch[:, :k] * other).max(1).mean())
    out["kl_silence"], out["changed_silence"] = float(kl[:, k].mean()), float(ch[:, k].mean())
    if share_sent is not None:
        w = np.asarray(share_sent, np.float64)[None, :] * other
        w = w / np.maximum(w.sum(1, keepdims=True), 1e-12)
        out["kl_swap_natural"] = float((kl[:, :k] * w).sum(1).mean())
        out["changed_swap_natural"] = float((ch[:, :k] * w).sum(1).mean())
    mat = np.full((k, k), np.nan)
    for r in range(k):
        m = real == r
        if m.any():
            mat[r] = kl[m, :k].mean(0)
    out["kl_matrix"] = mat.tolist()
    if "noise" in rows:
        out["noise_max"] = float(np.max(rows["noise"])) if n else 0.0
    out["real_share"] = (np.bincount(real, minlength=k)[:k] / n).tolist()
    for key in rows:
        if key.startswith("kl_") and key != "kl":
            name = key[3:]
            out.setdefault("references", {})[name] = {"kl": float(rows[key].mean()),
                                                      "changed": float(rows[f"changed_{name}"].mean())}
    return out


# ------------------------------------------------------------------------------------------------------ verdicts
def verdicts(use: dict[str, Any], mean: dict[str, Any] | None, lis: dict[str, Any] | None) -> dict[str, str]:
    """The three answers in plain words, from usage(), meaning() and listening()."""
    out: dict[str, str] = {}
    k = use["symbols"]
    used = sum(1 for s in use["share_sent"] if s >= 0.01)
    out["usage"] = (f"{used} of {k} symbols are sent (1% of decisions or more); a decision's message distribution "
                    f"holds {use['entropy_per_decision']:.2f} of {use['max_bits']:.2f} possible bits of uncertainty "
                    f"(its top symbol {use['top_prob']:.0%} on average), and the sent symbols together "
                    f"{use['entropy_sent']:.2f} bits.")
    if mean is not None:
        rows = [(n, r) for n, r in mean["labels"].items()]
        hits = sorted(((n, r) for n, r in rows if r["beats"] and r["excess"] >= MIN_BITS),
                      key=lambda t: -t[1]["excess"])
        soft = sorted(((n, r) for n, r in rows if r.get("beats_soft") and r.get("excess_soft", 0) >= MIN_BITS),
                      key=lambda t: -t[1]["excess_soft"])
        if hits:
            top = ", ".join(f"{n} ({r['excess']:.3f} bits, {r['share']:.0%} of it)" for n, r in hits[:3])
            out["meaning"] = (f"Yes - the symbols as sent carry information about the situation: {len(hits)} of "
                              f"{len(rows)} labels beat every time-shifted copy by {MIN_BITS} bits or more; the most "
                              f"about {top}.")
        elif soft:
            top = ", ".join(f"{n} ({r['excess_soft']:.3f} bits)" for n, r in soft[:3])
            out["meaning"] = (f"Only in the sampled symbols: the argmax symbols say nothing measurable, but drawing "
                              f"from each decision's distribution carries a little ({top}).")
        else:
            out["meaning"] = (f"No measurable information: no label's mutual information with the symbol beats its "
                              f"time-shifted copies by {MIN_BITS} bits.")
    if lis is not None and lis.get("pairs"):
        ch = max(lis["changed_swap"], lis["changed_silence"])
        kl = max(lis["kl_swap"], lis["kl_silence"])
        if ch >= LISTEN_CHANGE or kl >= LISTEN_KL:
            v = "Yes - team-mates listen"
        elif ch >= BARELY_CHANGE or kl >= BARELY_KL:
            v = "Barely - team-mates' play moves only a little with the message"
        else:
            v = "No - team-mates don't listen"
        out["listening"] = (f"{v}: swapping the symbol a receiver got for another changes its chosen action in "
                            f"{lis['changed_swap']:.2%} of decisions (KL {lis['kl_swap']:.4f} bits), silencing it in "
                            f"{lis['changed_silence']:.2%} (KL {lis['kl_silence']:.4f} bits); the worst swap per "
                            f"decision changes it in {lis['changed_any']:.2%}.")
    else:
        out["listening"] = "Not tested (no inputs, forward function and channel given, or no live team-mate slots)."
    return out


# --------------------------------------------------------------------------------------------------- the report
@dataclass
class MessageReport:
    """The three answers and their numbers. ``summary()`` for reading, ``to_dict()`` / ``save()`` for keeping."""

    usage: dict[str, Any]
    meaning: dict[str, Any] | None = None
    listening: dict[str, Any] | None = None
    verdicts: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def phrases(self, which: str = "sent") -> dict[int, str]:
        """A phrase per symbol (``which``: "sent", the symbols as played, or "soft", the distributions)."""
        if not self.meaning:
            return {}
        return {int(s): d["phrase"] for s, d in self.meaning["symbols"].get(which, {}).items()}

    def label_table(self) -> list[dict[str, Any]]:
        """One row per label, the most informative first."""
        if not self.meaning:
            return []
        rows = [{"label": n, **{k: v for k, v in r.items() if k != "levels"}} for n, r in self.meaning["labels"].items()]
        return sorted(rows, key=lambda r: -r["excess"])

    def summary(self, top: int = 8) -> str:
        u = self.usage
        lines = [f"Message channel: {u['symbols']} symbols, {u['moments']:,} decisions"]
        for k in ("usage", "meaning", "listening"):
            if k in self.verdicts:
                lines.append(f"  {k.upper()}: {self.verdicts[k]}")
        lines.append("  symbols as sent (share, then mean probability):")
        for s in range(u["symbols"]):
            ph = self.phrases("sent").get(s, "")
            lines.append(f"    {s}: {u['share_sent'][s]:6.1%}  {u['mean_prob'][s]:6.1%}  {ph}")
        if self.meaning:
            lines.append("  information about each label (bits: as sent / time-shifted baseline / sampled):")
            for r in self.label_table()[:top]:
                star = "*" if r["beats"] and r["excess"] >= MIN_BITS else " "
                soft = f"  {r['mi_soft']:.4f}" if "mi_soft" in r else ""
                lines.append(f"   {star}{r['label']:<16} {r['mi']:.4f} / {r['baseline']:.4f}{soft}   "
                             f"(label {r['entropy']:.2f} bits)")
        if self.listening and self.listening.get("pairs"):
            li = self.listening
            lines.append(f"  listening over {li['pairs']:,} receiver-sender pairs: swap KL {li['kl_swap']:.5f} bits, "
                         f"changed {li['changed_swap']:.3%}; silence KL {li['kl_silence']:.5f}, changed "
                         f"{li['changed_silence']:.3%}")
            for name, r in li.get("references", {}).items():
                lines.append(f"    reference '{name}': KL {r['kl']:.5f} bits, changed {r['changed']:.3%}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        def clean(o: Any) -> Any:
            if isinstance(o, dict):
                return {str(k): clean(v) for k, v in o.items()}
            if isinstance(o, (list, tuple)):
                return [clean(v) for v in o]
            if isinstance(o, np.ndarray):
                return clean(o.tolist())
            if isinstance(o, (np.floating, float)):
                return None if not np.isfinite(o) else float(o)
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, np.bool_):
                return bool(o)
            return o
        return clean({"usage": self.usage, "meaning": self.meaning, "listening": self.listening,
                      "verdicts": self.verdicts, "meta": self.meta})

    def save(self, path: str | os.PathLike[str]) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return p

    def figures(self) -> dict[str, str]:
        """SVG figures: symbol shares, information per label against its baseline, and listening."""
        from .viz import plots
        u = self.usage
        k = u["symbols"]
        figs = {"usage": plots.bars([f"symbol {s}" for s in range(k)], np.asarray(u["share_sent"]) * 100.0,
                                    "How often each symbol is sent (% of decisions)", fmt="{:.1f}",
                                    note=f"per-decision entropy {u['entropy_per_decision']:.2f} of "
                                         f"{u['max_bits']:.2f} bits")}
        if self.meaning:
            rows = self.label_table()[:12]
            if rows:
                figs["meaning"] = plots.bars([r["label"] for r in rows], [max(r["excess"], 0.0) for r in rows],
                                             "What the sent symbol tells about each label (bits above the shifted "
                                             "baseline)", fmt="{:.3f}")
        if self.listening and self.listening.get("pairs"):
            li = self.listening
            names = ["another symbol", "silence"] + [f"ref: {n}" for n in li.get("references", {})]
            vals = [li["changed_swap"], li["changed_silence"]] + [r["changed"] for r in
                                                                  li.get("references", {}).values()]
            figs["listening"] = plots.bars(names, np.asarray(vals) * 100.0,
                                           "A team-mate's chosen action changes (% of decisions) when its incoming "
                                           "message is ...", fmt="{:.2f}")
        return figs


def analyze(logits: Any, labels: dict[str, np.ndarray] | None = None, games: np.ndarray | None = None,
            sent: np.ndarray | None = None, order: np.ndarray | None = None, seq: np.ndarray | None = None,
            kinds: dict[str, str] | None = None, x: torch.Tensor | None = None,
            forward: Callable[[torch.Tensor], torch.Tensor] | None = None, channel: Channel | None = None,
            references: dict[str, Callable] | None = None, bins: int = 4, shuffles: int = 20, seed: int = 0,
            batch: int = 4096, max_listen: int | None = 20000, meta: dict[str, Any] | None = None) -> MessageReport:
    """Usage, meaning and listening in one call.

    Args:
        logits: [N, K] message logits per decision (array or tensor).
        labels: {name: [N]} the situation at each decision (the moments' labels); None skips the meaning.
        games / seq / order: the game of each moment, a sequence key (one car in one game) and the time order, for
            the time-shifted baseline (seq defaults to games).
        sent: the symbol played at each decision (default: the logits' argmax, as an argmax bot plays).
        x, forward, channel: the receivers' inputs [M, D] (any moments; rows with no live team-mate slot are
            skipped), the policy's action logits ``forward(x)``, and where the symbols sit (Channel) - for the
            listening test; ``references`` other edits to compare with (listen()). ``max_listen`` moments at most
            (an even sample).
    """
    lg = np.asarray(logits.detach().cpu().numpy() if isinstance(logits, torch.Tensor) else logits, np.float64)
    if lg.ndim != 2:
        raise ShapeError(f"logits should be [moments, symbols]; got shape {lg.shape}")
    probs = softmax(lg)
    s = lg.argmax(1) if sent is None else np.asarray(sent, np.int64)
    use = usage(probs, s)
    mean = None
    if labels:
        mean = meaning(s, labels, probs=probs, kinds=kinds, games=games, order=order, seq=seq, bins=bins,
                       shuffles=shuffles, seed=seed)
    lis = None
    if x is not None and forward is not None and channel is not None:
        if channel.symbols != lg.shape[1]:
            raise ShapeError(f"the channel has {channel.symbols} symbols, the logits {lg.shape[1]}")
        xx = torch.as_tensor(x)
        if max_listen and len(xx) > max_listen:
            xx = xx[np.linspace(0, len(xx) - 1, max_listen).astype(np.int64)]
        rows = listen(forward, xx, channel, batch=batch, references=references)
        lis = listening(rows, channel.symbols, use["share_sent"])
    rep = MessageReport(use, mean, lis, verdicts(use, mean, lis), dict(meta or {}))
    return rep


def situations(frames: Any, me: np.ndarray) -> dict[str, np.ndarray]:
    """Extra labels a message often speaks of, from the game state (``cds_interpret.frames.Frames``): ``closest``
    (the car is its team's closest to the ball), ``ball_half`` (the ball is in the opponents' half), ``mate_dist``
    (distance to the nearest team-mate / 5000; NaN without one)."""
    from .frames import canon, team_sign
    n = len(frames)
    rows = np.arange(n)
    me = np.asarray(me, np.int64)
    team = frames.car_team[rows, me]
    alive = (frames.car_present > 0.5) & (frames.car_demoed < 0.5)
    mates = (frames.car_team == team[:, None]) & alive
    d_ball = np.linalg.norm(frames.car_pos - frames.ball_pos[:, None], axis=-1)
    d_ball = np.where(mates, d_ball, np.inf)
    closest = (d_ball[rows, me] <= d_ball.min(1) + 1e-6) & alive[rows, me]
    by = canon(frames.ball_pos, team_sign(team))[:, 1]
    others = mates.copy()
    others[rows, me] = False
    d_mate = np.linalg.norm(frames.car_pos - frames.car_pos[rows, me][:, None], axis=-1)
    d_mate = np.where(others, d_mate, np.inf).min(1)
    return {"closest": closest.astype(np.float32), "ball_half": (by > 0).astype(np.float32),
            "mate_dist": np.where(np.isfinite(d_mate), d_mate / 5000.0, np.nan).astype(np.float32)}


KINDS = {"closest": "bin", "ball_half": "bin", "mate_dist": "reg", "ours_last": "bin", "touch_age": "reg"}
