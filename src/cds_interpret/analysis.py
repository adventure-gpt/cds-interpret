"""Reading a feature: how it relates to the labels, its top moments, and a plain description of a moment.

    rel = label_relations(act, labels, kinds)          # per label: AUROC (binary) or Pearson r (regression)
    top = top_moments(act, k=12, groups=game_ids, min_gap=8, steps=step_ids)   # the strongest, at most one a stretch
    describe(frame, me)                                # "ours at (x, y) 1840 uu/s on the ground, 34 boost; ball ..."
    guess_name(rel)                                    # the best-matching label, as a GUESS with its score
    causal_targets(features)                           # a causal test's target, group and controls, by a fixed rule

All game-independent except describe(), which reads a Frames moment (any bot's).
"""
from __future__ import annotations

import numpy as np

from .frames import Frames, canon, team_sign
from .probes import auroc


def label_relations(act: np.ndarray, labels: dict[str, np.ndarray], kinds: dict[str, str],
                    min_pos: int = 30) -> dict[str, dict]:
    """For a feature's activation act [N] (0 where it is off): per label, AUROC of act as a detector (binary labels)
    or Pearson r (regression labels), on the rows where the label is defined; also, for binary labels, P(label |
    feature on) against the base rate (the lift)."""
    act = np.asarray(act, np.float64)
    on = act > 0
    out = {}
    for k, y in labels.items():
        y = np.asarray(y, np.float64)
        v = ~np.isnan(y)
        if v.sum() < 50:
            continue
        a, yy = act[v], y[v]
        if kinds.get(k) == "bin":
            pos = int((yy > 0.5).sum())
            if pos < min_pos or pos == len(yy):
                continue
            base = pos / len(yy)
            p_on = float((yy[on[v]] > 0.5).mean()) if on[v].any() else float("nan")
            out[k] = {"auroc": auroc(a, yy), "base": base, "p_given_on": p_on,
                      "lift": p_on / base if base > 0 and np.isfinite(p_on) else float("nan"),
                      "recall": float((on[v] & (yy > 0.5)).sum() / pos), "coverage": float(v.mean())}
        else:
            if np.std(a) == 0 or np.std(yy) == 0:
                continue
            out[k] = {"r": float(np.corrcoef(a, yy)[0, 1]), "coverage": float(v.mean())}
    return out


def relation_strength(rel: dict) -> float:
    """One number per label relation, 0 = none, 1 = perfect: |2 AUROC - 1| or |r|."""
    return abs(2.0 * rel["auroc"] - 1.0) if "auroc" in rel else abs(rel["r"])


def guess_name(rel: dict[str, dict], top: int = 3, min_coverage: float = 0.3) -> list[tuple[str, float, str]]:
    """The labels a feature relates to most: [(label, strength, sign "+" / "-")] - a starting point for a name, not
    a finding: check the top moments. Labels defined on fewer than ``min_coverage`` of the moments (``threat_t`` has
    a value only when there is a threat) are left out: a relation measured on a small slice of moments makes a
    misleading name."""
    scored = []
    for k, r in rel.items():
        if r.get("coverage", 1.0) < min_coverage:
            continue
        s = relation_strength(r)
        sign = "+" if (r.get("auroc", 0.5) >= 0.5 if "auroc" in r else r["r"] >= 0) else "-"
        scored.append((k, s, sign))
    return sorted(scored, key=lambda t: -t[1])[:top]


def top_moments(act: np.ndarray, k: int = 12, groups: np.ndarray | None = None, steps: np.ndarray | None = None,
                min_gap: int = 8) -> np.ndarray:
    """Indices of the k strongest activations, at most one per stretch of min_gap steps of the same game (so the top
    list is not one moment repeated)."""
    order = np.argsort(-np.asarray(act), kind="mergesort")
    picked: list[int] = []
    taken: dict = {}
    for i in order:
        if act[i] <= 0 or len(picked) >= k:
            break
        if groups is not None and steps is not None:
            g, s = groups[i], steps[i]
            if any(abs(s - t) < min_gap for t in taken.get(g, [])):
                continue
            taken.setdefault(g, []).append(s)
        picked.append(int(i))
    return np.asarray(picked, np.int64)


def _where(xy) -> str:
    x, y = xy
    side = "left" if x < -1200 else "right" if x > 1200 else "middle"
    third = "own third" if y < -1700 else "attacking third" if y > 1700 else "midfield"
    return f"{third}, {side}"


def describe(fr: Frames, me: int, t: int = 0) -> str:
    """One moment (row t of fr) in plain words, from car slot me's side (its team attacks +y)."""
    team = float(fr.car_team[t, me])
    s = float(team_sign(np.array([team]))[0])
    pos = canon(fr.car_pos[t], s)
    vel = canon(fr.car_vel[t], s)
    b = canon(fr.ball_pos[t], s) + 0.0                  # (+ 0.0: no "-0" in the text)
    bv = canon(fr.ball_vel[t], s) + 0.0
    pos = pos + 0.0
    mine = pos[me]

    def car_txt(j):
        sp = float(np.linalg.norm(vel[j]))
        where = "demolished" if fr.car_demoed[t, j] > 0.5 else (
            "in the air" if fr.car_on_ground[t, j] < 0.5 else ("on a wall" if pos[j, 2] > 150 else "on the ground"))
        return f"({pos[j, 0]:.0f}, {pos[j, 1]:.0f}, {pos[j, 2]:.0f}) {sp:.0f} uu/s {where}, {fr.car_boost[t, j]:.0f} boost"
    parts = [f"us {car_txt(me)} [{_where(mine[:2])}]"]
    for j in range(fr.n_cars):
        if j == me or fr.car_present[t, j] < 0.5:
            continue
        who = "mate" if fr.car_team[t, j] == team else "opp"
        d = float(np.linalg.norm(pos[j] - mine))
        parts.append(f"{who} {car_txt(j)}, {d:.0f} uu from us")
    speed = float(np.linalg.norm(bv))
    heading = ("toward their goal" if bv[1] > 300 else "toward our goal" if bv[1] < -300 else "across the field")
    moving = "still" if speed < 50 else f"moving {speed:.0f} uu/s {heading}"
    parts.append(f"ball ({b[0]:.0f}, {b[1]:.0f}, {b[2]:.0f}) {moving}, {np.linalg.norm(b - mine):.0f} uu from us")
    return "; ".join(parts)


# ------------------------------------------------------------------------- many sparse features at once
def sparse_columns(idx: np.ndarray, val: np.ndarray, m: int):
    """Top-k codes (idx, val [n, k]) as columns: (rows, values, starts) - feature j fires on rows[starts[j]:starts[j+1]]
    with values[...] (positive activations only)."""
    flat_f = idx.reshape(-1).astype(np.int64)
    flat_v = val.reshape(-1).astype(np.float32)
    flat_r = np.repeat(np.arange(len(idx), dtype=np.int64), idx.shape[1])
    keep = flat_v > 0
    f, r, v = flat_f[keep], flat_r[keep], flat_v[keep]
    order = np.argsort(f, kind="stable")
    f, r, v = f[order], r[order], v[order]
    return r, v, np.searchsorted(f, np.arange(m + 1))


def _avg_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    _, start, counts = np.unique(x[order], return_index=True, return_counts=True)
    r = np.empty(len(x))
    r[order] = np.repeat(start + (counts - 1) / 2.0 + 1.0, counts)
    return r


def sparse_relations(rows_nz: np.ndarray, vals_nz: np.ndarray, labels: dict[str, np.ndarray], kinds: dict[str, str],
                     n: int, min_pos: int = 30) -> dict[str, dict]:
    """label_relations for a feature that is 0 except on rows_nz (values vals_nz) of n rows - in time proportional to
    how often it fires. Same outputs: AUROC / base / P(label | on) / lift / recall, or Pearson r."""
    out = {}
    for k, y in labels.items():
        kind = kinds.get(k)
        valid = ~np.isnan(y)
        nv = int(valid.sum())
        if nv < 50:
            continue
        yn = y[rows_nz]
        okn = ~np.isnan(yn)
        vv, yy = vals_nz[okn], yn[okn]
        if kind == "bin":
            n1 = int((y[valid] > 0.5).sum())
            n0 = nv - n1
            if n1 < min_pos or n0 == 0:
                continue
            pos = yy > 0.5
            nz1, nz0 = int(pos.sum()), int((~pos).sum())
            z1, z0 = n1 - nz1, n0 - nz0
            u_nz = (_avg_ranks(vv)[pos].sum() - nz1 * (nz1 + 1) / 2.0) if len(vv) else 0.0
            u = u_nz + nz1 * z0 + 0.5 * z1 * z0
            base = n1 / nv
            p_on = nz1 / len(yy) if len(yy) else float("nan")
            out[k] = {"auroc": float(u / (n1 * n0)), "base": base, "p_given_on": float(p_on),
                      "lift": float(p_on / base) if len(yy) else float("nan"), "recall": nz1 / n1,
                      "coverage": nv / max(n, 1)}
        else:
            yv = y[valid].astype(np.float64)
            my, sy = yv.mean(), yv.std()
            v64 = vv.astype(np.float64)
            ma = v64.sum() / nv
            sa = np.sqrt(max((v64 ** 2).sum() / nv - ma * ma, 0.0))
            if sa == 0 or sy == 0:
                continue
            cov = (v64 * yy).sum() / nv - ma * my
            out[k] = {"r": float(cov / (sa * sy)), "coverage": nv / max(n, 1)}
    return out


# ------------------------------------------------------------------------- choosing a causal test's targets
def causal_targets(features: dict, label: str = "demo_chance", also: str = "demo_soon", min_freq: float = 5e-4,
                   min_lift: float = 5.0, min_auroc: float = 0.75, n_group: int = 3, n_controls: int = 3,
                   seed: int = 0) -> dict:
    """A causal test's targets, chosen by a rule fixed BEFORE any game is played with an edit (so the result cannot
    steer the choice).

    features: a dictionary report's {id: {"freq", "relations": {label: {"auroc", "lift", ...}}}} (label_relations per
    feature, plus how often it fires). Target = the feature that best detects `label` (held-out AUROC) among those
    firing on >= min_freq of moments with lift >= min_lift and AUROC >= min_auroc; group = the best n_group such.
    Without any: the cleanest features (best |2 AUROC - 1| on any yes / no label, lift >= min_lift, firing on >= 0.1%
    of moments), each feature counted once, at its best label. Controls = n_controls features firing about as often
    as the target (within 30%), outside the group, that detect neither `label` nor `also` (AUROC < 0.6), drawn with
    `seed`. Returns {"rule", "target", "group", "controls", "candidates" (the top 10), "arms"}: the arms to play -
    base, the target ablated, the group ablated, the target scaled 3x, the controls ablated."""
    feats = {int(j): f for j, f in features.items()}

    def score(f, lab):
        r = f["relations"].get(lab, {})
        return r.get("auroc", 0.5), r.get("lift", 0.0)

    cands = sorted(((score(f, label)[0], j, label) for j, f in feats.items()
                    if f["freq"] >= min_freq and score(f, label)[1] >= min_lift and score(f, label)[0] >= min_auroc),
                   reverse=True)
    kind = label
    if not cands:
        kind = "cleanest"
        best: dict[int, tuple[float, str]] = {}
        for j, f in feats.items():
            if f["freq"] < 1e-3:
                continue
            for lab, r in f["relations"].items():
                if "auroc" in r and r.get("lift", 0) >= min_lift:
                    s = abs(2 * r["auroc"] - 1)
                    if s > best.get(j, (-1.0, ""))[0]:
                        best[j] = (s, lab)
        cands = sorted(((s, j, lab) for j, (s, lab) in best.items()), reverse=True)
    target = cands[0][1] if cands else None
    group = [c[1] for c in cands[:n_group]]
    controls: list[int] = []
    if target is not None:
        ft = feats[target]["freq"]
        pool = sorted(j for j, f in feats.items() if 0.7 * ft <= f["freq"] <= 1.3 * ft and j not in group
                      and score(f, label)[0] < 0.6 and score(f, also)[0] < 0.6)
        if pool:
            controls = sorted(int(c) for c in np.random.default_rng(seed).choice(
                pool, size=min(n_controls, len(pool)), replace=False))
    arms = ["base"]
    if target is not None:
        arms += [f"ablate:{target}", "ablate:" + ",".join(map(str, group)), f"scale3:{target}"]
        if controls:
            arms.append("ablate:" + ",".join(map(str, controls)))
    return {"rule": kind, "target": target, "group": group, "controls": controls,
            "candidates": [{"feature": c[1], "label": c[2], "score": round(float(c[0]), 4),
                            "score_is": "|2 AUROC - 1|" if kind == "cleanest" else "AUROC"} for c in cands[:10]],
            "arms": arms}
