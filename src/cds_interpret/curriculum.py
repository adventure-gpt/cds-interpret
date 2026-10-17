"""Interpretability for choosing what to practise: where a policy wins and loses, by situation; what comes before the
goals it concedes; where its critic is surprised and what explains it.

  next_goal_within(goal, team, steps, horizon)     did the side score / concede within `horizon` steps of each step
  pair_situations(fr, me)                          situations that need the other cars (an opponent dribbling at us ...)
  situation_outcomes(masks, for_, against, games)  per situation: how often, P(score soon), P(concede soon), the net,
                                                   with intervals over games
  enrichment(target, columns, base)                which labels / features are over-represented in a set of moments
  eta_squared(y, codes)                            the share of a quantity's variance a grouping explains
  kmeans(x, k)                                     a small k-means (numpy) for grouping failure moments

Frames and concept labels only (cds_interpret.frames, cds_interpret.labels): any bot, any recording.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .frames import Frames


def next_goal_within(goal: np.ndarray, team: int | np.ndarray, steps: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """For decision steps `steps` of one game (goal [T]: -1 or the scoring team at each step): whether the next goal
    comes within `horizon` steps and is `team`'s (for) or the other team's (against)."""
    goal = np.asarray(goal)
    steps = np.asarray(steps, np.int64)
    team = np.broadcast_to(np.asarray(team), steps.shape)
    gt = np.nonzero(goal >= 0)[0]
    f = np.zeros(len(steps), bool)
    a = np.zeros(len(steps), bool)
    if not len(gt):
        return f, a
    j = np.searchsorted(gt, steps)
    has = j < len(gt)
    t_next = gt[np.minimum(j, len(gt) - 1)]
    soon = has & (t_next - steps <= horizon)
    who = goal[t_next]
    f = soon & (who == team)
    a = soon & (who != team)
    return f, a


def pair_situations(fr: Frames, me: np.ndarray) -> dict[str, np.ndarray]:
    """Situations of car `me[i]` at moment i that depend on where the OTHER cars are (world frame in, team frame used):

      facing_carry   an opponent within 250 uu of a low ball (under 250 uu) that moves toward our goal at 300+ uu/s:
                     being dribbled at
      we_carry       we are within 250 uu of a low ball, no opponent within 1000 uu of it
      opp_aerial     an opponent airborne within 400 uu of a ball above 300 uu
      opp_closer     the nearest opponent is nearer the ball than we are by 1000+ uu, the ball in our half
      outnumbered    team modes: more opponents than team-mates within 2500 uu of the ball (us included)"""
    n = len(fr)
    me = np.asarray(me, np.int64)
    rows = np.arange(n)
    team = np.round(fr.car_team[rows, me]).astype(int)
    sign = np.where(team == 0, 1.0, -1.0)
    present = (fr.car_present > 0.5) & (fr.car_demoed < 0.5)
    opp = present & (np.round(fr.car_team) != team[:, None])
    mate = present & (np.round(fr.car_team) == team[:, None])
    d = np.linalg.norm(fr.car_pos - fr.ball_pos[:, None, :], axis=2)
    d_opp = np.where(opp, d, np.inf).min(1)
    d_me = d[rows, me]
    bz = fr.ball_pos[:, 2]
    bvy = fr.ball_vel[:, 1] * sign
    by = fr.ball_pos[:, 1] * sign
    opp_air = opp & (fr.car_on_ground < 0.5) & (d < 400.0)
    near = d < 2500.0
    return {
        "facing_carry": (d_opp < 250.0) & (bz < 250.0) & (bvy < -300.0),
        "we_carry": (d_me < 250.0) & (bz < 250.0) & (d_opp > 1000.0),
        "opp_aerial": opp_air.any(1) & (bz > 300.0),
        "opp_closer": (d_opp + 1000.0 < d_me) & (by < 0.0),
        "outnumbered": ((opp & near).sum(1) > (mate & near).sum(1)) & (present.sum(1) >= 4),
    }


def situation_outcomes(masks: dict[str, np.ndarray], for_: np.ndarray, against: np.ndarray, games: np.ndarray,
                       n_boot: int = 500, seed: int = 0) -> dict[str, dict]:
    """Per situation: moments (n), share of all moments, P(score within the horizon), P(concede), net = their
    difference, and a 95% interval of the net from resampling games."""
    games = np.asarray(games)
    ug, gi = np.unique(games, return_inverse=True)
    rng = np.random.default_rng(seed)
    boots = rng.integers(0, len(ug), (n_boot, len(ug)))
    f, a = np.asarray(for_, float), np.asarray(against, float)
    out: dict[str, dict[str, Any]] = {}
    for name, m in masks.items():
        m = np.asarray(m, bool)
        n = int(m.sum())
        if n == 0:
            out[name] = {"n": 0}
            continue
        nf = np.bincount(gi, weights=m * f, minlength=len(ug))
        na = np.bincount(gi, weights=m * a, minlength=len(ug))
        nn = np.bincount(gi, weights=m.astype(float), minlength=len(ug))
        bf = np.array([nf[b].sum() for b in boots])
        ba = np.array([na[b].sum() for b in boots])
        bn = np.array([nn[b].sum() for b in boots])
        net_b = (bf - ba) / np.maximum(bn, 1)
        out[name] = {"n": n, "share": n / len(m), "p_for": float(f[m].mean()), "p_against": float(a[m].mean()),
                     "net": float((f[m] - a[m]).mean()), "net_lo": float(np.percentile(net_b, 2.5)),
                     "net_hi": float(np.percentile(net_b, 97.5)), "games": int((nn > 0).sum())}
    return out


def enrichment(target: np.ndarray, columns: dict[str, np.ndarray], base: np.ndarray | None = None) -> dict[str, dict]:
    """For each column (a label, a feature's activation): its mean over the target moments and over the base moments
    (all others by default), their difference in units of the base's standard deviation, and for 0/1 columns the lift
    (the target's rate over the base's)."""
    t = np.asarray(target, bool)
    b = ~t if base is None else np.asarray(base, bool)
    out = {}
    for k, v in columns.items():
        v = np.asarray(v, float)
        vt, vb = v[t], v[b]
        vt, vb = vt[np.isfinite(vt)], vb[np.isfinite(vb)]
        if len(vt) < 5 or len(vb) < 5:
            continue
        sd = vb.std() if vb.std() > 0 else 1.0
        r = {"target": float(vt.mean()), "base": float(vb.mean()), "z": float((vt.mean() - vb.mean()) / sd)}
        if set(np.unique(vb)) <= {0.0, 1.0} and vb.mean() > 0:
            r["lift"] = float(vt.mean() / vb.mean())
        out[k] = r
    return out


def eta_squared(y: np.ndarray, codes: np.ndarray, min_count: int = 1) -> float:
    """Between-group share of the variance of y for categorical codes (groups smaller than min_count pooled)."""
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    y, codes = y[ok], np.asarray(codes)[ok]
    u, inv, cnt = np.unique(codes, return_inverse=True, return_counts=True)
    small = cnt[inv] < min_count
    inv = np.where(small, len(u), inv)
    s = np.bincount(inv, weights=y)
    c = np.bincount(inv).astype(float)
    means = s / np.maximum(c, 1)
    ss_between = float((c * (means - y.mean()) ** 2).sum())
    ss_total = float(((y - y.mean()) ** 2).sum())
    return ss_between / max(ss_total, 1e-12)


def kmeans(x: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(labels [N], centres [k, D]) of a plain k-means with k-means++ starts."""
    x = np.asarray(x, np.float64)
    rng = np.random.default_rng(seed)
    starts = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d2 = np.min([((x - cc) ** 2).sum(1) for cc in starts], axis=0)
        starts.append(x[rng.choice(len(x), p=d2 / d2.sum())])
    c = np.array(starts)
    lab = np.zeros(len(x), int)
    for _ in range(iters):
        d2 = ((x[:, None, :] - c[None]) ** 2).sum(2)
        new = d2.argmin(1)
        if (new == lab).all() and _ > 0:
            break
        lab = new
        for j in range(k):
            if (lab == j).any():
                c[j] = x[lab == j].mean(0)
    return lab, c


def car_situation_rows(fr: Frames, car: int, every: int = 3, horizon: int = 75) -> dict:
    """One car of one recorded game (frames in order, with goals): on every `every`-th step, the situations of
    cds_interpret.emergence.SITUATIONS (from state-only labels) and pair_situations, plus "all", and whether its team scores /
    concedes within `horizon` steps. Returns {"masks": {name: [n]}, "for": [n], "against": [n], "steps": [n]}."""
    from .emergence import situations
    from .labels import concept_labels
    ts = np.arange(0, len(fr), every)
    sub = fr.take(ts)
    me = np.full(len(ts), car)
    masks = situations(concept_labels(sub, me))
    masks.update(pair_situations(sub, me))
    masks["all"] = np.ones(len(ts), bool)
    team = int(round(float(fr.car_team[0, car])))
    f, a = next_goal_within(fr.goal, team, ts, horizon)
    return {"masks": masks, "for": f, "against": a, "steps": ts}


def situation_map(rows: list[dict], games: list[int], n_boot: int = 300) -> dict:
    """situation_outcomes over car_situation_rows of several games (one entry of `games` per entry of `rows`)."""
    if not rows:
        return {}
    masks = {k: np.concatenate([r["masks"][k] for r in rows]) for k in rows[0]["masks"]}
    g = np.concatenate([np.full(len(r["for"]), gi) for r, gi in zip(rows, games)])
    return situation_outcomes(masks, np.concatenate([r["for"] for r in rows]), np.concatenate([r["against"] for r in rows]),
                              g, n_boot=n_boot)
