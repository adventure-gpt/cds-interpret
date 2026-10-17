"""What in the game drives a decision (or a critic's value): attribution to the ENTITIES of the state - the ball,
the bot's own car (and its boost), the opponents, the team-mates - rather than to single input numbers.

Why entities and not input columns: a bot's observation is full of derived numbers (the ball relative to the car,
the opponent's distance), so a column belongs to two things at once and removing one column leaves an impossible
input. Here each moment's game state is split into entities (`FramesCodec.decompose`), an entity is swapped for the
same entity of ANOTHER recorded moment (a "donor": a resample ablation, so the input stays a real state), and the
input is rebuilt with the bot's own obs function (`FramesCodec.compose` -> `Policy.observe`).

    codec = FramesCodec(policy)
    parts = codec.decompose(frames, me)                                   # the moments to explain AND the donors
    donors = pick_donors(role_signature(parts), 4, groups=game, pool=donor_rows)   # same roles, other games
    tab = coalition_table(policy.log_probs, parts, donors, codec.groups(), codec.compose)
    phi = shapley(-kl_table(tab)[..., None])[..., 0]                      # [N, G]: shares of the policy's change
    occ = occlusion(tab)                                                  # f(x) - f(x with g swapped)
    faithfulness(kl_table(tab), {"shapley": order_from(phi)}, average_donors=True)

Coalition S is a bit mask over the groups: bit g set = group g is the moment's own, clear = it comes from the donor.
The table holds every coalition, so exact Shapley values, single-group occlusion, and the best and the average
("random") deletion order all come from one set of forward passes: 2^G per moment and donor (G = 5: 32).

Gradient methods (integrated_gradients, grad_x_input) work on input columns; group_sum() adds them up by group so
their rankings can be scored against the coalition table's (faithfulness). Measured on a 31M-weight self-play bot:
exact entity Shapley ranks what a decision rests on about twice as well as integrated gradients and 2.5x as well as
gradient x input, against the best possible order (0.46 / 0.26 / 0.19; random 0).
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Group:
    """One entity (or part of one) of a decomposed state.
    key: the part (or parts, moved together) it lives in, e.g. "ball" or ("cars", "cars_ctl");
    role: for set-valued parts [N, S, F] (cars), only the slots whose role (parts[key0 + ":role"] [N, S]) is this;
    fields: only these columns of the last axis (e.g. a car's boost), the rest stays the moment's own."""
    key: str | tuple[str, ...]
    role: int | None = None
    fields: tuple[int, ...] | None = None

    @property
    def keys(self) -> tuple[str, ...]:
        return (self.key,) if isinstance(self.key, str) else tuple(self.key)


def _take(parts: dict, rows) -> dict:
    return {k: v[rows] for k, v in parts.items()}


def swap(parts: dict, donor: dict, group: Group) -> dict:
    """parts with `group` replaced by the donor's (both dicts of arrays / tensors with the same rows). For a role
    group, the moment's slots of that role get the donor's slots of that role, in slot order (as many as both have)."""
    out = dict(parts)
    if group.role is None:
        for k in group.keys:
            a, b = parts[k], donor[k]
            if group.fields is None:
                out[k] = b.clone() if torch.is_tensor(b) else np.array(b, copy=True)
            else:
                a = a.clone() if torch.is_tensor(a) else np.array(a, copy=True)
                f = list(group.fields)
                a[..., f] = b[..., f]
                out[k] = a
        return out
    rk = group.keys[0] + ":role"
    ra = np.asarray(parts[rk].cpu() if torch.is_tensor(parts[rk]) else parts[rk])
    rb = np.asarray(donor[rk].cpu() if torch.is_tensor(donor[rk]) else donor[rk])
    ma, mb = ra == group.role, rb == group.role
    oa = np.argsort(~ma, axis=1, kind="stable")              # the slots of the role first, in slot order
    ob = np.argsort(~mb, axis=1, kind="stable")
    n_ok = np.minimum(ma.sum(1), mb.sum(1))
    rows = np.arange(len(ra))
    for k in group.keys:
        a = parts[k].clone() if torch.is_tensor(parts[k]) else np.array(parts[k], copy=True)
        b = donor[k]
        for j in range(ra.shape[1]):
            sel = rows[n_ok > j]
            if not len(sel):
                break
            sa, sb = oa[sel, j], ob[sel, j]
            if torch.is_tensor(a):
                ti, ta, tb = (torch.as_tensor(v, device=a.device) for v in (sel, sa, sb))
                if group.fields is None:
                    a[ti, ta] = b[ti, tb]
                else:
                    for fi in group.fields:
                        a[ti, ta, fi] = b[ti, tb, fi]
            else:
                if group.fields is None:
                    a[sel, sa] = b[sel, sb]
                else:
                    for fi in group.fields:
                        a[sel, sa, fi] = b[sel, sb, fi]
        out[k] = a
    return out


def hybrid(parts: dict, donor: dict, groups: Sequence[Group], keep_mask: int) -> dict:
    """The moment with every group whose bit is CLEAR in keep_mask taken from the donor."""
    out = parts
    for g, grp in enumerate(groups):
        if not (keep_mask >> g) & 1:
            out = swap(out, donor, grp)
    return out


def pick_donors(signature: np.ndarray, n_donors: int = 4, groups: np.ndarray | None = None, seed: int = 0,
                pool: np.ndarray | None = None) -> np.ndarray:
    """[n_donors, N] row indices: for each row a donor with the same signature (e.g. mode * 16 + roles present, so
    role swaps line up) from a different group (game) when groups is given; donors are drawn from `pool` (default
    every row) - pass a pool to keep test moments' donors among test moments."""
    rng = np.random.default_rng(seed)
    sig = np.asarray(signature)
    n = len(sig)
    pool = np.arange(n) if pool is None else np.asarray(pool)
    out = np.empty((n_donors, n), np.int64)
    by = {}
    for s in np.unique(sig):
        by[s] = pool[sig[pool] == s]
    for i in range(n):
        cand = by.get(sig[i], np.zeros(0, np.int64))
        if groups is not None and len(cand):
            c2 = cand[groups[cand] != groups[i]]
            cand = c2 if len(c2) else cand
        if not len(cand):
            raise ValueError(f"no donor with signature {sig[i]} for row {i}")
        out[:, i] = rng.choice(cand, size=n_donors, replace=len(cand) < n_donors)
    return out


# ---------------------------------------------------------------------------------------------- the coalition table
@torch.no_grad()
def coalition_table(f: Callable, parts: dict, donor_rows: np.ndarray, groups: dict[str, Group],
                    compose: Callable[[dict], object], batch: int = 512, log=None) -> np.ndarray:
    """f(compose(hybrid)) for every coalition of the groups, every row and every donor: [n_donors, N, 2^G, K]
    (float32). parts: the decomposed rows (a dict of arrays / tensors, row-major); donor_rows [n_donors, N]: row
    indices INTO parts of each row's donors (pick_donors); f returns [n, K] (or [n]) for a batch of model inputs."""
    names = list(groups)
    grp = [groups[k] for k in names]
    G = len(grp)
    n_d, n = donor_rows.shape
    out: np.ndarray | None = None
    for d in range(n_d):
        for i in range(0, n, batch):
            rows = np.arange(i, min(i + batch, n))
            p = _take(parts, rows)
            q = _take(parts, donor_rows[d, rows])
            for s in range(2 ** G):
                y = f(compose(hybrid(p, q, grp, s)))
                y = (y.detach().float().cpu().numpy() if torch.is_tensor(y) else np.asarray(y, np.float32))
                y = y.reshape(len(rows), -1)
                if out is None:
                    out = np.zeros((n_d, n, 2 ** G, y.shape[1]), np.float32)
                out[d, rows, s] = y
        if log is not None:
            log(f"  coalition table: donor {d + 1}/{n_d}")
    if out is None:
        raise ValueError("coalition_table: no rows to explain")
    return out


def _popcount(G: int) -> np.ndarray:
    return np.array([bin(s).count("1") for s in range(2 ** G)])


def shapley(tab: np.ndarray, mean_donors: bool = True) -> np.ndarray:
    """Exact Shapley values from a coalition table [n_donors, N, 2^G, K] -> [N, G, K] (donor-averaged) or
    [n_donors, N, G, K]. Each row's values add up to f(x) - f(all swapped) (efficiency)."""
    nd, n, S, K = tab.shape
    G = int(round(math.log2(S)))
    pc = _popcount(G)
    w = np.array([math.factorial(int(c)) * math.factorial(G - int(c) - 1) / math.factorial(G) for c in range(G)])
    phi = np.zeros((nd, n, G, K), np.float64)
    for g in range(G):
        bit = 1 << g
        without = np.array([s for s in range(S) if not s & bit])
        phi[:, :, g] = ((tab[:, :, without | bit] - tab[:, :, without]) * w[pc[without]][None, None, :, None]).sum(2)
    return phi.mean(0) if mean_donors else phi


def occlusion(tab: np.ndarray, mean_donors: bool = True) -> np.ndarray:
    """f(x) - f(x with only group g from the donor): [N, G, K] (or per donor)."""
    S = tab.shape[2]
    G = int(round(math.log2(S)))
    full = S - 1
    occ = np.stack([tab[:, :, full] - tab[:, :, full & ~(1 << g)] for g in range(G)], 2)
    return occ.mean(0) if mean_donors else occ


def kl_table(logp_tab: np.ndarray) -> np.ndarray:
    """KL(policy at the moment || policy at each coalition) from a table of LOG-PROBABILITIES [nd, N, 2^G, A]:
    [nd, N, 2^G] (0 at the full coalition)."""
    full = logp_tab[:, :, -1:]
    return (np.exp(full) * (full - logp_tab)).sum(-1)


# ---------------------------------------------------------------------------------------------------- faithfulness
def deletion_curve(metric: np.ndarray, order: np.ndarray) -> np.ndarray:
    """metric [N, 2^G] (e.g. a kl_table for one donor: how far the output moved with each coalition kept), order
    [N, G] (the groups from most to least important, by some method) -> [N, G + 1]: the metric after removing the
    first k groups of the order (k = 0 .. G)."""
    n, S = metric.shape
    G = order.shape[1]
    keep = np.full(n, S - 1, np.int64)
    out = np.zeros((n, G + 1))
    rows = np.arange(n)
    out[:, 0] = metric[rows, keep]
    for k in range(G):
        keep = keep & ~(1 << order[:, k].astype(np.int64))
        out[:, k + 1] = metric[rows, keep]
    return out


def best_and_mean_deletion(metric: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[N, G + 1] each: the most the metric can grow with k groups removed (the oracle order's curve is at most this;
    it is the best subset of each size) and its average over subsets of size k (= a random order's expected curve)."""
    S = metric.shape[1]
    G = int(round(math.log2(S)))
    pc = _popcount(G)
    removed = G - pc                                     # groups removed in each coalition
    best = np.stack([metric[:, removed == k].max(1) for k in range(G + 1)], 1)
    mean = np.stack([metric[:, removed == k].mean(1) for k in range(G + 1)], 1)
    return best, mean


def best_order_auc(metric: np.ndarray, max_perms: int = 50000, chunk: int = 512) -> np.ndarray:
    """[N]: the largest deletion AUC any ORDER of the groups reaches on each row (every permutation tried when there
    are at most max_perms of them, i.e. G <= 8): the true ceiling for a ranking method. (The best subset of each size,
    best_and_mean_deletion, is a looser ceiling: no single order needs to reach it at every size.)"""
    import itertools
    n, S = metric.shape
    G = int(round(math.log2(S)))
    if math.factorial(G) > max_perms:
        raise ValueError(f"{math.factorial(G)} orders of {G} groups: too many")
    perms = np.array(list(itertools.permutations(range(G))), np.int64)          # [P, G]
    keep = np.full(len(perms), S - 1, np.int64)
    kept = []
    for k in range(G - 1):
        keep = keep & ~(1 << perms[:, k])
        kept.append(keep.copy())
    masks = np.stack(kept, 1)                                                   # [P, G - 1]: kept after k + 1 removed
    best = np.full(n, -np.inf)
    for i in range(0, len(perms), chunk):
        auc = metric[:, masks[i:i + chunk]].mean(-1)                             # [N, chunk]
        best = np.maximum(best, auc.max(1))
    return best


def faithfulness(metric: np.ndarray, orders: dict[str, np.ndarray], average_donors: bool = False) -> dict[str, dict]:
    """Score rankings of the groups against what removing them does. metric [N, 2^G] (or [nd, N, 2^G], one per
    donor), orders: name -> [N, G] (or [nd, N, G]). AUC = the mean of a deletion curve over k = 1 .. G - 1 removed
    (the two ends are the same for every order). Per name:
      score         (AUC - random) / (best order - random), averaged over rows: 1 = the best possible ranking, 0 =
                    chance, < 0 worse than chance (best order: best_order_auc, every permutation tried)
      score_subset  the same against the best SUBSET of each size (a stricter ceiling no order may reach)
      score_pooled  (mean AUC - mean random) / (mean best order - mean random): the ratio of the means, steadier
                    than the mean of per-row ratios when many rows have little to rank
      curve         the mean deletion curve (k = 0 .. G).
    average_donors: score against the donor-averaged metric (one order per row, judged on what removing the groups
    does on average over the donors) - the fair test for an attribution that is itself donor-averaged."""
    if average_donors and metric.ndim == 3:
        metric = metric.mean(0)
        orders = {k: (v[0] if v.ndim == 3 else v) for k, v in orders.items()}
    m = metric if metric.ndim == 3 else metric[None]
    refs = []
    for d in range(m.shape[0]):
        b, r = best_and_mean_deletion(m[d])
        G = b.shape[1] - 1
        refs.append((best_order_auc(m[d]), b[:, 1:G].mean(1), r[:, 1:G].mean(1)))
    out = {}
    for name, od in orders.items():
        o = od if od.ndim == 3 else np.broadcast_to(od, (m.shape[0],) + od.shape)
        curves: list[np.ndarray] = []
        scores: list[np.ndarray] = []
        scores_s: list[np.ndarray] = []
        pooled: list[tuple[float, float, float]] = []
        for d in range(m.shape[0]):
            c = deletion_curve(m[d], o[d])
            G = c.shape[1] - 1
            auc = c[:, 1:G].mean(1)
            auc_o, auc_b, auc_r = refs[d]
            pooled.append((float(auc.mean()), float(auc_o.mean()), float(auc_r.mean())))
            for ceil, dst in ((auc_o, scores), (auc_b, scores_s)):
                den = ceil - auc_r
                ok = den > 1e-9
                dst.append(np.where(ok, (auc - auc_r) / np.where(ok, den, 1.0), np.nan))
            curves.append(c)
        sc, ss = np.concatenate(scores), np.concatenate(scores_s)
        pa, po, pr = (float(np.mean([p[i] for p in pooled])) for i in range(3))
        out[name] = {"score": float(np.nanmean(sc)), "score_median": float(np.nanmedian(sc)),
                     "score_subset": float(np.nanmean(ss)), "score_pooled": (pa - pr) / max(po - pr, 1e-12),
                     "auc": pa, "auc_best_order": po, "auc_random": pr,
                     "curve": np.concatenate(curves).mean(0).tolist()}
    return out


def order_from(attr: np.ndarray, absolute: bool = True) -> np.ndarray:
    """[N, G] scores (one number per group, e.g. Shapley on one output) -> [N, G] group indices, most important first."""
    a = np.abs(attr) if absolute else attr
    return np.argsort(-a, axis=1, kind="stable")


# ------------------------------------------------------------------------------------------------ gradient methods
def integrated_gradients(f_scalar: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, baseline: torch.Tensor,
                         steps: int = 32) -> torch.Tensor:
    """Integrated gradients of f_scalar (inputs [n, D] -> [n]) from baseline to x, midpoint rule: [n, D]. The
    columns add up to about f(x) - f(baseline) (completeness; check it with more steps if not)."""
    total = torch.zeros_like(x, dtype=torch.float32)
    diff = (x - baseline).float()
    for k in range(steps):
        a = (k + 0.5) / steps
        z = (baseline.float() + a * diff).requires_grad_(True)
        with torch.enable_grad():
            y = f_scalar(z).sum()
            (g,) = torch.autograd.grad(y, z)
        total += g
    return total * diff / steps


def grad_x_input(f_scalar: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor,
                 baseline: torch.Tensor | None = None) -> torch.Tensor:
    """Gradient at x times (x - baseline): [n, D] (baseline 0 when None)."""
    z = x.float().clone().requires_grad_(True)
    with torch.enable_grad():
        (g,) = torch.autograd.grad(f_scalar(z).sum(), z)
    return g * (x.float() - (0.0 if baseline is None else baseline.float()))


def group_sum(attr: np.ndarray | torch.Tensor, col_masks: dict[str, np.ndarray]) -> np.ndarray:
    """Column attributions [N, D] summed by group: col_masks name -> [D] bool (a fixed set of columns) or [N, D]
    bool (per row, e.g. the slots that are opponents in this moment) -> [N, G] in the masks' order."""
    a = attr.detach().cpu().numpy() if torch.is_tensor(attr) else np.asarray(attr)
    out = []
    for m in col_masks.values():
        m = np.asarray(m, bool)
        out.append((a * (m[None] if m.ndim == 1 else m)).sum(1))
    return np.stack(out, 1)


# -------------------------------------------------------------------------------------------------------- outputs
def policy_logp(logits_fn: Callable) -> Callable:
    """f for coalition_table: the model input -> log-probabilities of every action [n, A]."""
    def f(inp):
        return F.log_softmax(logits_fn(inp).float(), -1)
    return f


def control_logodds(logp: np.ndarray, table: np.ndarray, control: int, threshold: float = 0.5) -> np.ndarray:
    """log P(control on) - log P(control off) under a policy of log-probabilities [..., A], the action table [A, C]
    saying which actions have that control on (> threshold): e.g. the log-odds of boosting."""
    on = np.asarray(table)[:, control] > threshold
    lp = np.asarray(logp, np.float64)
    m = lp.max(-1, keepdims=True)
    p = np.exp(lp - m)
    p_on = (p * on).sum(-1).clip(1e-12)
    p_off = (p * ~on).sum(-1).clip(1e-12)
    return np.log(p_on) - np.log(p_off)


def shares(phi: np.ndarray) -> np.ndarray:
    """Shapley values [N, G] -> each group's share of the total |phi| per row [N, G] (rows with nothing: 0)."""
    a = np.abs(phi)
    s = a.sum(1, keepdims=True)
    return np.where(s > 0, a / np.where(s > 0, s, 1.0), 0.0)


def rank_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Mean Spearman correlation between two [N, G] attributions, row by row (rows with ties throughout skipped)."""
    def ranks(x):
        return np.argsort(np.argsort(x, axis=1, kind="stable"), axis=1).astype(np.float64)
    ra, rb = ranks(a), ranks(b)
    ra -= ra.mean(1, keepdims=True)
    rb -= rb.mean(1, keepdims=True)
    den = np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    ok = den > 0
    return float(((ra * rb).sum(1)[ok] / den[ok]).mean()) if ok.any() else float("nan")


def all_coalitions(G: int) -> list[tuple[int, ...]]:
    """Each coalition's groups (bit g set = group g kept), in table order: for reading a table by hand."""
    return [tuple(g for g in range(G) if (s >> g) & 1) for s in range(2 ** G)]


# ------------------------------------------------------------------------------------- any bot: the Frames codec
_VEC = ("car_pos", "car_vel", "car_angvel", "car_fwd", "car_up")
_SCAL = ("car_boost", "car_on_ground", "car_demoed", "car_has_flip", "car_supersonic", "car_team", "car_present")
ROLE_ME, ROLE_MATE, ROLE_OPP = 3, 1, 2


class FramesCodec:
    """Moments' game states (Frames) as swappable entities, rebuilt into a policy's input with its own obs function
    (``Policy.observe``): the decompose / compose pair for :func:`coalition_table`, for ANY bot with an obs function.

    Every moment is first turned into its own car's team frame (blue, attacking +y), so the ball or an opponent of
    one moment can be put into another moment as that moment's car would see it; compose turns it back into the
    world frame of the receiving moment before calling the obs function.

    parts: "ball" [N, 9] (pos, vel, angvel); "cars" [N, C, F] (the five car vectors, then the car scalars, then the
    previous controls when the frames have them) with "cars:role" [N, C] (3 = the bot's car, 1 team-mate,
    2 opponent, 0 absent); "sign" [N] and "me" [N] are never swapped, nor are the frames' other fields.

        codec = FramesCodec(policy)
        parts = codec.decompose(moments.frames.take(rows), moments.me[rows])
        groups = codec.groups()       # own car, own boost, ball, opponents, team-mates
        tab = coalition_table(policy.log_probs, parts, donor_rows, groups, codec.compose)
    """

    def __init__(self, policy: Any):
        self.policy = policy

    def decompose(self, frames: Any, me: Any) -> dict:
        from .frames import canon, team_sign
        me = np.broadcast_to(np.asarray(me, np.int64), (len(frames),))
        n, c = len(frames), frames.n_cars
        rows = np.arange(n)
        team_me = np.asarray(frames.car_team, np.float64)[rows, me]
        s = team_sign(team_me)
        ball = np.concatenate([canon(frames.data[k], s) for k in ("ball_pos", "ball_vel", "ball_angvel")], 1)
        cols = [canon(frames.data[k], s[:, None]) for k in _VEC]
        team = np.asarray(frames.car_team, np.float64)
        present = np.asarray(frames.car_present, np.float64) > 0.5
        same = np.abs(team - team_me[:, None]) < 0.5
        scal = []
        for k in _SCAL:
            v = np.asarray(frames.data[k], np.float64)
            if k == "car_team":
                v = np.where(same, 0.0, 1.0)                          # 0 = the bot's side, 1 = the other
            scal.append(v[..., None])
        cars = np.concatenate(cols + scal + ([np.asarray(frames.prev_controls, np.float64)]
                                             if frames.has("prev_controls") else []), -1)
        role = np.where(~present, 0, np.where(same, ROLE_MATE, ROLE_OPP))
        role[rows, me] = ROLE_ME
        parts = {"ball": ball.astype(np.float32), "cars": cars.astype(np.float32), "cars:role": role.astype(np.int8),
                 "sign": s.astype(np.float32), "me": me.copy()}
        for k, v in frames.data.items():
            if k not in _VEC + _SCAL + ("ball_pos", "ball_vel", "ball_angvel", "prev_controls"):
                parts["rest:" + k] = v
        self._has_prev = frames.has("prev_controls")
        self._c = c
        return parts

    def frames(self, parts: dict) -> Any:
        """The parts as world-frame Frames (each moment in its own bot car's world frame)."""
        from .frames import Frames, canon
        s = np.asarray(parts["sign"], np.float64)
        b = np.asarray(parts["ball"], np.float64)
        d = {k: canon(b[:, 3 * i:3 * i + 3], s) for i, k in enumerate(("ball_pos", "ball_vel", "ball_angvel"))}
        cars = np.asarray(parts["cars"], np.float64)
        for i, k in enumerate(_VEC):
            d[k] = canon(cars[..., 3 * i:3 * i + 3], s[:, None])
        o = 3 * len(_VEC)
        for i, k in enumerate(_SCAL):
            d[k] = cars[..., o + i]
        orange_me = s < 0
        d["car_team"] = np.where(orange_me[:, None], 1.0 - d["car_team"], d["car_team"])
        if cars.shape[-1] > o + len(_SCAL):
            d["prev_controls"] = cars[..., o + len(_SCAL):]
        for k, v in parts.items():
            if k.startswith("rest:"):
                d[k[5:]] = v
        return Frames({k: np.asarray(v, np.float32) if np.asarray(v).dtype.kind == "f" else np.asarray(v)
                       for k, v in d.items()})

    def compose(self, parts: dict) -> Any:
        return self.policy.observe(self.frames(parts), parts["me"])

    def groups(self, boost_apart: bool = True) -> dict[str, Group]:
        """own car (its boost apart), the ball, the opponents, the team-mates. Moments without team-mates leave that
        group's swaps empty (its share is 0)."""
        boost = 3 * len(_VEC) + _SCAL.index("car_boost")
        width = 3 * len(_VEC) + len(_SCAL) + (8 if getattr(self, "_has_prev", False) else 0)
        g: dict[str, Group] = {}
        if boost_apart:
            g["own car"] = Group("cars", role=ROLE_ME, fields=tuple(i for i in range(width) if i != boost))
            g["own boost"] = Group("cars", role=ROLE_ME, fields=(boost,))
        else:
            g["own car"] = Group("cars", role=ROLE_ME)
        g["ball"] = Group("ball")
        g["opponents"] = Group("cars", role=ROLE_OPP)
        g["team-mates"] = Group("cars", role=ROLE_MATE)
        return g


def role_signature(parts: dict) -> np.ndarray:
    """[N] a code of how many team-mates and opponents each moment has: donors with the same code line up."""
    r = np.asarray(parts["cars:role"])
    return (r == ROLE_MATE).sum(1) * 8 + (r == ROLE_OPP).sum(1)


__all__ = [
    "ROLE_MATE",
    "ROLE_ME",
    "ROLE_OPP",
    "FramesCodec",
    "Group",
    "all_coalitions",
    "best_and_mean_deletion",
    "best_order_auc",
    "coalition_table",
    "control_logodds",
    "deletion_curve",
    "faithfulness",
    "grad_x_input",
    "group_sum",
    "hybrid",
    "integrated_gradients",
    "kl_table",
    "occlusion",
    "order_from",
    "pick_donors",
    "policy_logp",
    "rank_agreement",
    "role_signature",
    "shapley",
    "shares",
    "swap",
]
