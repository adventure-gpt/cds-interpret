"""Linear probes: is a concept linearly readable from a layer, and how well, on games the probe never saw?

    train, test = group_split(game_ids, test_frac=0.2)           # whole games on one side
    res = ridge_suite(X, labels, kinds, train, test, groups=game_ids)    # every label at once; X [N, D]
    res["demo_chance"]["auroc"], res["land_t"]["r2"]               # held-out scores
    res["land_t"]["wnorm"], res["land_t"]["noise"]                 # how steady the readout is (for a live display)
    res = probe_layer(X, labels, kinds, train, test)               # one fit per label; logistic for yes / no labels

- :func:`ridge_suite` (the sweep): ridge regression for every label - for a yes / no label the linear-probability
  model, a least-squares (LDA-like) direction, scored by AUROC, which needs no calibration. One eigendecomposition
  of the layer's covariance serves all labels and penalties, so a sweep over every layer is cheap. The penalty is
  picked per label on a validation part of the training games.
- :func:`probe_layer` / :func:`fit_probe` (confirmations): ridge for number labels, L2 logistic regression (L-BFGS,
  full batch, class-balanced weights) for yes / no ones.
- The scale the penalty acts in (``scale=``): ``"global"`` (the default) centres every unit and divides them all by
  ONE number, the root of the layer's mean variance per unit, so the penalty weighs every unit's raw weight alike
  and a near-silent unit cannot earn a giant weight; ``"unit"`` divides each unit by its own standard deviation (the
  textbook standardisation, opt-in), which lets a unit that barely moves get a weight in the thousands - harmless
  offline, but in a live display tiny shifts in such units throw the readout far off (docs/howto/probes.md).
  Either way the fitted probe is returned in the layer's own units.
- Held-out metrics: R² (and Pearson r) for number labels; AUROC and average precision (against the base rate) for
  yes / no ones. Beside them, how steady each readout is (:func:`readout_stability`): its raw weight norm, how far
  it moves when every unit gets a little noise (N(0, noise_sd²)), and - given sequences (``seq=``, e.g. game * 64 +
  car, and ``order=``, e.g. the step) - how far it moves between consecutive moments, against how far the label does.
- Splits are by GAME: moments of one game are strongly correlated, and a split by moment would leak.

Baselines are the same probes on other features: the raw input, a random network of the same architecture, and the
constant predictor (R² 0, AUROC 0.5, AP = the base rate).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .errors import CDSInterpretError

LAMBDAS_REG = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
LAMBDAS_BIN = (1e-4, 1e-3, 1e-2)
SCALES = ("global", "unit")             # the penalty's scale: one number for the layer, or each unit's own sd
NOISE_SD = 0.02                         # the per-unit noise of the steadiness check (the layer's own units)


def group_split(groups: np.ndarray, test_frac: float = 0.2, seed: int = 0,
                strata: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Row indices (train, test) with whole groups (games) on one side; with strata (a per-row key, constant within a
    group) each stratum's groups are split in the same proportion."""
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    if len(uniq) < 2:
        raise CDSInterpretError(
            f"These moments come from {len(uniq)} game, but the toolkit needs at least 2: it learns on some games and "
            "scores on others. Record more - 10 or more gives numbers you can trust."
        )
    if strata is None:
        key = {g: 0 for g in uniq}
    else:
        strata = np.asarray(strata)
        first = {g: i for i, g in reversed(list(enumerate(groups)))}
        key = {g: strata[first[g]] for g in uniq}
    test_groups = []
    for s in sorted(set(key.values()), key=str):
        gs = np.array([g for g in uniq if key[g] == s])
        rng.shuffle(gs)
        k = max(1, int(round(test_frac * len(gs)))) if len(gs) > 1 else 0
        test_groups += list(gs[:k])
    is_test = np.isin(groups, np.asarray(test_groups))
    return np.nonzero(~is_test)[0], np.nonzero(is_test)[0]


MIN_TEST_POS = 10                       # a yes / no label needs this many "yes" and "no" moments in the held-out games
MIN_TEST_SPREAD = 0.25                  # a number label's held-out spread, as a share of its training spread


def _enough(y_train: np.ndarray, y_test: np.ndarray, kind: str, min_pos: int) -> bool:
    """Is there enough of a label on both sides to fit a probe and score it honestly?"""
    if kind == "bin":
        return bool((y_train > 0.5).sum() >= min_pos and (y_train <= 0.5).sum() >= min_pos
                    and (y_test > 0.5).sum() >= MIN_TEST_POS and (y_test <= 0.5).sum() >= MIN_TEST_POS)
    sd_tr, sd_te = float(np.std(y_train)), float(np.std(y_test))
    return sd_tr > 0 and sd_te >= MIN_TEST_SPREAD * sd_tr


def why_not_probed(y: np.ndarray, kind: str, train: np.ndarray, test: np.ndarray, min_pos: int = 20) -> str | None:
    """Why a label gets no probe on these rows, in words (None when it can have one)."""
    y = np.asarray(y, np.float64)
    tr, te = train[~np.isnan(y[train])], test[~np.isnan(y[test])]
    if len(tr) < 50 or len(te) < 20:
        return f"too few moments where it has a value ({len(tr)} in training games, {len(te)} held out)"
    if kind == "bin":
        pt, pe = int((y[tr] > 0.5).sum()), int((y[te] > 0.5).sum())
        nt, ne = len(tr) - pt, len(te) - pe
        if nt == 0 and ne == 0:
            return "always true here"
        if pt == 0 and pe == 0:
            return "never true here"
        if pt < min_pos or nt < min_pos:
            return f"{min(pt, nt)} {'yes' if pt < nt else 'no'} moments in the training games (needs {min_pos})"
        if pe < MIN_TEST_POS or ne < MIN_TEST_POS:
            return f"{min(pe, ne)} {'yes' if pe < ne else 'no'} moments in the held-out games (needs {MIN_TEST_POS})"
        return None
    sd_tr, sd_te = float(np.std(y[tr])), float(np.std(y[te]))
    if sd_tr == 0:
        return "the same value everywhere"
    if sd_te < MIN_TEST_SPREAD * sd_tr:
        return "it hardly varies in the held-out games, so a score would mean little"
    return None


def _tensor(x) -> torch.Tensor:
    return x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x, dtype=np.float32))


# ------------------------------------------------------------------------------------------------- metrics
def auroc(score: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y) > 0.5
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    s = np.asarray(score, np.float64)
    order = np.argsort(s, kind="mergesort")
    _, start, counts = np.unique(s[order], return_index=True, return_counts=True)
    r = np.empty(len(s))
    r[order] = np.repeat(start + (counts - 1) / 2.0 + 1.0, counts)            # average ranks of ties
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def average_precision(score: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y) > 0.5
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-np.asarray(score), kind="mergesort")
    yy = y[order]
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / yy.sum())


def r2(pred: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y, np.float64)
    ss = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ((y - pred) ** 2).sum() / ss) if ss > 0 else float("nan")


# --------------------------------------------------------------------------------------------------- the scale
def _divisors(var: torch.Tensor, scale: str = "global") -> torch.Tensor:
    """What each unit is divided by before a probe's penalty applies, from the units' variances [D]: "global" one
    number for all of them (the root of the mean variance per unit; 1 for a layer that never moves), "unit" each
    unit's own standard deviation (1 where it is under 1e-6)."""
    var = var.clamp_min(0)
    if scale == "unit":
        sd = var.sqrt()
        return torch.where(sd < 1e-6, torch.ones_like(sd), sd)
    if scale == "global":
        g = float(var.mean().sqrt()) if len(var) else 1.0
        return torch.full_like(var, g if g > 1e-12 else 1.0)
    raise ValueError(f"scale must be one of {SCALES}, not {scale!r}")


@dataclass
class Standardizer:
    """Every unit centred and divided by its divisor: one number for the whole layer (scale="global", the default) or
    each unit's own standard deviation (scale="unit")."""
    mean: torch.Tensor
    std: torch.Tensor

    @staticmethod
    def fit(x: torch.Tensor, scale: str = "global") -> Standardizer:
        m = x.double().mean(0)
        v = x.double().var(0) if len(x) > 1 else torch.zeros_like(m)
        return Standardizer(m.float(), _divisors(v, scale).float())

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


# ------------------------------------------------------------------------------------------- how steady a readout is
def _moves(v: np.ndarray, o: np.ndarray, same: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(v[o], axis=0), axis=1)[same]


def _steadiness(score, wnorms, noise_sd: float, rows: np.ndarray, seq=None, order=None, y=None,
                miss: bool = False) -> dict:
    """readout_stability's numbers from the readout's values on `rows` (score [n] or [n, k]), its weight norms (one a
    column) and the label on those rows (y [n] or [n, k], or None)."""
    sc = np.asarray(score, np.float64).reshape(len(score), -1)
    wn = [float(v) for v in wnorms]
    noise = float(noise_sd) * float(np.sqrt(sum(v * v for v in wn)))
    spread = float(np.sqrt(((sc - sc.mean(0)) ** 2).sum(1).mean())) if len(sc) else float("nan")
    out: dict = {"wnorm": wn[0] if len(wn) == 1 else wn, "noise": noise,
                 "noise_rel": noise / spread if spread > 0 else float("nan")}
    yy = None if y is None else np.asarray(y, np.float64).reshape(len(sc), -1)
    if miss and yy is not None and len(sc):
        d = np.linalg.norm(sc - yy, axis=1)
        out["miss_med"], out["miss_p90"] = float(np.median(d)), float(np.percentile(d, 90))
    if seq is not None and len(sc) > 1:
        s = np.asarray(seq)[rows]
        t = np.asarray(order)[rows] if order is not None else np.asarray(rows)
        o = np.lexsort((t, s))
        same = s[o][1:] == s[o][:-1]
        if same.any():
            out["jitter"] = float(np.median(_moves(sc, o, same)))
            if yy is not None:
                out["jitter_label"] = float(np.median(_moves(yy, o, same)))
                out["jitter_err"] = float(np.median(_moves(sc - yy, o, same)))
    return out


def readout_stability(w, b, x: torch.Tensor, rows: np.ndarray | None = None, y: np.ndarray | None = None,
                      seq: np.ndarray | None = None, order: np.ndarray | None = None,
                      noise_sd: float = NOISE_SD, chunk: int = 16384) -> dict:
    """How steady a linear readout x @ w + b of a layer is: what decides whether it can be shown LIVE, beside its
    held-out score. w [D] (one readout) or [D, k] (k read together, e.g. a position x, y: moves are then distances), b
    a number or [k]; x [N, D] the layer's own values; rows: the moments to use (default all); y: the label, [N] or
    [N, k] like the readout (optional).
      wnorm         |w| (a list, one a column, for k > 1): a norm far above what the layer's values would need says
                    the readout leans on units that barely move
      noise         how far the readout moves when every unit gets independent N(0, noise_sd^2) noise, in the
                    readout's units: noise_sd * |w| (the root-mean-square distance for k > 1)
      noise_rel     that, over the readout's own spread across the moments (its standard deviation)
      miss_med, miss_p90   (with y) the median and 90th-percentile distance from the label
    With seq (a per-row sequence key, e.g. game * 64 + car) the rows of each sequence are taken in `order` (a per-row
    time such as the step; default the row order):
      jitter        the median move of the readout between consecutive rows
      jitter_label  (with y) the same for the label: a steady readout moves about as much as what it reads
      jitter_err    (with y) the median change of its error between consecutive rows
    Use every moment of the games for jitter: a random sample of moments has few consecutive ones."""
    x = _tensor(x)
    wt = w.detach().cpu().double().numpy() if torch.is_tensor(w) else np.asarray(w, np.float64)
    W = wt.reshape(wt.shape[0], -1)
    bb = np.asarray(b.detach().cpu() if torch.is_tensor(b) else b, np.float64).reshape(-1)
    rows = np.arange(len(x)) if rows is None else np.asarray(rows)
    parts = [x[torch.as_tensor(rows[i:i + chunk])].double().numpy() @ W + bb for i in range(0, len(rows), chunk)]
    sc = np.concatenate(parts) if parts else np.zeros((0, W.shape[1]))
    yy = None if y is None else np.asarray(y, np.float64)[rows]
    return _steadiness(sc, np.linalg.norm(W, axis=0), noise_sd, rows, seq, order, yy, miss=yy is not None)


# ------------------------------------------------------------------------------------------------ the probes
@dataclass
class LinearProbe:
    """A fitted probe in the layer's own coordinates: score = x @ w + b. For a yes / no label the score is a
    least-squares readout of 0 / 1 (link "identity", ridge_suite) or a logit (link "logit", probe_layer's logistic
    fit)."""
    w: torch.Tensor                     # [D]
    b: float
    kind: str                           # "reg" | "bin"
    lam: float
    link: str = "identity"              # "identity" | "logit"

    def score(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() @ self.w + self.b

    def direction(self) -> torch.Tensor:
        """The probe's unit direction in the layer (for edits: intervene.scale_direction, steer, project_out)."""
        return self.w / self.w.norm().clamp_min(1e-12)

    def to_dict(self) -> dict:
        return {"w": self.w.detach().cpu().float().numpy().tolist(), "b": float(self.b), "kind": self.kind,
                "lam": float(self.lam), "link": self.link}

    @staticmethod
    def from_dict(d: dict) -> LinearProbe:
        return LinearProbe(torch.as_tensor(np.asarray(d["w"], np.float32)), float(d["b"]), str(d["kind"]),
                           float(d.get("lam", 0.0)), str(d.get("link", "identity")))


def _ridge_path(xs: torch.Tensor, y: torch.Tensor, lams) -> list[torch.Tensor]:
    """Ridge weights for each penalty (scaled xs [n, d], centred y [n, k]); penalty scaled by n."""
    n = xs.shape[0]
    xtx = (xs.T.double() @ xs.double()) / n
    xty = (xs.T.double() @ y.double()) / n
    evals, evecs = torch.linalg.eigh(xtx)
    proj = evecs.T @ xty
    return [(evecs @ (proj / (evals[:, None] + lam))).float() for lam in lams]


def _fit_logistic(xs: torch.Tensor, y: torch.Tensor, lam: float, iters: int = 150, w0=None) -> tuple[torch.Tensor, torch.Tensor]:
    """L2 logistic regression on scaled xs [n, d] and 0/1 y [n]; classes weighted to equal total weight."""
    n, d = xs.shape
    pos = float(y.sum())
    wpos = 0.5 * n / max(pos, 1.0)
    wneg = 0.5 * n / max(n - pos, 1.0)
    sw = torch.where(y > 0.5, wpos, wneg)
    w = torch.zeros(d, requires_grad=True) if w0 is None else w0[0].clone().requires_grad_(True)
    b = torch.zeros(1, requires_grad=True) if w0 is None else w0[1].clone().requires_grad_(True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-6, tolerance_change=1e-9)

    def closure():
        opt.zero_grad()
        z = xs @ w + b
        loss = (sw * torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none")).mean() + 0.5 * lam * (w * w).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach(), b.detach()


def _val_split(n: int, groups: np.ndarray | None, seed: int = 1):
    if groups is None or len(np.unique(groups)) < 2:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(n)
        k = n // 5
        return idx[k:], idx[:k]
    return group_split(groups, 0.2, seed)


def fit_probe(x: torch.Tensor, y: np.ndarray, kind: str, groups: np.ndarray | None = None,
              lams=None, scale: str = "global") -> LinearProbe:
    """One probe on training rows x [n, D] (NaN labels already removed), penalty picked on a group-wise validation
    part, then refit on all of them. scale: the scale the penalty acts in - "global" (one number for the layer: a
    near-silent unit cannot earn a giant weight) or "unit" (each unit's own sd; see the module's notes)."""
    y_t = torch.as_tensor(np.asarray(y, np.float32))
    st = Standardizer.fit(x, scale)
    xs = st(x.float())
    tr, va = _val_split(len(x), groups)
    if kind == "reg":
        lams = lams or LAMBDAS_REG
        mu = y_t[tr].mean()
        ws = _ridge_path(xs[tr], (y_t[tr] - mu)[:, None], lams)
        scores = [r2((xs[va] @ w[:, 0] + mu).numpy(), y_t[va].numpy()) for w in ws]
        lam = lams[int(np.nanargmax(scores))]
        mu = y_t.mean()
        w = _ridge_path(xs, (y_t - mu)[:, None], [lam])[0][:, 0]
        b = float(mu)
    else:
        lams = lams or LAMBDAS_BIN
        yb = (y_t > 0.5).float()
        if yb[tr].sum() < 2 or yb[va].sum() < 1:
            lam = lams[len(lams) // 2]
        else:
            scores = []
            for lam_ in lams:
                w_, b_ = _fit_logistic(xs[tr], yb[tr], lam_, iters=80)
                scores.append(average_precision((xs[va] @ w_ + b_).numpy(), yb[va].numpy()))
            lam = lams[int(np.nanargmax(scores))]
        w, bb = _fit_logistic(xs, yb, lam)
        b = float(bb)
    # back into the layer's own coordinates: score = ((x - m) / s) @ w + b
    w_raw = w / st.std
    b_raw = b - float((st.mean / st.std) @ w)
    return LinearProbe(w_raw, b_raw, kind, float(lam), "logit" if kind == "bin" else "identity")


def _metrics(s: np.ndarray, y: np.ndarray, kind: str) -> dict:
    y = np.asarray(y, np.float64)
    if kind == "reg":
        r = float(np.corrcoef(s, y)[0, 1]) if np.std(s) > 0 and np.std(y) > 0 else float("nan")
        return {"r2": r2(s, y), "r": r, "n": len(y)}
    return {"auroc": auroc(s, y), "ap": average_precision(s, y), "base": float((y > 0.5).mean()), "n": len(y),
            "pos": int((y > 0.5).sum())}


def evaluate(probe: LinearProbe, x: torch.Tensor, y: np.ndarray) -> dict:
    return _metrics(probe.score(x).numpy(), y, probe.kind)


def probe_layer(x: torch.Tensor, labels: dict[str, np.ndarray], kinds: dict[str, str], train: np.ndarray,
                test: np.ndarray, groups: np.ndarray | None = None, keep: bool = False,
                min_pos: int = 20, scale: str = "global", seq: np.ndarray | None = None,
                order: np.ndarray | None = None, noise_sd: float = NOISE_SD) -> dict[str, dict]:
    """Every label's probe on features x [N, D]: fitted on rows `train`, scored on rows `test` (NaN labels skipped
    on both sides). groups [N] (game ids) keep the penalty's validation split by game too. keep: also return the
    fitted LinearProbe under "probe". A binary label with fewer than min_pos positives in training is skipped.
    scale: the penalty's scale (fit_probe). Each result also says how steady the readout is on the test rows
    (readout_stability: wnorm, noise, noise_rel; jitter, jitter_label, jitter_err with seq [N] and order [N])."""
    x = _tensor(x)
    out = {}
    for name, y in labels.items():
        y = np.asarray(y, np.float32)
        tr = train[~np.isnan(y[train])]
        te = test[~np.isnan(y[test])]
        if len(tr) < 50 or len(te) < 20:
            continue
        kind = kinds[name]
        if not _enough(y[tr], y[te], kind, min_pos):
            continue
        p = fit_probe(x[tr], y[tr], kind, None if groups is None else np.asarray(groups)[tr], scale=scale)
        sc = p.score(x[te]).numpy()
        res = _metrics(sc, y[te], kind)
        res["lam"] = p.lam
        res.update(_steadiness(sc, [float(p.w.norm())], noise_sd, te, seq, order, y[te], miss=kind == "reg"))
        if keep:
            res["probe"] = p
        out[name] = res
    return out


# --------------------------------------------------------------------------------------------- the ridge sweep
def _moments(x: torch.Tensor, rows: np.ndarray, chunk: int = 16384):
    """(n, sum x, sum x x^T) over rows of x, in float64, chunk by chunk."""
    d = x.shape[1]
    s1 = torch.zeros(d, dtype=torch.float64)
    s2 = torch.zeros(d, d, dtype=torch.float64)
    for i in range(0, len(rows), chunk):
        xb = x[torch.as_tensor(rows[i:i + chunk])].float()
        s1 += xb.double().sum(0)
        s2 += (xb.T @ xb).double()                                 # (float32 products per chunk, summed in float64)
    return len(rows), s1, s2


def _cross(x: torch.Tensor, rows: np.ndarray, y: np.ndarray, chunk: int = 16384) -> torch.Tensor:
    """sum x y^T over rows: [d, k]."""
    out = torch.zeros(x.shape[1], y.shape[1], dtype=torch.float64)
    for i in range(0, len(rows), chunk):
        r = rows[i:i + chunk]
        out += x[torch.as_tensor(r)].double().T @ torch.as_tensor(y[r], dtype=torch.float64)
    return out


def _scores(x: torch.Tensor, rows: np.ndarray, w: torch.Tensor, b: torch.Tensor, chunk: int = 16384) -> np.ndarray:
    out = []
    for i in range(0, len(rows), chunk):
        out.append((x[torch.as_tensor(rows[i:i + chunk])].double() @ w + b).numpy())
    return np.concatenate(out) if out else np.zeros((0, w.shape[1]))


def ridge_suite(x: torch.Tensor, labels: dict[str, np.ndarray], kinds: dict[str, str], train: np.ndarray,
                test: np.ndarray, groups: np.ndarray | None = None, lams=LAMBDAS_REG, keep: bool = False,
                min_pos: int = 20, scale: str = "global", seq: np.ndarray | None = None,
                order: np.ndarray | None = None, noise_sd: float = NOISE_SD) -> dict[str, dict]:
    """Ridge probes for every label on features x [N, D] (rows `train` fit, rows `test` score; NaN labels skipped).
    Labels sharing the same rows (the same NaN pattern) share one covariance and its eigendecomposition. The penalty
    (times the number of rows) is picked per label on a game-wise validation part of the training rows, in the scale
    `scale` gives: "global" (the default: every unit centred and divided by one number, the root of the layer's mean
    variance per unit, so no unit's raw weight is cheaper than another's) or "unit" (each unit divided by its own sd:
    a unit that barely moves can then earn a giant raw weight - risky for a probe shown live). keep: also return each
    LinearProbe (in the layer's own units) under "probe". Each result also says how steady the readout is on the test
    rows (readout_stability): wnorm, noise (its move under N(0, noise_sd^2) noise on every unit), noise_rel, and with
    seq [N] (a sequence key, e.g. game * 64 + car) and order [N] (e.g. the step) jitter, jitter_label, jitter_err."""
    x = _tensor(x)
    names = [k for k in labels if k in kinds]
    pats: dict[bytes, list[str]] = {}
    valid = {}
    for k in names:
        v = ~np.isnan(np.asarray(labels[k], np.float32))
        valid[k] = v
        pats.setdefault(np.packbits(v).tobytes(), []).append(k)
    out: dict[str, dict] = {}
    for _, group in pats.items():
        v = valid[group[0]]
        tr, te = train[v[train]], test[v[test]]
        if len(tr) < 50 or len(te) < 20:
            continue
        fit_r, val_r = _val_split(len(tr), None if groups is None else np.asarray(groups)[tr])
        fit_r, val_r = tr[np.sort(fit_r)], tr[np.sort(val_r)]
        y = np.stack([np.nan_to_num(np.asarray(labels[k], np.float64)) for k in group], 1)
        n_f, s1_f, s2_f = _moments(x, fit_r)
        n_v, s1_v, s2_v = _moments(x, val_r)
        n_t, s1_t, s2_t = n_f + n_v, s1_f + s1_v, s2_f + s2_v
        mu = s1_t / n_t
        sd = _divisors(s2_t.diagonal() / n_t - mu * mu, scale)

        def solve(n, s1, s2, rows_y, y=y, sd=sd):          # (this group's labels and scale)
            m = s1 / n
            cov = (s2 / n - torch.outer(m, m)) / torch.outer(sd, sd)
            ym = torch.as_tensor(y[rows_y].mean(0))
            cxy = (_cross(x, rows_y, y) / n - torch.outer(m, ym)) / sd[:, None]
            ev, evec = torch.linalg.eigh(cov)
            return ev.clamp_min(0), evec, evec.T @ cxy, m, ym

        ev, evec, proj, m_f, ym_f = solve(n_f, s1_f, s2_f, fit_r)
        best: dict[str, tuple[int, float]] = {}
        for li, lam in enumerate(lams):
            w = evec @ (proj / (ev[:, None] + lam))                         # [d, k] scaled
            w_raw = w / sd[:, None]
            b = ym_f - (m_f / sd) @ w
            sc = _scores(x, val_r, w_raw, b)
            for j, k in enumerate(group):
                yv = y[val_r, j]
                if kinds[k] == "bin":
                    ok = (yv > 0.5).sum() >= 1 and (yv <= 0.5).sum() >= 1
                    met = auroc(sc[:, j], yv) if ok else float("nan")
                else:
                    met = r2(sc[:, j], yv)
                if k not in best or (np.isfinite(met) and (not np.isfinite(best[k][1]) or met > best[k][1])):
                    best[k] = (li, met)
        ev, evec, proj, m_t, ym_t = solve(n_t, s1_t, s2_t, tr)
        x_te = x[torch.as_tensor(te)]
        for j, k in enumerate(group):
            if not _enough(y[tr, j], y[te, j], kinds[k], min_pos):
                continue
            lam = lams[best[k][0]]
            w = evec @ (proj[:, j:j + 1] / (ev[:, None] + lam))
            w_raw = (w / sd[:, None])[:, 0]
            b = float(ym_t[j] - (m_t / sd) @ w[:, 0])
            p = LinearProbe(w_raw.float(), b, kinds[k], float(lam))
            sc_te = p.score(x_te).numpy()
            res = _metrics(sc_te, y[te, j], kinds[k])
            res["lam"] = float(lam)
            res.update(_steadiness(sc_te, [float(p.w.norm())], noise_sd, te, seq, order, y[te, j],
                                   miss=kinds[k] == "reg"))
            if keep:
                res["probe"] = p
            out[k] = res
    return out


def headline(res: dict) -> float:
    """The one number of a probe result: R^2 for a regression label, AUROC for a binary one."""
    return res["r2"] if "r2" in res else res["auroc"]


# ------------------------------------------------------------------------------------------- non-linear probes
def mlp_probe(x: torch.Tensor, y: np.ndarray, kind: str, train: np.ndarray, test: np.ndarray,
              groups: np.ndarray | None = None, hidden: int = 256, epochs: int = 40, batch: int = 1024,
              lr: float = 1e-3, weight_decay: float = 1e-4, patience: int = 4, seed: int = 0,
              device: str = "cpu") -> dict:
    """A one-hidden-layer MLP probe (ReLU, `hidden` units) for one label: is the concept present NON-linearly where a
    linear probe reads it poorly? Inputs centred and divided by one number for the layer (as scale="global"), Adam
    with early stopping on a game-wise validation part of the training rows (groups); the same held-out metrics as
    the linear probes (R^2 / AUROC + AP). Compare it with the same MLP on a random network's layer: an MLP can compute
    things itself, so only the gap over that control says the trained layer holds more."""
    y = np.asarray(y, np.float32)
    tr = train[~np.isnan(y[train])]
    te = test[~np.isnan(y[test])]
    fit_r, val_r = _val_split(len(tr), None if groups is None else np.asarray(groups)[tr], seed=seed + 1)
    fit_r, val_r = tr[np.sort(fit_r)], tr[np.sort(val_r)]
    xt = x[torch.as_tensor(fit_r)].float()
    st = Standardizer.fit(xt, "global")
    torch.manual_seed(seed)
    d = x.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    yt = torch.as_tensor(y)
    mu_y, sd_y = (float(yt[fit_r].mean()), float(yt[fit_r].std().clamp_min(1e-6))) if kind == "reg" else (0.0, 1.0)

    def prep(rows):
        return st(x[torch.as_tensor(rows)].float()).to(device)

    def predict(rows):
        out = []
        with torch.no_grad():
            for i in range(0, len(rows), 8192):
                out.append(net(prep(rows[i:i + 8192]))[:, 0].cpu())
        return torch.cat(out).numpy() * sd_y + mu_y

    def score(rows):
        s = predict(rows)
        return r2(s, y[rows]) if kind == "reg" else average_precision(s, y[rows])

    rng = np.random.default_rng(seed)
    best, best_state, bad, n_ep = -np.inf, None, 0, 0
    for ep in range(epochs):
        n_ep = ep + 1
        perm = rng.permutation(fit_r)
        net.train()
        for i in range(0, len(perm), batch):
            rb = perm[i:i + batch]
            xb = prep(rb)
            yb = ((yt[torch.as_tensor(rb)] - mu_y) / sd_y).to(device)
            out = net(xb)[:, 0]
            loss = (torch.nn.functional.mse_loss(out, yb) if kind == "reg"
                    else torch.nn.functional.binary_cross_entropy_with_logits(out, (yb > 0.5).float()))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        net.eval()
        v = score(val_r)
        if v > best + 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    res = _metrics(predict(te), y[te], kind)
    res["epochs"] = n_ep
    res["val"] = float(best)
    return res
