"""Readouts fit to be shown LIVE: calibrated probabilities, smoothing with a measured lag, feature flags that do not
flicker. Each comes with the number that says whether it helped, measured on recorded sequences of moments.

    cal = fit_isotonic(score_fit, y_fit)          # a monotone map: score -> probability (pool adjacent violators)
    p = apply_isotonic(cal, score)                # or fit_platt / apply_platt (two numbers)
    calibration(p, y)                             # expected calibration error, Brier score, reliability table
    s = ema_seq(x, seq, order, alpha)             # exponential smoothing within each sequence (game x car)
    smoothing_tradeoff(x, y, seq, order, alphas)  # per alpha: jitter, lag behind the label, accuracy
    f = hysteresis(pre, seq, order, on, off)      # a flag that turns on above `on` and off only below `off`
    switches_per_minute(f, seq, order, dt)        # flicker

Why: a binary probe fitted by least squares (the linear-probability model, what the ridge sweep fits) gives scores
that rank well (AUROC) but are not probabilities - for a rare event (a demo chance on ~1.5% of moments) a score of
0.4 may mean 10% or 80%; a feature flag read off one threshold flips on and off between neighbouring frames.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------------------------- calibration
def fit_platt(score: np.ndarray, y: np.ndarray, iters: int = 50, l2: float = 1e-6) -> tuple[float, float]:
    """(a, b) of p = sigmoid(a * score + b), maximum likelihood (Newton)."""
    s = np.asarray(score, np.float64)
    t = (np.asarray(y) > 0.5).astype(np.float64)
    a, b = 1.0, 0.0
    for _ in range(iters):
        z = np.clip(a * s + b, -40, 40)
        p = 1.0 / (1.0 + np.exp(-z))
        w = p * (1 - p) + 1e-12
        g = np.array([((p - t) * s).sum() + l2 * a, (p - t).sum()])
        H = np.array([[(w * s * s).sum() + l2, (w * s).sum()], [(w * s).sum(), w.sum()]])
        step = np.linalg.solve(H, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-9:
            break
    return float(a), float(b)


def apply_platt(ab: tuple[float, float], score: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(ab[0] * np.asarray(score, np.float64) + ab[1], -40, 40)))


def fit_isotonic(score: np.ndarray, y: np.ndarray, max_knots: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """A non-decreasing map score -> P(y), by pool adjacent violators on the sorted scores; returned as knots (x, p)
    for np.interp (at most max_knots, evenly spaced in rank, so a pack stays small)."""
    s = np.asarray(score, np.float64)
    t = (np.asarray(y) > 0.5).astype(np.float64)
    o = np.argsort(s, kind="mergesort")
    s, t = s[o], t[o]
    val, wt, xs = [], [], []
    for si, ti in zip(s, t):
        val.append(ti)
        wt.append(1.0)
        xs.append(si)
        while len(val) > 1 and val[-2] > val[-1]:
            v2, w2 = val.pop(), wt.pop()
            x2 = xs.pop()
            val[-1] = (val[-1] * wt[-1] + v2 * w2) / (wt[-1] + w2)
            wt[-1] += w2
            xs[-1] = x2                                   # the block's right end
    # blocks -> step function knots at each block's right end
    kx, kp = np.asarray(xs), np.asarray(val)
    if len(kx) > max_knots:
        cw = np.cumsum(wt)
        pick = np.unique(np.searchsorted(cw, np.linspace(0, cw[-1], max_knots), side="left").clip(0, len(kx) - 1))
        kx, kp = kx[pick], kp[pick]
    return kx, kp


def apply_isotonic(knots: tuple[np.ndarray, np.ndarray], score: np.ndarray) -> np.ndarray:
    return np.interp(np.asarray(score, np.float64), knots[0], knots[1])


def calibration(p: np.ndarray, y: np.ndarray, bins: int = 10) -> dict:
    """Expected calibration error (bins of equal count, so a rare event's high bins are not empty), Brier score, and
    the reliability table [(mean predicted, observed rate, n)]."""
    p = np.clip(np.asarray(p, np.float64), 0, 1)
    t = (np.asarray(y) > 0.5).astype(np.float64)
    o = np.argsort(p, kind="mergesort")
    parts = np.array_split(o, bins)
    table, ece = [], 0.0
    for part in parts:
        if not len(part):
            continue
        mp, fr = float(p[part].mean()), float(t[part].mean())
        table.append((mp, fr, int(len(part))))
        ece += len(part) / len(p) * abs(mp - fr)
    # the top 1% separately: where a rare event's alarm lives
    top = o[-max(1, len(o) // 100):]
    return {"ece": float(ece), "brier": float(((p - t) ** 2).mean()), "table": table,
            "top1pct_pred": float(p[top].mean()), "top1pct_rate": float(t[top].mean())}


# ----------------------------------------------------------------------------------------------- smoothing
def _seq_order(seq: np.ndarray, order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    o = np.lexsort((np.asarray(order), np.asarray(seq)))
    s = np.asarray(seq)[o]
    first = np.ones(len(o), bool)
    first[1:] = s[1:] != s[:-1]
    return o, first


def ema_seq(x: np.ndarray, seq: np.ndarray, order: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average within each sequence (in `order`), restarting at each sequence's first row:
    s_t = s_(t-1) + alpha (x_t - s_(t-1)). Returned in the input's row order."""
    x = np.asarray(x, np.float64)
    o, first = _seq_order(seq, order)
    xs = x[o]
    out = np.empty_like(xs)
    prev = 0.0
    for i in range(len(xs)):
        prev = xs[i] if first[i] else prev + alpha * (xs[i] - prev)
        out[i] = prev
    res = np.empty_like(out)
    res[o] = out
    return res


def _lag(a: np.ndarray, b: np.ndarray, first: np.ndarray, max_lag: int = 15) -> int:
    """The shift k >= 0 (rows) for which a[t] correlates best with b[t - k] within sequences (a, b in sequence
    order, first marking each sequence's first row): how far a readout trails what it reads."""
    pos = np.arange(len(a)) - np.maximum.accumulate(np.where(first, np.arange(len(a)), 0))
    best, arg = -np.inf, 0
    for k in range(max_lag + 1):
        i = np.nonzero(pos >= k)[0]
        aa, bb = a[i], b[i - k]
        if len(aa) < 10 or aa.std() == 0 or bb.std() == 0:
            continue
        c = np.corrcoef(aa, bb)[0, 1]
        if c > best:
            best, arg = c, k
    return arg


def smoothing_tradeoff(x: np.ndarray, y: np.ndarray, seq: np.ndarray, order: np.ndarray, alphas, dt: float,
                       kind: str = "reg") -> list[dict]:
    """For each alpha: the time constant tau = -dt / ln(1 - alpha), the median move between consecutive readings
    (jitter, against the label's own), how many rows the smoothed readout lags the label (best cross-correlation),
    and its accuracy against the label (R^2 for kind "reg", AUROC for "bin")."""
    from .probes import auroc, r2
    o, first = _seq_order(seq, order)
    yo = np.asarray(y, np.float64)[o]
    out = []
    for a in alphas:
        s = ema_seq(x, seq, order, a)[o]
        d = np.abs(np.diff(s))[~first[1:]]
        dl = np.abs(np.diff(yo))[~first[1:]]
        ok = ~np.isnan(yo)
        acc = r2(s[ok], yo[ok]) if kind == "reg" else auroc(s[ok], yo[ok])
        out.append({"alpha": float(a), "tau_s": float(-dt / np.log(1 - a)) if a < 1 else 0.0,
                    "jitter": float(np.median(d)), "jitter_p90": float(np.percentile(d, 90)),
                    "jitter_label": float(np.nanmedian(dl)), "lag_rows": _lag(s, np.nan_to_num(yo), first),
                    "accuracy": float(acc)})
    return out


# ------------------------------------------------------------------------------------------------- flags
def hysteresis(pre: np.ndarray, seq: np.ndarray, order: np.ndarray, on: float, off: float) -> np.ndarray:
    """A flag per row: turns on when pre > on, stays on until pre < off (off <= on); each sequence starts off."""
    pre = np.asarray(pre, np.float64)
    o, first = _seq_order(seq, order)
    p = pre[o]
    f = np.zeros(len(p), bool)
    state = False
    for i in range(len(p)):
        if first[i]:
            state = False
        state = (p[i] > on) if not state else (p[i] >= off)
        f[i] = state
    res = np.empty_like(f)
    res[o] = f
    return res


def hold(flag: np.ndarray, seq: np.ndarray, order: np.ndarray, rows: int) -> np.ndarray:
    """A flag kept on for `rows` more rows after it turns off (the overlay's 0.6 s hold), within sequences."""
    o, first = _seq_order(seq, order)
    f = np.asarray(flag, bool)[o]
    out = np.zeros(len(f), bool)
    left = 0
    for i in range(len(f)):
        if first[i]:
            left = 0
        left = rows if f[i] else max(left - 1, 0)
        out[i] = f[i] or left > 0
    res = np.empty_like(out)
    res[o] = out
    return res


def switches_per_minute(flag: np.ndarray, seq: np.ndarray, order: np.ndarray, dt: float) -> float:
    """How often the flag changes state, per minute of play (dt: seconds between consecutive rows)."""
    o, first = _seq_order(seq, order)
    f = np.asarray(flag, bool)[o]
    ch = (f[1:] != f[:-1]) & ~first[1:]
    minutes = (len(f) - first.sum()) * dt / 60.0
    return float(ch.sum() / max(minutes, 1e-9))


def flag_quality(flag: np.ndarray, truth: np.ndarray) -> dict:
    """Precision and recall of a flag against a reference (e.g. the full SAE's firing, or a label)."""
    f, t = np.asarray(flag, bool), np.asarray(truth, bool)
    tp = float((f & t).sum())
    return {"precision": tp / max(int(f.sum()), 1), "recall": tp / max(int(t.sum()), 1), "on_share": float(f.mean())}


def calibrate_pack_probe(pack: dict, name: str, a: float, b: float) -> dict:
    """Make an overlay pack show binary probe ``name`` as sigmoid(a * score + b) (fit_platt on held-out games): its
    weights are folded into a logit probe (kind "bin_logit"), which the overlay already reads - no other change."""
    for p in pack["probes"]:
        if p["name"] == name:
            p["w"] = (np.asarray(p["w"], np.float32) * np.float32(a)).astype(np.float32)
            p["b"] = float(a * float(p["b"]) + b)
            p["kind"] = "bin_logit"
            return pack
    raise KeyError(f"no probe named {name!r} in the pack (it has {[p['name'] for p in pack['probes']]})")
