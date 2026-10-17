"""Concept erasure: remove ALL linearly available information about a concept from a layer, then see whether the
policy notices (amnesic probing, Elazar et al. 2021, with LEACE, Belrose et al. 2023).

    er = fit_leace(x_train, z_train)            # x [n, d] the layer; z [n, k] the concept (0/1 or values)
    x2 = er(x)                                  # no linear read-out of z does better than a constant on x2
    edit = erase_edit(er)                       # an edit for intervene.edit_effect / intervene.editing
    ctl = random_eraser(x_train, rank=k)        # a same-rank erasure of a random direction (the control)

LEACE is the least-squares-smallest edit that makes the class-conditional (or concept-conditional) means equal, i.e.
the cross-covariance with z zero: r(x) = x - W+ P W (x - mean), W the whitening map of x, P the projection onto
the span of W cov(x, z). Rank k (the concept's dimension). Compared with projecting out a probe's direction (what
intervene.project_out does), it removes what EVERY linear probe could use, not just one direction - so if the
policy is unchanged after LEACE, the decision does not rest on the linearly readable concept at that layer (it may
still use a non-linear encoding of it: test with a non-linear probe, probes.mlp_probe).

Interpreting effect sizes needs the control: erasing any rank-k readable feature of the layer moves the policy too;
the concept's erasure matters only as far as it moves the policy MORE than erasing random readable features
(random_eraser, mode "feature") - and, for a behaviour tied to the concept, as far as it changes THAT behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Eraser:
    mean: torch.Tensor          # [d]
    U: torch.Tensor             # [d, r]: x' = x - ((x - mean) @ V) @ U.T
    V: torch.Tensor             # [d, r]
    rank: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        m, U, V = (t.to(x.device) for t in (self.mean, self.U, self.V))
        return (xf - ((xf - m) @ V) @ U.T).to(x.dtype)


def _whitening(x: torch.Tensor, rel_eps: float = 1e-5):
    """(mean, W [d, d], W+ [d, d]) of rows x: W = cov^(-1/2), W+ = cov^(1/2) on cov's range (tiny eigenvalues
    dropped)."""
    xd = x.double()
    mu = xd.mean(0)
    xc = xd - mu
    cov = xc.T @ xc / max(len(x) - 1, 1)
    ev, evec = torch.linalg.eigh(cov)
    keep = ev > rel_eps * ev.max().clamp_min(1e-30)
    s = ev.clamp_min(0).sqrt()
    inv = torch.where(keep, 1.0 / s.clamp_min(1e-30), torch.zeros_like(s))
    W = (evec * inv) @ evec.T
    Wp = (evec * torch.where(keep, s, torch.zeros_like(s))) @ evec.T
    return mu, W, Wp


@torch.no_grad()
def fit_leace(x: torch.Tensor, z: np.ndarray | torch.Tensor, rel_eps: float = 1e-5) -> Eraser:
    """LEACE eraser for concept z ([n] or [n, k], binary or real; NaN rows are dropped) on layer rows x [n, d]."""
    zt = torch.as_tensor(np.asarray(z, np.float64)).reshape(len(x), -1)
    ok = ~torch.isnan(zt).any(1)
    xs, zs = x[ok].double(), zt[ok]
    mu, W, Wp = _whitening(xs, rel_eps)
    zc = zs - zs.mean(0)
    sxz = (xs - mu).T @ zc / max(len(xs) - 1, 1)                    # [d, k]
    A = W @ sxz
    Q, R = torch.linalg.qr(A)
    keep = R.diagonal().abs() > 1e-10 * R.diagonal().abs().max().clamp_min(1e-30)
    Q = Q[:, keep]
    return Eraser(mu.float(), (Wp @ Q).float(), (W @ Q).float(), int(Q.shape[1]))


@torch.no_grad()
def random_eraser(x: torch.Tensor, rank: int = 1, seed: int = 0, rel_eps: float = 1e-5, mode: str = "feature") -> Eraser:
    """The control, a same-rank erasure of something that is NOT the concept:
    mode "feature" (the default): LEACE for `rank` random linear read-outs of the layer (z = x @ r, r ~ N(0, I) in
      the layer's own coordinates) - a placebo concept that lives where the layer varies, as real concepts do;
    mode "whitened": a random direction of the whitened layer (equivalently LEACE for a random Gaussian concept) -
      mostly the layer's low-variance directions, so its edits are small - a weak control (on a 31M-weight bot,
      every rank-1 concept erasure at the first hidden layer moved the policy ~45x more than it)."""
    g = torch.Generator().manual_seed(seed)
    if mode == "feature":
        r = torch.randn(x.shape[1], rank, generator=g)
        return fit_leace(x, (x.float() @ r).numpy(), rel_eps)
    xs = x.double()
    mu, W, Wp = _whitening(xs, rel_eps)
    Q, _ = torch.linalg.qr(torch.randn(x.shape[1], rank, generator=g, dtype=torch.float64))
    Q = W @ (Wp @ Q)                                                 # (stay in cov's range)
    Q, _ = torch.linalg.qr(Q)
    return Eraser(mu.float(), (Wp @ Q).float(), (W @ Q).float(), rank)


def erase_edit(er: Eraser):
    """An edit function (layer output -> new output) applying the eraser."""
    def edit(h):
        return er(h)
    return edit


def edit_size(er: Eraser, x: torch.Tensor) -> float:
    """Mean |x - er(x)| / mean |x - mean(x)|: how big the erasure is against the layer's own spread."""
    xf = x.float()
    d = (xf - er(xf)).norm(dim=1).mean()
    s = (xf - xf.mean(0)).norm(dim=1).mean()
    return float(d / s.clamp_min(1e-12))
