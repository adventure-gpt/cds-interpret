"""Which features appear as training goes: one dictionary over the SAME layer of several training snapshots.

    cc = Crosscoder(n_models=M, d=1536, m=8192, k=32).init_from(sample)   # sample [n, M, d]
    train_crosscoder(cc, batches, steps)                                   # batches(): [B, M, d] in layer units
    norms = cc.decoder_norms()                                             # [m, M]: how present each feature is
    prof = presence(norms)                                                 # per feature: share of its peak, by model

An acausal crosscoder (Lindsey et al. 2024, "Sparse crosscoders for cross-layer features and model diffing"), here
with a top-k code instead of an L1 penalty: a moment's activations in every snapshot are encoded together into one
sparse code z, and each snapshot has its own decoder: x_hat^(s) = W_dec^(s) z + b^(s). Each feature's decoder rows
over all snapshots are kept at unit norm together, so ||W_dec^(s)_j|| (decoder_norms) says how much feature j is
part of snapshot s's layer: a feature with norm ~0 in the early snapshots and large in the late ones was not
there early - it formed during training. Each snapshot's layer is divided by its own RMS first, so a snapshot whose
layer is simply larger does not take every feature.

Read with care: (1) a feature "absent" early may be present in another layer or another form there; (2) the moments
are those the LAST snapshot played (earlier snapshots might have played them differently); (3) like any dictionary,
features of a single run are only as solid as their seed stability (dict_eval.stability); (4) every feature's norm
can drift with the others across snapshots (on a 31M-weight bot over 7-37B steps the median live feature's norm rose
by a third): compare a feature with the median feature of the same snapshot, not with its own early value. There,
nearly every feature of one layer was present from the first snapshot (25 of 7,015 live features "emerged"); what
training changed was strength - the features about demolitions grew 13-15% against the median, those about kickoffs
and walls shrank.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Crosscoder(nn.Module):
    """One sparse code for a moment's activations in M snapshots, a decoder per snapshot (see the module docstring)."""

    scale: torch.Tensor                 # (buffers, registered in __init__)
    since_fired: torch.Tensor

    def __init__(self, n_models: int, d: int, m: int, k: int, k_aux: int | None = None):
        super().__init__()
        self.M, self.d, self.m, self.k = n_models, d, m, k
        self.k_aux = k_aux if k_aux is not None else min(m // 2, 4 * k)
        self.W_enc = nn.Parameter(torch.empty(n_models, d, m))
        self.b_enc = nn.Parameter(torch.zeros(m))
        self.W_dec = nn.Parameter(torch.empty(m, n_models, d))
        self.b_pre = nn.Parameter(torch.zeros(n_models, d))
        self.register_buffer("scale", torch.ones(n_models))
        self.register_buffer("since_fired", torch.zeros(m))
        nn.init.normal_(self.W_dec, std=1.0 / math.sqrt(d))
        self.normalize_decoder()
        with torch.no_grad():
            self.W_enc.copy_(self.W_dec.permute(1, 2, 0) * (1.0 / n_models))

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        n = self.W_dec.flatten(1).norm(dim=1).clamp_min(1e-8)
        self.W_dec.div_(n[:, None, None])

    @torch.no_grad()
    def init_from(self, sample: torch.Tensor) -> Crosscoder:
        """sample [n, M, d] (layer units): each snapshot's RMS as its scale, means as b_pre."""
        x = sample.float()
        self.scale.copy_(x.pow(2).mean(dim=(0, 2)).sqrt().clamp_min(1e-8))
        self.b_pre.copy_((x / self.scale[None, :, None]).mean(0))
        return self

    def _scaled(self, x):
        return x.float() / self.scale[None, :, None]

    def pre(self, x: torch.Tensor) -> torch.Tensor:
        xs = self._scaled(x) - self.b_pre
        return torch.einsum("bsd,sdm->bm", xs, self.W_enc) + self.b_enc

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        p = F.relu(self.pre(x))
        v, i = p.topk(self.k, dim=-1)
        return torch.zeros_like(p).scatter_(-1, i, v)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """[B, M, d] in each snapshot's layer units."""
        xh = torch.einsum("bm,msd->bsd", z, self.W_dec) + self.b_pre
        return xh * self.scale[None, :, None]

    def forward(self, x):
        return self.decode(self.encode(x))

    def decoder_norms(self) -> np.ndarray:
        """[m, M]: each feature's decoder norm in each snapshot (rows have unit norm over all snapshots)."""
        return self.W_dec.detach().norm(dim=2).cpu().numpy()

    def _loss(self, x, aux_coef: float = 1.0 / 32, dead_after: float = 2e5):
        xs = self._scaled(x) - self.b_pre                                     # [B, M, d]
        p = F.relu(torch.einsum("bsd,sdm->bm", xs, self.W_enc) + self.b_enc)
        v, i = p.topk(self.k, dim=-1)
        Wf = self.W_dec.flatten(1)                                             # [m, M d]
        xh = F.embedding_bag(i, Wf, per_sample_weights=v, mode="sum").view_as(xs)
        err = xs - xh
        var = (xs - xs.mean(0)).pow(2).sum(-1).mean(0).clamp_min(1e-8)       # [M]
        mse_s = err.pow(2).sum(-1).mean(0) / var                               # [M]: each snapshot's FVU
        mse = mse_s.mean()
        with torch.no_grad():
            fired = torch.zeros(self.m, dtype=torch.bool, device=x.device)
            fired[i[v > 0]] = True
            self.since_fired.add_(len(x))
            self.since_fired[fired] = 0
        dead = self.since_fired > dead_after
        n_dead = int(dead.sum())
        aux = torch.zeros((), device=x.device)
        if n_dead and aux_coef > 0:
            pd = p * dead[None].to(p.dtype)
            va, ia = pd.topk(min(self.k_aux, n_dead), dim=-1)
            eh = F.embedding_bag(ia, Wf, per_sample_weights=va, mode="sum").view_as(xs)
            e = err.detach()
            aux = (e - eh).pow(2).sum((-1, -2)).mean() / e.pow(2).sum((-1, -2)).mean().clamp_min(1e-8)
        return mse + aux_coef * aux, mse_s.detach(), n_dead


def train_crosscoder(cc: Crosscoder, batches, steps: int = 4000, lr: float = 2e-4, log=print, log_every: int = 200,
                     time_limit: float | None = None) -> dict:
    """batches: an iterator of [B, M, d] tensors (layer units, on cc's device). Adam; decoder rows renormalised (and
    their gradient made tangent to the unit sphere) each step."""
    opt = torch.optim.Adam(cc.parameters(), lr=lr)
    hist = []
    t0 = time.time()
    step = 0
    for step in range(1, steps + 1):
        x = next(batches)
        loss, mse_s, n_dead = cc._loss(x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            assert cc.W_dec.grad is not None
            g = cc.W_dec.grad.flatten(1)
            w = cc.W_dec.flatten(1)
            g -= (g * w).sum(1, keepdim=True) * w
        opt.step()
        cc.normalize_decoder()
        if step % log_every == 0 or step == 1:
            hist.append({"step": step, "fvu": [float(v) for v in mse_s], "dead": n_dead, "s": time.time() - t0})
            if log is not None:
                log(f"  crosscoder step {step}/{steps}: FVU by snapshot " + " ".join(f"{float(v):.3f}" for v in mse_s)
                    + f", dead {n_dead}, {time.time() - t0:.0f} s")
        if time_limit is not None and time.time() - t0 > time_limit:
            break
    return {"history": hist, "steps": step, "seconds": time.time() - t0}


@torch.no_grad()
def fvu_by_model(cc: Crosscoder, x: torch.Tensor, batch: int = 2048) -> np.ndarray:
    """[M]: fraction of variance unexplained in each snapshot's layer, rows x [n, M, d]."""
    dev = cc.W_enc.device
    mu = x.float().mean(0).to(dev)
    num = torch.zeros(cc.M, device=dev)
    den = torch.zeros(cc.M, device=dev)
    for i in range(0, len(x), batch):
        xb = x[i:i + batch].to(dev).float()
        num += (xb - cc(xb)).pow(2).sum((0, 2))
        den += (xb - mu).pow(2).sum((0, 2))
    return (num / den.clamp_min(1e-12)).cpu().numpy()


def presence(norms: np.ndarray, on: float = 0.5) -> dict:
    """From decoder norms [m, M]: per feature its profile (norm / its largest norm), the first snapshot where it
    reaches `on` of its peak ("birth"), the last where it does ("still there"), and a kind: "shared" (present in the
    first and last snapshots), "emerged" (not in the first, in the last), "faded" (in the first, not the last),
    "transient" (neither)."""
    prof = norms / np.maximum(norms.max(1, keepdims=True), 1e-12)
    present = prof >= on
    first = np.where(present.any(1), present.argmax(1), -1)
    last = np.where(present.any(1), norms.shape[1] - 1 - present[:, ::-1].argmax(1), -1)
    kind = np.where(present[:, 0] & present[:, -1], "shared",
                    np.where(~present[:, 0] & present[:, -1], "emerged",
                             np.where(present[:, 0] & ~present[:, -1], "faded", "transient")))
    return {"profile": prof, "birth": first, "last": last, "kind": kind}


def save(cc: Crosscoder, path, **meta) -> None:
    """Write a crosscoder as plain tensors and numbers (meta: JSON-like values)."""
    torch.save({"format": "cds-interpret.crosscoder", "M": cc.M, "d": cc.d, "m": cc.m, "k": cc.k, "k_aux": cc.k_aux,
                "state": {k: v.cpu() for k, v in cc.state_dict().items()}, "meta": meta}, path)


def load(path) -> Crosscoder:
    """A crosscoder written by save(). Loaded with torch.load(weights_only=True): no code in the file can run."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cc = Crosscoder(ck["M"], ck["d"], ck["m"], ck["k"], ck.get("k_aux"))
    cc.load_state_dict(ck["state"])
    cc.meta = ck.get("meta", {})
    return cc.eval()
