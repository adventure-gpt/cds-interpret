"""A top-k sparse autoencoder (SAE): a dictionary of features for one layer of a network.

    sae = TopKSAE(d=256, m=2048, k=16).init_from(sample)         # sample [n, d] of the layer's activations
    train_sae(sae, acts, steps=3000, batch=2048)                  # acts: [N, d] array or memmap (float16 / float32)
    z = sae.encode(x)                                             # [n, m], at most k non-zero per row
    x_hat = sae.decode(z)
    fvu(sae, x)                                                   # fraction of variance unexplained, held-out rows
    firing_thresholds(sae, x, [12, 40])                           # what counts as "on" for the live overlay
    downstream_check(sae, f, x)                                   # the policy on the reconstruction, from stored rows
    save(sae, "sae.pt"); load("sae.pt")                           # plain tensors: safe to load from anyone

Gao et al. 2024 ("Scaling and evaluating sparse autoencoders"): z = TopK(ReLU(W_enc (x - b_pre) + b_enc)),
x_hat = W_dec z + b_pre, decoder rows kept at unit norm, an auxiliary loss (AuxK) that lets dead latents learn
the residual. The layer is first divided by one scalar (its RMS over the training sample) so the loss is scale-free;
encode / decode take and give the layer's own units.

The check that matters for a policy (policy_kl): the network run with the layer replaced by the SAE's
reconstruction - how far its action distribution moves (KL), against replacing the layer by its mean (the scale
of "losing the layer"), and how often the argmax action changes.

Two variants:
  - BatchTopKSAE (Bussmann et al. 2024): the k * batch largest activations of a whole batch are kept in training, so
    a busy moment may use more features than a quiet one; at inference one learned threshold replaces the top-k.
  - a POLICY-AWARE loss: train_sae(..., downstream=f, kl_coef=c), f the rest of the network from this layer to the
    output (hooks.tail): the loss adds c * KL(policy from the real layer || policy from the reconstruction), so the
    dictionary spends its features on what changes decisions ("end-to-end" SAEs, Braun et al. 2024, with the
    reconstruction term kept). finetune_kl continues an MSE-trained dictionary that way.
Measured on a 31M-weight self-play bot's third hidden layer (1,536 units, 8,192 features, k 32, 30,000 held-out
moments): 1,000 steps of finetune_kl took the policy's KL from the reconstruction from 0.21 to 0.15 nats (KL recovered
0.90 -> 0.93, the chosen action changed on 32% -> 28% of moments) with the same features (decoder cosine 1.00) and the
same variance explained; trained from scratch with the KL term it reached 0.14 but explained less of the layer (FVU
0.25 against 0.14) with different features; BatchTopK matched TopK (KL recovered 0.90).
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    """A dictionary of m features for a layer of width d, at most k active on any row (see the module docstring).
    init_from(sample) before training; encode / decode / forward take and give the layer's own units."""

    scale: torch.Tensor                 # (buffers, registered in __init__)
    since_fired: torch.Tensor

    def __init__(self, d: int, m: int, k: int, k_aux: int | None = None):
        super().__init__()
        if not 0 < k <= m:
            raise ValueError(f"an SAE needs 0 < k <= m (active features per row <= features), got k={k}, m={m}")
        self.d, self.m, self.k = d, m, k
        self.k_aux = k_aux if k_aux is not None else min(m // 2, 4 * k)
        self.W_enc = nn.Parameter(torch.empty(d, m))
        self.b_enc = nn.Parameter(torch.zeros(m))
        self.W_dec = nn.Parameter(torch.empty(m, d))
        self.b_pre = nn.Parameter(torch.zeros(d))
        self.register_buffer("scale", torch.ones(()))
        self.register_buffer("since_fired", torch.zeros(m))       # samples since each latent last fired
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        self.normalize_decoder()
        with torch.no_grad():
            self.W_enc.copy_(self.W_dec.T)

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8))

    @torch.no_grad()
    def init_from(self, sample: torch.Tensor) -> TopKSAE:
        """Scale from the sample's RMS, b_pre from its mean (in the scaled units)."""
        x = torch.as_tensor(sample).float()
        self.scale.fill_(float(x.pow(2).mean().sqrt().clamp_min(1e-8)))
        self.b_pre.copy_((x / self.scale).mean(0))
        return self

    # ---- the map (layer units in and out)
    def pre(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() / self.scale - self.b_pre) @ self.W_enc + self.b_enc

    def encode(self, x: torch.Tensor, k: int | None = None) -> torch.Tensor:
        """[n, m] latent activations (scaled units), the top k positive ones per row, the rest 0."""
        p = F.relu(self.pre(x))
        v, i = p.topk(k or self.k, dim=-1)
        return torch.zeros_like(p).scatter_(-1, i, v)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return (z @ self.W_dec + self.b_pre) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def feature_direction(self, j: int) -> torch.Tensor:
        """Feature j's decoder direction in the layer's units (a unit vector times the scale)."""
        return self.W_dec[j] * self.scale

    # ---- training internals (scaled units)
    def _select(self, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(values, indices) [B, kk] of the latents kept for each row in training (top-k per row here)."""
        return p.topk(self.k, dim=-1)

    def _loss(self, xs: torch.Tensor, aux_coef: float = 1.0 / 32, dead_after: float = 2e5, want_xhat: bool = False):
        p = F.relu((xs - self.b_pre) @ self.W_enc + self.b_enc)
        v, i = self._select(p)
        x_hat = F.embedding_bag(i, self.W_dec, per_sample_weights=v, mode="sum") + self.b_pre
        err = xs - x_hat
        var = (xs - xs.mean(0)).pow(2).sum(-1).mean().clamp_min(1e-8)
        mse = err.pow(2).sum(-1).mean() / var
        with torch.no_grad():
            fired = torch.zeros(self.m, dtype=torch.bool, device=xs.device)
            fired[i[v > 0]] = True
            self.since_fired.add_(len(xs))
            self.since_fired[fired] = 0
        dead = self.since_fired > dead_after
        aux = torch.zeros((), device=xs.device)
        n_dead = int(dead.sum())
        if n_dead and aux_coef > 0:
            pd = p * dead[None].to(p.dtype)
            va, ia = pd.topk(min(self.k_aux, n_dead), dim=-1)
            e_hat = F.embedding_bag(ia, self.W_dec, per_sample_weights=va, mode="sum")
            aux = (err.detach() - e_hat).pow(2).sum(-1).mean() / err.detach().pow(2).sum(-1).mean().clamp_min(1e-8)
        if want_xhat:
            return mse, aux_coef * aux, n_dead, x_hat
        return mse + aux_coef * aux, float(mse.detach()), n_dead


class BatchTopKSAE(TopKSAE):
    """BatchTopK (Bussmann, Leask, Nanda 2024): in training the batch's k * B largest pre-activations are kept (a row
    may get more or fewer than k); a threshold theta (a running mean of the smallest kept value) is the inference
    rule: z = p where p > theta. The mean L0 is about k; a row keeps at most k_store (default 2k)."""

    theta: torch.Tensor                 # (buffers, registered in __init__)
    theta_n: torch.Tensor

    def __init__(self, d: int, m: int, k: int, k_aux: int | None = None, k_store: int | None = None):
        super().__init__(d, m, k, k_aux)
        self.k_store = k_store or 2 * k
        self.register_buffer("theta", torch.zeros(()))
        self.register_buffer("theta_n", torch.zeros(()))

    def _select(self, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = p.shape[0]
        flat = p.reshape(-1)
        v, idx = flat.topk(min(self.k * B, flat.numel()))
        if self.training:
            with torch.no_grad():
                n = float(self.theta_n)
                rate = 1.0 / (n + 1.0) if n < 1000 else 1e-3
                self.theta.mul_(1.0 - rate).add_(rate * v[-1].clamp_min(0))
                self.theta_n.add_(1)
        z = torch.zeros_like(flat).scatter_(0, idx, v).reshape(B, -1)
        return z.topk(min(self.k_store, self.m), dim=-1)

    def encode(self, x: torch.Tensor, k: int | None = None) -> torch.Tensor:
        p = F.relu(self.pre(x))
        z = p * (p > self.theta)
        kk = k or self.k_store
        if kk < self.m:
            v, i = z.topk(kk, dim=-1)
            z = torch.zeros_like(z).scatter_(-1, i, v)
        return z


def _batches(acts, batch: int, rng: np.random.Generator, rows: np.ndarray):
    while True:
        perm = rng.permutation(rows)
        for i in range(0, len(perm) - batch + 1, batch):
            idx = np.sort(perm[i:i + batch])
            yield torch.from_numpy(np.asarray(acts[idx], dtype=np.float32))


def _kl_rows(lp0: torch.Tensor, lp1: torch.Tensor) -> torch.Tensor:
    return (lp0.exp() * (lp0 - lp1)).sum(-1)


def train_sae(sae: TopKSAE, acts, rows: np.ndarray | None = None, steps: int = 3000, batch: int = 2048,
              lr: float = 2e-4, seed: int = 0, log=None, log_every: int = 200, time_limit: float | None = None,
              downstream=None, kl_coef: float = 0.0, mse_coef: float = 1.0) -> dict:
    """Adam on the normalised MSE (+ AuxK); decoder rows renormalised and their parallel gradient removed every step.
    acts: an [N, d] array, memmap or tensor of the layer's activations; rows: the training rows (default all);
    log: a function given a progress line now and then (print, say), or None; time_limit: stop after this many
    seconds. downstream (with kl_coef > 0): the rest of the network, layer activations [B, d] -> logits [B, A]
    (hooks.tail); the loss is then mse_coef * MSE + kl_coef * KL(policy(x) || policy(x_hat)) + AuxK (only the SAE's
    parameters are trained)."""
    if torch.is_tensor(acts):
        acts = acts.detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    rows = np.arange(len(acts)) if rows is None else np.asarray(rows)
    if len(rows) < 2:
        raise ValueError(f"train_sae needs at least 2 rows of activations, got {len(rows)}")
    batch = min(batch, len(rows))
    torch.manual_seed(seed)
    opt = torch.optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
    it = _batches(acts, batch, rng, rows)
    dev = sae.W_enc.device
    hist = []
    t0 = time.time()
    use_kl = downstream is not None and kl_coef > 0
    sae.train()
    for step in range(1, steps + 1):
        x = next(it).to(dev, non_blocking=True) / sae.scale
        kl_v = float("nan")
        if use_kl:
            mse_t, aux_t, n_dead, x_hat = sae._loss(x, want_xhat=True)
            with torch.no_grad():
                lp0 = F.log_softmax(downstream(x * sae.scale).float(), -1)
            lp1 = F.log_softmax(downstream(x_hat * sae.scale).float(), -1)
            kl_t = _kl_rows(lp0, lp1).mean()
            loss = mse_coef * mse_t + kl_coef * kl_t + aux_t
            mse, kl_v = float(mse_t.detach()), float(kl_t.detach())
        else:
            loss, mse, n_dead = sae._loss(x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():                                   # keep decoder rows on the unit sphere
            g = sae.W_dec.grad
            if g is not None:
                g -= (g * sae.W_dec).sum(1, keepdim=True) * sae.W_dec
        opt.step()
        sae.normalize_decoder()
        if step % log_every == 0 or step == 1:
            hist.append({"step": step, "mse": mse, "kl": kl_v, "dead": n_dead, "s": time.time() - t0})
            if log is not None:
                log(f"  sae step {step}/{steps}: normalised mse {mse:.4f}" + (f", KL {kl_v:.4f}" if use_kl else "")
                    + f", dead {n_dead}, {time.time() - t0:.0f} s")
        if time_limit is not None and time.time() - t0 > time_limit:
            break
    sae.eval()
    return {"history": hist, "steps": step, "seconds": time.time() - t0}


def finetune_kl(sae: TopKSAE, acts, downstream, rows=None, steps: int = 1000, batch: int = 2048, lr: float = 1e-4,
                kl_coef: float = 1.0, mse_coef: float = 1.0, seed: int = 1, log=None, **kw) -> dict:
    """Continue training a (MSE-trained) dictionary with the policy's KL in the loss: the same features, a
    reconstruction the policy can hardly tell from the layer."""
    return train_sae(sae, acts, rows=rows, steps=steps, batch=batch, lr=lr, seed=seed, log=log, downstream=downstream,
                     kl_coef=kl_coef, mse_coef=mse_coef, **kw)


@torch.no_grad()
def downstream_check(sae: TopKSAE, downstream, x: torch.Tensor, batch: int = 4096,
                     mean: torch.Tensor | None = None) -> dict:
    """From stored activations x [n, d] (layer units) and the rest of the network (downstream: layer -> logits, e.g.
    hooks.tail): the policy with the layer replaced by the reconstruction - mean KL, share of rows whose argmax
    changes, mean total variation - and with the layer replaced by its mean (the scale of losing the layer; mean
    defaults to x's), the KL recovered 1 - KL(recon) / KL(mean), the FVU and the mean L0."""
    dev = sae.W_enc.device
    x = torch.as_tensor(x)
    mu = (x.float().mean(0) if mean is None else torch.as_tensor(mean).float()).to(dev)
    acc = {"kl": 0.0, "argmax_changed": 0.0, "tv": 0.0, "kl_mean": 0.0, "fvu_num": 0.0, "fvu_den": 0.0, "l0": 0.0}
    for i in range(0, len(x), batch):
        xb = x[i:i + batch].to(dev).float()
        z = sae.encode(xb)
        xh = sae.decode(z)
        lp0 = F.log_softmax(downstream(xb).float(), -1)
        lp1 = F.log_softmax(downstream(xh).float(), -1)
        lpm = F.log_softmax(downstream(mu[None].expand_as(xb)).float(), -1)
        acc["kl"] += float(_kl_rows(lp0, lp1).sum())
        acc["kl_mean"] += float(_kl_rows(lp0, lpm).sum())
        acc["argmax_changed"] += float((lp0.argmax(-1) != lp1.argmax(-1)).float().sum())
        acc["tv"] += float(0.5 * (lp0.exp() - lp1.exp()).abs().sum(-1).sum())
        acc["fvu_num"] += float((xb - xh).pow(2).sum())
        acc["fvu_den"] += float((xb - mu).pow(2).sum())
        acc["l0"] += float((z > 0).float().sum())
    n = max(len(x), 1)
    out = {k: acc[k] / n for k in ("kl", "argmax_changed", "tv", "kl_mean", "l0")}
    out["kl_recovered"] = 1.0 - out["kl"] / max(out["kl_mean"], 1e-12)
    out["fvu"] = acc["fvu_num"] / max(acc["fvu_den"], 1e-12)
    return out


@torch.no_grad()
def fvu(sae: TopKSAE, x: torch.Tensor, batch: int = 8192) -> float:
    """Fraction of variance unexplained on rows x (layer units)."""
    num, den = 0.0, 0.0
    dev = sae.W_enc.device
    mu = x.float().mean(0).to(dev)
    for i in range(0, len(x), batch):
        xb = x[i:i + batch].to(dev).float()
        num += float((xb - sae(xb)).pow(2).sum())
        den += float((xb - mu).pow(2).sum())
    return num / max(den, 1e-12)


@torch.no_grad()
def latent_stats(sae: TopKSAE, x: torch.Tensor, batch: int = 8192) -> dict[str, torch.Tensor]:
    """Per latent over rows x: firing frequency, mean activation when firing, max activation."""
    dev = sae.W_enc.device
    cnt = torch.zeros(sae.m, device=dev)
    tot = torch.zeros(sae.m, device=dev)
    mx = torch.zeros(sae.m, device=dev)
    for i in range(0, len(x), batch):
        z = sae.encode(x[i:i + batch].to(dev))
        cnt += (z > 0).float().sum(0)
        tot += z.sum(0)
        mx = torch.maximum(mx, z.max(0).values)
    return {"freq": (cnt / len(x)).cpu(), "mean_when_on": (tot / cnt.clamp_min(1)).cpu(), "max": mx.cpu()}


@torch.no_grad()
def encode_all(sae: TopKSAE, x: torch.Tensor, batch: int = 8192) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k codes of every row: (indices [n, k] int32, values [n, k] float16) - the compact form of z."""
    idx, val = [], []
    dev = sae.W_enc.device
    for i in range(0, len(x), batch):
        xb = x[i:i + batch].to(dev)
        p = sae.encode(xb) if isinstance(sae, BatchTopKSAE) else F.relu(sae.pre(xb))
        v, j = p.topk(getattr(sae, "k_store", sae.k), dim=-1)
        idx.append(j.int().cpu())
        val.append(v.half().cpu())
    return torch.cat(idx), torch.cat(val)


def dense_column(idx: torch.Tensor, val: torch.Tensor, j: int) -> torch.Tensor:
    """Latent j's activation [n] from the compact codes."""
    hit = idx == j
    return (val.float() * hit).sum(1)


@torch.no_grad()
def firing_thresholds(sae: TopKSAE, x: torch.Tensor, features, q: float = 0.05, batch: int = 8192) -> dict[int, float]:
    """For each feature j: the q-quantile of its activation over the rows where the SAE's top-k let it through. A
    live display that computes only a few features (not all m, so not the top-k gate) calls one "on" when its
    pre-activation is above this - the feature's smallest typical firing value."""
    feats = [int(j) for j in features]
    vals: dict[int, list[torch.Tensor]] = {j: [] for j in feats}
    dev = sae.W_enc.device
    for i in range(0, len(x), batch):
        z = sae.encode(x[i:i + batch].to(dev))
        for j in feats:
            v = z[:, j]
            vals[j].append(v[v > 0].cpu())
    out = {}
    for j in feats:
        v = torch.cat(vals[j]) if vals[j] else torch.zeros(0)
        out[j] = float(torch.quantile(v.float(), q)) if len(v) else 0.0
    return out


def fingerprint(sae: TopKSAE) -> str:
    """A short code that is the same only for the same dictionary (the same weights, to the last bit). A names file
    keeps it, so names are never put on another dictionary's features by mistake."""
    import hashlib
    h = hashlib.sha256()
    for name, t in sorted(sae.state_dict().items()):
        if name == "since_fired":                    # (training bookkeeping, not part of the dictionary)
            continue
        h.update(name.encode())
        h.update(t.detach().cpu().float().contiguous().numpy().tobytes())
    return h.hexdigest()[:12]


def save(sae: TopKSAE, path, **meta) -> None:
    """Write an SAE as plain tensors and numbers (meta: JSON-like values: str, int, float, bool, lists, dicts)."""
    torch.save({"format": "cds-interpret.sae", "kind": type(sae).__name__, "k_store": getattr(sae, "k_store", None),
                "d": sae.d, "m": sae.m, "k": sae.k, "k_aux": sae.k_aux,
                "state": {k: v.detach().cpu() for k, v in sae.state_dict().items()}, "meta": meta}, path)


def load(path) -> TopKSAE:
    """An SAE written by save(). Loaded with torch.load(weights_only=True): no code in the file can run."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("kind") == "BatchTopKSAE":
        sae: TopKSAE = BatchTopKSAE(int(ck["d"]), int(ck["m"]), int(ck["k"]), ck.get("k_aux"), ck.get("k_store"))
    else:
        sae = TopKSAE(int(ck["d"]), int(ck["m"]), int(ck["k"]), ck.get("k_aux"))
    sae.load_state_dict(ck["state"])
    sae.meta = ck.get("meta", {})
    return sae.eval()
