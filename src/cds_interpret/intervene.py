"""Changing what a layer computes - for reconstruction checks and causal tests.

Edits are functions layer-output -> new output, applied through forward hooks:

=====================================  ==================================================================
replace_with(sae)                      the SAE's reconstruction instead of the layer
ablate_features(sae, [j, ...])         remove those features' contributions only (the rest of the layer,
                                       the SAE's error included, is untouched): x - sum_j z_j(x) d_j
clamp_feature(sae, j, value)           set feature j to ``value`` where it would be lower ... or always
steer(direction, alpha, where)         x + alpha * direction, on all rows or those where(x) is true
project_out(direction)                 remove x's component along a direction (a probe's, say)
scale_direction(direction, factor)     x's component along a direction times ``factor`` (0 removes it)
mean_ablate(mean)                      the layer's mean activation for every row (the cost of losing it)
=====================================  ==================================================================

Measuring an edit on recorded inputs::

    effect = edit_effect(policy, x, "mlp.3", {"no threat": scale_direction(probe.direction(), 0.0)})
    effect["no threat"]["kl"], effect["no threat"]["argmax_changed"]

Playing whole games with an edit on (in RocketSim, or in the real game)::

    with editing(policy, "mlp.3", ablate_features(sae, [12])):
        moments = record.rocketsim(policy, games=40)          # every decision of these games is edited

Then compare the edited bot with the original over enough games: :func:`bootstrap_share` gives a goal share with
an interval over games, :func:`rate_ratio_ci` compares how often something happens.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .errors import ShapeError
from .hooks import LayerTap

Edit = Callable[[torch.Tensor], torch.Tensor]


def replace_with(sae: Any) -> Edit:
    """The layer replaced by the SAE's reconstruction of it."""
    def edit(x: torch.Tensor) -> torch.Tensor:
        return sae(x).to(x.dtype)
    return edit


def ablate_features(sae: Any, features: Any, scale: float = 0.0) -> Edit:
    """Each listed feature's contribution z_j d_j removed (scale 0) or multiplied by ``scale`` (2 doubles it)."""
    feats = torch.as_tensor(list(features), dtype=torch.long)

    def edit(x: torch.Tensor) -> torch.Tensor:
        f = feats.to(x.device)
        z = sae.encode(x)[:, f]                                           # [n, f] scaled units
        dirs = sae.W_dec[f] * sae.scale                                   # [f, d] layer units
        return (x.float() + (scale - 1.0) * (z @ dirs)).to(x.dtype)
    return edit


def clamp_feature(sae: Any, j: int, value: float, mode: str = "max") -> Edit:
    """Feature j's activation set to ``value`` (the SAE's scaled units): mode "max" raises it to value where lower,
    "set" sets it everywhere (both add the difference along its decoder direction)."""
    if mode not in ("max", "set"):
        raise ValueError(f"mode must be 'max' or 'set', not {mode!r}")

    def edit(x: torch.Tensor) -> torch.Tensor:
        z = sae.encode(x)[:, j]
        target = torch.clamp_min(z, value) if mode == "max" else torch.full_like(z, value)
        return (x.float() + (target - z)[:, None] * (sae.W_dec[j] * sae.scale)[None]).to(x.dtype)
    return edit


def _unit(direction: Any) -> torch.Tensor:
    d = torch.as_tensor(direction).float().reshape(-1)
    return d / d.norm().clamp_min(1e-12)


def _fits(d: torch.Tensor, x: torch.Tensor) -> None:
    if x.shape[-1] != d.numel():
        raise ShapeError(f"This direction has {d.numel()} numbers, but the layer has {x.shape[-1]}. Make the "
                         "direction from the same layer you edit (a probe or feature of that layer).")


def steer(direction: Any, alpha: float, where: Callable[[torch.Tensor], torch.Tensor] | None = None) -> Edit:
    """x + alpha * direction (the direction as given, not normalised), on every row or where ``where(x)`` holds."""
    d = torch.as_tensor(direction).float().reshape(-1)

    def edit(x: torch.Tensor) -> torch.Tensor:
        _fits(d, x)
        add = alpha * d.to(x.device)[None].expand_as(x)
        if where is not None:
            add = add * where(x).float()[:, None]
        return (x.float() + add).to(x.dtype)
    return edit


def project_out(direction: Any) -> Edit:
    """x's component along the direction removed."""
    u = _unit(direction)

    def edit(x: torch.Tensor) -> torch.Tensor:
        _fits(u, x)
        xf = x.float()
        uu = u.to(x.device)
        return (xf - (xf @ uu)[:, None] * uu[None]).to(x.dtype)
    return edit


def scale_direction(direction: Any, factor: float) -> Edit:
    """x's component along a direction (a probe's, an SAE feature's decoder row) multiplied by ``factor``; the rest
    of x is untouched. factor 0 is project_out; 1 changes nothing."""
    u = _unit(direction)

    def edit(x: torch.Tensor) -> torch.Tensor:
        _fits(u, x)
        xf = x.float()
        uu = u.to(x.device)
        return (xf + (factor - 1.0) * (xf @ uu)[:, None] * uu[None]).to(x.dtype)
    return edit


def mean_ablate(mean: Any) -> Edit:
    """The layer's mean activation (``mean`` [D]) for every row."""
    m = torch.as_tensor(mean).float()

    def edit(x: torch.Tensor) -> torch.Tensor:
        return m.to(x.device, x.dtype)[None].expand_as(x).clone()
    return edit


def _net(policy_or_model: Any) -> torch.nn.Module:
    return getattr(policy_or_model, "net", policy_or_model)


def attach_edits(policy_or_model: Any, layer_path: str, edit: Edit) -> LayerTap:
    """The edit stays on the network (every forward pass) until ``.detach_hooks()``."""
    return LayerTap(_net(policy_or_model), {"edited": layer_path}, edits={"edited": edit}, read=False).attach()


@contextmanager
def editing(policy_or_model: Any, layer_path: str, edit: Edit) -> Iterator[Any]:
    """Within the ``with``, every forward pass of the network runs with the edit on its layer."""
    tap = attach_edits(policy_or_model, layer_path, edit)
    try:
        yield policy_or_model
    finally:
        tap.detach_hooks()


@torch.no_grad()
def policy_kl(model: torch.nn.Module, forward: Callable, x: torch.Tensor, layer_path: str, edits: dict[str, Edit],
              batch: int = 4096) -> dict[str, dict[str, float]]:
    """For each named edit: mean KL(original policy || edited policy) over rows x (``forward`` returns logits), the
    share of rows whose argmax action changes, and the mean total-variation distance."""
    out = {name: {"kl": 0.0, "argmax_changed": 0.0, "tv": 0.0} for name in edits}
    n = len(x)
    for i in range(0, n, batch):
        xb = x[i:i + batch]
        lp0 = F.log_softmax(forward(xb).float(), -1)
        for name, edit in edits.items():
            with LayerTap(model, {"e": layer_path}, edits={"e": edit}, read=False):
                lp1 = F.log_softmax(forward(xb).float(), -1)
            _accumulate(out[name], lp0, lp1)
    return _mean(out, n)


def _accumulate(acc: dict[str, float], lp0: torch.Tensor, lp1: torch.Tensor) -> None:
    acc["kl"] += float((lp0.exp() * (lp0 - lp1)).sum(-1).sum())
    acc["argmax_changed"] += float((lp0.argmax(-1) != lp1.argmax(-1)).float().sum())
    acc["tv"] += float(0.5 * (lp0.exp() - lp1.exp()).abs().sum(-1).sum())


def _mean(out: dict[str, dict[str, float]], n: int) -> dict[str, dict[str, float]]:
    for v in out.values():
        for k in v:
            v[k] /= max(n, 1)
    return out


@torch.no_grad()
def edit_effect(policy: Any, x: Any, layer_path: str, edits: dict[str, Edit], batch: int = 4096) -> dict[str, dict[str, float]]:
    """How far each edit moves a :class:`~cds_interpret.Policy` on inputs x. A discrete policy: mean KL(original ||
    edited), the share of moments whose argmax action changes, mean total-variation distance. A continuous one
    (output="controls"): the mean absolute change of the controls and the share of moments where any control
    moves by more than 0.5."""
    xt = policy.as_input(x)
    n = len(xt)
    if policy.discrete:
        out = {name: {"kl": 0.0, "argmax_changed": 0.0, "tv": 0.0} for name in edits}
    else:
        out = {name: {"control_change": 0.0, "changed": 0.0} for name in edits}
    for i in range(0, n, batch):
        xb = xt[i:i + batch]
        if policy.discrete:
            lp0 = policy.log_probs(xb)
        else:
            c0 = policy.forward(xb).float().clamp(-1, 1)
        for name, edit in edits.items():
            with editing(policy, layer_path, edit):
                if policy.discrete:
                    _accumulate(out[name], lp0, policy.log_probs(xb))
                else:
                    d = (policy.forward(xb).float().clamp(-1, 1) - c0).abs()
                    out[name]["control_change"] += float(d.mean(-1).sum())
                    out[name]["changed"] += float((d.max(-1).values > 0.5).float().sum())
    return _mean(out, n)


def rate_ratio_ci(a: float, na: float, b: float, nb: float, z: float = 1.96) -> tuple[float, float, float]:
    """Ratio of two Poisson rates (a events in na units vs b in nb) with a normal-approximation CI on the log scale
    (0.5 added to empty counts): (ratio, low, high)."""
    a_, b_ = max(a, 0.5), max(b, 0.5)
    rr = (a_ / na) / (b_ / nb)
    se = np.sqrt(1.0 / a_ + 1.0 / b_)
    return float(rr), float(rr * np.exp(-z * se)), float(rr * np.exp(z * se))


def bootstrap_share(goals_for: Any, goals_against: Any, n_boot: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """Goal share sum(for) / sum(for + against) over games, with a 95% bootstrap interval over games:
    (share, low, high)."""
    gf, ga = np.asarray(goals_for, np.float64), np.asarray(goals_against, np.float64)
    share = float(gf.sum()) / max(float(gf.sum() + ga.sum()), 1e-9)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(gf), (n_boot, len(gf)))
    f, a = gf[idx].sum(1), ga[idx].sum(1)
    s = f / np.maximum(f + a, 1e-9)
    return float(share), float(np.quantile(s, 0.025)), float(np.quantile(s, 0.975))
