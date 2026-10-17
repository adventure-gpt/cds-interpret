"""Concept erasure (LEACE) with its placebo control, and the non-linear probe."""

from __future__ import annotations

import numpy as np
import torch

from cds_interpret import erasure as ER
from cds_interpret.probes import auroc, mlp_probe, ridge_suite


def _layer(n=4000, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = (torch.rand(n, generator=g) < 0.3).float()
    x = torch.randn(n, d, generator=g) @ torch.randn(d, d, generator=g) + 2.0 * y[:, None] * torch.randn(1, d, generator=g)
    return x, y


def _ls_auroc(xtr, ytr, xte, yte):
    w = torch.linalg.lstsq(torch.cat([xtr, torch.ones(len(xtr), 1)], 1), ytr[:, None]).solution
    return auroc((torch.cat([xte, torch.ones(len(xte), 1)], 1) @ w)[:, 0].numpy(), yte.numpy())


def test_leace_removes_every_linear_read_out():
    x, y = _layer()
    er = ER.fit_leace(x[:3000], y[:3000].numpy())
    assert er.rank == 1
    assert _ls_auroc(x[:3000], y[:3000], x[3000:], y[3000:]) > 0.9
    xe = er(x)
    assert abs(_ls_auroc(xe[:3000], y[:3000], xe[3000:], y[3000:]) - 0.5) < 0.06
    assert torch.allclose(er(xe), xe, atol=1e-4)                        # erasing twice changes nothing more


def test_the_controls_have_the_same_rank_and_a_sensible_size():
    x, _ = _layer()
    for mode in ("feature", "whitened"):
        c = ER.random_eraser(x[:3000], rank=2, mode=mode)
        assert c.rank == 2 and 0 < ER.edit_size(c, x) < 1
    assert ER.edit_size(ER.random_eraser(x, 1, mode="feature"), x) > ER.edit_size(ER.random_eraser(x, 1, mode="whitened"), x)


def test_mlp_probe_reads_what_a_linear_probe_cannot():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(6000, 6, generator=g)
    y = (x[:, 0] * x[:, 1]).numpy().astype(np.float32)
    tr, te = np.arange(5000), np.arange(5000, 6000)
    assert ridge_suite(x, {"y": y}, {"y": "reg"}, tr, te)["y"]["r2"] < 0.05
    assert mlp_probe(x, y, "reg", tr, te, hidden=64, epochs=60, lr=3e-3)["r2"] > 0.6
