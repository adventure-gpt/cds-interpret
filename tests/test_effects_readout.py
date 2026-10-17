"""Feature effects (exact single-feature ablations, rankings, attribution patching) and live readouts (calibration,
smoothing, flag steadiness)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cds_interpret import effects as EF
from cds_interpret import readout as RO
from cds_interpret import sae as S
from cds_interpret.hooks import tail
from cds_interpret.intervene import ablate_features


def _data(n=800, d=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    dirs = torch.randn(24, d, generator=g)
    on = (torch.rand(n, 24, generator=g) < 0.12).float() * torch.rand(n, 24, generator=g) * 3
    return (on @ dirs).relu()


def test_tail_matches_the_network():
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(5, 8), torch.nn.ReLU(), torch.nn.Linear(8, 3))
    x = torch.randn(4, 5)
    assert torch.allclose(tail(net, 1)(net[1](net[0](x))), net(x), atol=1e-6)


def test_feature_effects_are_exact_single_feature_ablations():
    x = _data()
    sae = S.TopKSAE(12, 48, 3).init_from(x[:400])
    S.train_sae(sae, x.numpy(), steps=150, batch=256, lr=3e-3)
    torch.manual_seed(2)
    down = torch.nn.Sequential(torch.nn.Linear(12, 16), torch.nn.Tanh(), torch.nn.Linear(16, 5))
    table = np.zeros((5, 8), np.float32)
    table[[1, 3], 6] = 1.0                                               # actions 1 and 3 boost
    fx = EF.feature_effects(sae, down, x[:50], table=table, batch=16)
    i = 7
    r = int(np.argmax(fx["z"][i]))
    j = int(fx["idx"][i, r])
    with torch.no_grad():
        lp0 = torch.log_softmax(down(x[i:i + 1]), -1)
        lp1 = torch.log_softmax(down(ablate_features(sae, [j])(x[i:i + 1])), -1)
    assert fx["kl"][i, r] == pytest.approx(float((lp0.exp() * (lp0 - lp1)).sum()), abs=1e-5)
    assert fx["dctl"][i, r, 6] == pytest.approx(float(lp0.exp()[0, [1, 3]].sum() - lp1.exp()[0, [1, 3]].sum()), abs=1e-5)
    per = EF.per_feature(fx, sae.m, min_count=1)
    assert j in per and per[j]["count"] >= 1
    valid = fx["z"] > 0
    rnd = np.argsort(np.random.default_rng(0).random(fx["kl"].shape) + ~valid * 10, 1)
    ja = EF.joint_ablation(sae, down, x[:50], fx, {"exact": np.argsort(-fx["kl"], 1), "random": rnd}, ks=(1, 2))
    assert ja["exact"]["kl@1"] >= ja["random"]["kl@1"]
    assert EF.spearman_rows(EF.attribution_patching(sae, down, x[:50], fx), fx["dlogp"], valid) > 0.5
    assert EF.explain(fx, i, k=1)[0][0] == int(fx["idx"][i, np.argmax(fx["kl"][i])])


def test_readouts_calibrate_smooth_and_steady_flags():
    rng = np.random.default_rng(0)
    n = 20000
    s = rng.normal(size=n)
    y = rng.random(n) < 1 / (1 + np.exp(-(3 * s - 4)))
    raw = np.clip(0.02 + 0.1 * s, 0, 1)                                  # a linear-probability-like score
    iso = RO.fit_isotonic(s[:10000], y[:10000])
    pl = RO.fit_platt(s[:10000], y[:10000])
    c_raw = RO.calibration(raw[10000:], y[10000:])["ece"]
    assert RO.calibration(RO.apply_isotonic(iso, s[10000:]), y[10000:])["ece"] < c_raw
    assert RO.calibration(RO.apply_platt(pl, s[10000:]), y[10000:])["ece"] < c_raw and abs(pl[0] - 3) < 0.4
    seq = np.repeat(np.arange(20), 500)
    order = np.tile(np.arange(500), 20)
    sig = np.sin(order / 20.0)
    x = sig + rng.normal(scale=0.3, size=len(sig))
    tr = RO.smoothing_tradeoff(x, sig, seq, order, [1.0, 0.3, 0.1], dt=0.2)
    assert tr[0]["jitter"] > tr[1]["jitter"] > tr[2]["jitter"] and tr[2]["lag_rows"] >= tr[0]["lag_rows"]
    f_h = RO.hysteresis(x, seq, order, on=0.5, off=0.1)
    assert RO.switches_per_minute(f_h, seq, order, 0.2) < RO.switches_per_minute(x > 0.5, seq, order, 0.2)
    assert RO.hold(x > 0.5, seq, order, 3).sum() >= (x > 0.5).sum()


def test_a_calibrated_pack_reads_platt_probabilities():
    from cds_interpret.overlay.pack import Thoughts, make_pack
    rng = np.random.default_rng(0)
    w = rng.normal(size=8).astype(np.float32)
    pack = make_pack("x", probes={"p": {"w": w, "b": 0.1, "kind": "bin"}})
    h = rng.normal(size=8).astype(np.float32)
    RO.calibrate_pack_probe(pack, "p", 3.0, -1.0)
    got = Thoughts(pack).read(h)["probes"]["p"]
    got = got["value"] if isinstance(got, dict) else got
    assert got == pytest.approx(1 / (1 + np.exp(-(3.0 * (h @ w + 0.1) - 1.0))), rel=1e-4)
