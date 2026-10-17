"""BatchTopK and policy-aware (KL) dictionaries, and the dictionary scores (dict_eval)."""

from __future__ import annotations

import numpy as np
import torch

from cds_interpret import dict_eval as DE
from cds_interpret import sae as S


def _data(n=3000, d=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    dirs = torch.randn(24, d, generator=g)
    on = (torch.rand(n, 24, generator=g) < 0.12).float() * torch.rand(n, 24, generator=g) * 3
    return (on @ dirs).relu()


def test_kl_finetune_brings_the_reconstruction_closer_to_the_policy(tmp_path):
    x = _data()
    torch.manual_seed(1)
    down = torch.nn.Sequential(torch.nn.Linear(12, 16), torch.nn.Tanh(), torch.nn.Linear(16, 6))
    for p in down.parameters():
        p.requires_grad_(False)
    sae = S.TopKSAE(12, 48, 3).init_from(x[:500])
    S.train_sae(sae, x.numpy(), steps=300, batch=256, lr=3e-3)
    before = S.downstream_check(sae, down, x[2000:])
    assert 0 <= before["fvu"] < 1 and before["l0"] <= 3 + 1e-6
    S.finetune_kl(sae, x.numpy(), down, steps=300, batch=256, lr=1e-3, kl_coef=5.0)
    assert S.downstream_check(sae, down, x[2000:])["kl"] < before["kl"]


def test_batchtopk_varies_sparsity_by_row_and_saves(tmp_path):
    x = _data()
    bt = S.BatchTopKSAE(12, 48, 3).init_from(x[:500])
    S.train_sae(bt, x.numpy(), steps=300, batch=256, lr=3e-3)
    assert float(bt.theta) > 0
    l0 = (bt.encode(x[2000:]) > 0).float().sum(1)
    assert 1.0 < float(l0.mean()) < 8.0 and float(l0.std()) > 0
    idx, _ = S.encode_all(bt, x[:100])
    assert idx.shape == (100, bt.k_store)
    S.save(bt, tmp_path / "bt.pt")
    bt2 = S.load(tmp_path / "bt.pt")
    assert isinstance(bt2, S.BatchTopKSAE) and torch.allclose(bt2.encode(x[:20]), bt.encode(x[:20]))


def test_dictionary_scores():
    x = _data()
    a = S.TopKSAE(12, 48, 3).init_from(x[:500])
    S.train_sae(a, x.numpy(), steps=200, batch=256, lr=3e-3)
    st = DE.stability(a, a)
    assert st["summary"]["share_cos_0.9"] == 1.0 and st["summary"]["share_mutual_best"] == 1.0
    idx, val = S.encode_all(a, x)
    idx, val = idx.numpy(), val.float().numpy()
    j = int(np.bincount(idx[val > 0].ravel()).argmax())
    on = ((idx == j) & (val > 0)).any(1)
    labels = {"is_j": on.astype(np.float32), "noise": np.random.default_rng(0).random(len(x)).astype(np.float32)}
    al = DE.label_alignment(idx, val, labels, {"is_j": "bin", "noise": "reg"}, a.m)
    assert al["per_label"]["is_j"]["best_auroc"] > 0.99 and al["per_label"]["is_j"]["feature_auroc"] == j
    lv = DE.liveness(idx, val, a.m)
    assert lv["dead"] + (lv["freq"] > 0).sum() == a.m


def test_activation_match_finds_a_dictionary_in_itself():
    x = _data(1500)
    a = S.TopKSAE(12, 48, 3).init_from(x[:500])
    S.train_sae(a, x.numpy(), steps=100, batch=256, lr=3e-3)
    feats = [int(j) for j in np.nonzero(S.latent_stats(a, x)["freq"].numpy() > 0.01)[0][:5]]
    r, j = DE.activation_match(a, a, x, feats)
    assert np.allclose(r, 1.0, atol=1e-6) and list(j) == feats
