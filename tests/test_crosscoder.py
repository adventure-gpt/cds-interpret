"""The checkpoint crosscoder: a feature that only the last snapshot has comes out as "emerged"."""

from __future__ import annotations

import numpy as np
import torch

from cds_interpret import crosscoder as CC


def test_crosscoder_finds_a_feature_that_appears_later(tmp_path):
    g = torch.Generator().manual_seed(3)
    n, d, M = 4000, 10, 3
    shared = torch.randn(6, d, generator=g)
    late = torch.randn(4, d, generator=g)
    a_sh = (torch.rand(n, 6, generator=g) < 0.15).float() * torch.rand(n, 6, generator=g) * 2
    a_late = (torch.rand(n, 4, generator=g) < 0.15).float() * torch.rand(n, 4, generator=g) * 2
    X = torch.stack([a_sh @ shared + (a_late @ late if s == M - 1 else 0.0) + 0.01 * torch.randn(n, d, generator=g)
                     for s in range(M)], 1)                                # [n, M, d]
    torch.manual_seed(0)
    cc = CC.Crosscoder(M, d, 32, 3).init_from(X[:1000])

    def batches():
        rng = np.random.default_rng(0)
        while True:
            yield X[torch.as_tensor(rng.choice(n, 256, replace=False))]
    CC.train_crosscoder(cc, batches(), steps=600, lr=3e-3, log=None)
    assert (CC.fvu_by_model(cc, X) < 0.35).all()
    pres = CC.presence(cc.decoder_norms())
    z = cc.encode(X).detach().numpy()
    on_late = (a_late > 0).any(1).numpy()
    lift = (z[on_late] > 0).mean(0) / np.maximum((z[~on_late] > 0).mean(0), 1e-3)
    j = int(np.argmax(lift))
    assert pres["kind"][j] == "emerged" and pres["profile"][j, 0] < 0.5
    CC.save(cc, tmp_path / "cc.pt", note="test")
    cc2 = CC.load(tmp_path / "cc.pt")
    assert torch.allclose(cc2.encode(X[:50]), cc.encode(X[:50]))
