"""Entity attribution: swaps by role, the Frames codec's round trip through a policy's obs function, exact Shapley,
faithfulness scores."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cds_interpret import attribution as AT
from cds_interpret.frames import canon, team_sign
from cds_interpret.policy import Policy

from conftest import make_frames


def _obs(frames, me):
    """A tiny team-frame observation: the ball, the bot's car, every other car relative to it (as many bots do)."""
    n = len(frames)
    rows = np.arange(n)
    s = team_sign(frames.car_team[rows, me])
    ball = canon(frames.ball_pos, s) / 1000.0
    mine = canon(frames.car_pos[rows, me], s) / 1000.0
    others = canon(frames.car_pos, s[:, None]) / 1000.0 - mine[:, None]
    same = (np.abs(frames.car_team - frames.car_team[rows, me][:, None]) < 0.5).astype(np.float64)
    return np.concatenate([ball, mine, frames.car_boost[rows, me][:, None] / 100.0,
                           (others * frames.car_present[..., None]).reshape(n, -1), same], 1).astype(np.float32)


def _moments(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return make_frames(n, cars=2, ball_pos=rng.normal(size=(n, 3)) * 1000, car_pos=rng.normal(size=(n, 2, 3)) * 1000,
                       car_vel=rng.normal(size=(n, 2, 3)) * 500, car_boost=rng.uniform(0, 100, (n, 2)))


def test_codec_round_trip_and_swaps():
    fr = _moments()
    me = np.where(np.arange(len(fr)) % 2 == 0, 0, 1)                   # blue and orange moments
    pol = Policy(torch.nn.Linear(15, 3), obs=_obs)
    codec = AT.FramesCodec(pol)
    parts = codec.decompose(fr, me)
    assert torch.allclose(codec.compose(parts), pol.observe(fr, me), atol=1e-5)
    # the ball of moment 1 (orange) into moment 0 (blue): moment 0 sees it where moment 1's car saw it
    g = codec.groups()
    one = {k: v[[0]] for k, v in parts.items()}
    don = {k: v[[1]] for k, v in parts.items()}
    sw = codec.compose(AT.swap(one, don, g["ball"]))[0]
    assert torch.allclose(sw[:3], pol.observe(fr.take([1]), me[[1]])[0, :3], atol=1e-5)
    assert torch.allclose(sw[3:], pol.observe(fr.take([0]), me[[0]])[0, 3:], atol=1e-5)
    # the opponent swapped: only the other-car columns move
    so = codec.compose(AT.swap(one, don, g["opponents"]))[0]
    base = pol.observe(fr.take([0]), me[[0]])[0]
    assert torch.allclose(so[:10], base[:10], atol=1e-5) and not torch.allclose(so[10:13], base[10:13])


def test_shapley_from_the_coalition_table_is_exact():
    fr = _moments(60)
    me = np.zeros(len(fr), np.int64)
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(15, 16), torch.nn.Tanh(), torch.nn.Linear(16, 4))
    pol = Policy(net, obs=_obs)
    codec = AT.FramesCodec(pol)
    parts = codec.decompose(fr, me)
    donors = AT.pick_donors(AT.role_signature(parts), 2, seed=0, pool=np.arange(30, 60))[:, :30]
    groups = codec.groups()
    tab = AT.coalition_table(pol.log_probs, parts, donors, groups, codec.compose, batch=7)
    assert tab.shape == (2, 30, 2 ** len(groups), 4)
    phi = AT.shapley(tab, mean_donors=False)
    assert np.allclose(phi.sum(2), tab[:, :, -1] - tab[:, :, 0], atol=1e-4)
    kl = AT.kl_table(tab)
    assert np.allclose(kl[:, :, -1], 0, atol=1e-6) and (kl >= -1e-6).all()
    # team-mates: none in these 1v1 moments, so their Shapley value is exactly 0
    assert np.abs(AT.shapley(-kl[..., None])[:, list(groups).index("team-mates")]).max() < 1e-6
    phi_kl = AT.shapley(-kl[..., None])[..., 0]
    res = AT.faithfulness(kl, {"shapley": AT.order_from(phi_kl),
                               "random": np.stack([np.random.default_rng(i).permutation(len(groups)) for i in range(30)])},
                          average_donors=True)
    assert res["shapley"]["score_pooled"] > res["random"]["score_pooled"]


def test_faithfulness_is_one_for_the_true_order():
    rng = np.random.default_rng(0)
    G, n = 4, 50
    w = rng.uniform(0.1, 3.0, (n, G))
    metric = np.zeros((n, 2 ** G))
    for s in range(2 ** G):
        metric[:, s] = w[:, [g for g in range(G) if not (s >> g) & 1]].sum(1)
    res = AT.faithfulness(metric, {"good": AT.order_from(w), "bad": AT.order_from(-w, absolute=False)})
    assert res["good"]["score"] == pytest.approx(1.0) and res["bad"]["score"] < 0


def test_integrated_gradients_completeness():
    torch.manual_seed(1)
    net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    x, b = torch.randn(5, 4), torch.randn(5, 4)
    ig = AT.integrated_gradients(lambda z: net(z)[:, 0], x, b, steps=64)
    assert torch.allclose(ig.sum(1), (net(x) - net(b))[:, 0].detach(), atol=2e-3)


def test_study_attribution_on_the_demo_bot(tmp_path):
    import cds_interpret as ci
    pol = ci.demo.policy()
    m = ci.demo.synthetic_moments(n_games=16, per_game=40)
    study = ci.Study(pol, m, log=None)
    res = study.attribution(n=60, donors=2, pool=300)
    sh = res["shares"]["all moments"]
    assert set(sh) == {"own car", "own boost", "ball", "opponents", "team-mates"}
    assert abs(sum(sh.values()) - 1.0) < 1e-3                        # Shapley: the shares add up to the whole change
    assert res["faithfulness"]["Shapley"] > res["faithfulness"]["random"]
    page = study.report(tmp_path / "r.html").read_text(encoding="utf-8")
    assert "What decisions rest on" in page
