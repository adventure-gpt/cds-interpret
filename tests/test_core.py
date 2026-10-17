"""The game-independent pieces on tiny models and hand-made states: hooks, frames, labels, store, probes, SAE, edits,
analysis, behaviour counts and figures."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cds_interpret import analysis, intervene, probes
from cds_interpret import frames as F
from cds_interpret import labels as L
from cds_interpret import sae as S
from cds_interpret.errors import CDSInterpretError, LayerNotFound, ShapeError
from cds_interpret.hooks import LayerTap, default_layers, list_layers, run_layers, sequential_layers
from cds_interpret.store import MomentStore

from conftest import make_frames, tiny_mlp


# ------------------------------------------------------------------------------------------------- hooks
def test_layertap_reads_and_detaches():
    m = tiny_mlp()
    x = torch.randn(5, 6)
    ref = m(x)
    with LayerTap(m, {"h1": "1", "h2": "3"}) as tap:
        out = m(x)
    assert torch.equal(out, ref)                                   # a read-only tap changes nothing, bit for bit
    assert torch.allclose(tap.out["h1"], torch.relu(m[0](x)))
    assert tap.out["h2"].shape == (5, 8)
    with LayerTap(m, {"in2": "2"}, inputs=True) as tap:
        m(x)
    assert torch.allclose(tap.out["in2"], torch.relu(m[0](x)))
    assert not any(mod._forward_hooks or mod._forward_pre_hooks for mod in m.modules())
    assert torch.equal(m(x), ref)
    with LayerTap(m, {"x": ""}, inputs=True) as tap:              # "" is the whole network
        m(x)
    assert torch.equal(tap.out["x"], x)


def test_layertap_edits_and_friendly_errors():
    m = tiny_mlp()
    x = torch.randn(5, 6)
    with LayerTap(m, {"h2": "3"}, edits={"h2": lambda h: torch.zeros_like(h)}):
        out = m(x)
    assert torch.allclose(out, m[4].bias.expand(5, -1))
    acts = run_layers(m, m, x, {"h1": "1"}, inputs={"in4": "4"}, batch=2)
    assert acts["h1"].shape == (5, 16) and acts["in4"].shape == (5, 8)
    with pytest.raises(LayerNotFound, match="Did you mean"):
        LayerTap(m, {"h": "33"})
    with pytest.raises(LayerNotFound):
        LayerTap(m, {"h": "1"}, edits={"other": lambda h: h})


def test_any_network_gets_layer_names():
    outer = torch.nn.Module()
    outer.mlp = tiny_mlp()
    outs, ins = sequential_layers(outer.mlp, "mlp", "a")
    assert outs == {"a.h1": "mlp.1", "a.h2": "mlp.3", "a.out": "mlp.4"} and ins == {"a.in": "mlp.0"}
    x = torch.randn(7, 6)
    acts = run_layers(outer, lambda o: outer.mlp(o), x, outs, inputs=ins)
    assert acts["a.h2"].shape == (7, 8) and torch.allclose(acts["a.in"], x)
    infos = list_layers(outer, x, forward=lambda o: outer.mlp(o))
    shapes = {i.path: i.shape for i in infos}
    assert shapes["mlp.1"] == (7, 16) and shapes["mlp.4"] == (7, 4)
    assert default_layers(infos) == ["mlp.1", "mlp.3"]


# ------------------------------------------------------------------------------------------------ frames
def test_frames_and_canon():
    fr = make_frames(3)
    assert len(fr) == 3 and fr.n_cars == 2 and "3 moments" in repr(fr)
    assert fr.take([0, 2]).car_pos.shape == (2, 2, 3)
    v = np.array([[100.0, 200.0, 30.0]])
    assert np.allclose(F.canon(v, F.team_sign(np.array([1.0]))), [[-100.0, -200.0, 30.0]])
    with pytest.raises(ShapeError, match="lacks the fields"):
        F.Frames({"ball_pos": np.zeros((2, 3))})
    with pytest.raises(AttributeError, match="optional"):
        fr.goal
    two = F.Frames.concat([fr, fr])
    assert len(two) == 6


def test_from_rlgym_v1_and_v2():
    def phys(p):
        return SimpleNamespace(position=np.array(p, float), linear_velocity=np.zeros(3), angular_velocity=np.zeros(3),
                               forward=lambda: np.array([1.0, 0, 0]), up=lambda: np.array([0, 0, 1.0]))
    v1 = SimpleNamespace(ball=SimpleNamespace(position=np.array([0, 0, 93.0]), linear_velocity=np.zeros(3),
                                              angular_velocity=np.zeros(3)),
                         players=[SimpleNamespace(car_data=phys([0, -1000, 17]), boost_amount=0.33, on_ground=True,
                                                  is_demoed=False, has_flip=True, team_num=0, ball_touched=False)])
    fr = F.from_rlgym([v1, v1])
    assert fr.car_boost[0, 0] == pytest.approx(33.0) and fr.car_pos[1, 0, 1] == -1000
    car2 = SimpleNamespace(physics=SimpleNamespace(position=np.array([5.0, 6, 17]), linear_velocity=np.zeros(3),
                                                   angular_velocity=np.zeros(3), forward=np.array([1.0, 0, 0]),
                                                   up=np.array([0, 0, 1.0])),
                           boost_amount=40.0, on_ground=True, is_demoed=False, can_flip=True, team_num=1, ball_touches=1)
    v2 = SimpleNamespace(ball=v1.ball, cars={"a": car2})
    fr2 = F.from_rlgym([v2])
    assert fr2.car_boost[0, 0] == 40.0 and fr2.car_team[0, 0] == 1 and fr2.ev_touch[0, 0] == 1


def fake_packet(ball=(0.0, 0.0, 93.0), cars=((0.0, -1000.0, 17.0, 0), (0.0, 1000.0, 17.0, 1)), t=1.0, scores=(0, 0),
                touches=None, demos=None):
    def vec(x, y, z):
        return SimpleNamespace(x=x, y=y, z=z)

    players = []
    for j, (x, y, z, team) in enumerate(cars):
        players.append(SimpleNamespace(
            physics=SimpleNamespace(location=vec(x, y, z), velocity=vec(0, 0, 0), angular_velocity=vec(0, 0, 0),
                                    rotation=SimpleNamespace(pitch=0.0, yaw=np.pi / 2, roll=0.0)),
            boost=33.0, air_state=0, demolished_timeout=-1.0, dodge_timeout=-1.0, is_supersonic=False, team=team,
            latest_touch=None if not touches or touches[j] is None else SimpleNamespace(game_seconds=touches[j]),
            score_info=SimpleNamespace(demolitions=0 if not demos else demos[j]),
            last_input=SimpleNamespace(throttle=1.0, steer=0.0, pitch=0.0, yaw=0.0, roll=0.0, jump=False, boost=True,
                                       handbrake=False)))
    return SimpleNamespace(
        balls=[SimpleNamespace(physics=SimpleNamespace(location=vec(*ball), velocity=vec(0, 0, 0),
                                                       angular_velocity=vec(0, 0, 0)))],
        players=players, match_info=SimpleNamespace(seconds_elapsed=t, frame_num=int(t * 120),
                                                    match_phase="MatchPhase.Active"),
        teams=[SimpleNamespace(score=scores[0]), SimpleNamespace(score=scores[1])])


def test_from_rlbot_packet():
    fr = F.from_rlbot_packet(fake_packet())
    assert len(fr) == 1 and fr.n_cars == 2
    assert np.allclose(fr.car_fwd[0, 0], [0, 1, 0], atol=1e-6) and np.allclose(fr.car_up[0, 0], [0, 0, 1], atol=1e-6)
    assert fr.car_boost[0, 1] == 33.0 and fr.car_team[0, 1] == 1 and fr.time[0] == 1.0


# ------------------------------------------------------------------------------------------------ labels
def test_kickoff_threat_landing():
    fr = make_frames(1, car_pos=np.array([[[0.0, -4000, 17], [0.0, 4000, 17]]]))
    lab = L.concept_labels(fr, np.array([0]))
    assert lab["kickoff"][0] == 1 and lab["land_t"][0] == 0 and lab["threat"][0] == 0
    fr = make_frames(2, ball_pos=np.tile([0.0, -3000, 300], (2, 1)), ball_vel=np.tile([0.0, -2500, 0], (2, 1)),
                     car_pos=np.tile([[0.0, 0, 17], [0.0, 2000, 17]], (2, 1, 1)))
    lab = L.concept_labels(fr, np.array([0, 1]))
    assert lab["threat"][0] == 1 and lab["on_target"][0] == 0
    assert lab["threat"][1] == 0 and lab["on_target"][1] == 1
    assert 0.0 < lab["land_t"][0] < 1.5
    fr = make_frames(1, ball_pos=np.array([[0.0, 0, 1000]]))
    lab = L.concept_labels(fr, 0)
    assert lab["land_t"][0] == pytest.approx(np.sqrt(2 * (1000 - 92.75) / 650), abs=0.08)
    with pytest.raises(ShapeError, match="car slots"):
        L.concept_labels(fr, 5)


def test_labels_are_the_same_from_either_side():
    """A moment and its mirror (teams swapped, field turned half a turn) give each car the same labels."""
    rng = np.random.default_rng(0)
    n = 40
    pos = rng.uniform([-3000, -4000, 17], [3000, 4000, 17], (n, 2, 3))
    vel = rng.normal(0, 900, (n, 2, 3)) * [1, 1, 0]
    bp = rng.uniform([-3000, -4000, 93], [3000, 4000, 800], (n, 3))
    bv = rng.normal(0, 1500, (n, 3))
    flip = np.array([-1.0, -1.0, 1.0])
    a = make_frames(n, car_pos=pos, car_vel=vel, ball_pos=bp, ball_vel=bv)
    b = make_frames(n, car_pos=pos[:, ::-1] * flip, car_vel=vel[:, ::-1] * flip, ball_pos=bp * flip, ball_vel=bv * flip,
                    car_fwd=np.tile([0.0, -1.0, 0.0], (n, 2, 1)))
    la, lb = L.concept_labels(a, 0), L.concept_labels(b, 1)
    for k in ("threat", "on_target", "land_x", "land_y", "goal_side", "ball_vy", "ball_dist"):
        assert np.allclose(np.nan_to_num(la[k]), np.nan_to_num(lb[k]), atol=1e-3), k


def test_last_back_and_demo_chance():
    pos = np.array([[[0.0, -3000, 17], [0.0, -1000, 17], [0.0, 1000, 17], [0.0, 3000, 17]]])
    fr = make_frames(1, cars=4, car_pos=pos)
    lb = [L.concept_labels(fr, np.array([c]))["last_back"][0] for c in range(4)]
    assert lb == [1, 0, 0, 1]
    pos = np.array([[[0.0, -2000, 17], [0.0, -1000, 17]]])
    vel = np.array([[[0.0, 2250, 0], [0.0, 0, 0]]])
    far_ball = np.array([[3000.0, 3000, 93]])
    fr = make_frames(1, car_pos=pos, car_vel=vel, car_supersonic=np.array([[1.0, 0]]), ball_pos=far_ball)
    assert L.concept_labels(fr, np.array([0]))["demo_chance"][0] == 1
    fr = make_frames(1, car_pos=pos, car_vel=-vel, ball_pos=far_ball)
    assert L.concept_labels(fr, np.array([0]))["demo_chance"][0] == 0
    fr = make_frames(1, car_pos=pos, car_vel=vel, car_supersonic=np.array([[1.0, 0]]), ball_pos=np.array([[0.0, 0, 92.75]]))
    lab = L.concept_labels(fr, np.array([0]))
    assert lab["kickoff"][0] == 1 and lab["demo_chance"][0] == 0 and np.isnan(lab["demo_ttc"][0])


def test_soon_windows_and_future_labels():
    goal = np.array([-1, -1, -1, 0, -1, -1, -1, -1])
    ends = L.segment_ends(goal)
    assert list(ends) == [3, 3, 3, 3, 7, 7, 7, 7]
    ev = np.array([0, 0, 1, 0, 0, 1, 0, 0], np.float32)
    out = L._soon(ev, ends, 2)
    assert list(out[:7]) == [0, 1, 1, 0, 1, 1, 0] and np.isnan(out[7])
    t = 120
    fr = make_frames(t)
    fr.data.update({"ev_touch": np.zeros((t, 2), np.float32), "ev_demo_made": np.zeros((t, 2), np.float32),
                    "ev_demoed": np.zeros((t, 2), np.float32), "goal": np.full(t, -1)})
    fr.data["goal"][30] = 0
    fut = L.future_labels(fr, 0, 1 / 15)
    assert fut["goal_soon"][29] == 1 and fut["goal_soon"][31] == 0 and fut["outlook"][29] == 1
    assert np.isnan(fut["goal_soon"][-1])                          # the recording ends inside its window
    assert L.future_labels(fr, 1, 1 / 15)["concede_soon"][29] == 1
    with pytest.raises(ShapeError, match="events"):
        L.future_labels(make_frames(3), 0, 1 / 15)


def test_register_label():
    @L.register_label("ball_in_our_half", kind="bin", doc="the ball is in our half")
    def _(inp):
        return inp.ball[:, 1] < 0
    try:
        fr = make_frames(2, ball_pos=np.array([[0.0, -500, 93], [0.0, 500, 93]]))
        assert list(L.concept_labels(fr, 0)["ball_in_our_half"]) == [1, 0]
        assert list(L.concept_labels(fr, 1)["ball_in_our_half"]) == [0, 1]      # orange's half is +y
        assert L.LABEL_KIND["ball_in_our_half"] == "bin" and L.describe_label("ball_in_our_half").startswith("the ball")
        with pytest.raises(ValueError, match="already exists"):
            L.register_label("threat", "bin")(lambda inp: inp.ball[:, 0])
    finally:
        L.unregister_label("ball_in_our_half")
    assert "ball_in_our_half" not in L.concept_labels(make_frames(1), 0)


# ------------------------------------------------------------------------------------------------ store
def test_store_roundtrip(tmp_path):
    st = MomentStore(tmp_path / "m")
    st.write_shard({"a": np.arange(5), "x": np.arange(10.0).reshape(5, 2)}, meta={"k": 1})
    st.write_shard({"a": np.arange(5, 8), "x": np.zeros((3, 2))})
    st2 = MomentStore(tmp_path / "m")
    assert len(st2) == 8 and st2.shard_meta()[0]["k"] == 1
    assert list(st2.load(["a"], rows=np.array([6, 1, 7]))["a"]) == [6, 1, 7]
    assert [s for s, _ in st2.iter(["a"], batch=2)] == [0, 2, 4, 5, 7]
    with pytest.raises(CDSInterpretError):
        st2.write_shard({"a": np.arange(2)})
    other = MomentStore(tmp_path / "o")
    other.write_shard({"a": np.arange(2), "x": np.ones((2, 2))}, meta={"k": 2})
    assert st2.adopt(other) == ["shard_00002"] and len(MomentStore(tmp_path / "m")) == 10


# ------------------------------------------------------------------------------------------------ probes
def test_probe_metrics_and_split():
    rng = np.random.default_rng(0)
    s, y = rng.normal(size=300), rng.random(300) < 0.3
    brute = np.mean([[float(a > b) + 0.5 * float(a == b) for b in s[~y]] for a in s[y]])
    assert probes.auroc(s, y) == pytest.approx(brute)
    groups = np.repeat(np.arange(20), 10)
    tr, te = probes.group_split(groups, 0.25)
    assert not set(groups[tr]) & set(groups[te]) and len(tr) + len(te) == 200
    with pytest.raises(CDSInterpretError, match="needs at least 2"):
        probes.group_split(np.zeros(10), 0.2)


def test_ridge_suite_finds_linear_concepts():
    rng = np.random.default_rng(1)
    x = torch.from_numpy(rng.normal(size=(4000, 20)).astype(np.float32))
    w = rng.normal(size=20)
    lab = {"lin": (x.numpy() @ w).astype(np.float32), "noise": rng.normal(size=4000).astype(np.float32),
           "bin": ((x.numpy() @ w) > 1.0).astype(np.float32)}
    lab["lin"][:100] = np.nan
    groups = np.repeat(np.arange(400), 10)
    tr, te = probes.group_split(groups, 0.2)
    res = probes.ridge_suite(x, lab, {"lin": "reg", "noise": "reg", "bin": "bin"}, tr, te, groups=groups, keep=True)
    assert res["lin"]["r2"] > 0.99 and abs(res["noise"]["r2"]) < 0.05 and res["bin"]["auroc"] > 0.97
    p = res["lin"]["probe"]
    assert np.corrcoef(p.score(x).numpy()[100:], lab["lin"][100:])[0, 1] > 0.999
    res2 = probes.probe_layer(x.numpy(), {"bin": lab["bin"]}, {"bin": "bin"}, tr, te, groups=groups, keep=True)
    assert res2["bin"]["auroc"] > 0.98 and res2["bin"]["probe"].link == "logit"
    back = probes.LinearProbe.from_dict(p.to_dict())
    assert torch.allclose(back.w, p.w) and back.b == pytest.approx(p.b)


def _quiet_unit_layer(n_games: int = 200, per_game: int = 30, seed: int = 0):
    """16 ordinary units that carry most of a label, and one near-silent unit (sd 0.001) that carries the rest of it
    almost exactly: informative, but a unit that barely moves."""
    rng = np.random.default_rng(seed)
    n = n_games * per_game
    loud = rng.normal(size=(n, 16))
    s = loud @ rng.normal(size=16) / 4.0
    e = 0.5 * rng.normal(size=n)
    quiet = 0.002 * (e + 0.05 * rng.normal(size=n))
    x = torch.from_numpy(np.concatenate([loud, quiet[:, None]], 1).astype(np.float32))
    y = (s + e).astype(np.float32)
    return x, {"y": y, "pos": (y > 0).astype(np.float32)}, np.repeat(np.arange(n_games), per_game)


def test_a_near_silent_unit_gets_no_giant_weight_by_default():
    """Divided by its own sd, a near-silent but informative unit earns a raw weight in the hundreds, so noise of 0.02
    on every unit throws the readout off; the default (one scale for the whole layer) does not, and the steadiness
    numbers say which is which."""
    x, lab, groups = _quiet_unit_layer()
    tr, te = probes.group_split(groups, 0.2)
    kinds = {"y": "reg", "pos": "bin"}
    unit = probes.ridge_suite(x, {"y": lab["y"]}, kinds, tr, te, groups=groups, keep=True, scale="unit")["y"]
    glob = probes.ridge_suite(x, {"y": lab["y"]}, kinds, tr, te, groups=groups, keep=True)["y"]
    assert abs(float(unit["probe"].w[16])) > 200 and abs(float(glob["probe"].w[16])) < 5
    assert unit["noise"] == pytest.approx(0.02 * unit["wnorm"]) and unit["noise"] > 4 and glob["noise"] < 0.1
    assert unit["r2"] > 0.95 and glob["r2"] > 0.7
    noisy = x + 0.02 * torch.from_numpy(np.random.default_rng(5).normal(size=tuple(x.shape)).astype(np.float32))
    yt = lab["y"][te]
    assert probes.r2(unit["probe"].score(noisy[te]).numpy(), yt) < 0
    assert probes.r2(glob["probe"].score(noisy[te]).numpy(), yt) > 0.7
    pu = probes.probe_layer(x, lab, kinds, tr, te, groups=groups, keep=True, scale="unit")
    pg = probes.probe_layer(x, lab, kinds, tr, te, groups=groups, keep=True)
    for k in kinds:
        assert abs(float(pu[k]["probe"].w[16])) > 20 * abs(float(pg[k]["probe"].w[16])), k
        assert pg[k]["noise"] < pu[k]["noise"] / 20, k
    with pytest.raises(ValueError, match="scale"):
        probes.ridge_suite(x, {"y": lab["y"]}, kinds, tr, te, scale="per-unit")


def test_readout_steadiness_numbers():
    t = np.arange(10, dtype=np.float32)
    x = torch.from_numpy(np.stack([t, np.zeros_like(t)], 1))
    w = np.array([[3.0, 4.0], [0.0, 0.0]])
    y = np.stack([3 * t, 4 * t + 1.0], 1)
    seq = np.repeat([0, 1], 5)
    order = np.concatenate([np.arange(5)[::-1], np.arange(5)])
    rows = np.concatenate([np.arange(5)[::-1], np.arange(5, 10)])
    s = probes.readout_stability(w, [0.0, 0.0], x, rows, y=y, seq=seq, order=order)
    assert s["wnorm"] == pytest.approx([3.0, 4.0]) and s["noise"] == pytest.approx(0.1)
    assert s["jitter"] == pytest.approx(5.0) and s["jitter_label"] == pytest.approx(5.0)
    assert s["jitter_err"] == pytest.approx(0.0) and s["miss_med"] == pytest.approx(1.0)
    one = probes.readout_stability(torch.tensor([2.0, 0.0]), 1.0, x)
    assert one["wnorm"] == pytest.approx(2.0) and one["noise"] == pytest.approx(0.04) and "jitter" not in one


# ------------------------------------------------------------------------------------------------- SAE
def test_sae_learns_a_sparse_dictionary_and_edits(tmp_path):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    d, m_true, n = 16, 24, 4000
    dictionary = rng.normal(size=(m_true, d))
    dictionary /= np.linalg.norm(dictionary, axis=1, keepdims=True)
    codes = (rng.random((n, m_true)) < 0.08) * rng.exponential(1.0, (n, m_true))
    acts = (codes @ dictionary).astype(np.float32)
    sae = S.TopKSAE(d, 48, 4).init_from(acts[:2000])
    S.train_sae(sae, torch.from_numpy(acts), steps=1000, batch=256, lr=3e-3)
    x = torch.from_numpy(acts[-500:])
    assert S.fvu(sae, x) < 0.25
    z = sae.encode(x)
    assert int((z > 0).sum(1).max()) <= 4
    idx, val = S.encode_all(sae, x)
    j = int(idx[0, 0])
    assert torch.allclose(S.dense_column(idx, val, j), z[:, j], atol=1e-2, rtol=1e-2)
    out = intervene.ablate_features(sae, [j])(x)
    assert torch.allclose(x - out, z[:, j:j + 1] * sae.feature_direction(j)[None], atol=1e-5)
    th = S.firing_thresholds(sae, x, [j])
    assert 0 < th[j] <= float(z[:, j].detach().max())
    S.save(sae, tmp_path / "sae.pt", layer="h2")
    back = S.load(tmp_path / "sae.pt")
    assert torch.allclose(back(x), sae(x)) and back.meta["layer"] == "h2"
    with pytest.raises(ValueError, match="k <= m"):
        S.TopKSAE(4, 8, 9)


def test_direction_edits():
    torch.manual_seed(2)
    x = torch.randn(10, 5)
    u = torch.randn(5)
    uu = u / u.norm()
    comp = x @ uu
    assert torch.allclose(intervene.project_out(u)(x) @ uu, torch.zeros(10), atol=1e-5)
    assert torch.allclose(intervene.scale_direction(u, 3.0)(x) @ uu, 3.0 * comp, atol=1e-5)
    assert torch.allclose(intervene.scale_direction(u, 0.0)(x), intervene.project_out(u)(x), atol=1e-6)
    st = intervene.steer(uu, 2.0, where=lambda h: h[:, 0] > 0)(x)
    assert torch.allclose((st - x)[x[:, 0] > 0], 2.0 * uu.expand(int((x[:, 0] > 0).sum()), -1), atol=1e-6)
    assert torch.equal(st[x[:, 0] <= 0], x[x[:, 0] <= 0])
    assert torch.allclose(intervene.mean_ablate(torch.ones(5))(x), torch.ones(10, 5))


def test_an_edit_stays_on_and_comes_off():
    m = tiny_mlp()
    x = torch.randn(4, 6)
    ref = m(x)
    with intervene.editing(m, "3", lambda h: torch.zeros_like(h)):
        assert torch.allclose(m(x), m[4].bias.expand(4, -1))
    assert torch.equal(m(x), ref)
    r = intervene.policy_kl(m, m, torch.randn(64, 6), "3", {"same": lambda h: h, "zero": lambda h: torch.zeros_like(h)})
    assert r["same"]["kl"] == pytest.approx(0.0, abs=1e-7) and r["same"]["argmax_changed"] == 0 and r["zero"]["kl"] > 0
    rr, lo, hi = intervene.rate_ratio_ci(20, 100, 40, 100)
    assert rr == pytest.approx(0.5) and lo < 0.5 < hi
    share, lo, hi = intervene.bootstrap_share(np.array([2, 3, 1]), np.array([1, 1, 2]))
    assert share == pytest.approx(0.6) and lo <= share <= hi


# ------------------------------------------------------------------------------------------------ analysis
def test_sparse_relations_match_dense():
    rng = np.random.default_rng(3)
    n, k, m = 3000, 3, 12
    idx = np.stack([rng.permutation(m)[:k] for _ in range(n)])
    val = rng.exponential(1.0, (n, k)).astype(np.float32) * (rng.random((n, k)) < 0.7)
    lab = {"b": (rng.random(n) < 0.2).astype(np.float32), "r": rng.normal(size=n).astype(np.float32)}
    lab["b"][:50] = np.nan
    kinds = {"b": "bin", "r": "reg"}
    rows, vals, starts = analysis.sparse_columns(idx, val, m)
    for j in range(m):
        dense = (val * (idx == j)).sum(1)
        a = analysis.label_relations(dense, lab, kinds)
        b = analysis.sparse_relations(rows[starts[j]:starts[j + 1]], vals[starts[j]:starts[j + 1]], lab, kinds, n)
        assert a["b"]["auroc"] == pytest.approx(b["b"]["auroc"], abs=1e-9)
        assert a["r"]["r"] == pytest.approx(b["r"]["r"], abs=1e-6)
    top = analysis.top_moments(np.array([5.0, 4.0, 3.0, 0.0]), k=3, groups=np.array([0, 0, 1, 1]),
                               steps=np.array([0, 2, 0, 9]))
    assert list(top) == [0, 2]
    assert "ball" in analysis.describe(make_frames(1), 0)


def test_causal_targets_follow_the_fixed_rule():
    f = {0: {"freq": 0.01, "relations": {"demo_chance": {"auroc": 0.9, "lift": 9.0}}},
         1: {"freq": 0.01, "relations": {"demo_chance": {"auroc": 0.8, "lift": 6.0}}},
         2: {"freq": 0.0001, "relations": {"demo_chance": {"auroc": 0.99, "lift": 50.0}}},
         3: {"freq": 0.011, "relations": {"demo_chance": {"auroc": 0.5, "lift": 1.0}, "demo_soon": {"auroc": 0.55}}},
         4: {"freq": 0.009, "relations": {"demo_chance": {"auroc": 0.52, "lift": 1.0}}},
         5: {"freq": 0.5, "relations": {"demo_chance": {"auroc": 0.5, "lift": 1.0}}}}
    r = analysis.causal_targets(f, n_group=2)
    assert r["rule"] == "demo_chance" and r["target"] == 0 and r["group"] == [0, 1] and r["controls"] == [3, 4]


def test_profile_game_matches_online_counter():
    from cds_interpret.behavior import SideCounter, profile_game, summarize_profiles
    rng = np.random.default_rng(5)
    t_len, c = 60, 4
    pos = np.cumsum(rng.normal(0, 60, (t_len, c, 3)), 0) + np.array([[0.0, -3000, 17], [0, -1000, 17], [0, 1000, 17],
                                                                         [0, 3000, 17]])
    pos[..., 2] = 17
    vel = rng.normal(0, 900, (t_len, c, 3))
    vel[..., 2] = 0
    fr = make_frames(t_len, cars=c, car_pos=pos, car_vel=vel,
                     ball_pos=np.cumsum(rng.normal(0, 80, (t_len, 3)), 0) + [0, 0, 300],
                     ball_vel=rng.normal(0, 1500, (t_len, 3)))
    for k in ("ev_touch", "ev_demo_made", "ev_demoed"):
        fr.data[k] = (rng.random((t_len, c)) < 0.05).astype(np.float32)
    fr.data["goal"] = np.where(rng.random(t_len) < 0.03, 0, -1)
    side = np.array([True, True, False, False])
    off = summarize_profiles([profile_game(fr, side)], "A")
    cnt = SideCounter(1)
    for t in range(t_len):
        cnt.update(fr.take([t]), side[None])
    on = cnt.summary()["A"]
    for k in ("demos_made", "demos_suffered", "touches", "demo_chances", "boost_mean", "last_back_goal_side"):
        assert off[k] == pytest.approx(on[k]), k


# ------------------------------------------------------------------------------------------------ figures
def test_figures_are_svg_and_html():
    from cds_interpret.viz import field, gallery, plots
    svg = plots.lines({"a": ([0, 1, 2], [0.5, 0.7, 0.9]), "b": ([0, 1, 2], [0.4, 0.5, 0.6])}, "probe by layer")
    assert svg.startswith("<svg") and "prefers-color-scheme" in svg
    assert plots.heatmap(["x"], ["h1", "h2"], [[0.6, 0.9]], "AUROC").count("<rect") >= 2
    assert "<svg" in plots.bars(["a", "b"], [1.0, 2.0], "counts")
    fr = make_frames(3)
    m = field.moment_svg(fr, 0, 0, landing=(0.0, 500.0), title="kickoff")
    assert m.startswith("<svg") and "kickoff" in m
    clip = field.clip_data(fr, 0, 2, 0)
    page = gallery.gallery_html("features", "<p>x</p>", [gallery.feature_card("7", "a guess", 0.01, [("kickoff", "0.9")],
                                                                              [m], clip=clip)])
    assert page.startswith("<!doctype html>") and "data-clip" in page
