"""The overlay keeps going through bad frames; a bot that sees the field mirrored; the landing readout chosen by its
miss; precision for sparse features; carrying a pack's names to a newer checkpoint (and the whole refit); and the
display-only script that watches a bot you can't modify."""

import logging
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

import cds_interpret as ci
from cds_interpret import analysis as A
from cds_interpret import probes as P
from cds_interpret import refit as RF
from cds_interpret import sae as S
from cds_interpret._rlbot import FrameGuard
from cds_interpret.errors import CDSInterpretError
from cds_interpret.names import read_names, suggest_by_label
from cds_interpret.overlay import (
    Explainer,
    Thoughts,
    add_land,
    compute,
    load_pack,
    make_pack,
    match_cars,
    save_pack,
    watch_thoughts,
)
from cds_interpret.probes import LinearProbe

from test_core import fake_packet
from test_overlay import FakeRenderer, _fake_bot_class


# ------------------------------------------------------------------------------------------------- helpers
@pytest.fixture(scope="module")
def demo_study():
    pol = ci.demo.policy()
    m = ci.demo.synthetic_moments(n_games=24, per_game=60)
    st = ci.Study(pol, m, log=None)
    st.features("body.3", m=64, k=4, steps=300)
    return st


def newer_policy(seed: int = 5, size: float = 0.03) -> ci.Policy:
    """The demo bot "trained a little further": its weights nudged."""
    old = ci.demo.policy()
    net = ci.demo.DemoNet()
    net.load_state_dict(old.net.state_dict())
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(size * torch.randn(p.shape, generator=g))
    return ci.Policy(net.eval(), obs=ci.demo.obs, actions=ci.lookup_table(), name="newer")


def make_newer_policy():
    """For the CLI test: "test_watch_refit:make_newer_policy"."""
    return newer_policy()


def mirror_left(frames, me):
    """A bot that sees the field mirrored whenever its car is on the left of its own half."""
    rows = np.arange(len(frames))
    s = ci.frames.team_sign(frames.car_team[rows, me])
    return frames.car_pos[rows, me, 0] * s < 0


def team_packet(t, names, spectated=0, ball=True):
    cars = [(400.0 * j - 600, -1500.0 if j % 2 == 0 else 1500.0, 17.0, j % 2) for j in range(len(names))]
    pk = fake_packet(ball=(200.0, 300.0 + 40 * t, 500.0), cars=cars, t=t)
    for j, p in enumerate(pk.players):
        p.name, p.player_id = names[j], 100 + j
    pk.match_info.last_spectated = spectated
    if not ball:
        pk.balls = []
    return pk


# ----------------------------------------------------------------------------------------- bad frames, kept going
def test_frame_guard_logs_some_and_gives_up_only_on_a_long_run(tmp_path):
    seen = []
    bot = NS(logger=NS(warning=seen.append))
    g = FrameGuard(bot, "thing", limit=250, first=3, every=100, log_file=tmp_path / "e.log")
    for f in range(240):
        assert not g.failed(f, ValueError("bad"))
    assert len(seen) == 3 + 2                                    # the first three, then the 100th and 200th
    assert "Traceback" in seen[0] or "ValueError" in seen[0]
    g.ok()
    assert not g.failed(1, ValueError("again")) and g.in_a_row == 1
    g2 = FrameGuard(bot, "thing", limit=5)
    assert [g2.failed(f, KeyError(f)) for f in range(5)] == [False] * 4 + [True]
    assert "ValueError" in (tmp_path / "e.log").read_text(encoding="utf-8")


def test_the_overlay_keeps_going_through_bad_frames(tmp_path, monkeypatch):
    pytest.importorskip("rlbot")
    from cds_interpret.overlay import bot as OB
    from cds_interpret.overlay import with_thoughts
    from cds_interpret.overlay.panel import Panel
    Base = _fake_bot_class()
    sae = S.TopKSAE(8, 16, 2).init_from(torch.randn(50, 8))
    pack = make_pack("3", sae, {"f": 0}, None, {"P": LinearProbe(torch.ones(8), 0.0, "bin", 1.0)}, ci.lookup_table())
    path = save_pack(pack, tmp_path / "pack.npz")
    log = tmp_path / "overlay.log"
    bot = with_thoughts(Base, path, network="policy", worker=False, draw_every=1, error_log=log)()
    bot.initialize()
    draws = []
    real = Panel.draw
    fail = {"on": False}

    def draw(self, *a, **k):
        if fail["on"]:
            raise RuntimeError("a bad frame")
        draws.append(1)
        return real(self, *a, **k)
    monkeypatch.setattr(Panel, "draw", draw)
    replay = fake_packet(t=0.5)
    replay.balls = []                                            # the replay after a goal: no ball
    bot._packet_processor(replay)
    assert bot._cds_on and not draws and len(bot.sent) == 1     # skipped quietly; the bot still decided
    for f in range(1, 20):
        bot._packet_processor(fake_packet(t=f / 120))
    assert draws
    fail["on"] = True
    for f in range(20, 40):
        bot._packet_processor(fake_packet(t=f / 120))
    assert bot._cds_on                                           # logged, not switched off
    warnings = [e[1] for e in bot.events if isinstance(e, tuple)]
    assert any("a bad frame" in w for w in warnings) and "a bad frame" in log.read_text(encoding="utf-8")
    fail["on"] = False
    n = len(draws)
    for f in range(40, 60):
        bot._packet_processor(fake_packet(t=f / 120))
    assert len(draws) > n                                        # drawing again on the next good frames
    # a long run in a row: off, and its drawing taken down
    monkeypatch.setattr(OB, "MAX_FAILED_FRAMES", 5)
    bot2 = with_thoughts(Base, path, network="policy", worker=False, draw_every=1)()
    bot2.initialize()
    fail["on"] = True
    for f in range(1, 40):
        bot2._packet_processor(fake_packet(t=f / 120))
    assert not bot2._cds_on and len(bot2.sent) == 39
    assert any(c[0] == "clear_render_group" for c in bot2.renderer.calls)


def test_a_failing_or_stopped_worker_does_not_freeze_the_panel(tmp_path, monkeypatch):
    pytest.importorskip("rlbot")
    from cds_interpret.overlay import bot as OB
    from cds_interpret.overlay import with_thoughts
    started = []

    class FakeLink:
        def __init__(self, *a):
            started.append(self)
            self.ready, self.hello, self.alive = True, {}, True
            self.proc = NS(exitcode=1)
            self.answers = []

        def send(self, msg):
            self.answers.append({"error": "RuntimeError('boom')"})
            return True

        def poll(self):
            return self.answers.pop(0) if self.answers else None

        def close(self):
            pass
    monkeypatch.setattr(OB, "WorkerLink", FakeLink)
    monkeypatch.setattr(OB, "MAX_FAILED_READINGS", 4)
    Base = _fake_bot_class()
    pack = make_pack("3", None, None, None, {"P": LinearProbe(torch.ones(8), 0.0, "bin", 1.0)}, ci.lookup_table())
    path = save_pack(pack, tmp_path / "p.npz")
    bot = with_thoughts(Base, path, network="policy", worker=True)()
    bot.initialize()
    for f in range(1, 4):                                        # three bad readings: logged, still on
        bot._packet_processor(fake_packet(t=f / 120))
    assert bot._cds_on and any("boom" in e[1] for e in bot.events if isinstance(e, tuple))
    started[-1].alive = False                                    # the worker stopped: started again
    bot._packet_processor(fake_packet(t=5 / 120))
    assert len(started) == 2 and bot._cds_on
    for f in range(6, 30):                                       # a long run of failed readings: off, drawing down
        bot._packet_processor(fake_packet(t=f / 120))
    assert not bot._cds_on and any(c[0] == "clear_render_group" for c in bot.renderer.calls)
    assert len(bot.sent) == 28                                   # the bot played every tick


# ------------------------------------------------------------------------------------- a bot that sees it mirrored
def test_a_mirrored_landing_readout_is_turned_back_and_moves_are_said_as_sent():
    pol = ci.Policy(ci.demo.policy().net, obs=ci.demo.obs, actions=ci.lookup_table(), mirror=mirror_left)
    fr = ci.from_arrays(ball_pos=np.zeros((2, 3)), ball_vel=np.zeros((2, 3)),
                        car_pos=np.array([[[-500.0, 0, 17], [0, 2000, 17]], [[500.0, 0, 17], [0, 2000, 17]]]),
                        car_vel=np.zeros((2, 2, 3)), car_fwd=np.tile([0.0, 1, 0], (2, 2, 1)),
                        car_up=np.tile([0.0, 0, 1], (2, 2, 1)), car_boost=np.zeros((2, 2)),
                        car_on_ground=np.ones((2, 2)), car_team=np.array([[0.0, 1], [0, 1]]))
    assert pol.mirrored(fr, [0, 0]).tolist() == [True, False]
    assert ci.demo.policy().mirrored(fr, [0, 0]).tolist() == [False, False]
    table = ci.lookup_table()
    a = next(i for i, row in enumerate(table) if row[1] > 0.5 and row[5] < 0.5)   # a "steer right" move
    pack = make_pack("body.3", None, None, None, None, table)
    add_land(pack, "body.3", np.zeros((64, 2), np.float32), [700.0, 1200.0], frame="mirrored")
    th = Thoughts(load_pack(save_pack(pack, __import__("tempfile").mkdtemp() + "/m.npz")))
    assert th.land_frame == "mirrored"
    out = np.full((1, 90), -5.0, np.float32)
    out[0, a] = 5.0
    ex = Explainer(th, pol, freeze=False)
    for row, flip in ((0, True), (1, False)):
        state = {k: v[row] for k, v in fr.data.items()}
        res = compute(th, {"layer": np.zeros((1, 64), np.float32), "output": out, "cars": [0], "state": state}, None,
                      ex)["cars"][0]
        assert res["land_xy"] == pytest.approx([-700.0 if flip else 700.0, 1200.0])
        words = res["action"]["words"]
        assert ("steer left" in words) == flip and ("steer right" in words) == (not flip)
    res = compute(Thoughts(pack), {"layer": np.zeros((1, 64), np.float32), "output": out})
    assert res["cars"][0]["land_xy"] is None and any("mirrored" in n for n in res["notes"])   # no flag: no ring


# ---------------------------------------------------------------------------------------------- the landing fit
def test_landing_fit_picks_its_penalty_by_the_held_out_miss(demo_study):
    st = demo_study
    xy = np.stack([st.labels["land_x"], st.labels["land_y"]], 1)
    fit = P.fit_landing(st.activations("body.1"), xy, st.train, st.test, st.games, seq=st.moments.seq,
                        order=st.moments.step)
    assert fit["w"].shape == (64, 2) and fit["frame"] == "team" and fit["median_miss"] <= fit["p90_miss"]
    tried = {t["lambda"]: t["median_miss"] for t in fit["tried"]}
    assert fit["lambda"] == min(tried, key=tried.get) and len(tried) == len(P.LAMBDAS_LAND)
    best = st.landing()
    assert best["layer"] in st.layers and st.landing_results is best
    # a mirroring bot: the readout is fitted in its own frame (x negated where it sees the field mirrored)
    mir = np.zeros(len(xy), bool)
    mir[::2] = True
    fm = P.fit_landing(st.activations("body.1"), xy, st.train, st.test, st.games, mirrored=mir)
    flipped = xy.copy()
    flipped[mir, 0] *= -1
    ft = P.fit_landing(st.activations("body.1"), flipped, st.train, st.test, st.games)
    assert fm["frame"] == "mirrored" and np.allclose(fm["w"], ft["w"], atol=1e-4)
    pk = st.make_pack("body.3", probes={"Goal side": "goal_side"})
    assert pk["land"]["frame"] == "team" and pk["land"]["info"]["median_miss"] > 0
    assert pk["probes"][0]["label"] == "goal_side" and pk["meta"]["sae"]["m"] == 64


# --------------------------------------------------------------------------------- precision for sparse features
def test_a_sparse_feature_is_judged_by_its_precision():
    rng = np.random.default_rng(0)
    n = 20000
    y = (rng.random(n) < 0.02).astype(float)
    act = np.zeros(n)
    pos, neg = np.nonzero(y)[0], np.nonzero(1 - y)[0]
    act[rng.choice(pos, int(0.35 * len(pos)), replace=False)] = 2.0
    act[rng.choice(neg, int(0.003 * len(neg)), replace=False)] = 1.0
    r = A.label_relations(act, {"x": y}, {"x": "bin"})["x"]
    assert r["auroc"] < 0.7 and r["lift"] > 20                    # the trap: a weak AUROC for a clean detector
    assert A.detects(r) and not A.detects({**r, "ap": 0.05})
    assert A.relation_strength(r) == pytest.approx((r["ap"] - r["base"]) / (1 - r["base"]))
    nz = np.nonzero(act)[0]
    rs = A.sparse_relations(nz, act[nz], {"x": y}, {"x": "bin"}, n)["x"]
    assert rs["ap"] == pytest.approx(r["ap"]) and rs["auroc"] == pytest.approx(r["auroc"])
    # the moments where it is off count at the base rate, whatever order they are stored in
    perm = rng.permutation(n)
    assert A.label_relations(act[perm], {"x": y[perm]}, {"x": "bin"})["x"]["ap"] == pytest.approx(r["ap"])


def test_names_by_label_one_feature_each():
    rel = {3: {"kickoff": {"auroc": 0.62, "ap": 0.5, "base": 0.03, "lift": 20.0, "coverage": 1.0},
               "in_air": {"auroc": 0.9, "ap": 0.6, "base": 0.2, "lift": 3.5, "coverage": 1.0}},
           7: {"kickoff": {"auroc": 0.58, "ap": 0.2, "base": 0.03, "lift": 6.0, "coverage": 1.0},
               "in_air": {"auroc": 0.55, "ap": 0.22, "base": 0.2, "lift": 1.1, "coverage": 1.0}}}
    fs = NS(relations=rel, freq=np.array([0.0] * 3 + [0.05] + [0.0] * 3 + [0.04]), fingerprint="abc", codes=None)
    report: list = []
    out = suggest_by_label(fs, ["kickoff", "in_air"], report=report)
    assert [(s.feature, s.label) for s in out] == [(3, "kickoff")]       # 3 takes kickoff; 7 doesn't detect in_air
    assert report[1] == {"label": "in_air", "feature": 7, "auroc": 0.55, "ap": 0.22, "base": 0.2, "kept": False}
    assert "precision 0.50" in out[0].detail


# ------------------------------------------------------------------------------------------ carrying the names over
def test_names_are_carried_to_a_newer_checkpoint(demo_study, tmp_path):
    st = demo_study
    fs = st.feature_sets["body.3"]
    names = {f"idea {j}": int(j) for j in fs.top(3)}
    old_pack = st.make_pack("body.3", features=names, probes={"Goal side": "goal_side"}, donors=8)
    # the same network and dictionary: every name finds its own feature again (r under 1 only because the panel
    # reads each feature without the dictionary's top-k)
    rep = RF.carry_names(old_pack, st.activations("body.3"), fs)
    assert all(e["kept"] and e["new"] == e["old"] and e["r"] > 0.7 for e in rep)
    assert not any(e["kept"] for e in RF.carry_names(old_pack, st.activations("body.3"), fs, min_r=1.01))
    path = RF.write_carried_names(tmp_path / "names.toml", rep, fs)
    nm = read_names(path)
    assert nm.confirmed() == names and nm.dictionary == fs.fingerprint and "carried over" in nm.features[rep[0]["new"]].note
    with pytest.raises(CDSInterpretError, match="same moments"):
        RF.carry_names(old_pack, st.activations("body.3")[:10], fs)
    # the whole refit for a newer checkpoint, on these moments
    out = tmp_path / "refit"
    report = RF.refit_pack(old_pack, newer_policy(), st.policy, out, moments=st.moments, steps=300, log=None)
    assert {"features.sae", "names.toml", "refit.json", "thoughts.npz"} <= {p.name for p in out.iterdir()}
    th = Thoughts(load_pack(out / "thoughts.npz"))
    kept = [e["name"] for e in report["names"]["features"] if e["kept"]]
    assert kept and th.names == kept and th.fd is not None and th.donors and th.land is not None
    assert [q["name"] for q in th.pack["probes"]] == ["Goal side"] and "Goal side" in report["calibration"]
    again = RF.refit_pack(old_pack, newer_policy(), st.policy, out, moments=st.moments, steps=300, log=None)
    assert again["names"] == report["names"]                     # the dictionary is kept: the same answer


def test_refit_pack_command(demo_study, tmp_path, capsys):
    from cds_interpret.cli import main
    st = demo_study
    rec = tmp_path / "rec"
    st.moments.save(rec)
    pack = st.make_pack("body.3", features={"idea": int(st.feature_sets["body.3"].top(1)[0])}, donors=0)
    save_pack(pack, tmp_path / "old.npz")
    code = main(["refit-pack", str(tmp_path / "old.npz"), "--policy", "test_watch_refit:make_newer_policy",
                 "--old-policy", "cds_interpret.demo:policy", "--out", str(tmp_path / "new"), "--moments", str(rec)])
    assert code == 0 and (tmp_path / "new" / "thoughts.npz").exists()
    assert "names carried over" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------- watching a bot
def _watcher(tmp_path, **kw):
    st_pol = ci.demo.policy()
    m = ci.demo.synthetic_moments(n_games=12, per_game=40)
    st = ci.Study(st_pol, m, log=None)
    fs = st.features("body.3", m=32, k=4, steps=150)
    pack = st.make_pack("body.3", features={"idea": int(fs.top(1)[0])}, donors=0)
    path = save_pack(pack, tmp_path / "watch.npz")
    cls = watch_thoughts("cds_interpret.demo:policy", path, **kw)
    obj = cls.__new__(cls)
    obj.renderer, obj.logger = FakeRenderer(), logging.getLogger("cds-watch-test")
    obj.update_rendering_status = lambda *a, **k: None
    obj.ball_prediction = NS(slices=[])
    obj.initialize()
    return obj


def test_watch_a_bot_you_cannot_modify(tmp_path):
    pytest.importorskip("rlbot")
    names = ["SomeBot", "Other", "SomeBot (2)", "Other (2)"]
    assert match_cars(team_packet(1, names), "somebot") == [0, 2]
    assert match_cars(team_packet(1, names), [3, 9]) == [3] and match_cars(team_packet(1, names), lambda p: [1]) == [1]
    with pytest.raises(CDSInterpretError, match="which cars"):
        watch_thoughts("cds_interpret.demo:policy", "x.npz", cars="")
    obj = _watcher(tmp_path, cars="SomeBot")
    assert obj._cds_on
    drawn = []
    for f in range(1, 80):
        obj.renderer.calls.clear()
        obj.handle_packet(team_packet(f / 120, names, spectated=2))
        if any(c[0] == "draw_rect_2d" for c in obj.renderer.calls):
            drawn.append(obj.renderer.calls[:])
    assert drawn
    calls = drawn[-1]
    texts = [c[1][0] for c in calls if c[0] == "draw_string_2d"]
    assert "car 2 of 2 - on camera" in texts and "Next move" in texts
    joined = sorted(c[1][0].index for c in calls if c[0] == "draw_line_3d" and hasattr(c[1][0], "index"))
    assert joined == [0, 2]                                      # a ring for each of its cars, joined to its car
    label = [c[1][1].index for c in calls if c[0] == "draw_string_3d"]
    assert label in ([], [2])
    # its 2v2 / 3v3 brain only: nothing drawn in 1v1
    obj2 = _watcher(tmp_path, cars="SomeBot", team_sizes=[2, 3])
    for f in range(1, 40):
        obj2.handle_packet(team_packet(f / 120, ["SomeBot", "Other"]))
    assert not any(c[0] == "draw_rect_2d" for c in obj2.renderer.calls)


def test_watch_keeps_going_through_bad_frames(tmp_path, monkeypatch):
    pytest.importorskip("rlbot")
    from cds_interpret.overlay import watch as W
    names = ["SomeBot", "Other"]
    obj = _watcher(tmp_path, cars="SomeBot", error_log=tmp_path / "watch.log")
    obj.handle_packet(team_packet(1.0, names, ball=False))       # the replay: no ball, nothing drawn, no error
    assert not obj.renderer.calls and obj._cds_st["guard"].total == 0
    real = obj._cds_draw
    obj._cds_draw = lambda *a: (_ for _ in ()).throw(RuntimeError("a bad frame"))
    obj.handle_packet(team_packet(2.0, names))
    assert obj._cds_on and obj._cds_st["guard"].in_a_row == 1
    assert "a bad frame" in (tmp_path / "watch.log").read_text(encoding="utf-8")
    obj._cds_draw = real
    obj.handle_packet(team_packet(3.0, names))
    assert obj._cds_on and obj._cds_st["guard"].in_a_row == 0
    assert any(c[0] == "draw_rect_2d" for c in obj.renderer.calls)
    monkeypatch.setattr(W, "MAX_FAILED_DRAWS", 3)
    obj3 = _watcher(tmp_path, cars="SomeBot")
    obj3._cds_draw = lambda *a: (_ for _ in ()).throw(RuntimeError("always"))
    for f in range(1, 10):
        obj3.handle_packet(team_packet(float(f), names))
    assert not obj3._cds_on and any(c[0] == "clear_render_group" for c in obj3.renderer.calls)


def test_watch_template(tmp_path, capsys):
    from cds_interpret.cli import main
    assert main(["watch-template", str(tmp_path / "w")]) == 0
    assert (tmp_path / "w" / "watch.py").exists() and (tmp_path / "w" / "watch.script.toml").exists()
    assert main(["watch-template", str(tmp_path / "w")]) == 1                # never overwrites without --force
    import tomllib
    toml = tomllib.loads((tmp_path / "w" / "watch.script.toml").read_text(encoding="utf-8"))
    assert toml["settings"]["run_command"] == "python watch.py"
    compile((tmp_path / "w" / "watch.py").read_text(encoding="utf-8"), "watch.py", "exec")


# -------------------------------------------------------------------------- a landing readout relative to the car
def test_car_frame_offsets_round_trip_and_turn_with_the_car():
    from cds_interpret.overlay import car_frame_offsets, car_frame_point, heading, land_point
    rng = np.random.default_rng(3)
    pts, car = rng.uniform(-4000, 4000, (50, 2)), rng.uniform(-4000, 4000, (50, 2))
    yaw = rng.uniform(-np.pi, np.pi, 50)
    off = car_frame_offsets(pts, car, heading(yaw))
    assert np.allclose(car_frame_point(off, car, heading(yaw)), pts)
    assert np.allclose(np.linalg.norm(off, axis=1), np.linalg.norm(pts - car, axis=1))
    # a car at (100, 200) facing +x: a point 300 ahead and 50 to its left is at (400, 250)
    assert np.allclose(car_frame_offsets([[400.0, 250.0]], [[100.0, 200.0]], [[1.0, 0.0]]), [[300.0, 50.0]])
    # the same in the team frame (orange's turned half round) and the world's: the offsets do not change
    assert np.allclose(car_frame_offsets(-pts, -car, -heading(yaw)), off)
    phys = type("P", (), {})()
    phys.location = type("L", (), {"x": 100.0, "y": 200.0, "z": 17.0})()
    phys.rotation = type("R", (), {"pitch": 0.0, "yaw": np.pi / 2, "roll": 0.0})()
    assert land_point("car", [300.0, 50.0], 1, phys) == pytest.approx((50.0, 500.0))   # (facing +y: left is -x)
    assert land_point("car", [300.0, 50.0], 0, None) is None
    assert land_point("team", [300.0, 50.0], 1) == pytest.approx((-300.0, -50.0))


def test_a_car_frame_landing_readout_is_fitted_relative_to_the_car():
    from cds_interpret.overlay import car_frame_offsets, heading
    rng = np.random.default_rng(5)
    n = 4000
    car, yaw = rng.uniform(-3000, 3000, (n, 2)), rng.uniform(-np.pi, np.pi, n)
    true_off = rng.uniform(-2000, 2000, (n, 2))
    from cds_interpret.overlay import car_frame_point
    land = car_frame_point(true_off, car, heading(yaw))
    # a layer that holds the offset (as a car-centred network would), not the place on the field
    h = np.concatenate([true_off / 1000.0, rng.normal(0, 1, (n, 6))], 1).astype(np.float32)
    games = np.repeat(np.arange(40), n // 40)
    train, test = np.nonzero(games < 32)[0], np.nonzero(games >= 32)[0]
    fc = P.fit_landing(torch.from_numpy(h), land, train, test, games, car=(car, heading(yaw)))
    ft = P.fit_landing(torch.from_numpy(h), land, train, test, games)
    assert fc["frame"] == "car" and fc["median_miss"] < 5.0
    assert ft["frame"] == "team" and ft["median_miss"] > 500.0
    assert np.allclose(np.asarray(torch.from_numpy(h)[test].numpy() @ fc["w"] + fc["b"]),
                       car_frame_offsets(land[test], car[test], heading(yaw[test])), atol=10.0)
    pack = make_pack("body.3", None, None, None, None, ci.lookup_table())
    add_land(pack, "body.1", fc["w"], fc["b"], frame="car")
    assert Thoughts(load_pack(save_pack(pack, __import__("tempfile").mkdtemp() + "/c.npz"))).land_frame == "car"
    with pytest.raises(ci.CDSInterpretError):
        P.fit_landing(torch.from_numpy(h), land, train, test, games, car=(car, heading(yaw)),
                      mirrored=np.ones(n, bool))


def test_a_study_fits_its_landing_ring_relative_to_the_car_on_request(demo_study):
    fit = demo_study.landing("body.1", frame="car")
    assert fit["frame"] == "car" and fit["w"].shape == (64, 2) and fit["median_miss"] > 0
    pk = demo_study.make_pack("body.3", probes={"Goal side": "goal_side"}, land_layer="body.1", land_frame="car")
    assert pk["land"]["frame"] == "car"
    with pytest.raises(ci.CDSInterpretError):
        demo_study.landing("body.1", frame="sideways")
