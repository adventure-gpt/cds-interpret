"""The overlay in team modes and its two decision readings: the panel follows the camera, a landing marker per car,
"why this move" (attribution patching, checked against autograd by hand), "looking at" (exact Shapley shares over the
game's parts, checked against the coalition table), the pack's new fields, the config's new options and their
friendly errors, a panel that never changes size, and bots - a Hivemind and one-process-per-car Bots - whose controls
stay bit for bit the same."""

import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import cds_interpret as ci
from cds_interpret import attribution as AT
from cds_interpret import sae as S
from cds_interpret.errors import CDSInterpretError
from cds_interpret.frames import nearest_car
from cds_interpret.overlay import (
    Explainer,
    Follow,
    Panel,
    PanelConfig,
    RowSpec,
    Thoughts,
    WorkerLink,
    add_decoders,
    add_donors,
    add_land,
    check_panel,
    compute,
    landing_cars,
    load_pack,
    make_pack,
    save_pack,
    try_readings,
)
from cds_interpret.overlay import worker as W
from cds_interpret.overlay.explain import looking_shares
from cds_interpret.overlay.team import subtitle_text

from conftest import make_frames
from test_core import fake_packet
from test_overlay import FakeRenderer, fake_flat


# ------------------------------------------------------------------------------------------------- helpers
def team_obs(frames, me):
    """A tiny bot's input [N, 12]: the ball, its car, its nearest opponent and nearest team-mate (zeros if none), in
    its team's frame, so every part of the game can matter."""
    me = np.asarray(me, np.int64)
    rows = np.arange(len(frames))
    s = ci.frames.team_sign(frames.car_team[rows, me])
    parts = [ci.frames.canon(frames.ball_pos, s) / 4096, ci.frames.canon(frames.car_pos[rows, me], s) / 4096]
    for which in ("opponent", "mate"):
        j = nearest_car(frames, me, which)
        has = j >= 0
        pos = ci.frames.canon(frames.car_pos[rows, np.maximum(j, 0)], s) / 4096
        parts.append(np.where(has[:, None], pos, 0.0))
    return np.concatenate(parts, 1).astype(np.float32)


def random_moments(n, cars, seed=0):
    rng = np.random.default_rng(seed)
    fr = make_frames(n, cars, ball_pos=rng.uniform(-3000, 3000, (n, 3)) * [1, 1, 0.3] + [0, 0, 100],
                     ball_vel=rng.normal(0, 500, (n, 3)), car_pos=rng.uniform(-3000, 3000, (n, cars, 3)) * [1, 1, 0] + 17,
                     car_vel=rng.normal(0, 800, (n, cars, 3)))
    return ci.Moments(fr, np.zeros(n, np.int64), np.repeat(np.arange(n // 10 + 1), 10)[:n], np.arange(n))


def team_policy(seed=0):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(12, 32), torch.nn.ReLU(), torch.nn.Linear(32, 16), torch.nn.ReLU(),
                              torch.nn.Linear(16, 90))
    return ci.Policy(net.eval(), obs=team_obs, actions=ci.lookup_table())


def team_pack(pol, layer="3", width=16, donors=True, decoders=True):
    m1, m2 = random_moments(60, 2, 1), random_moments(60, 4, 2)
    x = pol.observe(m2.frames, m2.me)
    h = pol.activations(x, [layer])[layer]
    torch.manual_seed(0)
    sae = S.TopKSAE(width, 24, 4).init_from(h)
    pk = make_pack(layer, sae, {"a": 2, "b": 7, "c": 11}, None, None, pol.actions, decoders=decoders)
    if donors:
        add_donors(pk, [m1, m2], n=20)
    return pk, sae


def make_policy():
    """The factory the worker test names: team_policy() as "test_overlay_team:make_policy"."""
    return team_policy()


# ---------------------------------------------------------------------------------- whose thoughts, whose rings
def test_panel_follows_the_camera_then_the_last_then_a_pick():
    ours = [0, 2, 4]
    f = Follow(["camera", "last", "random"], seed=7)
    assert f.pick(ours, 2) == (2, "camera")
    assert f.pick(ours, 3) == (2, "last")                 # the camera went to an opponent: the last of ours
    assert f.pick(ours, 4) == (4, "camera") and f.pick(ours, 5) == (4, "last")
    g, g2 = Follow(["camera", "last", "random"], seed=7), Follow(["camera", "last", "random"], seed=7)
    car, rule = g.pick(ours, 5)                           # nothing of ours on camera yet: a pick, kept
    assert rule == "random" and car in ours and all(g.pick(ours, 1)[0] == car for _ in range(5))
    assert g2.pick(ours, 5)[0] == car                     # every copy of the bot picks the same car
    assert Follow([2]).pick(ours, 0) == (2, "car") and Follow(["first"]).pick(ours, 4) == (0, "first")
    assert Follow(["camera", "hide"]).pick(ours, 3) == (None, "hide")
    assert Follow([9]).pick(ours, 3) == (0, "first")       # no rule gives a car: the first
    assert Follow(["camera"]).pick([1], None) == (1, "first")   # a packet without last_spectated
    assert subtitle_text("car {car} of {cars} - {rule} ({name})", 4, ours, "last", "Bot (3)") == \
        "car 3 of 3 - last on camera (Bot (3))"


def test_which_cars_get_a_landing_marker():
    ours = [0, 2, 4]
    assert landing_cars("all", ours, 2, 3) == ours and landing_cars("none", ours, 2, 3) == []
    assert landing_cars("panel", ours, 4, 3) == [4] and landing_cars("camera", ours, 4, 2) == [2]
    assert landing_cars("camera", ours, 4, 3) == [] and landing_cars([1, 3], ours, 4, 3) == [0, 4]
    assert landing_cars([3], [1], 1, None) == [] and landing_cars("panel", ours, None, 2) == []
    m = PanelConfig.from_dict({"markers": {"landing": False, "landing_for": [1, 3]}}).markers
    assert m.landing_cars == "none"                       # landing = false still turns every ring off


# ------------------------------------------------------------------------------------------ the config
def test_new_options_and_their_friendly_errors():
    cfg = PanelConfig.from_dict({"panel": {"follow": ["camera", 2, "hide"], "subtitle": "{name}", "keep_size": False},
                                 "markers": {"landing_for": "panel", "car_label_from": "noticing"},
                                 "colors": {"team_mates": "#aa88ff"},
                                 "rows": [{"kind": "why", "max": 3, "min_push": 0.2, "against": False},
                                          {"kind": "looking", "every": 2, "donors": 5, "legend": False}]})
    assert cfg.follow == ["camera", 2, "hide"] and not cfg.keep_size and cfg.markers.landing_for == "panel"
    assert cfg.colors["team_mates"] == (170, 136, 255)
    assert cfg.wants == {"why": True, "looking": True, "every": 2, "donors": 5}
    assert RowSpec("why").max == 2 and RowSpec("why").empty == "no named idea" and RowSpec("noticing").max == 3
    assert RowSpec("looking").words == ("car", "ball", "opp", "mates")
    assert PanelConfig.from_dict({"panel": {"follow": "first"}}).follow == ["first"]
    assert PanelConfig.from_dict({"markers": {"landing_for": [1, 3]}}).markers.landing_for == [1, 3]
    bad = [({"panel": {"follow": ["camra"]}}, "Did you mean 'camera'"),
           ({"panel": {"follow": [0]}}, "start at 1"),
           ({"panel": {"follow": [True]}}, "not a rule"),
           ({"panel": {"subtitle": "car {nme}"}}, "Did you mean {name}"),
           ({"panel": {"keep_size": "yes"}}, "true or false"),
           ({"markers": {"landing_for": "pnael"}}, "Did you mean 'panel'"),
           ({"markers": {"landing_for": [0, 2]}}, "start at 1"),
           ({"markers": {"landing_for": True}}, "landing_for should be"),
           ({"markers": {"car_label_from": "probes"}}, "car_label_from"),
           ({"rows": [{"kind": "looking", "every": 0}]}, "leave it out"),
           ({"rows": [{"kind": "looking", "words": ["car", "ball"]}]}, "own car, ball, opponents and team-mates"),
           ({"rows": [{"kind": "looking", "evry": 2}]}, "Did you mean 'every'"),
           ({"rows": [{"kind": "why", "min_push": -1.0}]}, "0 or more"),
           ({"rows": [{"kind": "why", "against": "no"}]}, "true or false"),
           ({"rows": [{"kind": "wy"}]}, "Did you mean 'why'"),
           ({"colors": {"team_mate": [1, 2, 3]}}, "Did you mean 'team_mates'")]
    for d, msg in bad:
        with pytest.raises(CDSInterpretError, match=msg):
            PanelConfig.from_dict(d)


def test_check_panel_says_what_the_new_rows_need(tmp_path):
    pol = team_policy()
    pk, _ = team_pack(pol, donors=False, decoders=False)
    cfg = PanelConfig.from_dict({"rows": [{"kind": "why"}, {"kind": "looking"}, {"kind": "why", "features": ["d"]}]})
    out = check_panel(cfg, pk)
    assert any("decoder directions" in p and "add_decoders" in p for p in out)
    assert any("recorded moments" in p and "add_donors" in p for p in out)
    assert any("lists feature 'd'" in p for p in out)
    assert check_panel(None, pk) == []                     # the default panel just leaves them out
    pk2, _ = team_pack(pol)
    assert check_panel(cfg, pk2) == [next(p for p in out if "lists feature 'd'" in p)]


# --------------------------------------------------------------------------------------------------- the pack
def test_pack_keeps_decoders_and_donors(tmp_path):
    pol = team_policy()
    pk, sae = team_pack(pol)
    back = load_pack(save_pack(pk, tmp_path / "p.npz"))
    assert back["version"] == 2
    for f, g in zip(pk["features"], back["features"]):
        assert np.array_equal(f["d"], g["d"])
        assert np.allclose(g["d"], sae.feature_direction(f["index"]).detach().numpy())
    assert [g["cars"] for g in back["donors"]] == [2, 4]
    for a, b in zip(pk["donors"], back["donors"]):
        assert np.array_equal(a["me"], b["me"]) and set(a["data"]) == set(b["data"])
        assert all(np.array_equal(a["data"][k], b["data"][k]) for k in a["data"])
    th = Thoughts(back)
    assert th.fd.shape == (3, 16) and th.donor_cars() == [2, 4]
    # a pack without them (as 0.1 wrote them) still loads; add_decoders adds them, only from the features' own SAE
    old, _ = team_pack(pol, donors=False, decoders=False)
    th_old = Thoughts(load_pack(save_pack(old, tmp_path / "old.npz")))
    assert th_old.fd is None and th_old.donors == []
    with pytest.raises(CDSInterpretError, match="isn't the dictionary"):
        add_decoders(old, S.TopKSAE(16, 24, 4).init_from(torch.randn(50, 16)))
    add_decoders(old, sae)
    assert np.allclose(Thoughts(old).fd, th.fd)
    # donors: one group per number of cars, replaced when given again; kickoffs left out
    m = random_moments(30, 2, 5)
    m.frames.data["ball_pos"][:10] = [0.0, 0.0, 92.75]
    m.frames.data["ball_vel"][:10] = 0.0
    add_donors(pk, m, n=100)
    g2 = next(g for g in pk["donors"] if g["cars"] == 2)
    assert len(g2["me"]) == 20 and [g["cars"] for g in pk["donors"]] == [2, 4]
    with pytest.raises(CDSInterpretError, match="me="):
        add_donors(pk, m.frames)


# ------------------------------------------------------------------------------------------- why this move
def test_why_is_activation_times_direction_times_the_gradient():
    pol = team_policy()
    pk, sae = team_pack(pol)
    th = Thoughts(pk)
    ex = Explainer(th, pol, freeze=False)
    assert ex.available == {"why": True, "looking": [2, 4]} and ex.notes == []
    m = random_moments(5, 4, 9)
    x = pol.observe(m.frames, m.me)
    h = pol.activations(x, ["3"])["3"]
    out = pol.forward(x).detach()
    for i in range(3):
        r = th.read(h[i].numpy(), out[i].numpy())
        a = r["action"]["index"]
        why = {w["name"]: w["push"] for w in ex.why(x[i], h[i].numpy(), a)}
        # by hand: the gradient of log p(a) at the layer, through the rest of the network
        hh = h[i:i + 1].clone().requires_grad_(True)
        lp = torch.log_softmax(pol.net[4](hh), -1)[0, a]
        (g,) = torch.autograd.grad(lp, hh)
        for f, feat in zip(pk["features"], r["features"]):
            want = feat["act"] * float(sae.feature_direction(f["index"]).detach() @ g[0])
            assert why[f["name"]] == pytest.approx(want, rel=1e-4, abs=1e-6)
    assert all(p.requires_grad for p in pol.net.parameters())    # freeze=False leaves the bot's own net alone


def test_why_needs_its_pieces_and_says_so():
    pol = team_policy()
    pk, _ = team_pack(pol, decoders=False, donors=False)
    ex = Explainer(Thoughts(pk), pol)
    assert ex.available == {"why": False, "looking": []}
    assert any("add_decoders" in n for n in ex.notes) and any("add_donors" in n for n in ex.notes)
    pk2, _ = team_pack(pol)
    pk2["layer"] = "9"
    ex2 = Explainer(Thoughts(pk2), team_policy())
    assert not ex2.available["why"] and any("no layer '9'" in n for n in ex2.notes)
    assert Explainer(Thoughts(pk2), None).available == {"why": False, "looking": []}


# ---------------------------------------------------------------------------------------------- looking at
def test_looking_is_the_exact_shapley_split_over_the_parts():
    pol = team_policy()
    pk, _ = team_pack(pol)
    th = Thoughts(pk)
    ex = Explainer(th, pol, freeze=False)
    for cars, parts in ((2, {"own car", "ball", "opponents"}), (4, {"own car", "ball", "opponents", "team-mates"})):
        m = random_moments(3, cars, 11 + cars)
        state = {k: v[0] for k, v in m.frames.take([0]).data.items()}
        a = int(pol.act(pol.observe(m.frames.take([0]), [0]))[0])
        look = ex.looking(state, 0, a, donors=3)
        assert set(look) == parts and sum(look.values()) == pytest.approx(1.0) and min(look.values()) >= 0
    # against the coalition table and exact Shapley values, with one donor
    m = random_moments(2, 4, 21)
    live = m.frames.take([0])
    codec = AT.FramesCodec(pol)
    dparts = codec.decompose(*th.donors[1])
    a = int(pol.act(pol.observe(live, [0]))[0])
    got = looking_shares(pol, codec, live, 0, a, {k: v[:1] for k, v in dparts.items()},
                         AT.role_signature(dparts)[:1], donors=1)
    groups = codec.groups(boost_apart=False)
    p = codec.decompose(live, [0])
    both = {k: np.concatenate([np.asarray(p[k]), np.asarray(dparts[k][:1])]) for k in p}
    tab = AT.coalition_table(lambda inp: pol.log_probs(inp)[:, a], both, np.array([[1, 1]]), groups, codec.compose)
    phi = np.abs(AT.shapley(tab[:, :1])[0, :, 0])
    want = dict(zip(groups, phi / phi.sum()))
    assert got == pytest.approx(want, rel=1e-4, abs=1e-6)
    # no recorded moment of this mode (3v3): no reading, and no error
    m3 = random_moments(2, 6, 3)
    assert ex.looking({k: v[0] for k, v in m3.frames.take([0]).data.items()}, 0, 0) is None
    assert ex.looking_ok


def test_compute_gives_the_readings_to_the_panel_car_only():
    pol = team_policy()
    pk, _ = team_pack(pol)
    th = Thoughts(pk)
    ex = Explainer(th, pol, freeze=False)
    m = random_moments(2, 4, 4)
    m.frames.data["car_team"][:] = [0, 0, 1, 1]
    x = pol.observe(m.frames.take([0, 0]), [0, 1])
    h = pol.activations(x, ["3"])["3"].numpy()
    msg = {"frame": 5, "layer": h, "output": pol.forward(x).detach().numpy(), "obs": x.numpy(), "panel": 1,
           "why": True, "looking": True, "state": {k: v[0] for k, v in m.frames.take([0]).data.items()}, "me": 1,
           "donors": 2}
    res = compute(th, msg, None, ex)
    assert "why" not in res["cars"][0] and "looking" not in res["cars"][0]
    assert [w["name"] for w in res["cars"][1]["why"]] == ["a", "b", "c"]
    assert set(res["cars"][1]["looking"]) == {"own car", "ball", "opponents", "team-mates"}
    assert res["notes"] == []


def test_the_real_worker_builds_the_policy_and_answers(tmp_path):
    pol = team_policy()
    pk, _ = team_pack(pol)
    path = save_pack(pk, tmp_path / "w.npz")
    link = WorkerLink(W.serve, path, None, "test_overlay_team:make_policy")
    try:
        deadline = time.time() + 120
        while not link.ready and time.time() < deadline:
            link.poll()
            time.sleep(0.05)
        assert link.ready and link.hello["available"] == {"why": True, "looking": [2, 4]}
        m = random_moments(1, 2, 8)
        x = pol.observe(m.frames, m.me)
        msg = {"frame": 3, "layer": pol.activations(x, ["3"])["3"].numpy(), "output": pol.forward(x).detach().numpy(),
               "obs": x.numpy(), "panel": 0, "why": True, "looking": True, "me": 0,
               "state": {k: v[0] for k, v in m.frames.data.items()}}
        assert link.send(msg)
        got = None
        while got is None and time.time() < deadline:
            got = link.poll()
            time.sleep(0.01)
        car = got["cars"][0]
        assert len(car["why"]) == 3 and sum(car["looking"].values()) == pytest.approx(1.0)
    finally:
        link.close()


def test_try_readings_and_the_cli(tmp_path, capsys):
    pol = team_policy()
    pk, _ = team_pack(pol)
    path = save_pack(pk, tmp_path / "t.npz")
    t = try_readings(path, pol)
    assert t["problems"] == [] and t["why_ms"] > 0 and t["looking_ms"] > 0
    from cds_interpret.cli import main
    toml = tmp_path / "panel.toml"
    assert main(["panel", str(toml)]) == 0
    assert main(["check-panel", str(toml), str(path), "--policy", "test_overlay_team:make_policy"]) == 0
    out = capsys.readouterr().out
    assert "why this move: works" in out and "looking at: works" in out
    old, _ = team_pack(pol, decoders=False, donors=False)
    path2 = save_pack(old, tmp_path / "old.npz")
    assert main(["check-panel", str(toml), str(path2), "--policy", "test_overlay_team:make_policy"]) == 1
    assert "add_decoders" in capsys.readouterr().out


# --------------------------------------------------------------------------------------------------- drawing
RESULT = {"value": 0.4, "action": {"words": "boost", "p": 0.7}, "probes": {"Threat": 0.2},
          "looks": {"Threat": {"unit": "%"}},
          "features": [{"name": "about to hit", "act": 1.0, "on": True}, {"name": "ball high", "act": 0.0, "on": False},
                       {"name": "full tank", "act": 0.0, "on": False}],
          "why": [{"name": "ball high", "push": 0.9}, {"name": "about to hit", "push": 0.2},
                  {"name": "full tank", "push": -0.8}],
          "looking": {"own car": 0.5, "ball": 0.3, "opponents": 0.15, "team-mates": 0.05}}


def _texts(r):
    return [c[1][0] for c in r.calls if c[0] == "draw_string_2d"]


def test_several_markers_the_subtitle_and_the_new_rows():
    r = FakeRenderer()
    Panel().draw(r, fake_flat(), RESULT, car_index=2, marks=[(0, (10, 10)), (2, (20, 20)), (4, (30, 30))],
                 subtitle="car 2 of 3 - on camera", state={}, frame=10, landing_true=(0, 0, 93))
    names = [c[0] for c in r.calls]
    assert names.count("draw_polyline_3d") == 6                    # two rings a marker, three markers
    lines_to_cars = [c[1][0] for c in r.calls if c[0] == "draw_line_3d" and isinstance(c[1][0], tuple)]
    assert sorted(a[1] for a in lines_to_cars) == [0, 2, 4]        # each ring joined to its own car
    texts = _texts(r)
    assert "car 2 of 3 - on camera" in texts and "Why this move" in texts and "+ ball high" in texts
    assert "+ about to hit" in texts and "- full tank" in texts and "Looking at" in texts
    assert [t for t in texts if t.split(" ")[0] in
                                                          ("car", "ball", "opp", "mates") and t[-1].isdigit()] == ["car 50", "ball 30", "opp 15", "mates 5"]
    over = [c[1] for c in r.calls if c[0] == "draw_string_3d"]
    assert [o[0] for o in over] == ["ball high"] and over[0][1] == ("car", 2)   # the move's driver, above its car
    r2 = FakeRenderer()
    cfg = PanelConfig.from_dict({"markers": {"car_label_from": "noticing", "landing_line": False}})
    Panel(cfg).draw(r2, fake_flat(), RESULT, car_index=2, marks=[(0, (1, 1))], state={}, frame=1)
    assert [c[1][0] for c in r2.calls if c[0] == "draw_string_3d"] == ["about to hit"]
    assert not [c for c in r2.calls if c[0] == "draw_line_3d" and isinstance(c[1][0], tuple)]
    r3 = FakeRenderer()                                            # a copy without the panel: only its marker
    Panel().draw(r3, fake_flat(), None, marks=[(4, (5, 5))])
    assert [c[0] for c in r3.calls].count("draw_polyline_3d") == 2 and "draw_rect_2d" not in [c[0] for c in r3.calls]


def test_car_label_in_team_colour():
    """The word above the panel's car is in its team's colour, unless the config gives it a colour of its own."""
    def label_colour(cfg=None, team=None, th=RESULT):
        r = FakeRenderer()
        Panel(cfg or PanelConfig()).draw(r, fake_flat(), th, car_index=1, state={}, frame=1, team=team)
        return next(c[1][3] for c in r.calls if c[0] == "draw_string_3d")
    assert label_colour(team=0) == (80, 160, 255, 255) and label_colour(team=1) == (255, 150, 40, 255)
    assert label_colour() == (255, 196, 64, 255)                  # no team given: the notice colour, as before
    assert label_colour(PanelConfig.from_dict({"colors": {"team_orange": [200, 100, 0]}}), 1) == (200, 100, 0, 255)
    fixed = PanelConfig.from_dict({"markers": {"car_label_color": [1, 2, 3]}})
    assert label_colour(fixed, 0) == label_colour(fixed, 1) == (1, 2, 3, 255)
    styled = PanelConfig.from_dict({"features": {"ball high": {"color": [9, 9, 9]}}})
    assert label_colour(styled, 1) == (9, 9, 9, 255)              # a feature's own colour, when its style gives one


def test_why_holds_a_moment_and_looking_keeps_its_last_reading():
    p, state = Panel(), {}
    p.draw(FakeRenderer(), fake_flat(), RESULT, state=state, frame=10)
    later = dict(RESULT, why=[{"name": "ball high", "push": 0.05}, {"name": "about to hit", "push": 0.0},
                              {"name": "full tank", "push": 0.0}], looking=None)
    r = FakeRenderer()
    p.draw(r, fake_flat(), later, state=state, frame=40)          # 0.25 s later: still listed, the bar still there
    texts = _texts(r)
    assert "+ ball high" in texts and "car 50" in texts and "- full tank" not in texts
    r = FakeRenderer()
    p.draw(r, fake_flat(), later, state=state, frame=200)         # long after: gone
    assert "+ ball high" not in _texts(r) and "no named idea" in _texts(r)


def test_panel_keeps_its_size():
    """A move of one line or two, nothing or three features noticed, no driver or two and one against, a "looking at"
    reading or not yet, one car or a team: the background is the same height every time, so no row ever moves."""
    def height(words, feats, why, look, keep=True):
        r = FakeRenderer()
        th = {"value": 0.5, "action": {"words": words, "p": 0.5}, "probes": {"Threat": 0.2},
              "looks": {"Threat": {"unit": "%"}},
              "features": [{"name": f"f{i}", "act": 1.0, "on": i < feats} for i in range(4)],
              "why": why, "looking": look}
        cfg = PanelConfig(keep_size=keep)
        Panel(cfg).draw(r, fake_flat(), th, car_index=0, state={}, frame=1, available={"why", "looking"})
        return round(next(c[1][3] for c in r.calls if c[0] == "draw_rect_2d"), 9)   # the first is the background
    short, long_ = "boost", "throttle, steer left, nose down, roll left, boost, handbrake"
    whys = (None, [{"name": "f0", "push": 0.0}],
            [{"name": "f0", "push": 0.9}, {"name": "f1", "push": 0.8}, {"name": "f2", "push": -0.9}])
    looks = (None, {"own car": 0.5, "ball": 0.5}, {"own car": 0.4, "ball": 0.3, "opponents": 0.2, "team-mates": 0.1})
    hs = {height(w, f, why, look) for w in (short, long_) for f in (0, 1, 3) for why in whys for look in looks}
    assert len(hs) == 1, hs
    loose = {height(w, f, why, look, keep=False) for w in (short, long_) for f in (0, 3) for why in whys[1:]
             for look in looks[1:]}
    assert len(loose) > 1                                          # keep_size = false: rows shrink to what they hold


# ------------------------------------------------------------------------------------------------ the bots
def team_packet(t, spectated, n_cars=6):
    cars = [(500.0 * j - 1500, -2000.0 + 4000 * (j % 2), 17.0, j % 2) for j in range(n_cars)]
    pk = fake_packet(ball=(200.0, 300.0 + 50 * t, 400.0), cars=cars, t=t)
    pk.match_info.last_spectated = spectated
    for j, p in enumerate(pk.players):
        p.player_id, p.name = 1000 + j, f"bot{j}"
    return pk


def _hivemind_class():
    class FakeHivemind:
        """Stands in for rlbot.managers.Hivemind: three blue cars, one batched forward pass for all of them."""

        team, indices, ball_prediction = 0, [0, 2, 4], SimpleNamespace(slices=[])

        def __init__(self):
            self.sent, self.events, self.renderer = [], [], FakeRenderer()
            self.logger = SimpleNamespace(warning=lambda m: self.events.append(("warning", m)),
                                          info=lambda m: self.events.append(("info", m)))

        def initialize(self):
            self.policy = team_policy().net

        def get_outputs(self, packet):
            fr = ci.from_rlbot_packet(packet)
            x = torch.from_numpy(team_obs(ci.Frames({k: np.repeat(v, 3, 0) for k, v in fr.data.items()}),
                                          np.array(self.indices)))
            with torch.no_grad():
                return dict(zip(self.indices, self.policy(x).clone()))

        def _packet_processor(self, packet):
            self.sent.append(self.get_outputs(packet))

        def retire(self):
            pass

    return FakeHivemind


def _drawn(r):
    """(the car the panel's word floats over, the cars joined to a ring, the panel's texts) of one bot's drawing."""
    label = [c[1][1].index for c in r.calls if c[0] == "draw_string_3d"]
    rings = sorted(c[1][0].index for c in r.calls if c[0] == "draw_line_3d" and hasattr(c[1][0], "index"))
    return label, rings, _texts(r)


def test_a_hivemind_shows_the_car_on_camera_and_every_ring(tmp_path):
    pytest.importorskip("rlbot")
    from cds_interpret.overlay import with_thoughts
    pol = team_policy()
    pk, _ = team_pack(pol)
    add_donors(pk, random_moments(40, 6, 3), n=20)               # 3v3 moments: this match is 3v3
    add_land(pk, "1", np.ones((32, 2), np.float32) * 10, [100.0, 200.0])
    path = save_pack(pk, tmp_path / "team.npz")
    Base = _hivemind_class()
    plain = Base()
    plain.initialize()
    bot = with_thoughts(Base, path, network="policy", worker=False, draw_every=4,
                        policy=team_policy())()
    bot.initialize()
    seen = {}
    for f in range(1, 120):
        spec = 2 if f < 60 else (3 if f < 90 else 4)              # car 2 on camera, then an opponent, then car 4
        pk_ = team_packet(f / 120, spec)
        plain._packet_processor(pk_)
        bot.renderer.calls.clear()
        bot._packet_processor(pk_)
        if any(c[0] == "draw_rect_2d" for c in bot.renderer.calls):
            seen[f] = _drawn(bot.renderer)
    for a, b in zip(plain.sent, bot.sent):                        # bit for bit the same controls
        assert all(torch.equal(a[i], b[i]) for i in a)
    assert not [e for e in bot.events if isinstance(e, tuple) and e[0] == "warning"], bot.events
    early = [v for f, v in seen.items() if 20 < f < 60]
    mid = [v for f, v in seen.items() if 65 < f < 90]
    late = [v for f, v in seen.items() if f > 95]
    assert early and mid and late
    assert all(v[1] == [0, 2, 4] for v in early + mid + late)     # every car's ring, each joined to its car
    assert all("car 2 of 3 - on camera" in v[2] for v in early)
    assert all("car 2 of 3 - last on camera" in v[2] for v in mid)
    assert all("car 3 of 3 - on camera" in v[2] for v in late)
    assert any("Why this move" in v[2] for v in early) and any("Looking at" in v[2] for v in early)
    assert all(v[0] in ([], [2]) for v in early) and all(v[0] in ([], [4]) for v in late)


def _bot_class():
    class FakeBot:
        """Stands in for rlbot.managers.Bot, one process per car: index, team, player_id, match_config."""

        team, ball_prediction = 0, SimpleNamespace(slices=[])

        def __init__(self, index):
            self.index, self.player_id = index, 1000 + index
            self.sent, self.events, self.renderer = [], [], FakeRenderer()
            self.logger = SimpleNamespace(warning=lambda m: self.events.append(("warning", m)),
                                          info=lambda m: self.events.append(("info", m)))
            agents = ["me/bot", "other/bot", "me/bot", "other/bot", "someone/else", "other/bot"]
            self.match_config = SimpleNamespace(player_configurations=[
                SimpleNamespace(player_id=1000 + j, team=j % 2, variety=SimpleNamespace(agent_id=a))
                for j, a in enumerate(agents)])

        def initialize(self):
            self.policy = team_policy().net

        def get_output(self, packet):
            x = torch.from_numpy(team_obs(ci.from_rlbot_packet(packet), np.array([self.index])))
            with torch.no_grad():
                return self.policy(x)[0].clone()

        def _packet_processor(self, packet):
            self.sent.append(self.get_output(packet))

        def retire(self):
            pass

    return FakeBot


def test_one_process_per_car_agree_on_the_panel(tmp_path):
    pytest.importorskip("rlbot")
    from cds_interpret._rlbot import our_cars
    from cds_interpret.overlay import with_thoughts
    pol = team_policy()
    pk, _ = team_pack(pol)
    add_land(pk, "1", np.ones((32, 2), np.float32) * 10, [100.0, 200.0])
    path = save_pack(pk, tmp_path / "team.npz")
    Cls = with_thoughts(_bot_class(), path, network="policy", worker=False, draw_every=4)
    bots = {i: Cls(i) for i in (0, 2)}                            # two copies of "me/bot"; car 4 is someone else's
    for b in bots.values():
        b.initialize()
    assert our_cars(bots[0], team_packet(0.1, 0)) == [0, 2]
    panels = {0: [], 2: []}
    for f in range(1, 100):
        spec = 2 if f < 50 else 0
        pk_ = team_packet(f / 120, spec)
        for i, b in bots.items():
            b.renderer.calls.clear()
            b._packet_processor(pk_)
            names = [c[0] for c in b.renderer.calls]
            if "draw_polyline_3d" in names or "draw_rect_2d" in names:
                panels[i].append((f, "draw_rect_2d" in names, _drawn(b.renderer)))
    first = {i: [p for f, p, _ in v if 20 < f < 50] for i, v in panels.items()}
    second = {i: [p for f, p, _ in v if f > 70] for i, v in panels.items()}
    assert first[2] and all(first[2]) and first[0] and not any(first[0])      # car 2 on camera: its copy has the panel
    assert second[0] and all(second[0]) and second[2] and not any(second[2])  # then car 0's copy
    for i, v in panels.items():                                   # each copy draws only its own car's ring
        assert all(d[1] == [i] for _, _, d in v)
    assert any("car 2 of 2 - on camera" in d[2] for f, _, d in panels[2] if f < 50)
