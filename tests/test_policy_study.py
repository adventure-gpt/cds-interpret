"""Bring your own policy, the demo bot, Moments on disk, and a whole Study down to its report."""

import numpy as np
import pytest
import torch

import cds_interpret as ci
from cds_interpret.errors import CDSInterpretError, ShapeError
from cds_interpret.moments import Tracks

from conftest import make_frames, tiny_mlp


# ------------------------------------------------------------------------------------------------ the policy
def test_lookup_table_and_words():
    t = ci.lookup_table()
    assert t.shape == (90, 8) and len({tuple(r) for r in t}) == 90
    assert ci.describe_action([1, -1, 0, -1, 0, 0, 1, 0]) == "throttle, steer left, boost"
    assert ci.describe_action([0, 0, -1, 0, 0, 1, 0, 0]) == "flip forward"
    assert ci.describe_action(np.zeros(8)) == "coast"


def test_policy_outputs_and_actions():
    net = tiny_mlp(n_out=90)
    x = torch.randn(10, 6)
    pol = ci.Policy(net, actions=ci.lookup_table(), name="t")
    lp = pol.log_probs(x)
    assert torch.allclose(lp.exp().sum(-1), torch.ones(10), atol=1e-5)
    a = pol.act(x)
    assert a.shape == (10,) and np.array_equal(pol.controls(x), ci.lookup_table()[a])
    soft = torch.nn.Sequential(net, torch.nn.Softmax(-1))
    pol_p = ci.Policy(soft, output="probs", actions=ci.lookup_table())
    assert np.array_equal(pol_p.act(x), a)                        # the same choices from probabilities
    with pytest.raises(CDSInterpretError, match="build its input"):
        pol.observe(make_frames(2), 0)
    with pytest.raises(ShapeError, match=r"\[A, 8\]"):
        ci.Policy(net, actions=np.zeros((90, 7)))
    cont = ci.Policy(torch.nn.Sequential(torch.nn.Linear(6, 8), torch.nn.Tanh()), output="controls")
    assert cont.controls(x).shape == (10, 8) and not cont.discrete


def test_layers_random_baseline_and_custom_forward():
    class AC(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = tiny_mlp(0)
            self.critic = tiny_mlp(1, n_out=1)

    ac = AC()
    pol = ci.Policy(ac, forward=lambda x: ac.actor(x), value=lambda x: ac.critic(x)[:, 0])
    x = torch.randn(8, 6)
    assert pol.default_layers(x) == ["actor.1", "actor.3"]            # the critic is never called by forward
    assert pol.value(x).shape == (8,)
    rnd = pol.random_like(3)
    a, b = pol.activations(x, ["actor.3"])["actor.3"], rnd.activations(x, ["actor.3"])["actor.3"]
    assert a.shape == b.shape and not torch.allclose(a, b)
    assert torch.equal(pol.activations(x, ["actor.3"])["actor.3"], a)   # the original is untouched


@pytest.mark.filterwarnings("ignore:`torch.jit.script` is deprecated")
def test_eager_copy_of_torchscript():
    net = torch.nn.Sequential(torch.nn.Linear(6, 12), torch.nn.LeakyReLU(0.1), torch.nn.LayerNorm(12),
                              torch.nn.Linear(12, 4), torch.nn.Softmax(-1))
    scripted = torch.jit.script(net)
    with pytest.raises(CDSInterpretError, match="TorchScript"):
        ci.Policy(scripted)
    eager = ci.eager_copy(scripted)
    x = torch.randn(5, 6)
    assert torch.allclose(eager(x), net(x), atol=1e-6)
    assert ci.Policy(eager, output="probs").default_layers(x) == ["1"]


# ------------------------------------------------------------------------------------------------ the demo bot
def test_demo_bot_plays_like_its_teacher():
    pol = ci.demo.policy()
    m = ci.demo.synthetic_moments(n_games=6, per_game=50)
    x = pol.observe(m.frames, m.me)
    assert x.shape == (300, ci.demo.OBS_DIM)
    taught = ci.demo.nearest_actions(ci.demo.teacher(m.frames, m.me), ci.lookup_table())
    assert (pol.act(x) == taught).mean() > 0.8


# ------------------------------------------------------------------------------------------------ moments
def _moments_with_tracks(n_games=4, steps=40, every=2):
    rng = np.random.default_rng(0)
    t = n_games * steps
    fr = make_frames(t, car_pos=rng.uniform([-3000, -4000, 17], [3000, 4000, 17], (t, 2, 3)),
                     ball_pos=rng.uniform([-3000, -4000, 93], [3000, 4000, 500], (t, 3)))
    fr.data.update({"ev_touch": (rng.random((t, 2)) < 0.05).astype(np.float32),
                    "ev_demo_made": np.zeros((t, 2), np.float32), "ev_demoed": np.zeros((t, 2), np.float32),
                    "goal": np.where(rng.random(t) < 0.02, 0, -1).astype(np.int8)})
    game, step = np.repeat(np.arange(n_games), steps), np.tile(np.arange(steps), n_games)
    tr = Tracks(fr, game, step, 1 / 15)
    keep = np.nonzero(step % every == 0)[0]
    m = ci.Moments(fr.take(keep), 0, game[keep], step[keep], obs=rng.normal(size=(len(keep), 5)).astype(np.float32),
                   tracks=tr)
    return m.with_labels()


def test_moments_roundtrip_with_tracks_and_future_labels(tmp_path):
    m = _moments_with_tracks()
    m.save(tmp_path / "a")
    back = ci.Moments.load(tmp_path / "a")
    assert len(back) == len(m) and back.tracks is not None and np.allclose(back.obs, m.obs)
    assert "touch_soon" in back.labels and np.isfinite(back.labels["touch_soon"][:5]).all()
    m.save(tmp_path / "runs" / "b")
    m.save(tmp_path / "runs" / "c")
    both = ci.Moments.load_all(tmp_path / "runs")
    assert len(both) == 2 * len(m) and both.n_games == 2 * m.n_games
    assert len(np.unique(both.tracks.game)) == 2 * m.n_games
    with pytest.raises(CDSInterpretError, match="no recorded moments"):
        ci.Moments.load(tmp_path / "nothing")


# ------------------------------------------------------------------------------------------------ a whole study
@pytest.fixture(scope="module")
def study():
    pol = ci.demo.policy()
    m = ci.demo.synthetic_moments(n_games=24, per_game=50)
    return ci.Study(pol, m, log=None)


def test_study_probes_know_what_the_demo_bot_was_taught(study):
    assert study.layers == ["body.1", "body.3"]
    pr = study.probes()
    assert pr.score("body.3", "goal_side") > 0.9                   # its teacher turns on which side of the ball it is
    assert pr.score("body.3", "goal_side") > pr.score("random:body.3", "goal_side")
    assert "goal_side" in pr.summary()
    p = pr.probe("body.3", "goal_side")
    assert p.w.shape == (64,)
    with pytest.raises(CDSInterpretError, match=r"no label .not a label."):
        pr.probe("body.3", "not a label")


def test_study_features_edits_pack_and_report(study, tmp_path):
    fs = study.features("body.3", m=64, k=4, steps=300)
    assert 0 <= fs.fvu < 1 and fs.freq.shape == (64,)
    assert "reconstruction" in fs.effect and fs.effect["reconstruction"]["kl"] >= 0
    top = fs.top(5)
    assert top and isinstance(fs.name(top[0]), str)
    fs.name_feature(top[0], "my name")
    assert fs.name(top[0]) == "my name"
    eff = study.edits("body.3", {"no goal side": study.probe_edit("body.3", "goal_side", 0.0)})
    assert eff["no goal side"]["kl"] > 0
    pack = study.make_pack("body.3", features={"my name": top[0]}, probes={"Goal side": "goal_side"})
    assert pack["features"][0]["name"] == "my name" and pack["probes"][0]["unit"] == "%"
    path = study.report(tmp_path / "r" / "report.html")
    page = path.read_text(encoding="utf-8")
    for heading in ("What it knows", "How steady", "Features", "Edits", "my name"):
        assert heading in page, heading


def test_report_sections_are_extensible(study, tmp_path):
    from cds_interpret.report import SECTIONS, section

    @section("My technique", order=35)
    def _(s):
        return "<p>hello from a new technique</p>"
    try:
        page = study.report(tmp_path / "x.html").read_text(encoding="utf-8")
        assert "hello from a new technique" in page
    finally:
        SECTIONS[:] = [s for s in SECTIONS if s.title != "My technique"]


def test_tracking_across_snapshots(study):
    pol = study.policy
    rows = ci.tracking.track([("random", 0, pol.random_like(1)), ("trained", 100, lambda: pol)], study.x, ["body.3"],
                             {"goal_side": study.labels["goal_side"]}, {"goal_side": "bin"}, study.train, study.test,
                             groups=study.games)
    curve = ci.tracking.curves(rows, "body.3", ["goal_side"])["goal_side"]
    assert list(curve[0]) == [0, 100] and curve[1][1] > curve[1][0]
