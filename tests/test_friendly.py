"""Mistakes a newcomer makes get plain-words errors, and the numbers shown never rest on too few examples."""

import numpy as np
import pytest
import torch

import cds_interpret as ci
from cds_interpret.errors import CDSInterpretError, LayerNotFound, ShapeError

from conftest import make_frames, meshes_or_skip


@pytest.fixture(scope="module")
def small():
    pol = ci.demo.policy()
    m = ci.demo.synthetic_moments(n_games=16, per_game=40)
    st = ci.Study(pol, m, log=None)
    return pol, m, st


def test_wrong_input_width_is_explained():
    net = torch.nn.Sequential(torch.nn.Linear(26, 8), torch.nn.ReLU(), torch.nn.Linear(8, 90))
    pol = ci.Policy(net, obs=ci.demo.obs, actions=ci.lookup_table())          # demo.obs builds 30 numbers
    x = pol.observe(make_frames(4), 0)
    with pytest.raises(ShapeError, match="couldn't run on these inputs"):
        pol.layers(x)
    with pytest.raises(ShapeError, match="couldn't run on these inputs"):
        pol.act(x)


def test_action_table_must_match_the_outputs():
    pol = ci.Policy(torch.nn.Linear(30, 90), obs=ci.demo.obs, actions=ci.lookup_table()[:24])
    with pytest.raises(ShapeError, match="90 outputs, but the action table has 24"):
        pol.check_actions(90, exact=True)


def test_layer_and_label_mistakes(small):
    pol, m, st = small
    with pytest.raises(LayerNotFound, match="Did you mean"):
        st.features("body.33")
    with pytest.raises(CDSInterpretError, match="path in your network"):
        st.features(-1)
    with pytest.raises(LayerNotFound):
        ci.Study(pol, m, layers=["trunk.30"], log=None)
    with pytest.raises(CDSInterpretError, match="Did you mean 'threat'"):
        st.probes(labels=["threats"])
    with pytest.raises(CDSInterpretError, match="Did you mean 'goal_side'"):
        st.probe_edit("body.3", "goalside")


def test_skipped_labels_say_why(small):
    _, _, st = small
    pr = st.probes(baselines=False)
    assert "kickoff" not in pr.labels or "kickoff" in pr.skipped or pr.best("kickoff")
    for why in pr.skipped.values():
        assert isinstance(why, str) and why
    text = pr.summary()
    if pr.skipped:
        assert "Not scored:" in text
    assert st.why_not("goal_side") is None


def test_pack_checks(small):
    _, _, st = small
    st.features("body.3", m=32, k=4, steps=100)
    with pytest.raises(CDSInterpretError, match="don't exist"):
        st.make_pack("body.3", features={"oops": 9999})
    with pytest.raises(LayerNotFound):
        st.make_pack("body.4")
    pack = st.make_pack("body.3", probes={"Goal side": "goal_side"})
    assert pack["outlook"] is None                     # synthetic moments have no goals: no outlook bar


def test_edit_directions_must_fit_the_layer(small):
    _, _, st = small
    with pytest.raises(ShapeError, match="has 64"):
        st.edits("body.3", {"random": ci.intervene.project_out(torch.randn(65))})


def test_last_back_means_nothing_in_1v1():
    lab = ci.concept_labels(make_frames(3), 0)
    assert np.isnan(lab["last_back"]).all()
    four = ci.concept_labels(make_frames(3, cars=4, car_pos=np.tile([[0.0, -3000, 17], [0, -1000, 17],
                                                                     [0, 1000, 17], [0, 3000, 17]], (3, 1, 1))), 0)
    assert (four["last_back"] == 1).all()


def test_saving_never_mixes_recordings(tmp_path):
    m = ci.demo.synthetic_moments(n_games=3, per_game=5)
    m.save(tmp_path / "rec")
    with pytest.raises(CDSInterpretError, match="already holds a recording"):
        m.save(tmp_path / "rec")
    m2 = ci.demo.synthetic_moments(n_games=3, per_game=5, seed=2)
    m2.save(tmp_path / "rec", if_exists="append", renumber=True)
    back = ci.Moments.load(tmp_path / "rec")
    assert len(back) == 30 and back.n_games == 6
    m.save(tmp_path / "rec", if_exists="overwrite")
    assert len(ci.Moments.load(tmp_path / "rec")) == 15


@pytest.mark.sim
def test_rocketsim_rerun_into_the_same_folder(tmp_path):
    meshes_or_skip()
    pol = ci.demo.policy()
    ci.record.rocketsim(pol, games=1, seconds=4, opponent="idle", out=tmp_path / "r", log=None)
    with pytest.raises(CDSInterpretError, match="already holds a recording"):
        ci.record.rocketsim(pol, games=1, seconds=4, opponent="idle", out=tmp_path / "r", log=None)
    ci.record.rocketsim(pol, games=1, seconds=4, opponent="idle", out=tmp_path / "r", if_exists="append", log=None)
    assert ci.Moments.load(tmp_path / "r").n_games == 2
    bad = ci.Policy(pol.net, obs=ci.demo.obs, actions=ci.lookup_table()[:24])
    with pytest.raises(ShapeError, match="action table has 24"):
        ci.record.rocketsim(bad, games=1, seconds=2, log=None)
    with pytest.raises(CDSInterpretError, match="isn't one the recorder knows"):
        ci.record.rocketsim(pol, games=1, seconds=2, opponent="chaserr", log=None)
