"""Recorders: RocketSim games, the real game through RLBot (with a stand-in bot class), and RLGym v2."""

import numpy as np
import pytest
import torch

import cds_interpret as ci
from cds_interpret.errors import MeshesNotFound

from conftest import meshes_or_skip
from test_core import fake_packet
from test_overlay import _fake_bot_class


def test_meshes_not_found_says_how_to_get_them(tmp_path, monkeypatch):
    from cds_interpret.record import sim as R
    monkeypatch.setenv("CDS_INTERPRET_MESHES", str(tmp_path / "nowhere"))
    monkeypatch.chdir(tmp_path)
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(MeshesNotFound, match=r"cds-interpret\[rlgym\]"):
        R.find_meshes()


@pytest.mark.sim
def test_rocketsim_games_of_the_demo_bot(tmp_path):
    meshes_or_skip()
    pol = ci.demo.policy()
    m = ci.record.rocketsim(pol, games=2, seconds=20, opponent="chaser", every=3, out=tmp_path / "rec", log=None)
    assert len(m) == 2 * 100 and m.n_games == 2 and m.obs.shape == (200, ci.demo.OBS_DIM)   # 300 decisions a game
    assert m.output.shape == (200, 90) and m.tracks is not None and len(m.tracks.frames) == 2 * 300
    assert {"threat", "land_t", "touch_soon", "goal_soon"} <= set(m.labels)
    assert m.tracks.frames.ev_touch.sum() > 0                      # someone touched the ball in 40 s
    x = pol.observe(m.frames, m.me)                                # what it saw is what its obs function builds
    assert torch.allclose(x, torch.from_numpy(m.obs), atol=1e-5)
    back = ci.Moments.load(tmp_path / "rec")
    assert len(back) == len(m) and back.meta["source"] == "rocketsim"
    selfplay = ci.record.rocketsim(pol, games=1, seconds=4, log=None)  # both sides recorded against itself
    assert set(np.unique(selfplay.me)) == {0, 1}


def test_rlbot_recorder_writes_what_the_bot_saw(tmp_path):
    from cds_interpret.record.rlbot import with_recorder
    Base = _fake_bot_class()
    plain = Base()
    plain.initialize()
    bot = with_recorder(Base, out=tmp_path / "rlbot", network="policy", every=2, flush_every=50)()
    bot.initialize()
    for f in range(1, 121):
        touches = [None, 0.5] if f > 60 else None
        pk = fake_packet(ball=(0.0, 100.0 * f, 93.0), t=f / 120, scores=(1 if f > 100 else 0, 0), touches=touches)
        plain._packet_processor(pk)
        bot._packet_processor(pk)
    bot.retire()
    assert all(torch.equal(a, b) for a, b in zip(plain.sent, bot.sent))
    m = ci.Moments.load_all(tmp_path / "rlbot")
    assert len(m) == 60 and m.obs.shape == (60, 6) and m.output.shape == (60, 90)
    assert np.allclose(m.obs[:, 0], (np.arange(1, 121, 2) % 7))   # the input the network really got
    assert m.tracks is not None and len(m.tracks.frames) == 120
    assert m.tracks.frames.ev_touch[:, 1].sum() == 1 and (m.tracks.frames.goal == 0).sum() == 1
    assert m.labels["goal_soon"][49] == 1                          # (the goal falls in step 99: 1.5 s ahead of step 98)
    assert np.isnan(m.labels["goal_soon"][-1])                     # (the recording ends inside its 3 s window)


def test_rlgym_bridge_runs_an_obs_builder():
    pytest.importorskip("rlgym.rocket_league.api")
    from rlgym.rocket_league.obs_builders import DefaultObs

    from conftest import make_frames
    fr = make_frames(3, cars=4, car_pos=np.tile([[0.0, -1000, 17], [500, -2000, 17], [0, 1000, 17], [0, 2000, 17]],
                                                (3, 1, 1)))
    states, ids = ci.rlgym.to_rlgym(fr)
    assert ids[0] == {0: "blue-0", 1: "blue-1", 2: "orange-0", 3: "orange-1"}
    assert np.allclose(states[0].cars["blue-0"].physics.forward, [0, 1, 0], atol=1e-6)
    obs = ci.rlgym.obs_fn(DefaultObs(zero_padding=3))(fr, np.array([0, 2, 3]))
    assert obs.shape == (3, 172)
    assert obs[0, 52 + 1] == pytest.approx(-1000 / 2300, abs=1e-5)         # blue-0's y, blue's own frame
    assert obs[1, 52 + 1] == pytest.approx(-1000 / 2300, abs=1e-5)         # orange-0 at +1000, seen inverted


@pytest.mark.sim
def test_record_an_rlgym_env():
    pytest.importorskip("rlgym.rocket_league.api")
    meshes_or_skip()
    from rlgym.api import RLGym
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.reward_functions import GoalReward
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator, MutatorSequence
    env = RLGym(state_mutator=MutatorSequence(FixedTeamSizeMutator(1, 1), KickoffMutator()),
                obs_builder=DefaultObs(zero_padding=1), action_parser=RepeatAction(LookupTableAction(), repeats=8),
                reward_fn=GoalReward(), termination_cond=GoalCondition(), truncation_cond=TimeoutCondition(5.0),
                transition_engine=RocketSimEngine())
    torch.manual_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(92, 32), torch.nn.ReLU(), torch.nn.Linear(32, 90))
    pol = ci.Policy(net, actions=LookupTableAction.make_lookup_table())
    m = ci.record.rlgym(env, pol, episodes=2, log=None)
    assert m.n_games == 2 and m.obs.shape[1] == 92 and set(np.unique(m.me)) == {0, 1}
    assert "threat" in m.labels and m.tracks is not None
