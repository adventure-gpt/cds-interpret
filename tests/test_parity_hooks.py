"""Hooks for bots whose world is bigger than the generic game state: factories with arguments, a critic that sees
the bot's own state (with_thoughts extras), "looking at" on the network's input split by the policy's own codec
(input donors), action masks, mirror flags from the observation, and a watch script with its own observation hook."""

import logging
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

import cds_interpret as ci
from cds_interpret import attribution as AT
from cds_interpret._specs import build, is_spec
from cds_interpret.errors import CDSInterpretError
from cds_interpret.overlay import (
    Explainer,
    Thoughts,
    add_input_donors,
    compute,
    load_pack,
    load_policy,
    make_pack,
    save_pack,
    watch_thoughts,
)
from cds_interpret.overlay.explain import looking_shares_inputs
from cds_interpret.probes import LinearProbe

from test_core import fake_packet
from test_overlay import FakeRenderer, _fake_bot_class


# ---------------------------------------------------------------------------------------------- a toy codec
class ToyCodec:
    """An input of 12 numbers: own car (3), ball (3), two other cars (3 each), whose roles are its last two... no: the
    roles ride in their own array, as a real codec's do (the input itself says nothing of them here)."""

    def __init__(self, roles=(2, 1)):
        self.roles = np.asarray(roles, np.int8)

    def decompose(self, x):
        x = torch.as_tensor(np.asarray(x, np.float32))
        n = len(x)
        return {"own": x[:, 0:3].clone(), "ball": x[:, 3:6].clone(), "cars": x[:, 6:12].reshape(n, 2, 3).clone(),
                "cars:role": np.tile(self.roles, (n, 1))}

    def compose(self, parts):
        n = len(parts["own"])
        return torch.cat([torch.as_tensor(parts["own"]), torch.as_tensor(parts["ball"]),
                          torch.as_tensor(parts["cars"]).reshape(n, 6)], 1)

    def groups(self, parts):
        roles = np.asarray(parts["cars:role"][0])
        g = {"own car": AT.Group("own"), "ball": AT.Group("ball")}
        if (roles == 2).any():
            g["opponents"] = AT.Group("cars", role=2)
        if (roles == 1).any():
            g["team-mates"] = AT.Group("cars", role=1)
        return g


def toy_net(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(12, 16), torch.nn.ReLU(), torch.nn.Linear(16, 90))


def toy_policy(seed: int = 0, codec: bool = True) -> ci.Policy:
    """For the spec tests: "test_parity_hooks:toy_policy" with keyword arguments."""
    return ci.Policy(toy_net(seed).eval(), actions=ci.lookup_table(), codec=ToyCodec() if codec else None)


def state_critic(scale: float = 1.0):
    """A critic that reads the bot's own state from the game state it is given ("my_value" from extras)."""
    def value(obs, state=None, cars=None):
        return np.full(len(obs), scale * float(np.asarray(state["my_value"]).reshape(-1)[0]) + len(cars or []))
    return value


# ------------------------------------------------------------------------------------------------- the specs
def test_factories_take_keyword_arguments():
    assert is_spec("a:b") and is_spec(("a:b", {"x": 1})) and is_spec(["a:b", {}]) and not is_spec(("a:b", 1))
    pol = load_policy(("test_parity_hooks:toy_policy", {"seed": 3, "codec": False}))
    assert isinstance(pol, ci.Policy) and pol.codec is None
    assert torch.equal(pol.net[0].weight, toy_net(3)[0].weight)
    assert build(("test_parity_hooks:state_critic", {"scale": 2.0}), "critic")(np.zeros((2, 3)), {"my_value": 1.5},
                                                                               [0, 2]).tolist() == [5.0, 5.0]
    with pytest.raises(CDSInterpretError, match="doesn't take the arguments"):
        build(("test_parity_hooks:toy_policy", {"colour": 1}), "policy")
    with pytest.raises(CDSInterpretError, match="could not be loaded"):
        build("test_parity_hooks:nothing_here", "policy")


# --------------------------------------------------------------------------------- looking at, by the input codec
def _toy_pack(tmp=None):
    net = toy_net()
    pack = make_pack("1", None, None, None, {"P": LinearProbe(torch.ones(16) * 0.1, 0.0, "bin", 1.0)}, ci.lookup_table())
    rng = np.random.default_rng(0)
    add_input_donors(pack, {4: rng.normal(size=(50, 12)).astype(np.float32),
                            2: rng.normal(size=(40, 12)).astype(np.float32)}, n=20)
    if tmp is not None:
        pack = load_pack(save_pack(pack, tmp / "toy.npz"))
    return net, pack


def test_input_donors_round_trip_and_looking_by_the_codec(tmp_path):
    net, pack = _toy_pack(tmp_path)
    th = Thoughts(pack)
    assert [c for c, _ in th.input_donors] == [2, 4] and th.input_donors[1][1].shape == (20, 12)
    assert th.donor_cars() == [2, 4]
    pol = ci.Policy(net.eval(), actions=ci.lookup_table(), codec=ToyCodec())
    ex = Explainer(th, pol, freeze=False)
    assert ex.available == {"why": False, "looking": [2, 4]} and ex.input_codec is not None
    x = np.random.default_rng(5).normal(size=12).astype(np.float32)
    a = int(pol.act(x[None])[0])
    look = ex.looking(None, 0, a, donors=3, x=x, n_cars=4)
    assert set(look) == {"own car", "ball", "opponents", "team-mates"} and sum(look.values()) == pytest.approx(1.0)
    assert ex.looking(None, 0, a, x=x, n_cars=6) is None            # no recorded inputs of a 3v3 match
    # against the coalition table by hand, one donor
    codec = ToyCodec()
    dparts = codec.decompose(th.input_donors[1][1])
    sig = np.zeros(20, np.int64) + 8 + 1
    got = looking_shares_inputs(pol, codec, x, a, {k: v[:1] for k, v in dparts.items()}, sig[:1], donors=1)
    p = codec.decompose(x[None])
    both = {k: (torch.cat([torch.as_tensor(p[k]), torch.as_tensor(dparts[k][:1])]) if torch.is_tensor(p[k])
                else np.concatenate([p[k], dparts[k][:1]])) for k in p}
    groups = codec.groups(p)
    tab = AT.coalition_table(lambda inp: pol.log_probs(inp)[:, a], both, np.array([[1, 1]]), groups, codec.compose)
    phi = np.abs(AT.shapley(tab[:, :1])[0, :, 0])
    assert got == pytest.approx(dict(zip(groups, phi / phi.sum())), rel=1e-4, abs=1e-6)


def test_compute_masks_mirrors_and_hands_the_critic_its_state():
    _net, pack = _toy_pack()
    th = Thoughts(pack)
    out = np.full((2, 90), -5.0, np.float32)
    table = ci.lookup_table()
    right = next(i for i, r in enumerate(table) if r[1] > 0.5 and r[5] < 0.5)
    left = next(i for i, r in enumerate(table) if r[1] < -0.5 and r[5] < 0.5)
    out[:, right], out[:, left] = 5.0, 4.0
    mask = np.ones((2, 90), bool)
    mask[1, right] = False                                          # car 2 may not steer right
    msg = {"layer": np.zeros((2, 16), np.float32), "output": out, "mask": mask, "mirrored": [True, False],
           "obs": np.zeros((2, 12), np.float32), "state": {"my_value": np.array([0.25])}, "cars": [0, 2]}
    res = compute(th, msg, critic=state_critic(2.0))["cars"]
    assert res[0]["action"]["index"] == right and "steer left" in res[0]["action"]["words"]   # said as sent
    assert res[1]["action"]["index"] == left and res[1]["action"]["p"] > 0.9                  # among the allowed
    assert res[0]["value"] == pytest.approx(2.5) and res[1]["value"] == pytest.approx(2.5)


# --------------------------------------------------------------------------------- with_thoughts: extras
def test_with_thoughts_extras_reach_the_critic(tmp_path):
    pytest.importorskip("rlbot")
    from cds_interpret.overlay import with_thoughts
    Base = _fake_bot_class()
    pack = make_pack("3", None, None, None, {"P": LinearProbe(torch.ones(8), 0.0, "bin", 1.0)}, ci.lookup_table())
    path = save_pack(pack, tmp_path / "p.npz")
    calls = []

    def extras(bot, packet):
        calls.append(packet.match_info.frame_num)
        return {"my_value": np.array([float(packet.match_info.frame_num)])}
    bot = with_thoughts(Base, path, network="policy", worker=False, draw_every=1, extras=extras,
                        critic=("test_parity_hooks:state_critic", {"scale": 1.0}))()
    bot.initialize()
    plain = Base()
    plain.initialize()
    for f in range(1, 30):
        pk = fake_packet(t=f / 120)
        plain._packet_processor(pk)
        bot._packet_processor(pk)
    assert all(torch.equal(a, b) for a, b in zip(plain.sent, bot.sent)) and bot._cds_on
    assert calls and bot._cds_th_state["cars"][0]["value"] == pytest.approx(calls[-1] + 1)     # + one car
    with pytest.raises(CDSInterpretError, match="critic"):
        with_thoughts(Base, path, critic=lambda o: o)


# --------------------------------------------------------------------------------- watch: its own observation
def test_watch_with_its_own_observation_hook(tmp_path):
    pytest.importorskip("rlbot")
    net, pack = _toy_pack()
    path = save_pack(pack, tmp_path / "w.npz")
    table = ci.lookup_table()
    right = next(i for i, r in enumerate(table) if r[1] > 0.5 and r[5] < 0.5)
    seen = []

    class Observer:
        """Keeps a count between packets, as a real port keeps its timers."""

        def __init__(self):
            self.n = 0

        def __call__(self, packet, cars, field_info):
            self.n += 1
            seen.append((list(cars), field_info))
            m = np.zeros((len(cars), 90), bool)
            m[:, right] = True
            return {"obs": np.full((len(cars), 12), 0.1 * self.n, np.float32), "mirrored": [True] * len(cars),
                    "mask": m}
    cls = watch_thoughts(ci.Policy(net.eval(), actions=table), path, cars="somebot", observe=Observer())
    obj = cls.__new__(cls)
    obj.renderer, obj.logger = FakeRenderer(), logging.getLogger("cds-hooks-test")
    obj.update_rendering_status = lambda *a, **k: None
    obj.ball_prediction, obj.field_info = NS(slices=[]), "FIELD"
    obj.initialize()
    assert obj._cds_on
    drawn = []
    for f in range(1, 40):
        pk = fake_packet(t=f / 120)
        pk.players[0].name, pk.players[1].name = "SomeBot", "Other"
        pk.match_info.last_spectated = 0
        obj.renderer.calls.clear()
        obj.handle_packet(pk)
        if any(c[0] == "draw_rect_2d" for c in obj.renderer.calls):
            drawn.append([c[1][0] for c in obj.renderer.calls if c[0] == "draw_string_2d"])
    assert seen and seen[0] == ([0], "FIELD") and drawn
    assert any("steer left" in t for t in drawn[-1])                # the only allowed move, said as sent (mirrored)
