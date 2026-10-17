"""The thoughts overlay is display only: packs, their reading, the worker link, the panel, and an RLBot bot class
wrapped with it - whose controls stay bit for bit the same."""

import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import cds_interpret as ci
from cds_interpret import sae as S
from cds_interpret.errors import ShapeError
from cds_interpret.overlay import (
    Thoughts,
    WorkerLink,
    add_land,
    compute,
    draw_thoughts,
    load_pack,
    make_pack,
    save_pack,
)
from cds_interpret.overlay import worker as W
from cds_interpret.probes import LinearProbe

from conftest import tiny_mlp
from test_core import fake_packet


def fake_flat():
    return SimpleNamespace(Color=lambda r, g, b, a=255: (r, g, b, a),
                           Vector3=lambda x=0.0, y=0.0, z=0.0: SimpleNamespace(x=x, y=y, z=z),
                           CarAnchor=lambda i, v: ("car", i), TextHAlign=SimpleNamespace(Left=0, Center=1, Right=2))


class FakeRenderer:
    can_render = True

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *a, **k: self.calls.append((name, a))


def _sae(width=16, m=32):
    torch.manual_seed(0)
    return S.TopKSAE(width, m, 4).init_from(torch.randn(200, width))


def test_pack_roundtrip_and_reading(tmp_path):
    sae = _sae()
    probe = LinearProbe(torch.ones(16) * 0.1, 0.2, "bin", 1.0)
    logit = LinearProbe(torch.ones(16), -1.0, "bin", 1.0, "logit")
    pack = make_pack("body.3", sae, {"a feature": 3}, {"a feature": 0.5}, {"Chance": probe, "Logit": logit,
                     "Time": (LinearProbe(torch.zeros(16), 1.5, "reg", 1.0), {"unit": "s"})}, ci.lookup_table(),
                     title="T", outlook_probe=LinearProbe(torch.zeros(16), 0.3, "reg", 1.0))
    add_land(pack, "body.1", np.ones((8, 2), np.float32), [10.0, -10.0])
    path = save_pack(pack, tmp_path / "p")
    assert path.endswith(".npz")
    back = load_pack(path)
    th = Thoughts(back)
    h = torch.randn(16)
    r = th.read(h.numpy(), np.arange(90, dtype=np.float32))
    pre = (h / sae.scale - sae.b_pre) @ sae.W_enc[:, 3] + sae.b_enc[3]
    assert r["features"][0]["act"] == pytest.approx(max(float(pre.detach()), 0.0), abs=1e-4)
    assert r["features"][0]["on"] == (float(pre.detach()) > 0.5)
    assert r["probes"]["Chance"] == pytest.approx(float(np.clip(0.1 * h.sum() + 0.2, 0, 1)), abs=1e-5)
    assert r["probes"]["Logit"] == pytest.approx(float(torch.sigmoid(h.sum() - 1.0)), abs=1e-5)
    assert r["looks"]["Time"]["unit"] == "s" and r["value"] == pytest.approx(0.3)
    assert r["action"]["index"] == 89 and r["action"]["words"] == ci.describe_action(ci.lookup_table()[89])
    assert th.land_xy(np.ones(8)) == pytest.approx([18.0, -2.0])
    with pytest.raises(ShapeError, match="different widths"):
        make_pack("x", None, None, None, {"a": probe, "b": LinearProbe(torch.ones(5), 0.0, "reg", 1.0)})


def test_compute_and_panel():
    sae = _sae()
    pack = make_pack("body.3", sae, {"chasing": 1}, None, {"Chance": LinearProbe(torch.ones(16) * 0.01, 0.5, "bin", 1.0)},
                     ci.lookup_table())
    th = Thoughts(pack)
    msg = {"layer": np.random.default_rng(0).normal(size=(2, 16)).astype(np.float32),
           "output": np.zeros((2, 90), np.float32), "frame": 7}
    res = compute(th, msg)
    assert res["frame"] == 7 and len(res["cars"]) == 2 and res["cars"][0]["action"]["p"] == pytest.approx(1 / 90)
    msg["obs"] = np.zeros((2, 5), np.float32)
    res = compute(th, msg, critic=lambda obs: np.full(len(obs), 0.4))
    assert res["cars"][1]["value"] == pytest.approx(0.4)
    r = FakeRenderer()
    car = dict(res["cars"][0], features=[{"name": "chasing", "act": 1.0, "on": True}])
    state = {}
    draw_thoughts(r, fake_flat(), car, car_index=0, landing_true=(0, 0, 93), landing_think=(100, 100), state=state,
                  frame=10, title="A VERY LONG TITLE FOR A NARROW PANEL")
    names = [c[0] for c in r.calls]
    assert names[0] == "begin_rendering" and names[-1] == "end_rendering" and "draw_string_3d" in names
    assert "draw_polyline_3d" in names                            # the landing ring
    texts = [c[1][0] for c in r.calls if c[0] == "draw_string_2d"]
    assert any(t.startswith("> chasing") for t in texts) and any("Outlook" in t for t in texts)
    assert all(len(t) <= 33 for t in texts)                       # nothing wider than the panel
    r2 = FakeRenderer()                                           # the feature stays listed for a moment after it stops
    draw_thoughts(r2, fake_flat(), dict(car, features=[{"name": "chasing", "act": 0.0, "on": False}]), state=state,
                  frame=40)
    assert any(c[1][0].startswith("> chasing") for c in r2.calls if c[0] == "draw_string_2d")


def test_worker_link_never_blocks_on_a_slow_start():
    """A request bigger than a pipe's buffer, sent while the worker is still starting, would wait in the bot: the link
    sends nothing until the worker says it is ready."""
    link = WorkerLink(W.echo, 2.0)
    try:
        big = {"x": np.zeros(8000, np.float32)}                   # 32 KB
        t = time.time()
        assert link.send(big) is False                            # not ready yet: skipped, at once
        assert time.time() - t < 0.5
        deadline = time.time() + 60
        while not link.ready and time.time() < deadline:
            link.poll()
            time.sleep(0.05)
        assert link.ready and link.send(big) is True
        got = None
        while got is None and time.time() < deadline:
            got = link.poll()
            time.sleep(0.01)
        assert got is not None and got["x"].shape == (8000,)
    finally:
        link.close()


def test_the_real_worker_answers_after_saying_ready(tmp_path):
    sae = _sae()
    pack = make_pack("body.3", sae, {"a": 2}, None, {"P": LinearProbe(torch.ones(16) * 0.01, 0.5, "bin", 1.0)},
                     ci.lookup_table())
    path = save_pack(pack, tmp_path / "w.npz")
    link = WorkerLink(W.serve, path, None)
    try:
        deadline = time.time() + 120
        while not link.ready and time.time() < deadline:
            assert link.poll() is None
            time.sleep(0.05)
        assert link.ready
        h = np.random.default_rng(1).normal(size=(1, 16)).astype(np.float32)
        assert link.send({"layer": h, "output": np.zeros((1, 90), np.float32), "frame": 3})
        got = None
        while got is None and time.time() < deadline:
            got = link.poll()
            time.sleep(0.01)
        assert got["frame"] == 3 and got["cars"][0] == compute(Thoughts(load_pack(path)), {"layer": h,
                                                               "output": np.zeros((1, 90), np.float32)})["cars"][0]
    finally:
        link.close()


# ------------------------------------------------------------------------------------ an RLBot bot class, wrapped
def _fake_bot_class():
    class FakeBot:
        """Stands in for rlbot.managers.Bot: _packet_processor calls get_output, then sends the controls."""

        name, index, team = "fake", 0, 0
        ball_prediction = SimpleNamespace(slices=[])

        def __init__(self):
            self.sent, self.events, self.renderer = [], [], FakeRenderer()
            self.logger = SimpleNamespace(warning=lambda m: self.events.append(("warning", m)))

        def initialize(self):
            torch.manual_seed(0)
            self.policy = tiny_mlp(0, d_in=6, n_out=90)

        def get_output(self, packet):
            x = torch.full((1, 6), float(packet.match_info.frame_num % 7))
            with torch.no_grad():
                return self.policy(x)[0].clone()

        def _packet_processor(self, packet):
            self.events.append("decide")
            self.sent.append(self.get_output(packet))
            self.events.append("sent")

        def retire(self):
            self.events.append("retired")

    return FakeBot


def test_with_thoughts_changes_nothing_and_draws_after_the_controls(tmp_path):
    pytest.importorskip("rlbot")
    from cds_interpret.overlay import with_thoughts
    Base = _fake_bot_class()
    plain = Base()
    plain.initialize()
    sae = S.TopKSAE(8, 16, 2).init_from(torch.randn(50, 8))
    pack = make_pack("3", sae, {"f": 0}, None, {"P": LinearProbe(torch.ones(8), 0.0, "bin", 1.0)}, ci.lookup_table())
    path = save_pack(pack, tmp_path / "pack.npz")
    Cls = with_thoughts(Base, path, network="policy", worker=False, draw_every=1)
    bot = Cls()
    bot.initialize()
    for f in range(1, 40):
        pk = fake_packet(t=f / 120)
        plain._packet_processor(pk)
        bot._packet_processor(pk)
    assert all(torch.equal(a, b) for a, b in zip(plain.sent, bot.sent))      # bit for bit the same controls
    assert ("warning" not in [e[0] for e in bot.events if isinstance(e, tuple)]), bot.events
    assert any(c[0] == "begin_rendering" for c in bot.renderer.calls)
    bot.retire()
    assert bot.events[-1] == "retired"


def test_overlay_errors_switch_it_off_not_the_bot(tmp_path):
    pytest.importorskip("rlbot")
    from cds_interpret.overlay import with_thoughts
    Base = _fake_bot_class()
    pack = make_pack("99", None, None, None, {"P": LinearProbe(torch.ones(8), 0.0, "bin", 1.0)})
    path = save_pack(pack, tmp_path / "bad.npz")                  # a layer the network does not have
    bot = with_thoughts(Base, path, network="policy", worker=False)()
    bot.initialize()
    bot._packet_processor(fake_packet())
    assert len(bot.sent) == 1 and any(isinstance(e, tuple) and "overlay off" in e[1] for e in bot.events)
