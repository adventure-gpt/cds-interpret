"""The examples run: the RLBot demo bot plays from packets, and with the overlay its controls stay the same."""

import importlib.util
from pathlib import Path

import pytest
import torch

from test_core import fake_packet
from test_overlay import FakeRenderer

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ready(bot, index=0, team=0):
    bot.index, bot.team, bot.name = index, team, "demo"
    bot.renderer = FakeRenderer()
    sent = []
    bot._game_interface.send_msg = sent.append              # no RLBot server here: keep what would be sent
    bot.initialize()
    return sent


def test_rlbot_demo_bot_with_and_without_thoughts():
    pytest.importorskip("rlbot")
    mod = _load(ROOT / "examples" / "rlbot_demo_bot" / "bot.py", "demo_bot_example")
    plain = mod.DemoBot("test/demo")
    sent_plain = _ready(plain)
    from cds_interpret.overlay import with_thoughts
    pack = ROOT / "examples" / "rlbot_demo_bot" / "thoughts.npz"
    if not pack.exists():
        pytest.skip("thoughts.npz has not been made (examples/rlbot_demo_bot/make_pack.py)")
    shown = with_thoughts(mod.DemoBot, pack=pack, network="net", worker=False, draw_every=4)("test/demo")
    sent_shown = _ready(shown)
    for f in range(1, 50):
        pk = fake_packet(ball=(300.0, 1000.0 + 10 * f, 93.0), t=f / 120)
        plain._packet_processor(pk)
        shown._packet_processor(pk)
    inputs_plain = [m for m in sent_plain if hasattr(m, "controller_state")]
    inputs_shown = [m for m in sent_shown if hasattr(m, "controller_state")]
    assert len(inputs_plain) == len(inputs_shown) == 49
    assert all(repr(a.controller_state) == repr(b.controller_state) for a, b in zip(inputs_plain, inputs_shown))
    assert any(c[0] == "draw_string_2d" for c in shown.renderer.calls)
    assert isinstance(mod.bot_class(["--thoughts", "--record"]), type)
    torch.set_num_threads(2)
