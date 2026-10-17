"""The demo bot as an RLBot v5 bot, with the thoughts overlay and the recorder one flag away.

    python bot.py                       # what bot.toml runs: the demo bot, plain
    python bot.py --thoughts            # with the live overlay (thoughts.npz: make it with make_pack.py)
    python bot.py --record              # recording the game to recordings/ while it plays

Your own bot works the same way: keep your Bot class as it is, and in its entry point wrap the class with
``with_thoughts`` and / or ``with_recorder`` before you run it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from rlbot import flat
from rlbot.managers import Bot

import cds_interpret as ci

HERE = Path(__file__).resolve().parent
TICK_SKIP = 8                          # a decision every 8 ticks (15 a second), the controls held in between


class DemoBot(Bot):
    def initialize(self) -> None:
        torch.set_num_threads(1)
        self.policy = ci.demo.policy()
        self.net = self.policy.net                       # the network the overlay and the recorder read
        self.controls = flat.ControllerState()
        self.ticks = 0

    def get_output(self, packet: flat.GamePacket) -> flat.ControllerState:
        if self.ticks % TICK_SKIP == 0:
            frames = ci.from_rlbot_packet(packet)
            x = torch.from_numpy(ci.demo.obs(frames, np.array([self.index])))
            with torch.no_grad():
                action = int(self.net(x).argmax(-1))
            t, s, p, y, r, j, b, h = (float(v) for v in self.policy.actions[action])
            self.controls = flat.ControllerState(throttle=t, steer=s, pitch=p, yaw=y, roll=r, jump=j > 0.5,
                                                 boost=b > 0.5, handbrake=h > 0.5)
        self.ticks += 1
        return self.controls


def bot_class(argv: list[str]) -> type:
    cls: type = DemoBot
    if "--thoughts" in argv:
        cls = ci.overlay.with_thoughts(cls, pack=HERE / "thoughts.npz", network="net", title="DEMO BOT'S MIND")
    if "--record" in argv:
        cls = ci.record.with_recorder(cls, out=HERE / "recordings", network="net", tick_skip=TICK_SKIP)
    return cls


if __name__ == "__main__":
    bot_class(sys.argv[1:])("cds-interpret/demo-bot").run(wants_match_communications=False,
                                                          wants_ball_predictions=True)
