"""Record the real game: wrap your RLBot v5 bot class, and it writes what it saw and did while it plays - exactly as
it plays without the recorder.

    # run.py of your bot folder
    from cds_interpret.record.rlbot import with_recorder
    from my_bot import MyBot

    if __name__ == "__main__":
        with_recorder(MyBot, out="recordings/rlbot", network="policy").run(...)   # instead of MyBot(...).run(...)

    # later, anywhere
    moments = ci.Moments.load_all("recordings/rlbot")

How it stays out of the way: read-only forward hooks copy your network's input and output while your bot's own
forward pass runs (its outputs stay bit for bit the same), and everything else - the game state from the packet,
the events, writing to disk - happens after your controls have been sent. Any error in the recorder switches it off
and is logged; your bot plays on.

What is kept: every packet during play (kickoffs and live play; replays and pauses are skipped) as tracks, and at
every ``every``-th decision of your bot (a packet where its network ran) a moment with the network's input and
output and RLBot's ball prediction for the labels. Each match is one game, written to its own folder under ``out``
(one per bot and match), in chunks while it plays and at the end.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .._rlbot import PacketEvents, find_network, our_indices, packet_row, prediction_path, wrap_after_decision
from ..hooks import LayerTap
from ._builder import Recording, empty_events

_LIVE_PHASES = ("Kickoff", "Active")


def _live(packet: Any) -> bool:
    phase = getattr(packet.match_info, "match_phase", None)
    if phase is None:
        return True
    name = getattr(phase, "name", None) or str(phase).split(".")[-1]
    return name in _LIVE_PHASES


def _rows(tap: LayerTap, key: str) -> torch.Tensor | None:
    calls = tap.out.get(key)
    if not calls:
        return None
    x = torch.cat([c.reshape(c.shape[0] if c.ndim > 1 else 1, -1) for c in calls])
    return x.float().cpu()


def with_recorder(bot_class: type, out: str | os.PathLike[str] = "recordings/rlbot",
                  network: str | Callable[[Any], torch.nn.Module] | None = None, input_layer: str = "",
                  output_layer: str = "", every: int = 3, flush_every: int = 7200, tick_skip: int | None = None) -> type:
    """Your RLBot v5 bot class (a ``rlbot.managers.Bot`` or ``Hivemind`` subclass) with a recorder.

    Args:
        out: the folder recordings go under (one subfolder per bot and match).
        network: where your bot keeps its torch network - a dotted attribute name (``"policy"``,
            ``"agent.actor"``), a function bot -> module, or None to pick the largest nn.Module on the bot.
        input_layer: the module of that network whose INPUT is your observation ("" = the network itself).
        output_layer: the module whose output is the action output - logits or probabilities ("" = the network).
        every: keep a moment at every n-th decision.
        flush_every: write to disk every this many packets (and at the end).
        tick_skip: ticks between your bot's decisions, if it holds its controls between them (for the meta only).
    """
    out = Path(out)

    def init(bot: Any) -> None:
        net = find_network(bot, network)
        bot._cds_rec_taps = (LayerTap(net, {"x": input_layer}, inputs=True, keep="all").attach(),
                             LayerTap(net, {"y": output_layer}, keep="all").attach())
        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (bot.name or "bot"))
        bot._cds_rec_dir = out / f"{name}-{our_indices(bot)[0]}-{stamp}"
        bot._cds_rec = None
        bot._cds_rec_events = PacketEvents()
        bot._cds_rec_step = 0
        bot._cds_rec_decisions = 0
        bot._cds_rec_last = None
        bot._cds_rec_meta = {"source": "rlbot", "bot": bot.name, "every": every, "tick_skip": tick_skip,
                             "step_seconds": 1 / 120}

    def before(bot: Any, packet: Any) -> None:
        for t in bot._cds_rec_taps:
            t.clear()

    def flush(bot: Any) -> None:
        rec = bot._cds_rec
        if rec is None or len(rec) == 0:
            return
        last = bot._cds_rec_last
        if last is not None:                     # the newest step's events are not known yet: keep it for later
            return
        m = rec.finish(future=False)
        m.save(bot._cds_rec_dir, if_exists="append")         # (chunks of one match: the same game)
        bot._cds_rec = Recording(rec.c, rec.step_seconds, rec.meta)

    def after(bot: Any, packet: Any) -> None:
        tin, tout = bot._cds_rec_taps
        if not _live(packet):
            touch, made, demoed, goal = bot._cds_rec_events.update(packet)
            if bot._cds_rec_last is not None:                     # a goal ended live play: close the last step
                rec, ti = bot._cds_rec, bot._cds_rec_last
                if len(touch) == rec.c:
                    rec.set_events(ti, touch, made, demoed, goal)
                bot._cds_rec_last = None
            return
        row = packet_row(packet)
        c = len(row["car_boost"])
        if bot._cds_rec is None:
            bot._cds_rec = Recording(c, 1 / 120, bot._cds_rec_meta)
        rec = bot._cds_rec
        touch, made, demoed, goal = bot._cds_rec_events.update(packet)
        if bot._cds_rec_last is not None:
            if len(touch) == rec.c:
                rec.set_events(bot._cds_rec_last, touch, made, demoed, goal)
            bot._cds_rec_last = None
        if rec.n_steps >= flush_every:
            flush(bot)
            rec = bot._cds_rec
        if c != rec.c:                                            # a car joined or left: start a new chunk
            flush(bot)
            bot._cds_rec = rec = Recording(c, 1 / 120, bot._cds_rec_meta)
        ti = rec.add_step(0, bot._cds_rec_step, {**row, **empty_events(c), "goal": -1})
        bot._cds_rec_last = ti
        bot._cds_rec_step += 1
        x, y = _rows(tin, "x"), _rows(tout, "y")
        if x is None:
            return                                                # the network did not run on this packet
        bot._cds_rec_decisions += 1
        if (bot._cds_rec_decisions - 1) % every:
            return
        ours = our_indices(bot)
        if len(x) != len(ours):
            ours = ours[:len(x)]
        path = prediction_path(bot.ball_prediction, packet.match_info.seconds_elapsed)
        for k, idx in enumerate(ours):
            if idx < c:
                rec.add_moment(ti, idx, x[k].numpy(), None if y is None or len(y) <= k else y[k].numpy(), path)

    def retire(bot: Any) -> None:
        for t in bot._cds_rec_taps:
            t.detach_hooks()
        rec = bot._cds_rec
        if rec is not None and bot._cds_rec_last is not None:
            ti = bot._cds_rec_last
            z = np.zeros(rec.c, np.float32)
            rec.set_events(ti, z, z, z, -1)
            bot._cds_rec_last = None
        flush(bot)

    return wrap_after_decision(bot_class, "recorder", after, before, init, retire)
