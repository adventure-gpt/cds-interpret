"""What the RLBot recorder and the thoughts overlay share: finding your bot's network, read-only taps on it, the
events between two packets, the ball's path from RLBot's prediction, and running code after the controls are sent.

Nothing here imports rlbot at import time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from .errors import CDSInterpretError
from .frames import from_rlbot_packet
from .labels import PATH_T


def find_network(bot: Any, network: str | Callable[[Any], torch.nn.Module] | torch.nn.Module | None) -> torch.nn.Module:
    """Your bot's torch network: ``network`` as a dotted attribute path of the bot (``"agent.policy"``), a function
    of the bot, a module, or None to search the bot's attributes (two levels deep) for the largest nn.Module."""
    if isinstance(network, torch.nn.Module):
        return network
    if callable(network) and not isinstance(network, str):
        net = network(bot)
    elif isinstance(network, str):
        net = bot
        for part in network.split("."):
            if not hasattr(net, part):
                raise CDSInterpretError(f"the bot has no attribute {network!r} (stopped at {part!r}): pass network= the "
                                        "dotted name of the attribute that holds your torch network, as set in "
                                        "initialize()")
            net = getattr(net, part)
    else:
        found: list[tuple[int, str, torch.nn.Module]] = []

        def look(obj: Any, prefix: str, depth: int) -> None:
            for k, v in list(getattr(obj, "__dict__", {}).items()):
                if isinstance(v, torch.nn.Module):
                    found.append((sum(p.numel() for p in v.parameters()), prefix + k, v))
                elif depth > 0 and not k.startswith("_") and hasattr(v, "__dict__") and not isinstance(v, type):
                    look(v, f"{prefix}{k}.", depth - 1)
        look(bot, "", 1)
        if not found:
            raise CDSInterpretError("no torch network found on the bot: pass network= (the attribute that holds it, "
                                    "e.g. network=\"policy\", or a function bot -> module)")
        found.sort(key=lambda t: -t[0])
        net = found[0][2]
    if isinstance(net, torch.jit.ScriptModule):
        raise CDSInterpretError("the bot's network is a TorchScript module, whose layers PyTorch cannot hook. Load it "
                                "as an eager module in your bot (cds_interpret.eager_copy(net) for MLP blocks).")
    if not isinstance(net, torch.nn.Module):
        raise CDSInterpretError(f"network {network!r} is a {type(net).__name__}, not a torch.nn.Module")
    return net


class PacketEvents:
    """The events of the step between two packets: touches (a car's latest touch changed), demolitions (a car's
    demolition count went up; a car became demolished) and goals (a team's score went up)."""

    def __init__(self) -> None:
        self.prev: Any = None

    def update(self, packet: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        c = max(1, len(packet.players))
        touch, made, demoed = np.zeros(c, np.float32), np.zeros(c, np.float32), np.zeros(c, np.float32)
        goal = -1
        prev = self.prev
        if prev is not None and len(prev.players) == len(packet.players):
            for j, (a, b) in enumerate(zip(prev.players, packet.players)):
                ta, tb = getattr(a, "latest_touch", None), getattr(b, "latest_touch", None)
                if tb is not None and (ta is None or tb.game_seconds != ta.game_seconds):
                    touch[j] = 1.0
                da = getattr(getattr(a, "score_info", None), "demolitions", 0)
                db = getattr(getattr(b, "score_info", None), "demolitions", 0)
                made[j] = float(db > da)
                demoed[j] = float(b.demolished_timeout > 0 and not a.demolished_timeout > 0)
            for t, (ta2, tb2) in enumerate(zip(prev.teams, packet.teams)):
                if tb2.score > ta2.score:
                    goal = t
        self.prev = packet
        return touch, made, demoed, goal


def packet_row(packet: Any) -> dict[str, np.ndarray]:
    """One packet as a row of Frames fields (with prev_controls from each car's last input)."""
    fr = from_rlbot_packet(packet)
    row = {k: v[0] for k, v in fr.data.items()}
    ctl = np.zeros((fr.n_cars, 8), np.float32)
    for j, p in enumerate(packet.players):
        li = getattr(p, "last_input", None)
        if li is not None:
            ctl[j] = [li.throttle, li.steer, li.pitch, li.yaw, li.roll, float(li.jump), float(li.boost),
                      float(li.handbrake)]
    row["prev_controls"] = ctl
    return row


def prediction_path(prediction: Any, now: float) -> np.ndarray | None:
    """RLBot's ball prediction at labels.PATH_T seconds after ``now`` (nearest slices), [K, 3]; None without one."""
    slices = getattr(prediction, "slices", None)
    if not slices:
        return None
    ts = np.array([s.game_seconds for s in slices], np.float64) - float(now)
    pos = np.array([[s.physics.location.x, s.physics.location.y, s.physics.location.z] for s in slices], np.float32)
    if ts[-1] < PATH_T[-1] - 0.05:
        return None
    idx = np.clip(np.searchsorted(ts, PATH_T), 0, len(ts) - 1)
    return pos[idx]


def landing_point(prediction: Any, step: int = 4, below: float = 120.0) -> tuple[float, float, float] | None:
    """Where RLBot's prediction says the ball next comes down (first slice under ``below`` uu, every ``step``-th)."""
    slices = getattr(prediction, "slices", None)
    if not slices:
        return None
    for s in slices[::step]:
        loc = s.physics.location
        if loc.z < below:
            return (loc.x, loc.y, loc.z)
    return None


def wrap_after_decision(base: Any, name: str, after: Callable[[Any, Any], None],
                        before: Callable[[Any, Any], None] | None = None,
                        init: Callable[[Any], None] | None = None, retire: Callable[[Any], None] | None = None) -> type:
    """A subclass of an RLBot v5 Bot / Hivemind class whose decisions are the parent's own, with ``before(bot,
    packet)`` run just before the decision and ``after(bot, packet)`` run once its controls have been sent. Errors in
    the added code are logged and switch it off; they never reach the bot."""
    if not hasattr(base, "_packet_processor"):
        raise CDSInterpretError(f"{base.__name__} does not look like an RLBot v5 Bot or Hivemind class (no "
                                "_packet_processor). cds-interpret works with the rlbot package 2.x (RLBot v5).")

    def initialize(self: Any) -> None:
        base.initialize(self)
        self._cds_on = True
        if init is not None:
            try:
                init(self)
            except Exception as e:                    # the added code must never stop the bot
                _log(self, f"cds-interpret: {name} off ({e})")
                self._cds_on = False

    def _packet_processor(self: Any, packet: Any) -> None:
        on = getattr(self, "_cds_on", False)
        if on and before is not None:
            try:
                before(self, packet)
            except Exception as e:                    # (as above)
                _log(self, f"cds-interpret: {name} stopped ({e})")
                self._cds_on = on = False
        base._packet_processor(self, packet)          # the decision and its controls go out first
        if on:
            try:
                after(self, packet)
            except Exception as e:                    # (as above)
                _log(self, f"cds-interpret: {name} stopped ({e})")
                self._cds_on = False

    def retire_(self: Any) -> None:
        if retire is not None and getattr(self, "_cds_on", False):
            try:
                retire(self)
            except Exception as e:
                _log(self, f"cds-interpret: {name} could not finish ({e})")
        base.retire(self)

    return type(f"{base.__name__}With{name.title().replace(' ', '')}", (base,),
                {"initialize": initialize, "_packet_processor": _packet_processor, "retire": retire_,
                 "__doc__": f"{base.__name__}, playing exactly as it does, with cds-interpret's {name}."})


def _log(bot: Any, msg: str) -> None:
    logger = getattr(bot, "logger", None) or getattr(bot, "_logger", None)
    if logger is not None:
        logger.warning(msg)
    else:
        print(msg)


def our_indices(bot: Any) -> list[int]:
    """The car indices the bot controls (a Bot: [index]; a Hivemind: its indices)."""
    if hasattr(bot, "indices") and getattr(bot, "indices", None):
        return [int(i) for i in bot.indices]
    return [int(bot.index)]
