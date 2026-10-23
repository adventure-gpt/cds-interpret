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


class FrameGuard:
    """Keeps one bad frame from ending an add-on (the overlay): an error is logged - the first ``first`` of the
    match, then every ``every``-th, and any new kind of error, with its traceback - to the bot's log and, when
    ``log_file`` is given, to that file; the work goes on with the next frame. Only ``limit`` failures IN A ROW (no
    ``ok()`` between them) give up: ``failed`` then returns True.

        guard = FrameGuard(bot, "thoughts overlay", log_file="overlay_errors.log")
        try:
            work(packet); guard.ok()
        except Exception as e:
            if guard.failed(frame, e):
                ...switch it off..."""

    def __init__(self, bot: Any, name: str, limit: int = 600, first: int = 3, every: int = 200,
                 log_file: Any = None):
        self.bot, self.name, self.limit, self.first, self.every = bot, name, int(limit), int(first), int(every)
        self.log_file = log_file
        self.in_a_row = 0
        self.total = 0
        self._kinds: set[str] = set()

    def ok(self) -> None:
        """A frame went well: the run of failures is over."""
        self.in_a_row = 0

    def failed(self, frame: Any, e: BaseException) -> bool:
        """Count and (sometimes) log a failed frame; True when ``limit`` have failed in a row."""
        import traceback
        self.in_a_row += 1
        self.total += 1
        n = self.in_a_row
        kind = f"{type(e).__name__}: {e}"
        new = kind not in self._kinds and len(self._kinds) < 20
        self._kinds.add(kind)
        if self.total <= self.first or self.total % self.every == 0 or n >= self.limit or new:
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            self.write(f"cds-interpret: {self.name}: frame {frame} failed ({n} in a row, {self.total} in all; it "
                       f"carries on, and stops after {self.limit} in a row): {e!r}\n{tb}")
        return n >= self.limit

    def write(self, msg: str) -> None:
        _log(self.bot, msg)
        if self.log_file:
            import time
            from pathlib import Path
            try:
                p = Path(self.log_file)
                p.parent.mkdir(parents=True, exist_ok=True)
                with p.open("a", encoding="utf-8") as f:
                    f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + msg.rstrip() + "\n")
            except OSError:                           # (a log that can't be written must not stop anything)
                pass


def has_ball(packet: Any) -> bool:
    """False for a packet without a ball (after a goal, during the replay): there is nothing to read then."""
    balls = getattr(packet, "balls", None)
    return balls is None or len(balls) > 0


def _log(bot: Any, msg: str, level: str = "warning") -> None:
    logger = getattr(bot, "logger", None) or getattr(bot, "_logger", None)
    if logger is not None and hasattr(logger, level):
        getattr(logger, level)(msg)
    else:
        print(msg)


def our_indices(bot: Any) -> list[int]:
    """The car indices the bot controls (a Bot: [index]; a Hivemind: its indices)."""
    if hasattr(bot, "indices") and getattr(bot, "indices", None):
        return [int(i) for i in bot.indices]
    return [int(bot.index)]


def our_cars(bot: Any, packet: Any) -> list[int]:
    """The player indices of every car this bot drives in the match, lowest first: a Hivemind's own cars; for a Bot
    (RLBot runs one process per car), its own car and the cars of its team run by the same bot - the same agent id in
    the match configuration. Every copy of the bot works out the same list from the same packet."""
    mine = our_indices(bot)
    if getattr(bot, "indices", None):
        return sorted(set(mine))
    try:
        configs = list(getattr(getattr(bot, "match_config", None), "player_configurations", None) or [])
        me_id = getattr(bot, "player_id", None)
        agent = next((getattr(c.variety, "agent_id", None) for c in configs if c.player_id == me_id), None)
        if not agent:
            return mine
        ids = {c.player_id for c in configs
               if getattr(c.variety, "agent_id", None) == agent and int(c.team) == int(bot.team)}
        same = [j for j, p in enumerate(packet.players)
                if getattr(p, "player_id", None) in ids and int(p.team) == int(bot.team)]
        return sorted(set(same) | set(mine))
    except Exception:                                 # (an unusual packet or config: just this bot's own car)
        return mine


def spectated(packet: Any) -> int | None:
    """``packet.match_info.last_spectated``: the player the host's camera followed most recently (None when the
    packet has no such field). RLBot keeps it when the camera goes free, and it is 0 before anything was spectated."""
    v = getattr(getattr(packet, "match_info", None), "last_spectated", None)
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------------------------- whole messages off the socket
WHOLE_MESSAGES_SINCE = (2, 0, 0, 1, 56)
"""The rlbot version (2.0.0b56) whose SocketRelay reads each message whole by itself."""


def _version_key(version: str) -> tuple[int, int, int, int, int] | None:
    """A sortable key for an rlbot version: "2.0.0-beta.54" and "2.0.0b54" -> (2, 0, 0, 1, 54); a final release
    sorts after its betas (stage 3). None when it can't be read."""
    import re
    m = re.match(r"^\s*v?(\d+)\.(\d+)(?:\.(\d+))?(?:[-.]?(a|alpha|b|beta|rc|c)[-.]?(\d+))?", str(version), re.I)
    if not m:
        return None
    stage = {"a": 0, "alpha": 0, "b": 1, "beta": 1, "rc": 2, "c": 2}.get((m.group(4) or "").lower(), 3)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3) or 0), stage, int(m.group(5) or 0))


def needs_whole_messages(version: str) -> bool:
    """Whether an rlbot version's SocketRelay can lose its place in the stream (before 2.0.0b56: see
    read_message_whole)."""
    key = _version_key(version)
    return key is not None and key < WHOLE_MESSAGES_SINCE


def read_message_whole(relay: Any) -> bytes:
    """One message off an RLBot SocketRelay's socket, whole. RLBot's managers poll the socket without blocking; in
    rlbot before 2.0.0b56, a message that had only partly arrived made ``read_message`` give up halfway, the bytes it
    had read were lost, the stream lost its framing and the bot or script died a moment later ("Error while
    unpacking message ... Invalid offset"). Here the first byte is read the caller's way (nothing waiting: the
    socket's BlockingIOError, as before), and the rest of the 2-byte length and the message are then waited for,
    the socket's own blocking mode (or timeout) restored after."""
    sock = relay.socket
    first = sock.recv(1)
    if not first:
        raise EOFError
    timeout = sock.gettimeout()
    sock.settimeout(None)                                # (blocking)
    try:
        size = int.from_bytes(first + relay._read_exact(1), "big")
        return relay._read_exact(size)
    finally:
        sock.settimeout(timeout)


def fix_socket_reads() -> bool:
    """Make rlbot's SocketRelay read whole messages (read_message_whole) when the installed rlbot is older than
    2.0.0b56; newer versions are left alone. Safe to call more than once; the overlay, the watch script and the
    recorder call it. Returns whether the fix is in place."""
    try:
        from rlbot import interface
        from rlbot.version import __version__ as rlbot_version
    except Exception:                                    # (no rlbot here: nothing to fix)
        return False
    relay = interface.SocketRelay
    if getattr(relay.read_message, "_cds_whole", False):
        return True
    if not needs_whole_messages(rlbot_version):
        return False

    def read_message(self: Any) -> bytes:
        return read_message_whole(self)
    read_message._cds_whole = True                       # type: ignore[attr-defined]
    read_message.__doc__ = read_message_whole.__doc__
    relay._cds_original_read_message = relay.read_message   # type: ignore[attr-defined]
    relay.read_message = read_message                    # type: ignore[method-assign]
    return True
