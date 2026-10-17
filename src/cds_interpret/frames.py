"""Game frames: the one game-state format every label, habit count and drawing in this toolkit reads.

Labels are computed from the GAME STATE, never from one bot's observation, so any bot can be studied on the same
footing: a frame is what the game itself knows (positions, velocities, boost, who touched the ball), in the world
frame, in Unreal units (uu) and seconds. Blue (team 0) defends -y and orange (team 1) defends +y, as in RLBot, RLGym
and RocketSim alike.

A :class:`Frames` is a struct of numpy arrays with a leading batch axis N (the moments of one game in order, or any
batch of independent moments) and, for car fields, a car axis C. Absent car slots have ``car_present = 0``.

=================================  ==========  ======================================================
field                              shape       meaning
=================================  ==========  ======================================================
ball_pos, ball_vel, ball_angvel    [N, 3]      the ball
car_pos, car_vel, car_angvel       [N, C, 3]   each car
car_fwd, car_up                    [N, C, 3]   unit forward and up axes of each car
car_boost                          [N, C]      0..100
car_on_ground, car_demoed          [N, C]      0 / 1
car_has_flip                       [N, C]      1 = a jump or dodge is still available
car_supersonic                     [N, C]      0 / 1
car_team, car_present              [N, C]      0 blue / 1 orange; 1 = the slot holds a car
=================================  ==========  ======================================================

Optional fields, filled by the recorders (what happened in the step that FOLLOWS each frame):

=================================  ==========  ======================================================
ev_touch, ev_demo_made, ev_demoed  [N, C]      0 / 1
goal                               [N]         -1, or the team that scored
time                               [N]         seconds since the start of the game
prev_controls                      [N, C, 8]   each car's controls in the step before the frame
=================================  ==========  ======================================================

Controls are always the 8 RLBot / RLGym controls in this order: throttle, steer, pitch, yaw, roll, jump, boost,
handbrake.

Adapters to this format: :func:`from_arrays` (keyword arrays), :func:`from_rlgym` (RLGym v1 or v2 game states,
duck-typed), :func:`from_rlbot_packet` (RLBot v5 packets, duck-typed). The recorders in
:mod:`cds_interpret.record` produce Frames directly; :mod:`cds_interpret.rlgym` goes the other way, from Frames to
RLGym v2 game states.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import ShapeError

BALL_FIELDS = ("ball_pos", "ball_vel", "ball_angvel")
CAR_VEC_FIELDS = ("car_pos", "car_vel", "car_angvel", "car_fwd", "car_up")
CAR_FIELDS = ("car_boost", "car_on_ground", "car_demoed", "car_has_flip", "car_supersonic", "car_team", "car_present")
EVENT_FIELDS = ("ev_touch", "ev_demo_made", "ev_demoed")
REQUIRED = BALL_FIELDS + CAR_VEC_FIELDS + CAR_FIELDS
OPTIONAL = (*EVENT_FIELDS, "goal", "time", "prev_controls")
CONTROLS = ("throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake")
SUPERSONIC = 2200.0


@dataclass
class Frames:
    """N moments of game state (see the module's table for the fields). Read a field as an attribute:
    ``fr.ball_pos``, ``fr.car_boost``; ``len(fr)`` is N and ``fr.n_cars`` is C."""

    data: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        missing = [k for k in REQUIRED if k not in self.data]
        if missing:
            raise ShapeError(
                f"Frames lacks the fields {missing}. Build frames with frames.from_arrays(...), which fills the "
                "optional car fields for you, or see the table in cds_interpret.frames."
            )
        n = len(self.data["ball_pos"])
        c = self.data["car_pos"].shape[1] if self.data["car_pos"].ndim == 3 else None
        for k, v in self.data.items():
            if len(v) != n:
                raise ShapeError(f"Frames field {k} has {len(v)} rows, ball_pos has {n}")
            per_car = k.startswith(("car_", "ev_")) or k == "prev_controls"
            if per_car and (v.ndim < 2 or v.shape[1] != c):
                more = ", ..." if v.ndim > 2 else ""
                raise ShapeError(f"Frames field {k} has shape {v.shape}: expected [N, C{more}] with C = {c} cars "
                                 f"(from car_pos {self.data['car_pos'].shape})")

    def __getattr__(self, name: str) -> np.ndarray:
        d = self.__dict__.get("data")
        if d is not None and name in d:
            return d[name]
        if name in OPTIONAL:
            raise AttributeError(
                f"these frames have no '{name}' field (it is optional: recorded games have it, synthetic moments "
                "and single packets may not)"
            )
        raise AttributeError(name)

    def __len__(self) -> int:
        return len(self.data["ball_pos"])

    def __repr__(self) -> str:
        extra = [k for k in OPTIONAL if k in self.data]
        return f"Frames({len(self)} moments, {self.n_cars} car slots{', with ' + ', '.join(extra) if extra else ''})"

    @property
    def n_cars(self) -> int:
        return int(self.data["car_pos"].shape[1])

    def has(self, name: str) -> bool:
        return name in self.data

    def take(self, idx: Any) -> Frames:
        """The moments ``idx`` (an index array, a slice or a boolean mask)."""
        return Frames({k: v[idx] for k, v in self.data.items()})

    @staticmethod
    def concat(parts: Sequence[Frames]) -> Frames:
        """Moments of several Frames one after the other (fields all of them have; the same number of car slots)."""
        if not parts:
            raise ValueError("Frames.concat needs at least one Frames")
        cars = {p.n_cars for p in parts}
        if len(cars) > 1:
            raise ShapeError(f"cannot concatenate frames with different numbers of car slots: {sorted(cars)}")
        keys = set.intersection(*(set(p.data) for p in parts))
        return Frames({k: np.concatenate([p.data[k] for p in parts]) for k in keys})

    def astype(self, dtype: Any) -> Frames:
        return Frames({k: (v.astype(dtype) if v.dtype.kind == "f" else v) for k, v in self.data.items()})

    def to_dict(self) -> dict[str, np.ndarray]:
        return dict(self.data)


def team_sign(team: np.ndarray) -> np.ndarray:
    """+1 for blue, -1 for orange: multiply x and y by it to see the field as that team does (attacking +y)."""
    return np.where(np.asarray(team) > 0.5, -1.0, 1.0)


def canon(v: np.ndarray, sign: np.ndarray | float) -> np.ndarray:
    """Vectors [..., 3] in a team's frame: x and y times the team sign (a half turn about z for orange - a proper
    rotation, so every vector transforms alike). ``sign`` broadcasts against ``v[..., 0]``."""
    out = np.array(v, dtype=np.float64, copy=True)
    out[..., 0] *= sign
    out[..., 1] *= sign
    return out


def from_arrays(**arrays: Any) -> Frames:
    """Frames from keyword arrays. Required: ball_pos, ball_vel, car_pos, car_vel, car_fwd, car_up, car_boost,
    car_on_ground, car_team. Filled in when absent: ball_angvel and car_angvel (zeros), car_has_flip (1),
    car_demoed (0), car_present (1), car_supersonic (speed >= 2200 uu/s)."""
    a = {k: np.asarray(v) for k, v in arrays.items()}
    if "car_pos" not in a or "ball_pos" not in a:
        raise ShapeError("from_arrays needs at least ball_pos [N, 3] and car_pos [N, C, 3]")
    if a["car_pos"].ndim != 3:
        raise ShapeError(f"car_pos must be [N, C, 3] (moments, car slots, xyz), not {a['car_pos'].shape}")
    n, c = a["car_pos"].shape[:2]
    a.setdefault("ball_angvel", np.zeros((n, 3), np.float32))
    a.setdefault("car_angvel", np.zeros((n, c, 3), np.float32))
    a.setdefault("car_has_flip", np.ones((n, c), np.float32))
    a.setdefault("car_demoed", np.zeros((n, c), np.float32))
    a.setdefault("car_present", np.ones((n, c), np.float32))
    if "car_supersonic" not in a and "car_vel" in a:
        a["car_supersonic"] = (np.linalg.norm(a["car_vel"], axis=-1) >= SUPERSONIC).astype(np.float32)
    return Frames(a)


def _vec(v: Any) -> list[float]:
    """An RLGym / RLBot / RocketSim vector (numpy array, list, or an object with x, y, z) as three floats."""
    if hasattr(v, "x"):
        return [float(v.x), float(v.y), float(v.z)]
    return [float(x) for x in np.asarray(v, dtype=np.float64).reshape(3)]


def axes_from_rotator(pitch: float, yaw: float, roll: float) -> tuple[np.ndarray, np.ndarray]:
    """Forward and up unit vectors of a car from an Unreal rotator (radians), as RLBot gives it."""
    cp, sp, cy, sy, cr, sr = np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw), np.cos(roll), np.sin(roll)
    fwd = np.array([cp * cy, cp * sy, sp])
    up = np.array([-cr * cy * sp - sr * sy, -cr * sy * sp + sr * cy, cp * cr])
    return fwd, up


def _empty(n: int, c: int, with_events: bool = False) -> dict[str, np.ndarray]:
    out = {k: np.zeros((n, 3), np.float32) for k in BALL_FIELDS}
    out.update({k: np.zeros((n, c, 3), np.float32) for k in CAR_VEC_FIELDS})
    out.update({k: np.zeros((n, c), np.float32) for k in CAR_FIELDS})
    if with_events:
        out.update({k: np.zeros((n, c), np.float32) for k in EVENT_FIELDS})
    return out


def from_rlgym(states: Iterable[Any], max_cars: int | None = None) -> Frames:
    """Frames from a sequence of RLGym game states (duck-typed: RLGym itself is not imported).

    RLGym v1 / rlgym-sim (``GameState``): ``state.ball.position`` ..., ``state.players[i].car_data.position`` /
    ``linear_velocity`` / ``angular_velocity`` / ``forward()`` / ``up()``, ``boost_amount`` (0..1), ``on_ground``,
    ``is_demoed``, ``has_flip``, ``team_num``, ``ball_touched``.

    RLGym v2 (``rlgym.rocket_league.api.GameState``): ``state.cars`` (agent id -> Car) with ``car.physics.position``
    ..., ``forward`` / ``up``, ``boost_amount`` (0..100), ``on_ground``, ``is_demoed``, ``has_flip``, ``team_num``,
    ``ball_touches``. Cars are taken in ``state.cars`` order."""
    states = list(states)
    if not states:
        raise ValueError("from_rlgym got no states")
    rows = [list(s.cars.values()) if hasattr(s, "cars") else list(s.players) for s in states]
    c = max_cars or max(len(p) for p in rows)
    n = len(states)
    out = _empty(n, c)
    out["ev_touch"] = np.zeros((n, c), np.float32)
    for t, (s, players) in enumerate(zip(states, rows)):
        b = s.ball
        out["ball_pos"][t], out["ball_vel"][t] = _vec(b.position), _vec(b.linear_velocity)
        out["ball_angvel"][t] = _vec(getattr(b, "angular_velocity", np.zeros(3)))
        for j, p in enumerate(players[:c]):
            v2 = hasattr(p, "physics")
            ph = p.physics if v2 else p.car_data
            out["car_pos"][t, j], out["car_vel"][t, j] = _vec(ph.position), _vec(ph.linear_velocity)
            out["car_angvel"][t, j] = _vec(ph.angular_velocity)
            fwd = ph.forward if v2 else ph.forward()
            up = ph.up if v2 else ph.up()
            out["car_fwd"][t, j], out["car_up"][t, j] = _vec(fwd), _vec(up)
            boost = float(p.boost_amount)
            out["car_boost"][t, j] = boost if v2 else 100.0 * boost          # v1: 0..1, v2: 0..100
            out["car_on_ground"][t, j] = float(p.on_ground)
            out["car_demoed"][t, j] = float(p.is_demoed)
            out["car_has_flip"][t, j] = float(getattr(p, "has_flip", getattr(p, "can_flip", True)))
            out["car_team"][t, j] = float(p.team_num)
            out["car_present"][t, j] = 1.0
            out["car_supersonic"][t, j] = float(np.linalg.norm(out["car_vel"][t, j]) >= SUPERSONIC)
            out["ev_touch"][t, j] = float(bool(getattr(p, "ball_touched", getattr(p, "ball_touches", 0))))
    return Frames(out)


def from_rlbot_packet(packets: Any | Iterable[Any]) -> Frames:
    """Frames from RLBot v5 ``GamePacket`` s (``rlbot.flat``; duck-typed): ``packet.balls[0].physics``,
    ``packet.players[i].physics`` (location, velocity, angular_velocity, rotation pitch / yaw / roll), ``boost``,
    ``air_state``, ``demolished_timeout``, ``dodge_timeout``, ``is_supersonic``, ``team``. Cars in packet order.
    One packet or a sequence of them."""
    if hasattr(packets, "players"):
        packets = [packets]
    packets = list(packets)
    if not packets:
        raise ValueError("from_rlbot_packet got no packets")
    c = max(1, max(len(p.players) for p in packets))
    n = len(packets)
    out = _empty(n, c)
    out["time"] = np.zeros(n, np.float32)
    for t, pk in enumerate(packets):
        if len(pk.balls):
            b = pk.balls[0].physics
            out["ball_pos"][t], out["ball_vel"][t] = _vec(b.location), _vec(b.velocity)
            out["ball_angvel"][t] = _vec(b.angular_velocity)
        out["time"][t] = float(getattr(pk.match_info, "seconds_elapsed", 0.0))
        for j, p in enumerate(pk.players):
            ph = p.physics
            out["car_pos"][t, j], out["car_vel"][t, j] = _vec(ph.location), _vec(ph.velocity)
            out["car_angvel"][t, j] = _vec(ph.angular_velocity)
            fwd, up = axes_from_rotator(ph.rotation.pitch, ph.rotation.yaw, ph.rotation.roll)
            out["car_fwd"][t, j], out["car_up"][t, j] = fwd, up
            out["car_boost"][t, j] = float(p.boost)
            air = getattr(p, "air_state", None)
            on_ground = (int(air) == 0) if air is not None else bool(getattr(p, "has_wheel_contact", True))
            out["car_on_ground"][t, j] = float(on_ground)
            demoed = getattr(p, "demolished_timeout", -1.0) > 0 or bool(getattr(p, "is_demolished", False))
            out["car_demoed"][t, j] = float(demoed)
            out["car_has_flip"][t, j] = float(on_ground or getattr(p, "dodge_timeout", -1.0) != 0.0)
            out["car_team"][t, j] = float(p.team)
            out["car_present"][t, j] = 1.0
            sup = getattr(p, "is_supersonic", None)
            out["car_supersonic"][t, j] = float(sup if sup is not None
                                                else np.linalg.norm(out["car_vel"][t, j]) >= SUPERSONIC)
    return Frames(out)


def nearest_car(fr: Frames, me: Any, which: str = "opponent") -> np.ndarray:
    """[N] the car slot of the nearest opponent (``which="opponent"``) or team-mate (``"mate"``) of car slot me[i] at
    moment i, or -1 when there is none. Handy in an obs function: ``opp = nearest_car(frames, me)``."""
    if which not in ("opponent", "mate"):
        raise ValueError(f"which must be 'opponent' or 'mate', not {which!r}")
    n = len(fr)
    me = np.broadcast_to(np.asarray(me, dtype=np.int64), (n,))
    rows = np.arange(n)
    same = fr.car_team == fr.car_team[rows, me][:, None]
    mask = (~same if which == "opponent" else same) & (fr.car_present > 0.5)
    mask[rows, me] = False
    d = np.linalg.norm(fr.car_pos - fr.car_pos[rows, me][:, None], axis=-1)
    d = np.where(mask, d, np.inf)
    j = np.argmin(d, axis=1)
    return np.where(np.isfinite(d[rows, j]), j, -1).astype(np.int64)
