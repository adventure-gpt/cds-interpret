"""Concept labels from the game state, for one car's point of view.

Every label is a function of what the game knows - positions, velocities, boost, the ball's flight, what happened
next - never of a bot's observation, so the same labels serve any bot. Each label is computed for the car ``me[i]``
of moment i, in that car's team frame (it attacks +y and defends -y).

``LABEL_KIND`` says how a probe treats each label: ``"bin"`` (yes / no: logistic, scored by AUROC) or ``"reg"`` (a
number: least squares, scored by R²). ``describe_label(name)`` gives the one-line meaning of any label.

Labels of the moment (:func:`concept_labels`):

====================  ===================================================================================
land_t, land_x,       the ball's next contact with the floor if nobody touches it: seconds ahead, and where
land_y                (team frame). 0 when it is rolling on the floor now; NaN when it does not come down
                      within the path's horizon (on a wall, high bounces off the back wall)
land_dist             distance (xy) from the car to that landing point
ttb_me, ttb_opp,      time to ball: the earliest time the car could reach the ball's predicted path with a
ttb_mate              simple reach model (full throttle, boost while it has boost, a turning cost, more time
                      for a ball in the air); the nearest opponent's and team-mate's (NaN without
                      team-mates). Capped at the path's horizon. A demolished car cannot reach the ball
ttb_adv               ttb_opp - ttb_me, clipped to +-3 s (positive: we get there first)
first_to_ball         1 when no other car reaches the ball sooner
last_back             1 when the car is its team's rearmost car (closest to its own goal line) among cars
                      that are not demolished. NaN in 1v1, where it would always be 1
goal_side             1 when the car is between the ball and its own goal along the field (car y < ball y)
boost                 boost / 100
boost_adv             (boost - the opponents' mean boost) / 100
threat                the untouched ball enters our goal within 3 s (by the ball's path)
threat_t              seconds until it does (NaN when it does not)
on_target             the untouched ball enters the opponent's goal within 3 s
demo_chance           an opponent is within reach on a line at our speed: within 2300 uu and 1.1 s at the
                      present closing speed, within 18 degrees of our direction of travel, our speed 1700+
                      uu/s and either supersonic already or boost left, both cars near the same height,
                      the opponent not demolished; never during a kickoff (two cars racing at each other
                      for the ball meet the same geometry)
demo_ttc              time to contact of the best such opponent (NaN when there is none)
kickoff               the ball sits still on the centre spot (kickoff, before the first touch)
in_air, on_wall,      wheels off everything; wheels on a wall (above 150 uu, up axis within 45 degrees of
on_ceiling            horizontal, below 1800 uu); wheels on the ceiling
height, speed         car z / 2044; car speed / 2300
supersonic            the game's supersonic flag
ball_height,          ball z / 2044; car-to-ball distance / 5000
ball_dist
ball_vy               the ball's velocity along the field toward the opponent's goal / 6000
has_flip              a jump or dodge is still available
====================  ===================================================================================

Future labels (:func:`future_labels`, from recorded play: what happened next; NaN where the recording ends first;
a goal ends the window, as the kickoff that follows resets the game):

====================  ===================================================================================
shot_soon             within 1 s the car touches the ball and leaves it on target for the opponent's goal
touch_soon            within 0.5 s the car touches the ball
demo_soon             within 1 s the car demolishes an opponent
demoed_soon           within 1 s the car is demolished
goal_soon             within 3 s its team scores
concede_soon          within 3 s the other team scores
outlook               goal_soon - concede_soon (-1, 0, +1): a stand-in for a critic's value, for bots
                      that have none
====================  ===================================================================================

Your own labels join all of these through :func:`register_label`::

    @register_label("ball_on_my_side", kind="bin", doc="the ball is in our half")
    def _(inp):                                   # inp: LabelInput (frames, me, team sign, ball path ...)
        return inp.ball[:, 1] < 0                 # inp.ball: the ball's position in the team frame

The ball's path: pass ``path`` [N, K, 3] (world positions at times ``path_t`` [K]) from the best predictor you have
(the RocketSim recorder passes RocketSim's ball prediction; the RLBot recorder RLBot's); without one,
:func:`ball_path` integrates a simple ball (gravity, drag, bounces off floor, walls and ceiling, the goal mouths
open; no ramps or corners).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .errors import ShapeError
from .frames import Frames, canon, team_sign

FIELD_X, FIELD_Y, FIELD_Z = 4096.0, 5120.0, 2044.0
GOAL_HALF_W, GOAL_H = 892.755, 642.775
BALL_R = 92.75
GRAVITY = 650.0
BALL_DRAG = 0.03
BALL_RESTITUTION = 0.6
BALL_MAX = 6000.0
CAR_MAX, SUPERSONIC, THROTTLE_MAX = 2300.0, 2200.0, 1410.0
REACH = 140.0                       # car-centre to ball-centre distance that counts as reaching it
PATH_T = np.arange(1, 121) / 30.0   # default path times: every 4 ticks out to 4 s

LABEL_KIND: dict[str, str] = {
    "land_t": "reg", "land_x": "reg", "land_y": "reg", "land_dist": "reg",
    "ttb_me": "reg", "ttb_opp": "reg", "ttb_mate": "reg", "ttb_adv": "reg", "first_to_ball": "bin",
    "last_back": "bin", "goal_side": "bin", "boost": "reg", "boost_adv": "reg",
    "threat": "bin", "threat_t": "reg", "on_target": "bin", "demo_chance": "bin", "demo_ttc": "reg",
    "kickoff": "bin", "in_air": "bin", "on_wall": "bin", "on_ceiling": "bin", "height": "reg", "speed": "reg",
    "supersonic": "bin", "ball_height": "reg", "ball_dist": "reg", "ball_vy": "reg", "has_flip": "bin",
    "shot_soon": "bin", "touch_soon": "bin", "demo_soon": "bin", "demoed_soon": "bin", "goal_soon": "bin",
    "concede_soon": "bin", "outlook": "reg",
}
"""How a probe treats each label: "bin" (yes / no) or "reg" (a number). Registered labels are added here."""

FUTURE_LABELS = ("shot_soon", "touch_soon", "demo_soon", "demoed_soon", "goal_soon", "concede_soon", "outlook")

LABEL_DOC: dict[str, str] = {
    "land_t": "seconds until the untouched ball next touches the floor",
    "land_x": "x of that landing point (team frame)",
    "land_y": "y of that landing point (team frame: + is toward their goal)",
    "land_dist": "distance from the car to that landing point (uu)",
    "ttb_me": "seconds this car needs to reach the ball",
    "ttb_opp": "seconds the nearest opponent needs to reach the ball",
    "ttb_mate": "seconds the nearest team-mate needs to reach the ball",
    "ttb_adv": "how much sooner this car reaches the ball than the opponents (s)",
    "first_to_ball": "no other car reaches the ball sooner",
    "last_back": "this car is its team's rearmost",
    "goal_side": "this car is between the ball and its own goal",
    "boost": "boost / 100",
    "boost_adv": "boost minus the opponents' mean boost, / 100",
    "threat": "the untouched ball goes into our goal within 3 s",
    "threat_t": "seconds until it does",
    "on_target": "the untouched ball goes into their goal within 3 s",
    "demo_chance": "an opponent can be demolished right now",
    "demo_ttc": "seconds to contact with that opponent",
    "kickoff": "kickoff: the ball still on the centre spot",
    "in_air": "wheels off everything",
    "on_wall": "driving on a wall",
    "on_ceiling": "driving on the ceiling",
    "height": "car height / 2044",
    "speed": "car speed / 2300",
    "supersonic": "supersonic",
    "ball_height": "ball height / 2044",
    "ball_dist": "car-to-ball distance / 5000",
    "ball_vy": "ball speed toward their goal / 6000",
    "has_flip": "a jump or dodge is still available",
    "shot_soon": "within 1 s this car puts the ball on target",
    "touch_soon": "within 0.5 s this car touches the ball",
    "demo_soon": "within 1 s this car demolishes an opponent",
    "demoed_soon": "within 1 s this car is demolished",
    "goal_soon": "within 3 s our team scores",
    "concede_soon": "within 3 s the other team scores",
    "outlook": "goal_soon - concede_soon",
}


LABEL_SHORT: dict[str, str] = {
    "land_t": "time to landing", "land_x": "landing x", "land_y": "landing y", "land_dist": "distance to landing",
    "ttb_me": "time to the ball", "ttb_opp": "opponent's time to ball", "ttb_mate": "mate's time to ball",
    "ttb_adv": "lead to the ball", "first_to_ball": "first to the ball", "last_back": "last back",
    "goal_side": "goal side", "boost": "boost", "boost_adv": "boost lead", "threat": "threat on goal",
    "threat_t": "time to the threat", "on_target": "shot on target", "demo_chance": "demo chance",
    "demo_ttc": "time to the demo", "kickoff": "kickoff", "in_air": "in the air", "on_wall": "on a wall",
    "on_ceiling": "on the ceiling", "height": "height", "speed": "speed", "supersonic": "supersonic",
    "ball_height": "ball height", "ball_dist": "distance to the ball", "ball_vy": "ball heading their way",
    "has_flip": "flip ready", "shot_soon": "about to shoot", "touch_soon": "about to touch",
    "demo_soon": "going for a demo", "demoed_soon": "about to be demoed", "goal_soon": "about to score",
    "concede_soon": "about to concede", "outlook": "outlook",
}
"""Short display names (for the overlay's panel, which fits about 28 characters a line)."""


def short_label(name: str) -> str:
    """A label's short display name ("threat" -> "threat on goal"); unknown names with _ turned into spaces."""
    return LABEL_SHORT.get(name, name.replace("_", " "))


def describe_label(name: str) -> str:
    """The one-line meaning of a label (built in or registered)."""
    return LABEL_DOC.get(name, _REGISTRY[name].doc if name in _REGISTRY else name)


# ------------------------------------------------------------------------------------------------- your own labels
@dataclass
class LabelInput:
    """What a registered label function receives: the frames, the car of each moment, and some views computed
    once for all labels. Positions and velocities here are in the car's TEAM frame (attacking +y)."""

    frames: Frames
    me: np.ndarray                      # [N] car slot of each moment
    sign: np.ndarray                    # [N] +1 blue, -1 orange
    path: np.ndarray                    # [N, K, 3] the ball's path (WORLD frame)
    path_t: np.ndarray                  # [K] its times, seconds ahead
    car_pos: np.ndarray                 # [N, C, 3] team frame
    car_vel: np.ndarray                 # [N, C, 3] team frame
    ball: np.ndarray                    # [N, 3] the ball's position, team frame
    ball_vel: np.ndarray                # [N, 3] team frame
    labels: dict[str, np.ndarray] = field(default_factory=dict)   # the built-in labels, already computed

    @property
    def my_pos(self) -> np.ndarray:
        return self.car_pos[np.arange(len(self.me)), self.me]

    @property
    def my_vel(self) -> np.ndarray:
        return self.car_vel[np.arange(len(self.me)), self.me]


@dataclass
class _Registered:
    fn: Callable[[LabelInput], np.ndarray]
    kind: str
    doc: str


_REGISTRY: dict[str, _Registered] = {}


def register_label(name: str, kind: str, doc: str = "", replace: bool = False) -> Callable:
    """Add your own concept label: a decorator for ``fn(inp: LabelInput) -> array [N]``. ``kind`` is "bin" (yes /
    no) or "reg" (a number). The label is then computed by :func:`concept_labels` beside the built-in ones, probed,
    related to features and shown in reports like them."""
    if kind not in ("bin", "reg"):
        raise ValueError(f"kind must be 'bin' or 'reg', not {kind!r}")
    if not replace and (name in _REGISTRY or name in LABEL_DOC):
        raise ValueError(f"a label named {name!r} already exists (pass replace=True to override yours)")

    def deco(fn: Callable[[LabelInput], np.ndarray]) -> Callable[[LabelInput], np.ndarray]:
        _REGISTRY[name] = _Registered(fn, kind, doc or name)
        LABEL_KIND[name] = kind
        return fn
    return deco


def unregister_label(name: str) -> None:
    """Remove a label added with register_label."""
    _REGISTRY.pop(name, None)
    if name not in LABEL_DOC:
        LABEL_KIND.pop(name, None)


def registered_labels() -> dict[str, str]:
    """The labels added with register_label: name -> kind."""
    return {k: v.kind for k, v in _REGISTRY.items()}


# ---------------------------------------------------------------------------------------------------- ball path
def ball_path(pos: np.ndarray, vel: np.ndarray, times: np.ndarray = PATH_T, dt: float = 1.0 / 120.0) -> np.ndarray:
    """A simple untouched-ball flight: pos / vel [N, 3] (world) -> positions [N, K, 3] at ``times`` (seconds,
    rising). Gravity, linear drag, the 6000 uu/s cap; bounces (restitution 0.6, some friction) off the floor, the
    ceiling, the side walls and the back walls outside the goal mouths; a ball inside a goal mouth goes on into the
    net and stops there. No ramps, corners or spin: a hint where no simulator's prediction is at hand."""
    p = np.array(pos, dtype=np.float64, copy=True).reshape(-1, 3)
    v = np.array(vel, dtype=np.float64, copy=True).reshape(-1, 3)
    times = np.asarray(times, dtype=np.float64)
    out = np.zeros((len(p), len(times), 3))
    stopped = np.zeros(len(p), bool)
    t, k = 0.0, 0
    n_steps = int(np.ceil(times[-1] / dt))
    for _ in range(n_steps + 1):
        while k < len(times) and times[k] <= t + 1e-9:
            out[:, k] = p
            k += 1
        if k >= len(times):
            break
        live = ~stopped
        v[live, 2] -= GRAVITY * dt
        v[live] *= (1.0 - BALL_DRAG * dt)
        sp = np.linalg.norm(v, axis=1, keepdims=True)
        v = np.where(sp > BALL_MAX, v * BALL_MAX / np.maximum(sp, 1e-9), v)
        p[live] += v[live] * dt
        low = live & (p[:, 2] < BALL_R)
        p[low, 2] = BALL_R
        v[low, 2] = np.abs(v[low, 2]) * BALL_RESTITUTION
        v[low, :2] *= 0.97
        v[low & (v[:, 2] < 40.0), 2] = 0.0                       # comes to roll
        high = live & (p[:, 2] > FIELD_Z - BALL_R)
        p[high, 2] = FIELD_Z - BALL_R
        v[high, 2] = -np.abs(v[high, 2]) * BALL_RESTITUTION
        side = live & (np.abs(p[:, 0]) > FIELD_X - BALL_R)
        p[side, 0] = np.sign(p[side, 0]) * (FIELD_X - BALL_R)
        v[side, 0] *= -BALL_RESTITUTION
        mouth = (np.abs(p[:, 0]) < GOAL_HALF_W - BALL_R * 0.5) & (p[:, 2] < GOAL_H - BALL_R * 0.5)
        back = live & (np.abs(p[:, 1]) > FIELD_Y - BALL_R) & ~mouth
        p[back, 1] = np.sign(p[back, 1]) * (FIELD_Y - BALL_R)
        v[back, 1] *= -BALL_RESTITUTION
        in_net = live & (np.abs(p[:, 1]) > FIELD_Y + BALL_R + 1.0)
        stopped |= in_net
        v[in_net] = 0.0
        t += dt
    while k < len(times):
        out[:, k] = p
        k += 1
    return out


def goal_entry(path: np.ndarray, times: np.ndarray, sign: np.ndarray, horizon: float = 3.0):
    """For paths [N, K, 3] (world) and each moment's team sign: (enters own goal, seconds until; enters the
    opponent's goal, seconds until) within ``horizon``. A ball is in when its centre is past the goal line by half a
    radius, inside the mouth."""
    y = path[..., 1] * np.asarray(sign)[:, None]
    in_mouth = (np.abs(path[..., 0]) < GOAL_HALF_W + 20.0) & (path[..., 2] < GOAL_H + 20.0)
    within = (np.asarray(times) <= horizon + 1e-9)[None, :]
    line = FIELD_Y + 0.5 * BALL_R
    own = (y < -line) & in_mouth & within
    opp = (y > line) & in_mouth & within

    def first(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hit = m.any(1)
        k = np.argmax(m, axis=1)
        return hit, np.where(hit, np.asarray(times)[k], np.nan)
    return first(own) + first(opp)


def landing(path: np.ndarray, times: np.ndarray, ball_pos: np.ndarray, ball_vel: np.ndarray):
    """(land_t, land_xy [N, 2] world) of the next floor contact on paths [N, K, 3]; 0 and the ball's own spot when it
    is rolling now; NaN when it does not come down within the path."""
    z = path[..., 2]
    times = np.asarray(times)
    low = z < BALL_R + 30.0
    dz_prev = np.diff(z, axis=1, prepend=ball_pos[:, 2:3])
    dz_next = np.diff(z, axis=1, append=z[:, -1:] + 1.0)
    dip = (z < BALL_R + 150.0) & (dz_prev <= 0.0) & (dz_next > 0.0)
    hit = low | dip
    any_hit = hit.any(1)
    k = np.argmax(hit, axis=1)
    t = np.where(any_hit, times[k], np.nan)
    xy = np.where(any_hit[:, None], path[np.arange(len(path)), k, :2], np.nan)
    rolling = (ball_pos[:, 2] < BALL_R + 30.0) & (np.abs(ball_vel[:, 2]) < 150.0)
    t = np.where(rolling, 0.0, t)
    xy = np.where(rolling[:, None], ball_pos[:, :2], xy)
    return t, xy


# ------------------------------------------------------------------------------------------------ time to the ball
def arrival_time(car_pos, car_vel, car_fwd, boost, on_ground, demoed, target) -> np.ndarray:
    """Seconds for cars [..., 3] to reach targets [..., 3] with a simple model: full throttle, boost (to 2300 uu/s)
    while it has 5+ boost, else to 1410; a turning cost of 1 s per 2.2 rad of heading change on the ground; an extra
    jump / climb cost for a target above 250 uu. inf for a demolished car."""
    d = target - car_pos
    dist3 = np.linalg.norm(d, axis=-1)
    dirn = d / np.maximum(dist3, 1e-6)[..., None]
    dist = np.maximum(dist3 - REACH, 0.0)
    v0 = np.clip((car_vel * dirn).sum(-1), 0.0, CAR_MAX)
    has_boost = boost > 5.0
    vmax = np.where(has_boost, CAR_MAX, THROTTLE_MAX)
    acc = np.where(has_boost, 1150.0, 650.0)
    v0 = np.minimum(v0, vmax)
    t_acc = (vmax - v0) / acc
    d_acc = v0 * t_acc + 0.5 * acc * t_acc * t_acc
    t_run = np.where(dist <= d_acc, (-v0 + np.sqrt(v0 * v0 + 2.0 * acc * dist)) / acc, t_acc + (dist - d_acc) / vmax)
    fxy = car_fwd[..., :2] / np.maximum(np.linalg.norm(car_fwd[..., :2], axis=-1, keepdims=True), 1e-6)
    dxy = d[..., :2] / np.maximum(np.linalg.norm(d[..., :2], axis=-1, keepdims=True), 1e-6)
    ang = np.arccos(np.clip((fxy * dxy).sum(-1), -1.0, 1.0))
    t_turn = np.where(on_ground > 0.5, ang / 2.2, 0.15)
    climb = np.maximum(target[..., 2] - 250.0, 0.0)
    t_climb = np.where(climb > 0.0, 0.25 + climb / 1100.0, 0.0)
    t = t_run + t_turn + t_climb
    return np.where(demoed > 0.5, np.inf, t)


def time_to_ball(fr: Frames, path: np.ndarray, times: np.ndarray) -> np.ndarray:
    """[N, C] the earliest path time each car can be at the ball (capped at the last path time; inf for absent or
    demolished cars)."""
    times = np.asarray(times)
    n, c = fr.car_pos.shape[:2]
    pts = np.concatenate([fr.ball_pos[:, None, :], path], axis=1)                     # [N, K+1, 3]
    tk = np.concatenate([[0.0], times])
    out = np.full((n, c), np.inf)
    for j in range(c):
        a = arrival_time(fr.car_pos[:, j, None], fr.car_vel[:, j, None], fr.car_fwd[:, j, None],
                         fr.car_boost[:, j, None], fr.car_on_ground[:, j, None], fr.car_demoed[:, j, None], pts)
        ok = a <= tk[None, :]
        k = np.argmax(ok, axis=1)
        t = np.where(ok.any(1), tk[k], times[-1])
        t = np.where((fr.car_present[:, j] > 0.5) & (fr.car_demoed[:, j] < 0.5), t, np.inf)
        out[:, j] = t
    return out


def is_kickoff(fr: Frames) -> np.ndarray:
    """[N] the ball sits still on the centre spot."""
    return ((np.abs(fr.ball_pos[:, 0]) < 2.0) & (np.abs(fr.ball_pos[:, 1]) < 2.0)
            & (fr.ball_pos[:, 2] < BALL_R + 10.0) & (np.linalg.norm(fr.ball_vel, axis=1) < 1.0))


# --------------------------------------------------------------------------------------------------------- labels
def concept_labels(fr: Frames, me: Any, path: np.ndarray | None = None,
                   path_t: np.ndarray | None = None, registered: bool = True) -> dict[str, np.ndarray]:
    """The labels of the moment (the module's first table) for moments ``fr`` seen by car slot ``me[i]`` (an int
    array [N], or one int for every moment), plus your registered labels. ``path`` [N, K, 3]: the ball's predicted
    world positions at ``path_t`` [K] seconds (default: :func:`ball_path`'s simple flight)."""
    n, c = len(fr), fr.n_cars
    slots: np.ndarray = np.broadcast_to(np.asarray(me, dtype=np.int64), (n,)).copy()
    me = slots
    if n and (me.min() < 0 or me.max() >= c):
        raise ShapeError(f"me holds car slots {me.min()}..{me.max()}, but the frames have {c} car slots")
    rows = np.arange(n)
    if path is None:
        path_t = PATH_T if path_t is None else np.asarray(path_t)
        path = ball_path(fr.ball_pos, fr.ball_vel, path_t)
    elif path_t is None:
        raise ShapeError("concept_labels: pass path_t (the path's times, seconds ahead) with path")
    path_t = np.asarray(path_t)
    if path.shape[:2] != (n, len(path_t)):
        raise ShapeError(f"path has shape {path.shape}; expected [N={n}, K={len(path_t)}, 3]")
    team = fr.car_team[rows, me]
    sign = team_sign(team)
    present = (fr.car_present > 0.5) & (fr.car_demoed < 0.5)
    mate_mask = (fr.car_team == team[:, None]) & (fr.car_present > 0.5)
    opp_mask = (fr.car_team != team[:, None]) & (fr.car_present > 0.5)
    not_me = np.ones((n, c), bool)
    not_me[rows, me] = False

    pos = canon(fr.car_pos, sign[:, None])                                               # [N, C, 3] team frame
    vel = canon(fr.car_vel, sign[:, None])
    bpos, bvel = canon(fr.ball_pos, sign), canon(fr.ball_vel, sign)
    my_pos, my_vel = pos[rows, me], vel[rows, me]
    lab: dict[str, np.ndarray] = {}

    # the ball's landing and its way into either goal
    lt, lxy = landing(path, path_t, fr.ball_pos, fr.ball_vel)
    lxy = lxy * sign[:, None]
    lab["land_t"], lab["land_x"], lab["land_y"] = lt, lxy[:, 0], lxy[:, 1]
    lab["land_dist"] = np.linalg.norm(lxy - my_pos[:, :2], axis=1)
    own_in, own_t, opp_in, _ = goal_entry(path, path_t, sign)
    lab["threat"], lab["threat_t"], lab["on_target"] = own_in.astype(np.float32), own_t, opp_in.astype(np.float32)

    # time to ball
    ttb = time_to_ball(fr, path, path_t)                                                # [N, C]
    cap = float(path_t[-1])
    lab["ttb_me"] = np.minimum(ttb[rows, me], cap)
    t_opp = np.where(opp_mask, ttb, np.inf).min(1) if c else np.full(n, np.inf)
    t_mate = np.where(mate_mask & not_me, ttb, np.inf).min(1) if c else np.full(n, np.inf)
    has_mate = (mate_mask & not_me).any(1)
    lab["ttb_opp"] = np.minimum(t_opp, cap)
    lab["ttb_mate"] = np.where(has_mate, np.minimum(t_mate, cap), np.nan)
    lab["ttb_adv"] = np.clip(lab["ttb_opp"] - lab["ttb_me"], -3.0, 3.0)
    lab["first_to_ball"] = (ttb[rows, me] <= np.minimum(t_opp, t_mate)).astype(np.float32)

    # positioning
    y_mates = np.where(mate_mask & present, pos[..., 1], np.inf)
    back = ((my_pos[:, 1] <= y_mates.min(1) + 1e-3) & present[rows, me]).astype(np.float32)
    lab["last_back"] = np.where(has_mate, back, np.nan)              # (always true in 1v1: left out there)
    lab["goal_side"] = (my_pos[:, 1] < bpos[:, 1]).astype(np.float32)

    # boost
    boost = fr.car_boost[rows, me]
    opp_boost = np.where(opp_mask, fr.car_boost, 0.0).sum(1) / np.maximum(opp_mask.sum(1), 1)
    lab["boost"], lab["boost_adv"] = boost / 100.0, (boost - opp_boost) / 100.0

    # a demolition within reach
    d = pos - my_pos[:, None]
    dist = np.linalg.norm(d, axis=-1)
    dirn = d / np.maximum(dist, 1e-6)[..., None]
    speed = np.linalg.norm(my_vel, axis=1)
    heading = np.where(speed[:, None] > 300.0, my_vel / np.maximum(speed, 1e-6)[:, None],
                       canon(fr.car_fwd[rows, me], sign))
    cos = (dirn * heading[:, None]).sum(-1)
    closing = ((my_vel[:, None] - vel) * dirn).sum(-1)
    ttc = dist / np.maximum(closing, 1.0)
    can_go = (fr.car_supersonic[rows, me] > 0.5) | (boost > 8.0)
    ok = (opp_mask & present & (dist < 2300.0) & (cos > np.cos(np.radians(18.0))) & (closing > 300.0) & (ttc < 1.1)
          & (np.abs(pos[..., 2] - my_pos[:, None, 2]) < 200.0))
    ok &= ((speed > 1700.0) & can_go & present[rows, me])[:, None]
    kick = is_kickoff(fr)
    chance = ok.any(1) & ~kick
    lab["demo_chance"] = chance.astype(np.float32)
    lab["demo_ttc"] = np.where(chance, np.where(ok, ttc, np.inf).min(1), np.nan)

    # phase and car state
    lab["kickoff"] = kick.astype(np.float32)
    ground = fr.car_on_ground[rows, me] > 0.5
    up_z = fr.car_up[rows, me, 2]
    z = fr.car_pos[rows, me, 2]
    alive = present[rows, me]
    lab["in_air"] = (~ground & alive).astype(np.float32)
    lab["on_wall"] = (ground & (z > 150.0) & (z < 1800.0) & (np.abs(up_z) < 0.7071)).astype(np.float32)
    lab["on_ceiling"] = (ground & (z > 1800.0) & (up_z < -0.7071)).astype(np.float32)
    lab["height"] = z / FIELD_Z
    lab["speed"] = speed / CAR_MAX
    lab["supersonic"] = fr.car_supersonic[rows, me].astype(np.float32)
    lab["ball_height"] = fr.ball_pos[:, 2] / FIELD_Z
    lab["ball_dist"] = np.linalg.norm(bpos - my_pos, axis=1) / 5000.0
    lab["ball_vy"] = bvel[:, 1] / BALL_MAX
    lab["has_flip"] = fr.car_has_flip[rows, me].astype(np.float32)
    out = {k: np.asarray(v, dtype=np.float32) for k, v in lab.items()}

    if registered and _REGISTRY:
        inp = LabelInput(fr, me, sign, path, path_t, pos, vel, bpos, bvel, dict(out))
        for name, reg in _REGISTRY.items():
            val = np.asarray(reg.fn(inp), dtype=np.float32).reshape(-1)
            if len(val) != n:
                raise ShapeError(f"registered label {name!r} returned {len(val)} values for {n} moments")
            out[name] = val
    return out


# ------------------------------------------------------------------------------------------------- future labels
def _soon(ev: np.ndarray, seg_end: np.ndarray, horizon: int) -> np.ndarray:
    """ev [T] 0/1 events of the step after each decision; seg_end [T] the last step of each step's segment (a goal
    ends one). -> [T] 1 if an event falls in steps t .. min(t + horizon - 1, seg_end[t]), 0 if none, NaN if the
    recording ends inside the window with no event and no goal."""
    t_len = len(ev)
    cs = np.concatenate([[0.0], np.cumsum(ev, dtype=np.float64)])
    t = np.arange(t_len)
    u = np.minimum(t + horizon - 1, seg_end)
    hits = cs[u + 1] - cs[t]
    out = (hits > 0).astype(np.float32)
    censored = (t + horizon - 1 > t_len - 1) & (seg_end >= t_len - 1) & (hits == 0)
    out[censored] = np.nan
    return out


def segment_ends(goal: np.ndarray) -> np.ndarray:
    """[T] for each step the last step of its segment: the next step with a goal (inclusive), else T - 1."""
    t_len = len(goal)
    ends = np.full(t_len, t_len - 1, np.int64)
    nxt = t_len - 1
    for t in range(t_len - 1, -1, -1):
        if goal[t] >= 0:
            nxt = t
        ends[t] = nxt
    return ends


def future_labels(game: Frames, car: int, step_seconds: float, shot: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Future labels (the module's second table) for car slot ``car`` at every step of ONE game's frames in order
    (with the event fields ev_touch, ev_demo_made, ev_demoed and goal). ``step_seconds``: the time between
    consecutive frames (8 ticks: 8 / 120). ``shot`` [T, C]: 1 where that car's touch in the step left the ball on
    target (None: worked out with the simple ball path from the frame after the touch)."""
    missing = [k for k in ("ev_touch", "ev_demo_made", "ev_demoed", "goal") if not game.has(k)]
    if missing:
        raise ShapeError(f"future_labels needs recorded events; these frames lack {missing}")
    t_len = len(game)
    goal = game.goal.astype(np.int64)
    ends = segment_ends(goal)
    team = int(round(float(game.car_team[0, car])))
    h1, h05, h3 = (max(1, int(round(s / step_seconds))) for s in (1.0, 0.5, 3.0))
    touch = game.ev_touch[:, car] > 0.5
    if shot is None:
        nxt = np.minimum(np.arange(t_len) + 1, t_len - 1)
        after = game.take(nxt)
        sgn = np.full(t_len, 1.0 if team == 0 else -1.0)
        _, _, on_t, _ = goal_entry(ball_path(after.ball_pos, after.ball_vel), PATH_T, sgn)
        shot_c = touch & on_t
    else:
        shot_c = np.asarray(shot)[:, car] > 0.5
    out = {
        "shot_soon": _soon(shot_c.astype(np.float32), ends, h1),
        "touch_soon": _soon(touch.astype(np.float32), ends, h05),
        "demo_soon": _soon((game.ev_demo_made[:, car] > 0.5).astype(np.float32), ends, h1),
        "demoed_soon": _soon((game.ev_demoed[:, car] > 0.5).astype(np.float32), ends, h1),
        "goal_soon": _soon((goal == team).astype(np.float32), ends, h3),
        "concede_soon": _soon(((goal >= 0) & (goal != team)).astype(np.float32), ends, h3),
    }
    out["outlook"] = out["goal_soon"] - out["concede_soon"]
    return out


def kinds_for(names) -> dict[str, str]:
    """{name: kind} for the given label names (unknown names are treated as numbers, "reg")."""
    return {k: LABEL_KIND.get(k, "reg") for k in names}
