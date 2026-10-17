"""RLGym v2 bridges: frames as RLGym game states, and your RLGym ObsBuilder as a Policy's obs function.

    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.action_parsers import LookupTableAction
    import cds_interpret as ci

    policy = ci.Policy(net, obs=ci.rlgym.obs_fn(DefaultObs(zero_padding=3)),
                       actions=LookupTableAction.make_lookup_table())
    x = policy.observe(moments.frames, moments.me)      # your bot's own input for any moment

    states, agent_of_slot = ci.rlgym.to_rlgym(frames)   # the game states themselves

What frames do not carry is filled neutrally: every boost pad up, no jump, flip or handbrake in progress, an Octane
hitbox. An ObsBuilder that keeps history between steps (stacked frames, previous actions) is reset before every
moment, so it sees each moment cold; record from your RLGym env (:func:`cds_interpret.record.rlgym`) to keep the
exact inputs your bot saw in play.

Needs RLGym v2: ``pip install "cds-interpret[rlgym]"`` (Python 3.11-3.12).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .errors import require
from .frames import Frames

N_BOOST_PADS = 34


def _api() -> Any:
    return require("rlgym.rocket_league.api", "turning frames into RLGym game states", extra="rlgym",
                   pip_name="rlgym-rocket-league")


def _physics(api: Any, pos: np.ndarray, vel: np.ndarray, angvel: np.ndarray, fwd: np.ndarray | None = None,
             up: np.ndarray | None = None) -> Any:
    ph = api.PhysicsObject()
    ph.position = np.asarray(pos, np.float32)
    ph.linear_velocity = np.asarray(vel, np.float32)
    ph.angular_velocity = np.asarray(angvel, np.float32)
    if fwd is None or up is None:
        ph.rotation_mtx = np.eye(3, dtype=np.float32)
    else:
        f = np.asarray(fwd, np.float64)
        u = np.asarray(up, np.float64)
        right = np.cross(u, f)                       # forward x right x up, Rocket League's left-handed axes
        ph.rotation_mtx = np.stack([f, right, u], axis=1).astype(np.float32)
    return ph


def agent_ids(fr: Frames, t: int) -> dict[int, str]:
    """Car slot -> RLGym agent id ("blue-0", "blue-1", ..., "orange-0", ...) of the cars present at moment t."""
    out = {}
    nb = no = 0
    for j in range(fr.n_cars):
        if fr.car_present[t, j] < 0.5:
            continue
        if fr.car_team[t, j] < 0.5:
            out[j] = f"blue-{nb}"
            nb += 1
        else:
            out[j] = f"orange-{no}"
            no += 1
    return out


def to_rlgym(fr: Frames) -> tuple[list[Any], list[dict[int, str]]]:
    """Frames as RLGym v2 GameStates, one a moment: (states, agent_of_slot), where agent_of_slot[i] maps each present
    car slot of moment i to its agent id in states[i].cars (blue cars first, in slot order)."""
    api = _api()
    states, ids = [], []
    slots = set(getattr(api.Car, "__slots__", ()))
    for t in range(len(fr)):
        gs = api.GameState()
        gs.tick_count = int(round(float(fr.time[t]) * 120)) if fr.has("time") else 0
        gs.goal_scored = bool(fr.goal[t] >= 0) if fr.has("goal") else False
        cfg = api.GameConfig()
        cfg.gravity, cfg.boost_consumption, cfg.dodge_deadzone = 1.0, 1.0, 0.5
        gs.config = cfg
        gs.ball = _physics(api, fr.ball_pos[t], fr.ball_vel[t], fr.ball_angvel[t])
        gs.boost_pad_timers = np.zeros(N_BOOST_PADS, np.float32)
        idmap = agent_ids(fr, t)
        cars = {}
        for j, aid in idmap.items():
            on_ground = bool(fr.car_on_ground[t, j] > 0.5)
            has_flip = bool(fr.car_has_flip[t, j] > 0.5)
            values = {
                "team_num": int(fr.car_team[t, j] > 0.5), "hitbox_type": 0,
                "ball_touches": int(fr.ev_touch[t, j] > 0.5) if fr.has("ev_touch") else 0,
                "bump_victim_id": None, "demo_respawn_timer": 3.0 if fr.car_demoed[t, j] > 0.5 else 0.0,
                "wheels_with_contact": (on_ground,) * 4,
                "supersonic_time": 0.5 if fr.car_supersonic[t, j] > 0.5 else 0.0,
                "boost_amount": float(fr.car_boost[t, j]), "boost_active_time": 0.0, "handbrake": 0.0,
                "is_jumping": False, "has_jumped": not on_ground, "is_holding_jump": False, "jump_time": 0.0,
                "has_flipped": (not on_ground) and not has_flip, "has_double_jumped": False,
                "air_time_since_jump": 0.0, "flip_time": 0.0, "flip_torque": np.zeros(3, np.float32),
                "is_autoflipping": False, "autoflip_timer": 0.0, "autoflip_direction": 1.0,
                "physics": _physics(api, fr.car_pos[t, j], fr.car_vel[t, j], fr.car_angvel[t, j], fr.car_fwd[t, j],
                                    fr.car_up[t, j]),
                "_inverted_physics": None,
            }
            car = api.Car()
            for k, v in values.items():
                if not slots or k in slots:
                    setattr(car, k, v)
            cars[aid] = car
        gs.cars = cars
        states.append(gs)
        ids.append(idmap)
    return states, ids


def obs_fn(builder: Any, shared_info: dict[str, Any] | None = None) -> Callable[[Frames, np.ndarray], np.ndarray]:
    """An RLGym v2 ObsBuilder as a Policy's ``obs`` function: ``obs(frames, me) -> [N, D]``, the builder's
    observation of moment i for car slot me[i]. The builder is reset before every moment."""
    def obs(frames: Frames, me: np.ndarray) -> np.ndarray:
        states, ids = to_rlgym(frames)
        info: dict[str, Any] = {} if shared_info is None else shared_info
        me = np.asarray(me).reshape(-1)
        out = []
        for i, (gs, idmap) in enumerate(zip(states, ids)):
            if int(me[i]) not in idmap:
                raise ValueError(f"moment {i}: car slot {int(me[i])} holds no car")
            agents = list(gs.cars)
            builder.reset(agents, gs, info)
            got = builder.build_obs(agents, gs, info)
            out.append(np.asarray(got[idmap[int(me[i])]], np.float32).reshape(-1))
        return np.stack(out) if out else np.zeros((0, 0), np.float32)
    return obs
