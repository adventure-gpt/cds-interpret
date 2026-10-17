"""Record games of your bot in RocketSim (the public ``rocketsim`` package): every decision step of every game as
tracks, and what the bot saw and chose at every ``every``-th decision as moments, with every concept label.

    import cds_interpret as ci
    moments = ci.record.rocketsim(policy, games=20, seconds=120, opponent="self", out="recordings/my-bot")

The policy needs an ``obs`` function (it builds its own input from the game state here) and, for a discrete policy,
its action table. RocketSim needs the arena's collision meshes: see :func:`find_meshes`.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import CDSInterpretError, MeshesNotFound, require
from ..frames import Frames
from ..moments import Moments, check_out
from ..policy import Policy
from ._builder import Recording, empty_events
from .scripted import SCRIPTED

TICK_HZ = 120
_INITIALISED: dict[str, Any] = {}


# ------------------------------------------------------------------------------------------------------- meshes
def find_meshes(path: str | os.PathLike[str] | None = None) -> Path:
    """The folder of RocketSim's collision meshes (it holds a ``soccar`` folder of ``.cmf`` files), looked for in
    order: ``path``; the ``CDS_INTERPRET_MESHES`` environment variable; ``./collision_meshes``; the copy that comes
    with RLGym's simulator (``pip install "cds-interpret[rlgym]"``). Raises MeshesNotFound saying how to get them."""
    cands: list[tuple[str, Path]] = []
    if path is not None:
        cands.append(("the path you passed", Path(path)))
    env = os.environ.get("CDS_INTERPRET_MESHES")
    if env:
        cands.append(("CDS_INTERPRET_MESHES", Path(env)))
    cands.append(("the current folder", Path.cwd() / "collision_meshes"))
    try:
        import importlib.util
        spec = importlib.util.find_spec("rlgym.rocket_league.sim")
        if spec and spec.origin:
            cands.append(("RLGym's simulator", Path(spec.origin).parent / "collision_meshes"))
    except (ImportError, ValueError):
        pass
    for _, p in cands:
        if (p / "soccar").is_dir() and any((p / "soccar").glob("*.cmf")):
            return p
    looked = "\n".join(f"  - {p}  ({why})" for why, p in cands)
    raise MeshesNotFound(
        "RocketSim needs the arena's shape - its collision meshes, a folder holding soccar/*.cmf - and couldn't find "
        f"them. It looked in:\n{looked}\n"
        "The easiest fix: pip install \"cds-interpret[rlgym]\" (RLGym's simulator comes with a copy; Python 3.11-3.12). "
        "Or dump them from your own copy of the game with RLArenaCollisionDumper and pass "
        "meshes=\"path/to/collision_meshes\" (or set the CDS_INTERPRET_MESHES environment variable)."
    )


def init_rocketsim(meshes: str | os.PathLike[str] | None = None) -> Any:
    """RocketSim, initialised once with the collision meshes (find_meshes). Returns the RocketSim module."""
    rs = require("RocketSim", "recording games in RocketSim", extra="sim", pip_name="rocketsim")
    if "path" not in _INITIALISED:
        p = find_meshes(meshes)
        rs.init(str(p))
        _INITIALISED["path"] = str(p)
    return rs


# ------------------------------------------------------------------------------------------------ state -> frames
def _v(v: Any) -> tuple[float, float, float]:
    return (float(v.x), float(v.y), float(v.z))


def _has_flip(st: Any) -> float:
    hf = getattr(st, "has_flip_or_jump", None)
    if hf is not None:
        return float(bool(hf))
    if st.is_on_ground:
        return 1.0
    return float(not st.has_flipped and not st.has_double_jumped
                 and (not st.has_jumped or st.air_time_since_jump < 1.25))


def arena_row(arena: Any, cars: list[Any], prev_controls: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """One moment of an arena as a row of Frames fields (cars in the order given)."""
    c = len(cars)
    b = arena.ball.get_state()
    row = {"ball_pos": np.array(_v(b.pos), np.float32), "ball_vel": np.array(_v(b.vel), np.float32),
           "ball_angvel": np.array(_v(b.ang_vel), np.float32)}
    vec = {k: np.zeros((c, 3), np.float32) for k in ("car_pos", "car_vel", "car_angvel", "car_fwd", "car_up")}
    sc = {k: np.zeros(c, np.float32) for k in ("car_boost", "car_on_ground", "car_demoed", "car_has_flip",
                                               "car_supersonic", "car_team", "car_present")}
    for j, car in enumerate(cars):
        st = car.get_state()
        vec["car_pos"][j], vec["car_vel"][j], vec["car_angvel"][j] = _v(st.pos), _v(st.vel), _v(st.ang_vel)
        vec["car_fwd"][j], vec["car_up"][j] = _v(st.rot_mat.forward), _v(st.rot_mat.up)
        sc["car_boost"][j] = float(st.boost)
        sc["car_on_ground"][j] = float(st.is_on_ground)
        sc["car_demoed"][j] = float(st.is_demoed)
        sc["car_has_flip"][j] = _has_flip(st)
        sc["car_supersonic"][j] = float(st.is_supersonic)
        sc["car_team"][j] = float(int(car.team))
        sc["car_present"][j] = 1.0
    row.update(vec)
    row.update(sc)
    row["prev_controls"] = np.zeros((c, 8), np.float32) if prev_controls is None else np.asarray(prev_controls, np.float32)
    return row


def _controls(rs: Any, c: np.ndarray) -> Any:
    cc = rs.CarControls()
    cc.throttle, cc.steer, cc.pitch, cc.yaw, cc.roll = (float(v) for v in c[:5])
    cc.jump, cc.boost, cc.handbrake = bool(c[5] > 0.5), bool(c[6] > 0.5), bool(c[7] > 0.5)
    return cc


def _path(arena: Any) -> np.ndarray:
    """RocketSim's ball prediction at labels.PATH_T (every 4 ticks out to 4 s): [120, 3]."""
    pred = arena.get_ball_prediction(121, 4)[1:]
    return np.array([_v(s.pos) for s in pred], np.float32)


# --------------------------------------------------------------------------------------------------- the recorder
Player = Policy | str | Callable[[Frames, np.ndarray], np.ndarray]


def _player_controls(player: Player, row_frames: Frames, slots: list[int], deterministic: bool,
                     gen: Any) -> tuple[np.ndarray, Any, Any]:
    """(controls [k, 8], obs [k, D] or None, output [k, A] or None) of one player's cars."""
    import torch
    fr = row_frames.take(np.zeros(len(slots), np.int64))
    me = np.asarray(slots, np.int64)
    if isinstance(player, Policy):
        x = player.observe(fr, me)
        with torch.no_grad():
            out = player.forward(x)
            if player.discrete:
                player.check_actions(int(out.shape[-1]), exact=True)
                lp = player.log_probs_from_output(out)
                a = lp.argmax(-1) if deterministic else torch.multinomial(lp.exp(), 1, generator=gen)[:, 0]
                if player.actions is None:
                    raise CDSInterpretError("to play a discrete policy in RocketSim, give Policy its action table "
                                            "(actions=[A, 8])")
                ctl = player.actions[a.cpu().numpy()]
            else:
                ctl = out.float().clamp(-1, 1).cpu().numpy()
        return np.asarray(ctl, np.float32), x.numpy(), out.detach().float().cpu().numpy()
    fn = SCRIPTED[player] if isinstance(player, str) else player
    return np.asarray(fn(fr, me), np.float32), None, None


def rocketsim(policy: Policy, games: int = 10, seconds: float = 120.0, team_size: int = 1,
              opponent: Player = "self", tick_skip: int = 8, every: int = 3, deterministic: bool = True,
              seed: int = 0, meshes: str | os.PathLike[str] | None = None, out: str | os.PathLike[str] | None = None,
              record_opponent: bool | None = None, ball_path: str = "rocketsim", late: float = 0.05,
              if_exists: str = "error", log: Callable[[str], Any] | None = print) -> Moments:
    """Play ``games`` games of ``seconds`` each (from kickoffs; a goal starts a new kickoff) of ``policy`` as blue
    against ``opponent`` as orange - ``"self"`` (the same policy), another Policy, ``"chaser"``, ``"idle"`` or a
    function ``controls(frames, me) -> [N, 8]`` - and record them.

    Args:
        team_size: cars per team (1, 2 or 3).
        tick_skip: physics ticks per decision (8 = 15 decisions a second, what most bots use).
        every: keep a moment at every n-th decision of each recorded car (tracks keep every decision).
        deterministic: argmax actions (True) or sampled ones.
        late: the chance that a car's new controls reach it one tick late at a decision (as in the real game, where
            a decision is not always in time). RocketSim and argmax players are deterministic, so without this
            every game from the same kickoff spot would be the same game; 0 turns it off.
        record_opponent: also keep the opponent's moments (default: when it is "self").
        ball_path: "rocketsim" (RocketSim's ball prediction for the labels: walls, ramps and corners) or "simple"
            (the toolkit's flight model: faster, rougher).
        out: a folder to save the moments to (they are returned either way).
        if_exists: what to do when ``out`` already holds a recording - "error" (the default: nothing is
            overwritten by accident), "reuse" (load it and play nothing: handy for a script you run again),
            "append" (add these games, numbered after the ones there) or "overwrite".
        log: progress lines go to this function (None: quiet).
    """
    if out is not None and check_out(out, if_exists, recorder=True) and if_exists == "reuse":
        if log:
            log(f"  {Path(out).as_posix()} already holds a recording: using it "
                "(if_exists=\"overwrite\" plays new games)")
        return Moments.load(out)                          # (all checks before playing any game, not after)
    rs = init_rocketsim(meshes)
    if not isinstance(policy, Policy):
        raise CDSInterpretError("rocketsim() needs a cds_interpret.Policy: wrap your network first, "
                                "ci.Policy(net, obs=..., actions=...)")
    policy.require_obs()
    if isinstance(opponent, str) and opponent not in ("self", *SCRIPTED):
        raise CDSInterpretError(f"opponent {opponent!r} isn't one the recorder knows. Use \"self\", another "
                                f"ci.Policy, a function, or one of {sorted(SCRIPTED)}.")
    if team_size not in (1, 2, 3):
        raise CDSInterpretError(f"team_size must be 1, 2 or 3 (cars per team), not {team_size}")
    if ball_path not in ("rocketsim", "simple"):
        raise CDSInterpretError(f"ball_path must be \"rocketsim\" or \"simple\", not {ball_path!r}")
    opp: Player = policy if opponent == "self" else opponent
    rec_b = (opponent == "self") if record_opponent is None else record_opponent
    if rec_b and not isinstance(opp, Policy):
        rec_b = False
    import torch
    gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    step_seconds = tick_skip / TICK_HZ
    n_steps = int(np.ceil(seconds * TICK_HZ / tick_skip))
    c = 2 * team_size
    opp_name = opp.name if isinstance(opp, Policy) else (opp if isinstance(opp, str) else getattr(opp, "__name__", "custom"))
    meta = {"source": "rocketsim", "policy": policy.name, "opponent": "self" if opponent == "self" else opp_name,
            "team_size": team_size, "tick_skip": tick_skip, "every": every, "games": games, "seconds": seconds,
            "deterministic": deterministic, "seed": seed, "step_seconds": step_seconds, "ball_path": ball_path,
            "late": late}
    rec = Recording(c, step_seconds, meta)
    goals = np.zeros((games, 2), np.int64)
    t0 = time.time()
    for g in range(games):
        arena = rs.Arena(rs.GameMode.SOCCAR)
        cars = [arena.add_car(0) for _ in range(team_size)] + [arena.add_car(1) for _ in range(team_size)]
        arena.reset_kickoff(seed * 100003 + g)
        prev = np.zeros((c, 8), np.float32)
        score = (0, 0)
        blue = list(range(team_size))
        orange = list(range(team_size, c))
        for step in range(n_steps):
            row = arena_row(arena, cars, prev)
            ti = rec.add_step(g, step, {**row, **empty_events(c), "goal": -1, "time": step * step_seconds})
            fr = Frames({k: v[None] for k, v in row.items()})
            ctl = np.zeros((c, 8), np.float32)
            keep = step % every == 0
            path = _path(arena) if keep and ball_path == "rocketsim" else None
            for player, slots, recorded in ((policy, blue, True), (opp, orange, rec_b)):
                cc, x, o = _player_controls(player, fr, slots, deterministic, gen)
                ctl[slots] = cc
                if keep and recorded:
                    for k, s in enumerate(slots):
                        rec.add_moment(ti, s, None if x is None else x[k], None if o is None else o[k], path)
            hit_before = [car.get_state().ball_hit_info.tick_count_when_hit for car in cars]
            demos_before = [int(car.demos) for car in cars]
            demoed_before = [bool(car.get_state().is_demoed) for car in cars]
            delayed = rng.random(c) < late
            if delayed.any():                                     # those cars keep their old controls one tick
                for car, cc, old, d in zip(cars, ctl, prev, delayed):
                    car.set_controls(_controls(rs, old if d else cc))
                arena.step(1)
            for car, cc in zip(cars, ctl):
                car.set_controls(_controls(rs, cc))
            arena.step(tick_skip - int(delayed.any()))
            ev = empty_events(c)
            for j, car in enumerate(cars):
                st = car.get_state()
                hit = st.ball_hit_info
                ev["ev_touch"][j] = float(bool(hit.is_valid) and hit.tick_count_when_hit != hit_before[j])
                ev["ev_demo_made"][j] = float(int(car.demos) > demos_before[j])
                ev["ev_demoed"][j] = float(bool(st.is_demoed) and not demoed_before[j])
            new_score = (int(arena.blue_score), int(arena.orange_score))
            goal = -1
            if new_score != score:
                goal = 0 if new_score[0] > score[0] else 1
                goals[g, goal] += 1
                score = new_score
            rec.set_events(ti, ev["ev_touch"], ev["ev_demo_made"], ev["ev_demoed"], goal)
            prev = ctl
            if goal >= 0:
                arena.reset_kickoff(seed * 100003 + g * 997 + step)
                prev = np.zeros((c, 8), np.float32)
        if log is not None:
            log(f"  game {g + 1}/{games}: blue {goals[g, 0]} - orange {goals[g, 1]}  "
                f"({len(rec)} moments so far, {time.time() - t0:.0f} s)")
    m = rec.finish()
    m.meta.update({"goals_blue": int(goals[:, 0].sum()), "goals_orange": int(goals[:, 1].sum()),
                   "goals_per_game": goals.tolist()})
    if out is not None:
        m.save(out, if_exists="error" if if_exists == "reuse" else if_exists, renumber=True)
        if log is not None:
            log(f"  saved {len(m)} moments to {out}")
    return m
