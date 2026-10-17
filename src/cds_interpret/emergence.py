"""The pieces of an emergence tracker: what a policy does, and how its layers are, check-in after check-in.

Three kinds of reading, each comparable from one training snapshot to the next:

  1. Habits in its own games (tier 1, the game state only): `game_habits` counts, per recorded game and side, demos,
     supersonic time, wall and air play, high-ball and aerial touches, hard hits, shots, flip resets, ceiling drops,
     air dribbles, wavedash-like landings, kickoffs, ball contact. `summarize_habits` turns a list of games into rates
     with bootstrap intervals over games.
  2. Dispositions on a FIXED bank of recorded moments (tier 2): the same moments for every snapshot, so a change is the
     policy's and not the sample's. `situations` picks moments by concept labels (a demo chance, an offensive aerial
     chance, a 50/50 ...), `dispositions` gives the probability mass a policy puts on an action class (boost, jump,
     dodge, powerslide) in each.
  3. Health of the layers on the same bank: `layer_health` (dead and dormant units, effective rank), `linear_cka`
     between two snapshots' layers, `weight_norms`, and `kl_rows` between two policies' action distributions.

Nothing here knows any one bot: games are `Frames` (cds_interpret.frames), labels are cds_interpret.labels' names, actions a
table of controls [A, 8] (throttle, steer, pitch, yaw, roll, jump, boost, handbrake), layers plain tensors.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from .errors import ShapeError
from .frames import Frames

# ------------------------------------------------------------------------------------------------ 1. habits in games
WALL_Z, CEILING_Z = 250.0, 1900.0
HIGH_BALL_Z, AIR_TOUCH_BALL_Z = 800.0, 250.0
HARD_HIT, HARDER_HIT = 3200.0, 4000.0
CONTACT_DIST = 200.0
WAVEDASH_GAIN = 350.0
KICK_FOLLOW_S = 1.5
KICK_GOAL_S = 5.0
AERIAL_GOAL_S = 3.0


def _team_of(game: Frames, side: np.ndarray) -> int | None:
    t = game.car_team[0][side & (game.car_present[0] > 0.5)]
    return int(round(float(t[0]))) if len(t) else None


def game_habits(game: Frames, side_a: np.ndarray, shot: np.ndarray | None = None, step_seconds: float = 8 / 120,
                labels: bool = True, labels_every: int = 3) -> dict[str, dict[str, float]]:
    """Counts for one recorded game (frames in order; each frame the state BEFORE a decision step, with that step's
    events and goal): side_a [C] which car slots are side A, shot [T, C] (optional) 1 where a car's touch left the ball
    on target. Returns {"A": {...}, "B": {...}} of sums; `summarize_habits` makes rates of them.

    Definitions (decision steps; car-steps count present, not demolished cars; a touch is the first step of a
    contact - the steps of one carry or push after it are not counted again):
      on a wall    wheels down above 250 uu (and below 1900 uu); on the ceiling: wheels down above 1900 uu
      airborne     wheels off everything
      air touch    a touch while airborne with the ball above 250 uu; offensive when the ball is in the opponent's half
      high touch   a touch with the ball above 800 uu (after the step)
      hard hit     a touch by this side alone that leaves the ball above 3200 / 4000 uu/s
      flip reset   an airborne car touches the ball and has its jump back at the next step, still airborne, having had
                   none before
      ceiling drop wheels on the ceiling, then airborne
      air dribble  the 2nd+ touch of one car in one airborne stretch
      wavedash     a landing with a speed gain of 350+ uu/s at that step
      contact      car centre within 200 uu of the ball centre
      kickoff      after the start and each goal: the side of the first touch, and whether the ball is in the side's
                   attacking half 1.5 s after it (a goal before then counts for the side that scored)
      kickoff goal a goal within 5 s of a kickoff's first touch; the other goals are open play (open_goal_share: the
                   goal share without kickoff goals - a kickoff routine can carry a goal share on its own)
      aerial goal  the side scores within 3 s of one of its air touches
    With labels, also behavior.profile_game's label-based habits (demo chances and their conversion, goal-side when the
    ball heads in, last man back), its labels computed on every labels_every-th step (3: ~1 s a 300 s game)."""
    T, C = len(game), game.n_cars
    present = game.car_present > 0.5
    alive = present & (game.car_demoed < 0.5)
    ground = game.car_on_ground > 0.5
    z = game.car_pos[..., 2]
    speed = np.linalg.norm(game.car_vel, axis=2)
    nxt = np.minimum(np.arange(T) + 1, T - 1)
    ball_after = game.ball_vel[nxt]
    ball_speed_after = np.linalg.norm(ball_after, axis=1)
    ball_z_after = game.ball_pos[nxt, 2]
    touching = (game.ev_touch > 0.5) & present if game.has("ev_touch") else np.zeros((T, C), bool)
    touched = touching.copy()                                                # a touch = its first step of contact
    touched[1:] &= ~touching[:-1]
    goal = game.goal if game.has("goal") else np.full(T, -1)
    on_ceiling = ground & (z > CEILING_Z) & alive
    on_wall = ground & (z > WALL_Z) & ~on_ceiling & alive
    airborne = ~ground & alive
    team = np.round(game.car_team[0]).astype(int)
    sign = np.where(team == 0, 1.0, -1.0)                                    # +y attacks for blue
    ball_y_team = game.ball_pos[:, 1][:, None] * sign[None, :]               # [T, C]: ball y in each car's frame
    dist = np.linalg.norm(game.car_pos - game.ball_pos[:, None, :], axis=2)
    has_flip = game.car_has_flip > 0.5
    # per car events
    air_touch = touched & airborne & (game.ball_pos[:, 2][:, None] > AIR_TOUCH_BALL_Z)
    flip_reset = np.zeros((T, C), bool)
    flip_reset[:-1] = touching[:-1] & airborne[:-1] & airborne[1:] & ~has_flip[:-1] & has_flip[1:]
    ceiling_drop = np.zeros((T, C), bool)
    ceiling_drop[:-1] = on_ceiling[:-1] & ~ground[1:] & alive[1:]
    landing = np.zeros((T, C), bool)
    landing[:-1] = ~ground[:-1] & ground[1:] & alive[:-1] & alive[1:]
    gain = np.zeros((T, C))
    gain[:-1] = speed[1:] - speed[:-1]
    wavedash = landing & (gain >= WAVEDASH_GAIN)
    air_dribble = np.zeros((T, C), bool)
    for c in range(C):
        n_in = 0
        for t in range(T):
            if not airborne[t, c]:
                n_in = 0
                continue
            if touched[t, c]:
                n_in += 1
                if n_in >= 2:
                    air_dribble[t, c] = True
    boost = game.car_boost
    used = np.zeros((T, C))
    keep = alive[:-1] & alive[1:] & (goal[:-1] < 0)[:, None]
    used[:-1] = np.where(keep, np.clip(boost[:-1] - boost[1:], 0.0, None), 0.0)
    # kickoffs: the first touch after each start
    starts = [0] + [int(t) + 1 for t in np.nonzero(goal >= 0)[0] if t + 1 < T]
    follow = max(1, int(round(KICK_FOLLOW_S / step_seconds)))
    kick_goal_w = max(1, int(round(KICK_GOAL_S / step_seconds)))
    aerial_window = int(round(AERIAL_GOAL_S / step_seconds))
    out = {}
    for s, sm in (("A", side_a), ("B", ~side_a)):
        tm = _team_of(game, sm)
        m = present & sm[None, :]
        ma = alive & sm[None, :]
        alone = (touched & m).any(1) & ~(touched & ~m & present).any(1)
        o = {
            "goals": float((goal == tm).sum()) if tm is not None else 0.0,
            "goals_against": float(((goal >= 0) & (goal != tm)).sum()) if tm is not None else 0.0,
            "car_steps": float(ma.sum()),
            "demos_made": float(((game.ev_demo_made > 0.5) & m).sum()) if game.has("ev_demo_made") else 0.0,
            "demos_suffered": float(((game.ev_demoed > 0.5) & m).sum()) if game.has("ev_demoed") else 0.0,
            "touches": float((touched & m).sum()),
            "supersonic_steps": float(((game.car_supersonic > 0.5) & ma).sum()),
            "speed_sum": float((speed * ma).sum()),
            "boost_sum": float((boost * ma).sum()),
            "empty_steps": float(((boost < 10.0) & ma).sum()),
            "boost_used": float((used * m).sum()),
            "air_steps": float((airborne & sm[None, :]).sum()),
            "high_air_steps": float((airborne & (z > 300.0) & sm[None, :]).sum()),
            "wall_steps": float((on_wall & sm[None, :]).sum()),
            "ceiling_steps": float((on_ceiling & sm[None, :]).sum()),
            "wall_touches": float((touched & on_wall & sm[None, :]).sum()),
            "air_touches": float((air_touch & sm[None, :]).sum()),
            "air_touches_off": float((air_touch & (ball_y_team > 0) & sm[None, :]).sum()),
            "air_touches_def": float((air_touch & (ball_y_team <= 0) & sm[None, :]).sum()),
            "high_touches": float((touched & m & (ball_z_after > HIGH_BALL_Z)[:, None]).sum()),
            "hard_hits": float((alone & (ball_speed_after > HARD_HIT)).sum()),
            "harder_hits": float((alone & (ball_speed_after > HARDER_HIT)).sum()),
            "flip_resets": float((flip_reset & sm[None, :]).sum()),
            "ceiling_drops": float((ceiling_drop & sm[None, :]).sum()),
            "air_dribble_touches": float((air_dribble & sm[None, :]).sum()),
            "wavedashes": float((wavedash & sm[None, :]).sum()),
            "contact_steps": float(((dist < CONTACT_DIST) & ma).sum()),
            "steps": float(T),
        }
        if shot is not None:
            sh = (shot > 0) & m
            o["shots"] = float(sh.sum())
            o["shot_speed_sum"] = float((sh.any(1) * ball_speed_after).sum())
        # aerial goals: this side scores within 3 s of one of its air touches
        at = np.nonzero((air_touch & sm[None, :]).any(1))[0]
        ag = 0
        if tm is not None and len(at):
            for g_t in np.nonzero(goal == tm)[0]:
                ag += int(((at <= g_t) & (at >= g_t - aerial_window)).any())
        o["aerial_goals"] = float(ag)
        # kickoffs
        k_first = k_half = k_n = k_goals = k_against = 0.0
        for i, t0 in enumerate(starts):
            t_end = starts[i + 1] if i + 1 < len(starts) else T
            tt = np.nonzero(touched[t0:t_end].any(1))[0]
            if not len(tt):
                continue
            tc = t0 + int(tt[0])
            if tm is not None:
                gw = goal[tc:min(tc + kick_goal_w, t_end)]
                k_goals += float((gw == tm).sum())
                k_against += float(((gw >= 0) & (gw != tm)).sum())
            k_n += 1
            first_us = bool((touched[tc] & sm).any())
            first_them = bool((touched[tc] & ~sm & present[tc]).any())
            k_first += 1.0 if first_us and not first_them else (0.5 if first_us and first_them else 0.0)
            tf = tc + follow
            g_between = np.nonzero(goal[tc:min(tf, t_end)] >= 0)[0]
            if len(g_between):
                k_half += float(goal[tc + int(g_between[0])] == tm)
            elif tf < t_end and tm is not None:
                k_half += float(game.ball_pos[tf, 1] * (1.0 if tm == 0 else -1.0) > 0)
            else:
                k_n -= 1
                k_first -= 1.0 if first_us and not first_them else (0.5 if first_us and first_them else 0.0)
        o["kickoffs"], o["kickoff_first"], o["kickoff_their_half"] = float(k_n), float(k_first), float(k_half)
        o["kickoff_goals"], o["kickoff_goals_against"] = k_goals, k_against
        o["open_goals"], o["open_goals_against"] = o["goals"] - k_goals, o["goals_against"] - k_against
        out[s] = o
    if labels:
        from .behavior import profile_game
        prof = profile_game(game, side_a, step_seconds=step_seconds, labels_every=labels_every)
        for s in "AB":
            for k in ("demo_chances", "chance_stretches", "chance_converted", "threat_car_steps", "threat_goal_side",
                      "last_back_steps", "last_back_goal_side"):
                out[s][k] = prof[s][k]
    return out


# rate name -> (numerator key, denominator key or a unit). Units: "game", "car_min" (a car-minute), "share" of
# car-steps, or another key (a ratio of two sums)
HABIT_RATES = {
    "goal_share": ("goals", "@goals_total"),
    "open_goal_share": ("open_goals", "@open_total"),
    "kickoff_goals_a_game": ("kickoff_goals", "game"),
    "kickoff_goals_against_a_game": ("kickoff_goals_against", "game"),
    "open_goals_a_game": ("open_goals", "game"),
    "open_goals_against_a_game": ("open_goals_against", "game"),
    "goals_a_game": ("goals", "game"),
    "goals_against_a_game": ("goals_against", "game"),
    "demos_made_a_game": ("demos_made", "game"),
    "demos_suffered_a_game": ("demos_suffered", "game"),
    "demo_chances_a_game": ("demo_chances", "game"),
    "demo_conversion": ("chance_converted", "chance_stretches"),
    "threat_goal_side": ("threat_goal_side", "threat_car_steps"),
    "supersonic_share": ("supersonic_steps", "car_steps"),
    "mean_speed": ("speed_sum", "car_steps"),
    "mean_boost": ("boost_sum", "car_steps"),
    "boost_empty_share": ("empty_steps", "car_steps"),
    "boost_used_a_car_min": ("boost_used", "car_min"),
    "air_share": ("air_steps", "car_steps"),
    "high_air_share": ("high_air_steps", "car_steps"),
    "wall_share": ("wall_steps", "car_steps"),
    "ceiling_share": ("ceiling_steps", "car_steps"),
    "touches_a_car_min": ("touches", "car_min"),
    "contact_share": ("contact_steps", "car_steps"),
    "wall_touches_a_game": ("wall_touches", "game"),
    "air_touches_a_car_min": ("air_touches", "car_min"),
    "air_touches_off_a_car_min": ("air_touches_off", "car_min"),
    "air_touches_def_a_car_min": ("air_touches_def", "car_min"),
    "high_touches_a_game": ("high_touches", "game"),
    "hard_hits_a_game": ("hard_hits", "game"),
    "harder_hits_a_game": ("harder_hits", "game"),
    "shots_a_car_min": ("shots", "car_min"),
    "mean_shot_speed": ("shot_speed_sum", "shots"),
    "aerial_goals_a_game": ("aerial_goals", "game"),
    "aerial_goal_share": ("aerial_goals", "goals"),
    "flip_resets_a_car_min": ("flip_resets", "car_min"),
    "ceiling_drops_a_car_min": ("ceiling_drops", "car_min"),
    "air_dribble_touches_a_car_min": ("air_dribble_touches", "car_min"),
    "wavedashes_a_car_min": ("wavedashes", "car_min"),
    "kickoff_first_touch": ("kickoff_first", "kickoffs"),
    "kickoff_their_half": ("kickoff_their_half", "kickoffs"),
    "last_back_goal_side": ("last_back_goal_side", "last_back_steps"),
}


def _rate(tot: dict, num: str, den: str, n_games: float, step_seconds: float) -> float:
    a = tot.get(num, float("nan"))
    if den == "game":
        b = n_games
    elif den == "car_min":
        b = tot.get("car_steps", 0.0) * step_seconds / 60.0
    elif den == "@goals_total":
        b = tot.get("goals", 0.0) + tot.get("goals_against", 0.0)
    elif den == "@open_total":
        b = tot.get("open_goals", 0.0) + tot.get("open_goals_against", 0.0)
    else:
        b = tot.get(den, float("nan"))
    return float(a / b) if b and b > 0 and math.isfinite(a) else float("nan")


def summarize_habits(games: list[dict], side: str = "A", step_seconds: float = 8 / 120, n_boot: int = 1000,
                     seed: int = 0) -> dict[str, dict[str, float]]:
    """Rates of HABIT_RATES over a list of game_habits results for one side: {rate: {"v": value, "lo", "hi"}} with a
    95% bootstrap interval over games (ratio of sums, games resampled)."""
    keys = sorted({k for g in games for k in g[side]})
    arr = {k: np.array([g[side].get(k, np.nan) for g in games], float) for k in keys}
    n = len(games)
    rng = np.random.default_rng(seed)
    boots = rng.integers(0, n, (n_boot, n)) if n else np.zeros((0, 0), int)
    out = {}
    for name, (num, den) in HABIT_RATES.items():
        if num not in arr:
            continue
        tot = {k: float(np.nansum(v)) for k, v in arr.items()}
        v = _rate(tot, num, den, n, step_seconds)
        draws = []
        for b in boots:
            tb = {k: float(np.nansum(vv[b])) for k, vv in arr.items() if k in (num, den, "car_steps", "goals", "goals_against",
                                                                   "open_goals", "open_goals_against")}
            draws.append(_rate(tb, num, den, n, step_seconds))
        bs = np.asarray(draws, float)
        bs = bs[np.isfinite(bs)]
        out[name] = {"v": v, "lo": float(np.percentile(bs, 2.5)) if len(bs) else float("nan"),
                     "hi": float(np.percentile(bs, 97.5)) if len(bs) else float("nan")}
    out["games"] = {"v": float(n), "lo": float(n), "hi": float(n)}
    return out


# ------------------------------------------------------------------------------------------- 2. fixed-bank dispositions
def action_classes(table: np.ndarray) -> dict[str, np.ndarray]:
    """Boolean masks over the action table's rows: boost, jump (any), dodge-like (a jump with a direction: a dodge
    when airborne with a flip in hand, a directional jump on the ground), plain jump, powerslide (handbrake without a
    jump), reverse (throttle < 0 on the ground), coast (no throttle, no boost, no jump)."""
    t = np.asarray(table, float)
    if t.ndim != 2 or t.shape[1] != 8:
        raise ShapeError(f"the action table must be [A, 8] controls (throttle, steer, pitch, yaw, roll, jump, boost, "
                         f"handbrake), got {t.shape}")
    thr, pitch, yaw, roll, jump, boost, hb = t[:, 0], t[:, 2], t[:, 3], t[:, 4], t[:, 5], t[:, 6], t[:, 7]
    direction = (np.abs(pitch) + np.abs(yaw) + np.abs(roll)) > 0
    return {"boost": boost > 0.5, "jump": jump > 0.5, "dodge": (jump > 0.5) & direction,
            "plain_jump": (jump > 0.5) & ~direction, "powerslide": (hb > 0.5) & (jump < 0.5),
            "reverse": (thr < -0.5) & (jump < 0.5) & (boost < 0.5), "coast": (np.abs(thr) < 0.5) & (boost < 0.5) & (jump < 0.5)}


# Situations from concept labels (cds_interpret.labels units: heights / 2044, distances / 5000, speed / 2300, boost / 100,
# land_y in uu in the car's team frame, ttb_adv in s - positive when we reach the ball first). Each value: (rule, the
# action classes that say most about it).
SITUATIONS = {
    "demo_chance": ("an opponent in reach on our line at speed (labels.demo_chance)", ("boost",)),
    "fast_no_chance": ("as fast and boosting-able, on the ground, no opponent in line (the demo chance's control)", ("boost",)),
    "off_aerial": ("on the ground, ball 350-1600 uu up within 2500 uu, we are first by 0.25 s+, it comes down in their half",
                   ("jump", "boost")),
    "def_aerial": ("on the ground, ball 350-1600 uu up within 2500 uu, heading into our goal or coming down deep in our half",
                   ("jump", "boost")),
    "wall_up": ("on a wall, the ball 300+ uu above us within 2000 uu", ("jump", "boost")),
    "air_flip_near_ball": ("airborne with a flip in hand, the ball within 400 uu", ("dodge",)),
    "fifty": ("on the ground, ball low within 1500 uu, neither side clearly first (|ttb_adv| < 0.15 s), not a kickoff",
              ("boost", "dodge")),
    "carry": ("ball on or at our nose (within 250 uu, low), on the ground, a touch due within 0.5 s", ("jump", "dodge")),
    "beaten": ("they are 0.8 s+ first, the ball coming down in our half, we are not goal-side", ("boost",)),
    "low_boost_quiet": ("boost under 15, no threat, the ball 3000+ uu away, on the ground", ("boost", "powerslide")),
    "kickoff": ("the kickoff phase", ("boost", "dodge")),
    "threat": ("the ball heading into our goal within 3 s", ("boost", "jump")),
}


def situations(lab: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Boolean masks [N] of SITUATIONS from label columns (keys as cds_interpret.labels, with or without a 'lab_' prefix).
    Without the future label touch_soon (state-only labels), 'carry' is the geometry alone."""
    g = {k[4:] if k.startswith("lab_") else k: np.asarray(v, float) for k, v in lab.items()}
    nan0 = {k: np.nan_to_num(v, nan=0.0) for k, v in g.items()}
    ground = (nan0["in_air"] < 0.5) & (nan0["on_wall"] < 0.5) & (nan0.get("on_ceiling", 0 * nan0["in_air"]) < 0.5)
    bz = g["ball_height"] * 2044.0
    bd = g["ball_dist"] * 5000.0
    cz = g["height"] * 2044.0
    land_y = np.nan_to_num(g["land_y"], nan=0.0)
    kick = nan0["kickoff"] > 0.5
    chance = nan0["demo_chance"] > 0.5
    up_ball = ground & (bz >= 350) & (bz <= 1600) & (bd < 2500) & ~kick
    return {
        "demo_chance": chance,
        "fast_no_chance": ~chance & (g["speed"] * 2300 >= 1700) & ((g["boost"] > 0.08) | (nan0["supersonic"] > 0.5)) & ground & ~kick,
        "off_aerial": up_ball & (g["ttb_adv"] > 0.25) & (land_y > 0) & (nan0["threat"] < 0.5),
        "def_aerial": up_ball & ((nan0["threat"] > 0.5) | (land_y < -2500)),
        "wall_up": (nan0["on_wall"] > 0.5) & (bz > cz + 300) & (bd < 2000),
        "air_flip_near_ball": (nan0["in_air"] > 0.5) & (nan0["has_flip"] > 0.5) & (bd < 400),
        "fifty": ground & (np.abs(g["ttb_adv"]) < 0.15) & (bd < 1500) & (bz < 300) & ~kick,
        "carry": ground & (bd < 250) & (bz < 250) & ((nan0["touch_soon"] > 0.5) if "touch_soon" in nan0 else True) & ~kick,
        "beaten": (g["ttb_adv"] < -0.8) & (land_y < 0) & (nan0["goal_side"] < 0.5) & ~kick,
        "low_boost_quiet": ground & (g["boost"] < 0.15) & (nan0["threat"] < 0.5) & (bd > 3000) & ~kick,
        "kickoff": kick,
        "threat": nan0["threat"] > 0.5,
    }


def dispositions(probs: np.ndarray, table: np.ndarray, masks: dict[str, np.ndarray],
                 entropy: np.ndarray | None = None) -> dict[str, dict[str, float]]:
    """For each situation: the mean probability the policy puts on each action class, its mean entropy (nats) and the
    number of moments. probs [N, A] (rows sum to 1)."""
    cls = action_classes(table)
    p = np.asarray(probs, np.float64)
    if p.ndim != 2 or p.shape[1] != len(np.asarray(table)):
        raise ShapeError(f"probs must be [N, A] with A = the action table's {len(np.asarray(table))} rows, got {p.shape}")
    if entropy is None:
        entropy = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum(1)
    mass = {c: p[:, m].sum(1) for c, m in cls.items()}
    out = {}
    for name, sm in masks.items():
        sm = np.asarray(sm, bool)
        n = int(sm.sum())
        d = {"n": n, "entropy": float(entropy[sm].mean()) if n else float("nan")}
        for c, v in mass.items():
            d["p_" + c] = float(v[sm].mean()) if n else float("nan")
        out[name] = d
    return out


def kl_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(p || q) per row (nats), probability arrays [N, A]."""
    p = np.clip(np.asarray(p, np.float64), 1e-12, 1.0)
    q = np.clip(np.asarray(q, np.float64), 1e-12, 1.0)
    return (p * (np.log(p) - np.log(q))).sum(1)


# --------------------------------------------------------------------------------------------------- 3. layer health
def layer_health(acts: torch.Tensor, tau: float = 0.025, rank_rows: int = 4096, seed: int = 0) -> dict[str, float]:
    """For one layer's activations [N, D] (after its nonlinearity): the share of units that never fire on these rows
    (dead), the share whose mean |activation| is at most tau of the layer's mean (dormant, Sokar et al. 2023 - dead
    units included), the effective rank (exp of the entropy of the normalised singular values of the centred
    activations; Roy & Vetterli 2007) on up to rank_rows rows, and that rank over the width; also the share of units
    active on under 1% of rows (rare) and the share of the layer's total mean |activation| held by its top 1% of units
    (a few giant units make 'dormant' read high: check top1pct_share beside it)."""
    if acts.ndim != 2:
        raise ShapeError(f"layer_health takes activations [N, D], got {tuple(acts.shape)}")
    x = acts.float()
    a = x.abs().mean(0)
    dead = (x.abs().amax(0) <= 0).float().mean()
    rare = ((x > 0).float().mean(0) < 0.01).float().mean()
    score = a / a.mean().clamp_min(1e-12)
    dormant = (score <= tau).float().mean()
    g = torch.Generator().manual_seed(seed)
    rows = torch.randperm(len(x), generator=g)[:rank_rows]
    xs = x[rows]
    xs = xs - xs.mean(0)
    s = torch.linalg.svdvals(xs)
    p = s / s.sum().clamp_min(1e-12)
    erank = float(torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()))
    top = torch.sort(a, descending=True).values
    top_share = float(top[: max(1, len(top) // 100)].sum() / top.sum().clamp_min(1e-12))
    return {"dead": float(dead), "rare": float(rare), "dormant": float(dormant), "erank": erank,
            "erank_frac": erank / x.shape[1], "mean_abs": float(a.mean()), "top1pct_share": top_share}


def linear_cka(x: torch.Tensor, y: torch.Tensor, standardize: bool = False) -> float:
    """Linear CKA (Kornblith et al. 2019) between two representations of the same rows [N, D1], [N, D2]. standardize:
    every unit to unit variance first (so a few large units do not decide it)."""
    if len(x) != len(y):
        raise ShapeError(f"linear_cka needs the same rows in both: {len(x)} and {len(y)}")
    x = x.double() - x.double().mean(0)
    y = y.double() - y.double().mean(0)
    if standardize:
        x = x / x.std(0).clamp_min(1e-8)
        y = y / y.std(0).clamp_min(1e-8)
    xy = (y.T @ x).pow(2).sum()
    xx = (x.T @ x).pow(2).sum().sqrt()
    yy = (y.T @ y).pow(2).sum().sqrt()
    return float(xy / (xx * yy).clamp_min(1e-30))


def weight_norms(model: torch.nn.Module, prefixes: tuple[str, ...] | None = None) -> dict[str, float]:
    """Frobenius norm of every Linear weight (or only those under the given module prefixes, e.g. ("actor",
    "critic")), and the square root of their summed squares: "<prefix>:total" per prefix, or "total"."""
    out: dict[str, float] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and (prefixes is None or name.startswith(prefixes)):
            out[name] = float(mod.weight.detach().float().norm())
    if prefixes is None:
        out["total"] = float(math.sqrt(sum(v * v for v in out.values())))
        return out
    for p in prefixes:
        out[p + ":total"] = float(math.sqrt(sum(v * v for k, v in out.items() if k.startswith(p + ".") and ":" not in k)))
    return out


# ---------------------------------------------------------------------------------------------- critic calibration
def first_goal_returns(goal: np.ndarray, team: int, steps: np.ndarray, gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """For decision steps `steps` of one game: the discounted outcome of the NEXT goal - +gamma^k when `team` scores
    it k steps later, -gamma^k when the other team does - and a mask of steps with a goal still to come (the game's end
    truncates the others). goal [T] (-1 or the scoring team)."""
    gt = np.nonzero(np.asarray(goal) >= 0)[0]
    out = np.zeros(len(steps))
    ok = np.zeros(len(steps), bool)
    if not len(gt):
        return out, ok
    j = np.searchsorted(gt, steps)
    has = j < len(gt)
    k = np.where(has, gt[np.minimum(j, len(gt) - 1)] - steps, 0)
    sgn = np.where(np.asarray(goal)[gt[np.minimum(j, len(gt) - 1)]] == team, 1.0, -1.0)
    out[has] = (sgn * gamma ** k)[has]
    ok[has] = True
    return out, ok


def calibration(value: np.ndarray, ret: np.ndarray, groups: dict[str, np.ndarray] | None = None) -> dict:
    """How the critic's value tracks the realised outcome: Pearson r, the least-squares slope of value on return
    (the critic's scale), the AUROC of the value for 'we score the next goal', and per group the mean value, the mean
    return and the residual of the overall fit (value above what the fit gives = the critic is optimistic there)."""
    from .probes import auroc
    v, r = np.asarray(value, float), np.asarray(ret, float)
    ok = np.isfinite(v) & np.isfinite(r)
    v, r = v[ok], r[ok]
    if len(v) < 10:
        return {"n": int(len(v))}
    cv = np.cov(r, v)
    slope = float(cv[0, 1] / max(cv[0, 0], 1e-12))
    icpt = float(v.mean() - slope * r.mean())
    res: dict[str, Any] = {"n": int(len(v)), "r": float(np.corrcoef(v, r)[0, 1]), "slope": slope, "intercept": icpt,
           "auroc_next_goal": auroc(v, (r > 0).astype(float)), "mean_value": float(v.mean()), "mean_return": float(r.mean())}
    if groups:
        res["groups"] = {}
        for name, m in groups.items():
            m = np.asarray(m, bool)[ok]
            if m.sum() < 20:
                continue
            res["groups"][name] = {"n": int(m.sum()), "mean_value": float(v[m].mean()), "mean_return": float(r[m].mean()),
                                   "residual": float((v[m] - (slope * r[m] + icpt)).mean()),
                                   "r": float(np.corrcoef(v[m], r[m])[0, 1]) if m.sum() > 2 and r[m].std() > 0 else float("nan")}
    return res


def interval_changes(old: dict, new: dict, value: str = "v", lo: str = "lo", hi: str = "hi",
                     skip: tuple[str, ...] = ("games",)) -> list[dict]:
    """What moved between two measurements beyond its intervals: the new interval and the old one do not overlap (for
    95% intervals from resampling games, roughly p < 0.01).

    `old` and `new` map a name to {value, lo, hi}: the layout `summarize_habits` returns, or, with value="net",
    lo="net_lo", hi="net_hi", one side of a `curriculum.situation_map`. Names missing on either side, or without
    finite numbers, are left out. Returns a list of {name, old, new, old_ci, new_ci, dir ("up" / "down"), ratio},
    the largest relative moves first (a change of sign counts as the largest)."""
    out = []
    for k, n in new.items():
        o = old.get(k)
        if k in skip or not isinstance(n, dict) or not isinstance(o, dict):
            continue
        try:
            nv, nl, nh = float(n[value]), float(n[lo]), float(n[hi])
            ov, ol, oh = float(o[value]), float(o[lo]), float(o[hi])
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(x) for x in (nv, nl, nh, ov, ol, oh)):
            continue
        if nl > oh or nh < ol:
            out.append({"name": k, "old": ov, "new": nv, "old_ci": [ol, oh], "new_ci": [nl, nh],
                        "dir": "up" if nv > ov else "down", "ratio": nv / ov if ov else float("inf")})

    def size(d: dict) -> float:
        if (d["old"] >= 0) != (d["new"] >= 0):
            return float("inf")
        return abs(math.log(max(abs(d["new"]), 1e-9) / max(abs(d["old"]), 1e-9)))

    return sorted(out, key=lambda d: -size(d))
