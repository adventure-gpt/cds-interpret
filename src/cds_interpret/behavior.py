"""Tier 1: what a bot DOES, from the game state alone - for any bot you can watch (a bot in a simulator, RLBot games,
recorded games), no network needed.

    cnt = SideCounter(n_games)
    for each decision step:  cnt.update(frames, side_a)       # frames: Frames of every game's state before the step,
                                                              #   with the step's events and goals; side_a [games, C]
    cnt.summary()            # per side: goals, demos made / suffered, a game; goal share; habits (below)
    cnt.per_game("A")        # per-game arrays, for intervals

Habits counted per side (car-steps are decision steps of one car):
  demos_made / demos_suffered     a game
  demo_chances                    car-steps with a demo within reach (labels.demo_chance), a game
  demo_conversion                 share of demo-chance stretches followed by a demolition within 1 s
  last_back_goal_side             when a car is its team's last man back (team modes), the share of that time it is
                                  between the ball and its own goal
  threat_goal_side                when the ball is heading into its goal (labels.threat), the share of that time the
                                  side has a car goal-side
  boost_mean, supersonic_share, air_share
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .frames import Frames
from .labels import concept_labels


def profile_game(game: Frames, side_a: np.ndarray, step_seconds: float = 8 / 120, labels_every: int = 1) -> dict:
    """The same habits as SideCounter, for one recorded game at once (frames in order, with events and goals):
    side_a [C] which car slots are side A. Returns {"A": {...}, "B": {...}} of sums (car-steps, events), vectorised
    over the whole game - the offline path for recorded games and packets."""
    t_len = len(game)
    out: dict[str, dict[str, float]] = {s: {} for s in "AB"}
    present = game.car_present > 0.5
    alive = present & (game.car_demoed < 0.5)
    ts = np.arange(0, t_len, labels_every)
    rows_t, rows_c = np.nonzero(present[ts])
    rows_t = ts[rows_t]
    lab = concept_labels(game.take(rows_t), rows_c)
    team_mode = present.sum(1)[rows_t] >= 4
    h1 = max(1, int(round(1.0 / step_seconds)))
    for s, sm in (("A", side_a), ("B", ~side_a)):
        o = out[s]
        m = present & sm[None, :]
        o["demos_made"] = float((game.ev_demo_made * m).sum())
        o["demos_suffered"] = float((game.ev_demoed * m).sum())
        o["touches"] = float((game.ev_touch * m).sum())
        o["car_steps"] = float((m & alive).sum())
        o["boost_sum"] = float((game.car_boost * (m & alive)).sum())
        o["supersonic_steps"] = float((game.car_supersonic * (m & alive)).sum())
        o["air_steps"] = float(((game.car_on_ground < 0.5) & m & alive).sum())
        teams = np.unique(game.car_team[0][sm & present[0]])
        o["goals"] = float(sum((game.goal == int(tm)).sum() for tm in teams)) if game.has("goal") else 0.0
        r = sm[rows_c]
        lb = r & (lab["last_back"] > 0.5) & team_mode
        o["last_back_steps"] = float(lb.sum())
        o["last_back_goal_side"] = float((lb & (lab["goal_side"] > 0.5)).sum())
        thr = r & (lab["threat"] > 0.5)
        o["threat_car_steps"] = float(thr.sum())
        o["threat_goal_side"] = float((thr & (lab["goal_side"] > 0.5)).sum())
        o["demo_chances"] = float((r & (lab["demo_chance"] > 0.5)).sum())
        # chance stretches (a car's run of demo-chance steps) and those followed by its demolition within 1 s
        stretches = converted = 0
        for c in np.nonzero(sm)[0]:
            sel = rows_c == c
            tt, ch = rows_t[sel], lab["demo_chance"][sel] > 0.5
            starts = tt[ch & ~np.concatenate([[False], ch[:-1]])]
            dm = np.nonzero(game.ev_demo_made[:, c] > 0.5)[0]
            stretches += len(starts)
            converted += int(sum(((dm >= t0) & (dm < t0 + h1)).any() for t0 in starts))
        o["chance_stretches"], o["chance_converted"] = float(stretches), float(converted)
    return out


def summarize_profiles(profiles: list[dict], side: str) -> dict:
    """Per-game rates and shares over a list of profile_game results."""
    tot: dict[str, float] = {}
    for p in profiles:
        for k, v in p[side].items():
            tot[k] = tot.get(k, 0.0) + v
    g = max(len(profiles), 1)
    return {"games": len(profiles), "goals": tot.get("goals", 0) / g, "demos_made": tot["demos_made"] / g,
            "demos_suffered": tot["demos_suffered"] / g, "touches": tot["touches"] / g,
            "demo_chances": tot["demo_chances"] / g,
            "demo_conversion": tot["chance_converted"] / max(tot["chance_stretches"], 1),
            "last_back_goal_side": tot["last_back_goal_side"] / max(tot["last_back_steps"], 1) if tot["last_back_steps"] else float("nan"),
            "threat_goal_side": tot["threat_goal_side"] / max(tot["threat_car_steps"], 1),
            "boost_mean": tot["boost_sum"] / max(tot["car_steps"], 1),
            "supersonic_share": tot["supersonic_steps"] / max(tot["car_steps"], 1),
            "air_share": tot["air_steps"] / max(tot["car_steps"], 1)}


class SideCounter:
    KEYS = ("goals", "demos_made", "demos_suffered", "touches", "car_steps", "demo_chances", "chance_stretches",
            "chance_converted", "last_back_steps", "last_back_goal_side", "threat_steps", "threat_goal_side",
            "boost_sum", "supersonic_steps", "air_steps")

    def __init__(self, n_games: int, labels_every: int = 1):
        self.n = n_games
        self.c = {s: {k: np.zeros(n_games) for k in self.KEYS} for s in "AB"}
        self.step = 0
        self.labels_every = labels_every
        self._open: dict = {}             # (game, car) -> step a demo chance stretch started (awaiting a demo in 1 s)

    def update(self, fr: Frames, side_a: np.ndarray, path=None, path_t=None) -> None:
        present = fr.car_present > 0.5
        alive = present & (fr.car_demoed < 0.5)
        for s, m in (("A", side_a & present), ("B", ~side_a & present)):
            c = self.c[s]
            c["demos_made"] += (fr.ev_demo_made * m).sum(1)
            c["demos_suffered"] += (fr.ev_demoed * m).sum(1)
            c["touches"] += (fr.ev_touch * m).sum(1)
            c["car_steps"] += (m & alive).sum(1)
            c["boost_sum"] += (fr.car_boost * (m & alive)).sum(1)
            c["supersonic_steps"] += (fr.car_supersonic * (m & alive)).sum(1)
            c["air_steps"] += ((fr.car_on_ground < 0.5) & m & alive).sum(1)
            team = np.where(m.any(1), (fr.car_team * m).sum(1) / np.maximum(m.sum(1), 1), -1)
            if fr.has("goal"):
                c["goals"] += (fr.goal >= 0) & (fr.goal == np.round(team))
        if self.step % self.labels_every == 0:
            games, cars = np.nonzero(present)
            lab = concept_labels(fr.take(games), cars, None if path is None else path[games], path_t)
            side = side_a[games, cars]
            for s, sm in (("A", side), ("B", ~side)):
                c = self.c[s]
                g = games[sm]
                ch = lab["demo_chance"][sm] > 0.5
                np.add.at(c["demo_chances"], g[ch], 1)
                for gi, car, on in zip(g, cars[sm], ch):
                    key = (gi, car)
                    if on and key not in self._open:
                        self._open[key] = self.step
                        c["chance_stretches"][gi] += 1
                lb = (lab["last_back"][sm] > 0.5) & (fr.car_team[games[sm], cars[sm]] >= 0)
                team_mode = (present.sum(1) >= 4)[g]
                lb &= team_mode
                np.add.at(c["last_back_steps"], g[lb], 1)
                np.add.at(c["last_back_goal_side"], g[lb & (lab["goal_side"][sm] > 0.5)], 1)
                thr = lab["threat"][sm] > 0.5
                # a side's threat steps counted once per game-step: any of its cars sees the threat
                for gi in np.unique(g[thr]):
                    c["threat_steps"][gi] += 1
                    c["threat_goal_side"][gi] += float(((g == gi) & thr & (lab["goal_side"][sm] > 0.5)).any())
        # demo chance stretches converted: a demolition by that car within 15 steps (1 s) of the stretch's start
        done = []
        for (gi, car), t0 in self._open.items():
            s = "A" if side_a[gi, car] else "B"
            if fr.ev_demo_made[gi, car] > 0.5:
                self.c[s]["chance_converted"][gi] += 1
                done.append((gi, car))
            elif self.step - t0 > 15:
                done.append((gi, car))
        for k in done:
            del self._open[k]
        self.step += 1

    def per_game(self, side: str) -> dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self.c[side].items()}

    def summary(self) -> dict:
        out: dict[str, Any] = {}
        for s in "AB":
            c = self.c[s]
            g = max(self.n, 1)
            out[s] = {
                "goals": float(c["goals"].sum() / g), "demos_made": float(c["demos_made"].sum() / g),
                "demos_suffered": float(c["demos_suffered"].sum() / g), "touches": float(c["touches"].sum() / g),
                "demo_chances": float(c["demo_chances"].sum() / g),
                "demo_conversion": float(c["chance_converted"].sum() / max(c["chance_stretches"].sum(), 1)),
                "last_back_goal_side": float(c["last_back_goal_side"].sum() / max(c["last_back_steps"].sum(), 1)),
                "threat_goal_side": float(c["threat_goal_side"].sum() / max(c["threat_steps"].sum(), 1)),
                "boost_mean": float(c["boost_sum"].sum() / max(c["car_steps"].sum(), 1)),
                "supersonic_share": float(c["supersonic_steps"].sum() / max(c["car_steps"].sum(), 1)),
                "air_share": float(c["air_steps"].sum() / max(c["car_steps"].sum(), 1)),
            }
        ga, gb = self.c["A"]["goals"].sum(), self.c["B"]["goals"].sum()
        out["goal_share_A"] = float(ga / max(ga + gb, 1))
        out["games"] = self.n
        return out
