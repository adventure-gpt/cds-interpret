"""What every recorder shares: tracks (every decision step of every game) and moments (what the bot saw and did at
some of them), turned into Moments with their labels."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..frames import CAR_FIELDS, CAR_VEC_FIELDS, EVENT_FIELDS, Frames
from ..labels import PATH_T, concept_labels
from ..moments import Moments, Tracks, add_future_labels

STEP_FIELDS = ("ball_pos", "ball_vel", "ball_angvel", *CAR_VEC_FIELDS, *CAR_FIELDS)


class Recording:
    """Rows of a recording, added one decision step at a time. ``finish()`` makes Moments."""

    def __init__(self, n_cars: int, step_seconds: float, meta: dict[str, Any] | None = None):
        self.c = n_cars
        self.step_seconds = step_seconds
        self.meta = dict(meta or {})
        self.steps: dict[str, list[np.ndarray]] = {}
        self.step_game: list[int] = []
        self.step_index: list[int] = []
        self.m_track: list[int] = []
        self.m_me: list[int] = []
        self.m_obs: list[np.ndarray] = []
        self.m_out: list[np.ndarray] = []
        self.m_path: list[np.ndarray | None] = []

    def __len__(self) -> int:
        return len(self.m_track)

    @property
    def n_steps(self) -> int:
        return len(self.step_game)

    def add_step(self, game: int, step: int, row: dict[str, Any]) -> int:
        """One decision step: ``row`` holds one moment of every Frames field (ball [3], cars [C, ...]), the events of
        the step that follows (ev_touch, ev_demo_made, ev_demoed [C]; goal: -1 or the team), time and
        prev_controls [C, 8] when known. Returns the step's track index."""
        for k, v in row.items():
            self.steps.setdefault(k, []).append(np.asarray(v, np.float32 if k != "goal" else np.int8))
        self.step_game.append(int(game))
        self.step_index.append(int(step))
        return len(self.step_game) - 1

    def set_events(self, track_index: int, ev_touch: np.ndarray, ev_demo_made: np.ndarray, ev_demoed: np.ndarray,
                   goal: int) -> None:
        """Fill in the events of a step once the step has been simulated (or the next packet seen)."""
        for k, v in (("ev_touch", ev_touch), ("ev_demo_made", ev_demo_made), ("ev_demoed", ev_demoed)):
            self.steps[k][track_index] = np.asarray(v, np.float32)
        self.steps["goal"][track_index] = np.asarray(goal, np.int8)

    def add_moment(self, track_index: int, me: int, obs: Any = None, output: Any = None,
                   path: np.ndarray | None = None) -> None:
        """A moment the bot (car slot ``me``) decided at, with its input and output, and the ball's path [K, 3] at
        labels.PATH_T (None: the simple flight model)."""
        self.m_track.append(int(track_index))
        self.m_me.append(int(me))
        if obs is not None:
            self.m_obs.append(np.asarray(obs, np.float32).reshape(-1))
        if output is not None:
            self.m_out.append(np.asarray(output, np.float32).reshape(-1))
        self.m_path.append(None if path is None else np.asarray(path, np.float32))

    def _track_frames(self) -> Frames:
        return Frames({k: np.stack(v) for k, v in self.steps.items()})

    def finish(self, future: bool = True) -> Moments:
        if not self.m_track:
            raise ValueError("nothing was recorded (no decision of the recorded bot)")
        tf = self._track_frames()
        tracks = Tracks(tf, np.asarray(self.step_game, np.int64), np.asarray(self.step_index, np.int64),
                        self.step_seconds)
        idx = np.asarray(self.m_track, np.int64)
        fr = tf.take(idx)
        me = np.asarray(self.m_me, np.int64)
        game = tracks.game[idx]
        step = tracks.step[idx]
        n = len(idx)
        obs = np.stack(self.m_obs) if len(self.m_obs) == n else None
        out = np.stack(self.m_out) if len(self.m_out) == n else None
        labels = moment_labels(fr, me, self.m_path)
        m = Moments(fr, me, game, step, obs, out, labels, dict(self.meta), tracks)
        if future:
            add_future_labels(m)
        return m


def moment_labels(fr: Frames, me: np.ndarray, paths: list[np.ndarray | None]) -> dict[str, np.ndarray]:
    """concept_labels with each moment's own ball path where one was given (the simple flight model elsewhere)."""
    have = np.array([p is not None for p in paths], bool)
    if have.all():
        return concept_labels(fr, me, np.stack(paths), PATH_T)          # type: ignore[arg-type]
    if not have.any():
        return concept_labels(fr, me)
    a = concept_labels(fr.take(have), me[have], np.stack([p for p in paths if p is not None]), PATH_T)
    b = concept_labels(fr.take(~have), me[~have])
    out = {}
    for k in a:
        v = np.zeros(len(fr), np.float32)
        v[have], v[~have] = a[k], b[k]
        out[k] = v
    return out


def empty_events(c: int) -> dict[str, np.ndarray]:
    return {k: np.zeros(c, np.float32) for k in EVENT_FIELDS}
