"""Simple scripted players for recorded games: something to play against when you have one bot.

Each is a function ``controls(frames, me) -> [N, 8]`` on Frames, like a Policy's obs function, so the recorders can
use it as an opponent: ``record.rocketsim(policy, opponent="chaser")``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..frames import Frames


def _heading_error(fr: Frames, me: np.ndarray, target_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(signed angle from the car's forward axis to the target, in radians, + = to the car's right; distance)."""
    rows = np.arange(len(fr))
    pos = fr.car_pos[rows, me]
    fwd = fr.car_fwd[rows, me]
    d = target_xy - pos[:, :2]
    ang_t = np.arctan2(d[:, 1], d[:, 0])
    ang_f = np.arctan2(fwd[:, 1], fwd[:, 0])
    err = (ang_t - ang_f + np.pi) % (2 * np.pi) - np.pi
    # Rocket League's axes are left-handed: a positive yaw turns toward +y from +x, which is steering RIGHT
    return err, np.linalg.norm(d, axis=1)


def chase(fr: Frames, me: np.ndarray) -> np.ndarray:
    """Drive at the ball: steer toward it, boost on a straight line from far away, jump at a ball in the air
    nearby."""
    me = np.asarray(me, np.int64)
    rows = np.arange(len(fr))
    err, dist = _heading_error(fr, me, fr.ball_pos[:, :2])
    out = np.zeros((len(fr), 8), np.float32)
    out[:, 0] = 1.0
    out[:, 1] = np.clip(err * 2.5, -1, 1)
    out[:, 3] = out[:, 1]
    out[:, 6] = ((np.abs(err) < 0.3) & (dist > 1200) & (fr.car_on_ground[rows, me] > 0.5)).astype(np.float32)
    high = (fr.ball_pos[:, 2] > 250) & (dist < 400) & (fr.car_on_ground[rows, me] > 0.5)
    out[:, 5] = high.astype(np.float32)
    out[:, 7] = (np.abs(err) > 1.8).astype(np.float32)
    return out


def idle(fr: Frames, me: np.ndarray) -> np.ndarray:
    """Do nothing at all."""
    return np.zeros((len(fr), 8), np.float32)


SCRIPTED: dict[str, Callable[[Frames, np.ndarray], np.ndarray]] = {"chaser": chase, "idle": idle}
"""Opponents the recorders know by name."""
