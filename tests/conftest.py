"""Shared helpers for the tests: small hand-made moments and networks, no simulator needed."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from cds_interpret import frames as F

torch.set_num_threads(int(os.environ.get("CDSI_TEST_THREADS", "2")))


def tiny_mlp(seed: int = 0, d_in: int = 6, n_out: int = 4) -> torch.nn.Sequential:
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Linear(d_in, 16), torch.nn.ReLU(), torch.nn.Linear(16, 8), torch.nn.ReLU(),
                               torch.nn.Linear(8, n_out))


def make_frames(n: int = 1, cars: int = 2, **over) -> F.Frames:
    d = {"ball_pos": np.tile([0.0, 0.0, 92.75], (n, 1)), "ball_vel": np.zeros((n, 3)),
         "car_pos": np.zeros((n, cars, 3)), "car_vel": np.zeros((n, cars, 3)),
         "car_fwd": np.tile([0.0, 1.0, 0.0], (n, cars, 1)), "car_up": np.tile([0.0, 0.0, 1.0], (n, cars, 1)),
         "car_boost": np.full((n, cars), 50.0), "car_on_ground": np.ones((n, cars)),
         "car_team": np.tile(np.arange(cars) >= cars // 2, (n, 1)).astype(float)}
    d.update(over)
    return F.from_arrays(**d)


def meshes_or_skip():
    pytest.importorskip("RocketSim")
    from cds_interpret.errors import MeshesNotFound
    from cds_interpret.record import find_meshes
    try:
        return find_meshes()
    except MeshesNotFound:
        pytest.skip("RocketSim's collision meshes are not available here")
