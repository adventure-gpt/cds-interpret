"""When did a concept or a feature form? The same probes on the same recorded moments, for each training snapshot
of a network.

    rows = track(snapshots, x, layers, labels, kinds, train, test, groups=game_ids)

``snapshots``: a list of ``(name, step, policy)`` - ``policy`` a :class:`~cds_interpret.Policy` or a function that
loads one (so only one snapshot is in memory at a time) - optionally with a 4th element, ``transform(x)``, that
turns the recorded inputs into that snapshot's input layout (when the observation changed during training).
``x``: the recorded inputs [N, D] (from the snapshot that played the games). ``targets``: extra number targets,
name -> [N] (e.g. the final network's SAE feature activations: how well a linear readout of each earlier snapshot
recovers them = when that feature's information was there).

Returns one dict per (snapshot, layer): ``{"name", "step", "layer", "metrics": {label: probe result}}``. Plot with
:func:`cds_interpret.viz.plots.lines`. The moments are those the final snapshot played: an earlier snapshot is probed
on states it might never have reached itself - say so when reading the curves.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .probes import headline, ridge_suite


def track(snapshots: Sequence[tuple], x: Any, layers: list[str], labels: dict[str, np.ndarray], kinds: dict[str, str],
          train: np.ndarray, test: np.ndarray, groups: np.ndarray | None = None,
          targets: dict[str, np.ndarray] | None = None, log: Callable[[str], Any] | None = None) -> list[dict]:
    out = []
    labs = dict(labels)
    knd = dict(kinds)
    for k, v in (targets or {}).items():
        labs[f"target:{k}"] = np.asarray(v, np.float32)
        knd[f"target:{k}"] = "reg"
    for snap in snapshots:
        name, step, pol = snap[0], snap[1], snap[2]
        transform = snap[3] if len(snap) > 3 else None
        policy = pol() if callable(pol) and not hasattr(pol, "net") else pol
        x_in = transform(x) if transform is not None else x
        acts = policy.activations(x_in, layers)
        for layer in layers:
            res = ridge_suite(acts[layer], labs, knd, train, test, groups=groups)
            out.append({"name": name, "step": int(step), "layer": layer, "metrics": res})
            if log is not None:
                log(f"{name} {layer}: " + " ".join(f"{k} {headline(v):.3f}" for k, v in sorted(res.items())))
        del acts, policy
    return out


def curves(rows: list[dict], layer: str, labels: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """track()'s rows as plot series for one layer: label -> (steps, headline score)."""
    sel = sorted((r for r in rows if r["layer"] == layer), key=lambda r: r["step"])
    out = {}
    for k in labels:
        xs = np.array([r["step"] for r in sel if k in r["metrics"]], float)
        ys = np.array([headline(r["metrics"][k]) for r in sel if k in r["metrics"]], float)
        if len(xs):
            out[k] = (xs, ys)
    return out
