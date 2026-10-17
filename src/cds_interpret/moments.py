"""Moments: what a bot saw and did at a set of game moments, with the game state and its concept labels.

    m = Moments.load("recordings/my-bot")       # written by a recorder (RocketSim, RLBot, RLGym) or by m.save()
    m.frames                                    # Frames: the game state of every moment (all cars)
    m.me                                        # [N] the car slot of the bot at each moment
    m.game, m.step                              # [N] which game, which decision step of it
    m.obs                                       # [N, D] the network's input (None: build it with policy.observe)
    m.output                                    # [N, A] what the network output (None if not recorded)
    m.labels                                    # name -> [N] concept labels (labels.concept_labels and future ones)
    m.tracks                                    # every step of every game (for clips and future labels), or None

Moments of one recording come from whole games, so probes can be trained and scored on different games.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .errors import CDSInterpretError, ShapeError
from .frames import Frames
from .labels import FUTURE_LABELS, concept_labels, future_labels
from .store import MomentStore

_FR, _LAB = "fr_", "lab_"
IF_EXISTS = ("error", "append", "overwrite", "reuse")


def check_out(path: str | os.PathLike[str], if_exists: str, recorder: bool = False) -> bool:
    """Whether a recording already sits at path; "error" raises then, "overwrite" removes it first ("reuse", for a
    recorder: the caller loads it instead of playing)."""
    import shutil
    if if_exists not in IF_EXISTS or (if_exists == "reuse" and not recorder):
        known = IF_EXISTS if recorder else IF_EXISTS[:3]
        raise CDSInterpretError(f"if_exists must be one of {', '.join(repr(k) for k in known)}, not {if_exists!r}")
    root = Path(path)
    there = (root / "manifest.json").exists()
    if there and if_exists == "error":
        use = (f"To use it instead of playing new games, pass if_exists=\"reuse\" (or load it yourself: "
               f"moments = ci.Moments.load({root.as_posix()!r})). " if recorder else
               f"(To study it: moments = ci.Moments.load({root.as_posix()!r}).) ")
        raise CDSInterpretError(
            f"{root} already holds a recording. {use}To add these games to it, pass if_exists=\"append\"; to "
            "replace it, if_exists=\"overwrite\"; or pick a new folder.")
    if there and if_exists == "overwrite":
        shutil.rmtree(root)                              # (only ever a folder holding a recording's manifest)
        return False
    return there


@dataclass
class Tracks:
    """Every decision step of every recorded game (all cars, with events and goals), for animated clips and
    future labels. ``frames`` rows are grouped by game, in step order."""

    frames: Frames
    game: np.ndarray
    step: np.ndarray
    step_seconds: float = 8 / 120

    def __post_init__(self) -> None:
        change = np.nonzero(np.diff(self.game))[0] + 1
        starts = np.concatenate([[0], change]).astype(np.int64)
        ends = np.concatenate([change, [len(self.game)]]).astype(np.int64)
        self._span = {int(self.game[s]): (int(s), int(e)) for s, e in zip(starts, ends)} if len(self.game) else {}

    @property
    def games(self) -> list[int]:
        return list(self._span)

    def game_frames(self, game: int) -> Frames:
        s, e = self._span[int(game)]
        return self.frames.take(slice(s, e))

    def window(self, game: int, t0: int, t1: int) -> tuple[Frames, int]:
        """(Frames of steps t0..t1 of the game, clipped to it; the step of its first row)."""
        s, e = self._span[int(game)]
        first = int(self.step[s])
        a, b = s + max(t0 - first, 0), min(s + (t1 - first) + 1, e)
        return self.frames.take(slice(a, b)), int(self.step[a])


@dataclass
class Moments:
    """N moments of a bot's play (see the module's docstring)."""

    frames: Frames
    me: np.ndarray
    game: np.ndarray
    step: np.ndarray
    obs: np.ndarray | None = None
    output: np.ndarray | None = None
    labels: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    tracks: Tracks | None = None

    def __post_init__(self) -> None:
        n = len(self.frames)
        self.me = np.broadcast_to(np.asarray(self.me, dtype=np.int64), (n,)).copy()
        self.game = np.broadcast_to(np.asarray(self.game, dtype=np.int64), (n,)).copy()
        self.step = np.broadcast_to(np.asarray(self.step, dtype=np.int64), (n,)).copy()
        for name, arr in (("obs", self.obs), ("output", self.output)):
            if arr is not None and len(arr) != n:
                raise ShapeError(f"Moments.{name} has {len(arr)} rows for {n} moments")
        for k, v in self.labels.items():
            if len(v) != n:
                raise ShapeError(f"label {k} has {len(v)} values for {n} moments")

    def __len__(self) -> int:
        return len(self.frames)

    def __repr__(self) -> str:
        games = len(np.unique(self.game)) if len(self) else 0
        parts = [f"{len(self)} moments", f"{games} games", f"{len(self.labels)} labels"]
        if self.obs is not None:
            parts.append(f"obs {self.obs.shape[1]}-d")
        if self.tracks is not None:
            parts.append("tracks")
        return f"Moments({', '.join(parts)})"

    @property
    def seq(self) -> np.ndarray:
        """A sequence key per moment (the game and the car): consecutive moments of one car in one game share it."""
        return self.game * 64 + self.me

    @property
    def n_games(self) -> int:
        return len(np.unique(self.game))

    def take(self, idx: Any) -> Moments:
        return Moments(self.frames.take(idx), self.me[idx], self.game[idx], self.step[idx],
                       None if self.obs is None else self.obs[idx], None if self.output is None else self.output[idx],
                       {k: v[idx] for k, v in self.labels.items()}, dict(self.meta), self.tracks)

    def with_labels(self, path: np.ndarray | None = None, path_t: np.ndarray | None = None,
                    overwrite: bool = False) -> Moments:
        """These moments with the labels of the moment computed (concept_labels; future labels need a recording).
        Labels already present are kept unless overwrite."""
        lab = concept_labels(self.frames, self.me, path, path_t)
        for k, v in lab.items():
            if overwrite or k not in self.labels:
                self.labels[k] = v
        return self

    # ---- files
    def save(self, path: str | os.PathLike[str], meta: dict[str, Any] | None = None, if_exists: str = "error",
             renumber: bool = False) -> Path:
        """Write these moments to a folder (tracks go in its 'tracks' subfolder). When the folder already holds a
        recording: if_exists="error" (the default) stops, "append" adds these moments to it (renumber=True numbers
        their games after the ones there), "overwrite" replaces it."""
        root = Path(path)
        if check_out(root, if_exists) and if_exists == "append" and renumber:
            have = MomentStore(root).load(["game"])["game"]
            offset = int(have.max()) + 1 if len(have) else 0
            self.game = self.game + offset
            if self.tracks is not None:
                self.tracks = Tracks(self.tracks.frames, self.tracks.game + offset, self.tracks.step,
                                     self.tracks.step_seconds)
        cols: dict[str, np.ndarray] = {f"{_FR}{k}": v for k, v in self.frames.data.items()}
        cols.update({"me": self.me.astype(np.int16), "game": self.game.astype(np.int32),
                     "step": self.step.astype(np.int32)})
        if self.obs is not None:
            cols["obs"] = np.asarray(self.obs, np.float32)
        if self.output is not None:
            cols["output"] = np.asarray(self.output, np.float32)
        cols.update({f"{_LAB}{k}": np.asarray(v, np.float32) for k, v in self.labels.items()})
        st = MomentStore(root)
        st.write_shard(cols, meta=meta)
        st.set_meta(**self.meta)
        if self.tracks is not None:
            tc = {f"{_FR}{k}": v for k, v in self.tracks.frames.data.items()}
            tc.update({"game": self.tracks.game.astype(np.int32), "step": self.tracks.step.astype(np.int32)})
            ts = MomentStore(root / "tracks")
            ts.write_shard(tc, meta=meta)
            ts.set_meta(step_seconds=self.tracks.step_seconds)
        return root

    @classmethod
    def load(cls, path: str | os.PathLike[str], rows: np.ndarray | None = None) -> Moments:
        """Moments from a folder written by save() or a recorder (all rows, or the rows ``rows``)."""
        st = MomentStore(path)
        if not st.exists():
            raise CDSInterpretError(f"There are no recorded moments in {str(path)!r} (no manifest.json there). Record "
                                    "some first - see docs/howto/record-moments.md.")
        names = list(st.columns())
        c = st.load(names, rows)
        fr = Frames({k[len(_FR):]: v for k, v in c.items() if k.startswith(_FR)})
        labels = {k[len(_LAB):]: v for k, v in c.items() if k.startswith(_LAB)}
        tracks = None
        tp = Path(path) / "tracks"
        if (tp / "manifest.json").exists():
            ts = MomentStore(tp)
            tcols = ts.load(list(ts.columns()))
            tr_fr = Frames({k[len(_FR):]: v for k, v in tcols.items() if k.startswith(_FR)})
            order = np.lexsort((tcols["step"], tcols["game"]))
            tracks = Tracks(tr_fr.take(order), tcols["game"][order].astype(np.int64),
                            tcols["step"][order].astype(np.int64), float(ts.meta.get("step_seconds", 8 / 120)))
        m = cls(fr, c["me"], c["game"], c["step"], c.get("obs"), c.get("output"), labels, dict(st.meta), tracks)
        if tracks is not None and any(k not in labels for k in FUTURE_LABELS):
            add_future_labels(m)
        return m

    @classmethod
    def load_all(cls, root: str | os.PathLike[str]) -> Moments:
        """Every recording under a folder (each folder with a manifest.json, at any depth) as one Moments, game ids
        renumbered so they never collide - e.g. the folders the RLBot recorder writes, one per bot and match."""
        root = Path(root)
        found = sorted(p.parent for p in root.rglob("manifest.json") if p.parent.name != "tracks")
        if not found:
            raise CDSInterpretError(f"no recordings under {root} (no manifest.json found)")
        parts = [cls.load(p) for p in found]
        offset = 0
        for p in parts:
            ids = np.unique(p.game)
            if p.tracks is not None:
                ids = np.unique(np.concatenate([ids, p.tracks.game]))
            remap = {int(g): offset + i for i, g in enumerate(ids)}
            p.game = np.array([remap[int(g)] for g in p.game], np.int64)
            if p.tracks is not None:
                p.tracks = Tracks(p.tracks.frames, np.array([remap[int(g)] for g in p.tracks.game], np.int64),
                                  p.tracks.step, p.tracks.step_seconds)
            offset += len(ids)
        return cls.concat(parts)

    @staticmethod
    def concat(parts: list[Moments]) -> Moments:
        """Several Moments as one (their game ids must not collide: load_all renumbers them). Tracks are joined when
        every part has them, with the same number of car slots."""
        if not parts:
            raise ValueError("Moments.concat needs at least one Moments")
        if len(parts) == 1:
            return parts[0]
        tracks = None
        if all(p.tracks is not None for p in parts) and len({p.tracks.frames.n_cars for p in parts}) == 1:  # type: ignore[union-attr]
            tracks = Tracks(Frames.concat([p.tracks.frames for p in parts]),  # type: ignore[union-attr]
                            np.concatenate([p.tracks.game for p in parts]),  # type: ignore[union-attr]
                            np.concatenate([p.tracks.step for p in parts]),  # type: ignore[union-attr]
                            parts[0].tracks.step_seconds)  # type: ignore[union-attr]
        if len({p.frames.n_cars for p in parts}) > 1:
            raise CDSInterpretError("these recordings have different numbers of cars (1v1 and 2v2, say): study them "
                                    "separately, or load them one mode at a time")
        keys = set.intersection(*(set(p.labels) for p in parts))
        has_obs = all(p.obs is not None for p in parts)
        has_out = all(p.output is not None for p in parts)
        return Moments(Frames.concat([p.frames for p in parts]), np.concatenate([p.me for p in parts]),
                       np.concatenate([p.game for p in parts]), np.concatenate([p.step for p in parts]),
                       np.concatenate([p.obs for p in parts]) if has_obs else None,  # type: ignore[misc]
                       np.concatenate([p.output for p in parts]) if has_out else None,  # type: ignore[misc]
                       {k: np.concatenate([p.labels[k] for p in parts]) for k in keys}, dict(parts[0].meta), tracks)


def add_future_labels(m: Moments) -> Moments:
    """Future labels (labels.future_labels) for moments that have tracks: per game and car, from every step."""
    if m.tracks is None or not m.tracks.frames.has("ev_touch"):
        return m
    tr = m.tracks
    fut = {k: np.full(len(m), np.nan, np.float32) for k in FUTURE_LABELS}
    for g in np.unique(m.game):
        if int(g) not in tr._span:
            continue
        gf = tr.game_frames(int(g))
        s0 = int(tr.step[tr._span[int(g)][0]])
        sel = np.nonzero(m.game == g)[0]
        for car in np.unique(m.me[sel]):
            fl = future_labels(gf, int(car), tr.step_seconds)
            rows = sel[m.me[sel] == car]
            pos = np.clip(m.step[rows] - s0, 0, len(gf) - 1)
            for k in FUTURE_LABELS:
                fut[k][rows] = fl[k][pos]
    for k, v in fut.items():
        m.labels[k] = v
    return m
