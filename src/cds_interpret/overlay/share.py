"""Two overlays in one match: both teams' thoughts, one panel.

Every overlay of this package (``with_thoughts`` in a bot, ``watch_thoughts`` in a script) draws its panel where its
config says. When there are more - the same bot on both teams, or two different bots that each show their thoughts -
they share that one panel: it shows the thoughts of the car the host's camera follows, whichever overlay has that car;
when the camera is on a car none of them has, the one of their cars that was on camera last; before any of their cars
has been on camera, one of them picked at random (every overlay picks the same). Only the overlay with that car draws
the panel; every overlay still draws its own cars' landing rings, and while overlays of both teams run, each title and
ring takes its team's colour. ``[panel] share`` chooses: ``"one"`` (that), ``"sides"`` (each overlay its own panel, on
its team's side of the screen: blue left, orange right) or ``"none"`` (each as if alone).

How they agree: each overlay leaves a small note in a folder every other overlay on the machine reads (RLBot runs a
match's bots and scripts on one machine): its cars, their teams, when one of them was last on camera, and a random
number for the pick - one small JSON file per overlay process (``<pid>.json`` in ``$CDS_INTERPRET_OVERLAYS``, by
default ``cds-interpret-overlays`` in the temp folder), rewritten twice a second and whenever its cars come on or off
camera, removed when the overlay stops. A note not rewritten for a few seconds (a process that ended without removing
it) counts for nothing and is cleaned up.
"""
from __future__ import annotations

import contextlib
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

BEAT = 0.5          # seconds between rewrites of this overlay's note (and at once when its cars come on / off camera)
ALIVE = 5.0         # a note rewritten within this many seconds is a running overlay
GONE = 120.0        # notes older than this are removed (left behind by a process that ended without removing its own)
READ_EVERY = 0.25   # seconds between reads of the others' notes
SHARE_MODES = ("one", "sides", "none")


def overlay_dir() -> Path:
    """Where the overlays leave their notes: $CDS_INTERPRET_OVERLAYS, else cds-interpret-overlays in the temp folder."""
    env = os.environ.get("CDS_INTERPRET_OVERLAYS")
    return Path(env).expanduser() if env else Path(tempfile.gettempdir()) / "cds-interpret-overlays"


class Neighbours:
    """This overlay's note, and what the other overlays' notes say. ``beat(cars, teams, spectated)`` on every frame
    the overlay runs (it writes at most twice a second, and at once when its cars come on or off camera);
    ``owns_panel(cars, spectated)``: does this overlay draw the shared panel now? ``crowded(teams)``: is an overlay of
    another team running? Never raises: a folder that can't be written or read just means no neighbours."""

    def __init__(self, folder: str | os.PathLike[str] | None = None, pid: int | None = None,
                 pick: float | None = None):
        self.folder = Path(folder) if folder is not None else overlay_dir()
        self.pid = int(os.getpid() if pid is None else pid)
        self.path = self.folder / f"{self.pid}.json"
        self.pick = float(random.random() if pick is None else pick)   # (the random pick: the highest wins)
        self.cars: tuple[int, ...] = ()
        self.teams: tuple[int, ...] = ()
        self.cam_t = 0.0                              # when one of its cars was last on camera (time.time())
        self._on_cam = False
        self.box: tuple[float, ...] | None = None     # (x, y, width, height) of its own panel, once measured
        self._wrote = -1e18
        self._read = -1e18
        self._others: list[dict[str, Any]] = []
        self.ok = True

    def beat(self, cars: Any, teams: Any, spectated: int | None, box: Any = None, now: float | None = None) -> None:
        """This overlay is running with these cars (player indices; [] for none now) of these teams; spectated:
        the player the host's camera follows (packet.match_info.last_spectated); box: its own panel's (x, y, width,
        height) once it knows it (Panel.measure)."""
        now = time.time() if now is None else float(now)
        self.cars = tuple(sorted(int(c) for c in cars))
        self.teams = tuple(sorted({int(t) for t in teams}))
        if box is not None:
            self.box = tuple(float(v) for v in box)
        on = spectated is not None and int(spectated) in self.cars
        if on:
            self.cam_t = now
        changed, self._on_cam = on != self._on_cam, on
        if changed or now - self._wrote >= BEAT:
            self._write(now)

    def alive(self, now: float | None = None) -> None:
        """Still running, nothing new to say (a frame it reads nothing on: a goal's replay)."""
        now = time.time() if now is None else float(now)
        if now - self._wrote >= BEAT:
            self._write(now)

    def _write(self, now: float) -> None:
        self._wrote = now
        note = {"pid": self.pid, "cars": list(self.cars), "teams": list(self.teams), "cam_t": self.cam_t,
                "pick": self.pick, "box": list(self.box) if self.box is not None else None, "t": now}
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(f".{self.pid}.tmp")
            tmp.write_text(json.dumps(note), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            self.ok = False

    def others(self, now: float | None = None) -> list[dict[str, Any]]:
        """The other running overlays' notes (read at most four times a second)."""
        now = time.time() if now is None else float(now)
        if now - self._read < READ_EVERY:
            return self._others
        self._read = now
        out = []
        try:
            files = list(self.folder.glob("*.json"))
        except OSError:
            files = []
        for f in files:
            if f == self.path:
                continue
            try:
                note = json.loads(f.read_text(encoding="utf-8"))
                t = float(note.get("t", 0.0))
            except (OSError, ValueError, TypeError, AttributeError):
                continue
            if now - t > GONE:
                with contextlib.suppress(OSError):
                    f.unlink()
                continue
            if now - t <= ALIVE:
                out.append(note)
        self._others = out
        return out

    def owns_panel(self, cars: Any, spectated: int | None, now: float | None = None) -> bool:
        """Does this overlay draw the one shared panel now? Overlays with the same cars (copies of one bot, one
        process a car) count as one: the panel's rules choose among their cars as before. Among the rest: the one
        with the car on camera; else the one whose car was on camera last; else the highest random pick."""
        mine = tuple(sorted(int(c) for c in cars))
        if not mine:
            return False
        groups: dict[tuple[int, ...], list[float]] = {mine: [self.cam_t, self.pick]}
        for n in self.others(now):
            g = tuple(sorted(int(c) for c in (n.get("cars") or [])))
            if not g:
                continue
            e = groups.setdefault(g, [0.0, 0.0])
            e[0] = max(e[0], float(n.get("cam_t", 0.0) or 0.0))
            e[1] = max(e[1], float(n.get("pick", 0.0) or 0.0))
        if len(groups) == 1:
            return True
        if spectated is not None:
            on = [g for g in groups if int(spectated) in g]
            if on:
                return max(on, key=lambda g: groups[g][1]) == mine
        seen = [g for g in groups if groups[g][0] > 0]
        if seen:
            return max(seen, key=lambda g: (groups[g][0], groups[g][1])) == mine
        return max(groups, key=lambda g: groups[g][1]) == mine

    def shared_box(self, own: Any, now: float | None = None) -> tuple[float, float, float, float]:
        """The panel every overlay of the match shares: the smallest box holding its own (x, y, width, height) and
        every other running overlay's - the biggest of them, so it keeps its size whoever draws it."""
        boxes = [tuple(float(v) for v in own)]
        for n in self.others(now):
            b = n.get("box")
            if isinstance(b, list) and len(b) == 4 and n.get("cars"):
                try:
                    boxes.append(tuple(float(v) for v in b))
                except (TypeError, ValueError):
                    continue
        x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x1, y1 = max(b[0] + b[2] for b in boxes), max(b[1] + b[3] for b in boxes)
        return x0, y0, x1 - x0, y1 - y0

    def crowded(self, teams: Any, now: float | None = None) -> bool:
        """Is an overlay with cars of another team running?"""
        mine = {int(t) for t in teams}
        if not mine:
            return False
        return any(any(int(t) not in mine for t in (n.get("teams") or [])) for n in self.others(now))

    def close(self) -> None:
        """Remove this overlay's note (when it stops)."""
        with contextlib.suppress(OSError):
            self.path.unlink()
