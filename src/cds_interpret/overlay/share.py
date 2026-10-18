"""Two overlays in one match: both teams' thoughts on screen at once.

Every overlay of this package (``with_thoughts`` in a bot, ``watch_thoughts`` in a script) draws its panel where its
config says. When the other team has an overlay too - the same bot on both teams, or two different bots that each
show their thoughts - the two panels would land on the same spot. So each overlay leaves a small note in a folder
every other overlay on the machine reads (RLBot runs a match's bots and scripts on one machine): which team's car its
panel shows, and when it last drew one. When another overlay is showing the OTHER team, the panels go to their
teams' sides of the screen - blue on the left, orange on the right, as the game's scoreboard has them - mirror
images of the configured position, and each panel's title and landing rings take its team's colour. Alone, nothing
changes. ``[panel] side`` chooses: ``"auto"`` (that), ``"team"`` (always the teams' sides) or ``"fixed"`` (always as
configured).

The notes are one small JSON file per overlay process (``<pid>.json`` in ``$CDS_INTERPRET_OVERLAYS``, by default
``cds-interpret-overlays`` in the temp folder), rewritten about once a second, removed when the overlay stops; a note
not rewritten for a few seconds (a process that ended without removing it) counts for nothing and is cleaned up.
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

BEAT = 1.0          # seconds between rewrites of this overlay's note
ALIVE = 5.0         # a note rewritten within this many seconds is a running overlay
SHOWN = 30.0        # ... that counts as showing a team while it drew that team's panel within this many seconds
GONE = 120.0        # notes older than this are removed (left behind by a process that ended without removing its own)
READ_EVERY = 1.0    # seconds between reads of the others' notes


def overlay_dir() -> Path:
    """Where the overlays leave their notes: $CDS_INTERPRET_OVERLAYS, else cds-interpret-overlays in the temp folder."""
    env = os.environ.get("CDS_INTERPRET_OVERLAYS")
    return Path(env).expanduser() if env else Path(tempfile.gettempdir()) / "cds-interpret-overlays"


class Neighbours:
    """This overlay's note, and what the other overlays' notes say. ``beat(team, showing)`` on every frame the overlay
    runs (it writes at most once a second); ``crowded(team)``: is another overlay showing the other team now?
    Never raises: a folder that can't be written or read just means no neighbours."""

    def __init__(self, folder: str | os.PathLike[str] | None = None, pid: int | None = None):
        self.folder = Path(folder) if folder is not None else overlay_dir()
        self.pid = int(os.getpid() if pid is None else pid)
        self.path = self.folder / f"{self.pid}.json"
        self.team: int | None = None
        self.shown = 0.0                              # when this overlay last drew a panel (time.time())
        self._wrote = -1e18
        self._read = -1e18
        self._others: list[dict[str, Any]] = []
        self.ok = True

    def beat(self, team: int | None, showing: bool, now: float | None = None) -> None:
        """This overlay is running; ``team``: the team of the car its panel shows (None: no panel); ``showing``: it
        draws a panel this frame."""
        now = time.time() if now is None else float(now)
        if showing and team is not None:
            self.team, self.shown = int(team), now
        if now - self._wrote < BEAT:
            return
        self._wrote = now
        note = {"pid": self.pid, "team": self.team, "shown": self.shown, "t": now}
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(f".{self.pid}.tmp")
            tmp.write_text(json.dumps(note), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            self.ok = False

    def others(self, now: float | None = None) -> list[dict[str, Any]]:
        """The other running overlays' notes (read at most once a second)."""
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

    def crowded(self, team: int | None, now: float | None = None) -> bool:
        """Is another overlay showing the other team (a panel drawn within the last 30 s, still running)?"""
        if team is None:
            return False
        now = time.time() if now is None else float(now)
        for note in self.others(now):
            t = note.get("team")
            if t is not None and int(t) != int(team) and now - float(note.get("shown", 0.0)) <= SHOWN:
                return True
        return False

    def close(self) -> None:
        """Remove this overlay's note (when it stops)."""
        with contextlib.suppress(OSError):
            self.path.unlink()
