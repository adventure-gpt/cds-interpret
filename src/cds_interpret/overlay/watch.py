"""Thoughts for a bot you can't modify: a display-only RLBot v5 SCRIPT that draws the thoughts panel and landing
rings over a bot that runs on its own - a compiled program, a bot in another language, anyone's bot you have the
weights of - without touching it.

The real bot drives its cars as always. The script reads each game packet, rebuilds the bot's input from it with an
observation function you provide, runs the bot's network (its weights, loaded into the script's memory - nothing is
written anywhere), and draws the same panel as ``with_thoughts`` over the cars it matches:

    from cds_interpret.overlay import watch_thoughts

    if __name__ == "__main__":
        watch_thoughts("my_port:make_policy", "thoughts.npz", cars="SomeBot")("me/watch-somebot").run(
            wants_match_communications=False, wants_ball_predictions=True)

``make_policy()`` returns a ``cds_interpret.Policy``: the bot's network with its weights, its obs function (how the
bot builds its input from the game - a port of the bot's own code), its action table, and ``mirror=`` if it sees the
field mirrored at times. How faithful the panel is depends on that port: check it plays like the real bot.

What it can show: everything the panel shows, with these limits - the outlook needs a critic (``Policy(value=...)``)
or an outlook probe in the pack, "looking at" needs the pack's recorded moments of the mode, and a bot that keeps a
memory between decisions (a recurrent network, previous actions it doesn't send) is seen only as far as the obs
function can rebuild it. Rows that can't be computed take no room. ``cds-interpret watch-template FOLDER`` writes a
ready script and its RLBot file; docs/howto/watch-a-bot.md is the guide.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from .._rlbot import FrameGuard, _log, fix_socket_reads, has_ball, landing_point, packet_row, spectated
from ..errors import CDSInterpretError, require
from ..frames import Frames
from ..hooks import LayerTap
from .bot import STALE, _fresh, _rows
from .pack import Thoughts, land_point, load_pack
from .panel import GROUP, Panel, PanelConfig, check_panel, smooth_point
from .share import Neighbours
from .team import Follow, landing_cars, match_seed, subtitle_text
from .worker import compute

MAX_FAILED_DRAWS = 75                   # draws failing in a row (about 5 s) before the script switches itself off


def match_cars(packet: Any, cars: Any) -> list[int]:
    """The player indices of the watched bot's cars: ``cars`` as a piece of their name (case does not matter:
    "SomeBot" matches "SomeBot" and "SomeBot (2)"), a list of player indices, or a function packet -> indices."""
    players = list(getattr(packet, "players", []) or [])
    if callable(cars):
        found = [int(i) for i in cars(packet)]
    elif isinstance(cars, (list, tuple, set)):
        found = [int(i) for i in cars]
    else:
        word = str(cars).strip().lower()
        found = [i for i, p in enumerate(players) if word and word in str(getattr(p, "name", "")).lower()]
    return sorted(i for i in set(found) if 0 <= i < len(players))


def watch_thoughts(policy: Any, pack: str | os.PathLike[str], cars: Any, panel: Any = None,
                   hooks: dict[str, Callable] | None = None, title: str | None = None,
                   team_sizes: list[int] | None = None, error_log: str | os.PathLike[str] | None = None,
                   observe: Any = None) -> type:
    """An RLBot v5 Script class that draws a bot's thoughts without being that bot (see the module's notes).

    Args:
        policy: the bot's ``cds_interpret.Policy`` or ``"package.module:factory"`` returning one (built when the
            script starts, in its own process).
        pack: a thoughts pack made for that network.
        cars: which cars are the bot: a piece of their name, a list of player indices, or a function packet ->
            indices (match_cars).
        panel, hooks, title: as with_thoughts: the panel's TOML file or PanelConfig (follow, landing_for, keep_size,
            colours ... all work the same), functions for custom rows, the title (default: the pack's).
        team_sizes: draw only when the watched car's team has this many players (e.g. [2, 3] for a pack made for
            a bot's 2v2 / 3v3 brain); default: always.
        error_log: a file the script also writes its errors to.
        observe: for a bot whose input needs more of the game than the policy's obs function gets from game state
            (boost pads, timers it keeps between packets, ...): a function, or ``"package.module:factory"`` (with its
            keyword arguments as a pair) building one when the script starts - ``observe(packet, cars, field_info)``
            returns the bot's inputs for those cars [n, ...], or a dict with "obs" and optionally "mirrored" [n]
            (which cars see the field mirrored), "mask" [n, A] (the actions each may take: the move is chosen among
            them) and "state" (more fields for the game state a critic or codec gets). It may keep what it needs
            between packets; it is called on draw frames only.
    """
    Script: Any = require("rlbot.managers", "watching a bot in RLBot").Script
    fix_socket_reads()                                   # (rlbot before 2.0.0b56: read whole messages)

    pack_path = os.fspath(pack)
    if cars is None or (isinstance(cars, str) and not cars.strip()):
        raise CDSInterpretError("watch_thoughts: say which cars are the bot - cars=\"part of its name\" (or a list of "
                                "player indices)")

    class WatchThoughts(Script):
        """Draws the thoughts of the bot's cars; display only."""

        def initialize(self) -> None:
            self._cds_on = False
            try:
                self._cds_setup()
                self._cds_on = True
                th = self._cds_th
                _log(self, f"cds-interpret watch: on - {len(th.pack['features'])} named features at "
                           f"{th.layer}, watching {cars!r}", "info")
            except Exception as e:                    # a display script: say why, draw nothing
                _log(self, f"cds-interpret watch: off ({e})")

        def _cds_setup(self) -> None:
            from .explain import Explainer, load_policy
            torch.set_num_threads(1)
            pk = load_pack(pack_path)
            th = Thoughts(pk)
            pol = load_policy(policy)
            if pol is None:
                raise CDSInterpretError("watch_thoughts needs the bot's policy")
            pol.net.eval()
            for p in pol.net.parameters():           # (the script's own copy: no gradients to its weights)
                p.requires_grad_(False)
            conf = panel if isinstance(panel, PanelConfig) else PanelConfig.load(panel) if panel else PanelConfig()
            conf = replace(conf, title=title if title is not None else
                           conf.title if conf.title is not None else pk["meta"].get("title", "THOUGHTS"))
            self._cds_panel = Panel(conf, hooks)
            for problem in check_panel(conf, pk):
                _log(self, f"cds-interpret watch: {problem}")
            self._cds_th, self._cds_pol = th, pol
            self._cds_ex = Explainer(th, pol, freeze=False)
            for note in self._cds_ex.notes:
                _log(self, f"cds-interpret watch: {note}", "info")
            self._cds_ex.notes.clear()
            taps = {"h": pk["layer"]}
            out_layer = pk["meta"].get("output_layer", "")
            if out_layer:
                taps["y"] = out_layer
            if th.land is not None and th.land_module != pk["layer"]:
                taps["land_h"] = th.land_module
            self._cds_taps = [LayerTap(pol.net, {k: v}, keep="all").attach() for k, v in taps.items()]
            self._cds_critic = pol.value if pol.value_fn is not None else None
            from .._specs import build, is_spec
            self._cds_observe = build(observe, "observe") if observe is not None and is_spec(observe) else observe
            self._cds_st: dict[str, Any] = {
                "drawn": -10 ** 9, "follow": Follow(conf.follow), "seeded": False, "think": {}, "draw": {},
                "readings": 0, "cleared": True, "panel_car": None,
                "guard": FrameGuard(self, "watch", MAX_FAILED_DRAWS, log_file=error_log)}
            self._cds_scale = float(pk["meta"].get("value_scale", 1.0))
            self._cds_share = Neighbours()             # (the other team's overlay, if any: overlay.share)
            self.update_rendering_status(True)

        def handle_packet(self, packet: Any) -> None:
            if not getattr(self, "_cds_on", False):
                return
            st = self._cds_st
            self._cds_share.alive()                    # (this overlay is running; what it shows: _cds_draw)
            frame = int(packet.match_info.frame_num)
            if 0 <= frame - st["drawn"] < self._cds_panel.config.draw_every or not self.renderer.can_render:
                return
            st["drawn"] = frame
            if not has_ball(packet):                    # (the replay after a goal: nothing to read)
                return
            try:
                self._cds_draw(packet, frame)
                st["guard"].ok()
            except Exception as e:                    # one bad packet must not end (or freeze) the panel
                if st["guard"].failed(frame, e):
                    self._cds_on = False
                    try:
                        self.renderer.clear_render_group(GROUP)
                    finally:
                        _log(self, f"cds-interpret watch: off after {MAX_FAILED_DRAWS} failed draws in a row")

        def _clear(self) -> None:
            if not self._cds_st["cleared"]:
                self.renderer.clear_render_group(GROUP)
                self._cds_st["cleared"] = True

        def _cds_draw(self, packet: Any, frame: int) -> None:
            st, pnl, th, pol = self._cds_st, self._cds_panel, self._cds_th, self._cds_pol
            players = packet.players
            watched = match_cars(packet, cars)
            if team_sizes:
                size = {t: sum(1 for p in players if int(p.team) == t) for t in (0, 1)}
                watched = [i for i in watched if size[int(players[i].team)] in team_sizes]
            share = self._cds_share
            spec = spectated(packet)
            share.beat(watched, [int(players[i].team) for i in watched], spec, box=st.get("box"))
            if not watched:
                self._clear()
                return
            if not st["seeded"]:
                st["follow"].seed = match_seed(packet, watched)
                st["seeded"] = True
            panel_car, rule = st["follow"].pick(watched, spec)
            conf = pnl.config
            # (overlay.share) more overlays in the match: one panel, the camera's rules over all their cars
            shared_out = conf.share == "one" and not share.owns_panel(watched, spec)
            if shared_out:
                panel_car, rule = None, "hide"
            if panel_car != st["panel_car"]:
                st["panel_car"], st["readings"] = panel_car, 0
            # the bot's input for each of its cars, rebuilt from the packet, and its network on it
            row = packet_row(packet)
            n = len(watched)
            seen: dict[str, Any] = {}
            if self._cds_observe is not None:
                got_obs = self._cds_observe(packet, watched, getattr(self, "field_info", None))
                seen = dict(got_obs) if isinstance(got_obs, dict) else {"obs": got_obs}
                x = torch.as_tensor(np.asarray(seen["obs"].detach().cpu() if torch.is_tensor(seen["obs"])
                                               else seen["obs"], np.float32))
                row.update({kk: np.asarray(v) for kk, v in (seen.get("state") or {}).items()})
            else:
                fr = Frames({k: np.repeat(np.asarray(v)[None], n, 0) for k, v in row.items()})
                x = pol.observe(fr, np.array(watched))
            for t in self._cds_taps:
                t.clear()
            with torch.no_grad():
                out = pol.forward(x)
            got = {k: r for t in self._cds_taps for k in t.layers if (r := _rows(t, k)) is not None}
            if "h" not in got:
                raise CDSInterpretError(f"the policy's forward never went through the pack's layer {th.layer!r}: is it "
                                        "the same network the pack was made from?")
            wants = pnl.config.wants
            k = watched.index(panel_car) if panel_car in watched else None
            msg: dict[str, Any] = {"frame": frame, "layer": got["h"],
                                   "output": got.get("y", out.detach().float().cpu().numpy().reshape(n, -1)),
                                   "obs": x.numpy(), "land_h": got.get("land_h"), "panel": k, "cars": watched,
                                   "state": row, "why": wants["why"], "me": panel_car, "donors": wants["donors"]}
            for key in ("mirrored", "mask"):                # (from the observe hook: its own flags and masks)
                if seen.get(key) is not None:
                    v = seen[key]
                    msg[key] = np.asarray(v.detach().cpu() if torch.is_tensor(v) else v)
            msg["looking"] = bool(k is not None and wants["looking"] and st["readings"] % max(1, wants["every"]) == 0)
            st["readings"] += 1
            res = compute(th, msg, self._cds_critic, self._cds_ex)
            for note in res.get("notes") or []:
                _log(self, f"cds-interpret watch: {note}")
            result = dict(zip(watched, res["cars"]))
            dt = pnl.config.draw_every / 120.0
            markers = pnl.config.markers
            marks: list[tuple[int | None, Any]] = []
            for c in landing_cars(markers.landing_cars, watched, panel_car, spec):
                xy = result[c].get("land_xy")
                if xy is None:
                    continue
                new = land_point(th.land_frame, xy, int(players[c].team), players[c].physics)   # (team or car frame)
                if new is None:
                    continue
                prev = st["think"].get(c)
                fresh = prev is None or frame - prev[1] > STALE * 120
                pt = smooth_point(None if fresh else prev[0], new, dt, markers.landing_smooth)
                st["think"][c] = (pt, frame)
                marks.append((c, pt))
            thc = result.get(panel_car) if panel_car is not None else None
            from rlbot import flat
            teams = sorted({int(players[i].team) for i in watched})
            crowded = conf.share != "none" and share.crowded(teams)
            if conf.share == "one" and frame - st.get("measured", -10 ** 9) >= 60:
                res_m = thc if thc is not None else result.get(watched[0])
                if res_m is not None:            # (its own panel's size, for the shared panel: as big as the biggest)
                    st["box"] = pnl.measure(flat, res_m, "-" if len(watched) > 1 and conf.subtitle else None,
                                            {"why": self._cds_ex.available["why"],
                                             "looking": len(players) in self._cds_ex.available["looking"]},
                                            teams[0] if len(teams) == 1 else None, crowded)
                    st["measured"] = frame
            if thc is None and not marks:
                self._clear()
                return
            st["cleared"] = False
            team = int(players[panel_car].team) if panel_car is not None else None
            mates = [i for i in watched if int(players[i].team) == team] if team is not None else []
            sub = None
            if thc is not None and len(mates) > 1 and pnl.config.subtitle:
                sub = subtitle_text(pnl.config.subtitle, panel_car, mates, rule, str(getattr(players[panel_car],
                                                                                              "name", "")))
            true = (landing_point(getattr(self, "ball_prediction", None)) if markers.prediction and not shared_out
                    else None)
            av = self._cds_ex.available
            box = share.shared_box(st["box"]) if conf.share == "one" and st.get("box") is not None else None
            if team is None and len(teams) == 1:
                team = teams[0]
            pnl.draw(self.renderer, flat, thc, car_index=panel_car, landing_true=true,
                     state=_fresh(st["draw"], panel_car, frame) if thc is not None else None, frame=frame,
                     value_scale=self._cds_scale, team=team, marks=marks, subtitle=sub,
                     available={"why": av["why"], "looking": len(players) in av["looking"]},
                     crowded=crowded, box=box)

        def retire(self) -> None:
            for t in getattr(self, "_cds_taps", []):
                t.detach_hooks()
            if getattr(self, "_cds_share", None) is not None:
                self._cds_share.close()

    return WatchThoughts


WATCH_PY = '''"""The thoughts of a bot you can't modify, drawn over its cars while it plays - display only: the bot drives as
always; this script reads the game, rebuilds the bot's input and runs its network to draw the panel.

RLBot: add watch.script.toml (in this folder) to your match as a SCRIPT, beside the bot itself, turn rendering on.
Guide: https://github.com/adventure-gpt/cds-interpret/blob/main/docs/howto/watch-a-bot.md
"""

from cds_interpret.overlay import watch_thoughts

PACK = "thoughts.npz"                 # EDIT: a thoughts pack made for the bot's network
POLICY = "my_port:make_policy"        # EDIT: "module:function" returning a cds_interpret.Policy - the bot's network
                                      #       with its weights, and its obs function (how it builds its input)
CARS = "MyBot"                        # EDIT: a piece of the bot's name in the game (or a list of player numbers)
PANEL = None                          # or "panel.toml" (cds-interpret panel panel.toml writes one)

if __name__ == "__main__":
    watch_thoughts(POLICY, PACK, cars=CARS, panel=PANEL, error_log="watch_errors.log")(
        "cds-interpret/watch").run(wants_match_communications=False, wants_ball_predictions=True)
'''

SCRIPT_TOML = '''#:schema https://rlbot.org/schemas/script.json
# Thoughts for a bot you can't modify - display only. Add this file to your match as a SCRIPT (not a bot), beside the
# bot itself, and turn on rendering. Edit watch.py first: the pack, the bot's policy and its name.
[settings]
name = "Thoughts (display only)"
run_command = "python watch.py"
run_command_linux = "python3 watch.py"
agent_id = "cds-interpret/watch"
'''


def write_template(folder: str | os.PathLike[str], force: bool = False) -> list[str]:
    """Write watch.py and watch.script.toml into ``folder`` (``cds-interpret watch-template``); the paths written."""
    from pathlib import Path
    d = Path(folder)
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for name, text in (("watch.py", WATCH_PY), ("watch.script.toml", SCRIPT_TOML)):
        p = d / name
        if p.exists() and not force:
            raise CDSInterpretError(f"{p} already exists - pass --force to overwrite it")
        p.write_text(text, encoding="utf-8")
        out.append(str(p))
    return out
