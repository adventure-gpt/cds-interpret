"""The live thoughts overlay for any RLBot v5 bot whose brain is a PyTorch network.

    # run.py of your bot folder
    from cds_interpret.overlay import with_thoughts
    from my_bot import MyBot

    if __name__ == "__main__":
        with_thoughts(MyBot, pack="thoughts.npz", network="policy").run(...)    # instead of MyBot(...).run(...)

Why it cannot change how your bot plays:

1. nothing is added to the decision path: the layers the overlay shows are copied by read-only forward hooks while
   your bot's own forward pass runs (its outputs stay bit for bit the same - the tests check it), and everything
   else - the features, the probes, the drawing - happens after your controls have been sent;
2. the reading runs in a separate process (``worker=True``, the default) with one thread at below-normal priority,
   so it never shares your bot's thread pool; at most one request is in flight and a busy worker just skips a
   decision;
3. nothing is drawn on the tick of a decision, and nothing is sent to the worker before it has started (a request
   bigger than the pipe's buffer would otherwise wait - in the bot - for a worker still loading);
4. an error in one frame is logged and the overlay carries on with the next (the first few, then every 200th, with
   the traceback; ``error_log=`` copies them to a file); only a long run of failures in a row switches it off, and
   then it takes its drawing down. A worker that stops is started again (three times at most). Your bot plays on
   whatever happens, and packets without a ball (the replay after a goal) are skipped.

Team modes: the panel shows one of your bot's cars, chosen by the panel's ``follow`` rules (the car the host's camera
follows, by default; see :mod:`cds_interpret.overlay.team`), and every car gets its own landing marker. A Hivemind's
network rows are matched to its ``indices`` in order. A plain Bot runs once per car: every copy works out the same
panel car from the same packet, that copy draws the panel and each copy draws its own marker.

The match must allow rendering (RLBot's "enable rendering" setting), and the pack must be made from the same network
your bot runs (its module paths name its layers).
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from .._rlbot import (
    FrameGuard,
    _log,
    find_network,
    has_ball,
    landing_point,
    our_cars,
    our_indices,
    packet_row,
    spectated,
    wrap_after_decision,
)
from ..errors import CDSInterpretError
from ..hooks import LayerTap
from .pack import Thoughts, land_point, load_pack
from .panel import GROUP, Panel, PanelConfig, check_panel, smooth_point
from .share import Neighbours
from .team import Follow, landing_cars, match_seed, subtitle_text
from .worker import WorkerLink, compute, load_critic, serve

STALE = 0.5                             # seconds without a draw after which a car's smoothing starts afresh
MAX_FAILED_FRAMES = 600                 # frames failing IN A ROW (5 s) before the overlay switches itself off
MAX_FAILED_READINGS = 50                # the worker's answers failing in a row before it does
MAX_RESTARTS = 3                        # times a worker that stopped is started again


class GiveUp(RuntimeError):
    """The overlay switches itself off (too many failures in a row, a worker that keeps stopping)."""


def _rows(tap: LayerTap, key: str, flat: bool = True) -> np.ndarray | None:
    calls = tap.out.get(key)
    if not calls:
        return None
    if flat:
        calls = [c.reshape(c.shape[0] if c.ndim > 1 else 1, -1) for c in calls]
    else:
        calls = [c if c.ndim > 1 else c.reshape(1, -1) for c in calls]
    return torch.cat(calls).float().cpu().numpy()


def _fresh(store: dict[int, dict[str, Any]], car: int, frame: int) -> dict[str, Any]:
    """A car's own drawing state (its smoothing, the features it has held on screen), started afresh when it has not
    been drawn for a while - so switching the panel to another car never shows that car's old numbers."""
    st = store.get(car)
    if st is None or frame - st.get("_drawn", frame) > STALE * 120:
        st = store[car] = {}
    st["_drawn"] = frame
    return st


def with_thoughts(bot_class: type, pack: str | os.PathLike[str], network: str | Callable[[Any], torch.nn.Module] | None = None,
                  panel: str | os.PathLike[str] | PanelConfig | None = None, hooks: dict[str, Callable] | None = None,
                  input_layer: str = "", critic: Any = None, worker: bool = True, title: str | None = None,
                  draw_every: int | None = None, x: float | None = None, y: float | None = None,
                  policy: Any = None, error_log: str | os.PathLike[str] | None = None,
                  extras: Callable[[Any, Any], dict[str, Any]] | None = None) -> type:
    """Your RLBot v5 bot class (a ``rlbot.managers.Bot`` or ``Hivemind`` subclass) with the thoughts overlay.

    Args:
        pack: the thoughts pack (overlay.save_pack) for this network.
        network: where your bot keeps its torch network - a dotted attribute name, a function bot -> module, or
            None to pick the largest nn.Module on the bot.
        panel: how the panel looks - a TOML file (``cds-interpret panel`` writes an example of every option) or a
            PanelConfig; None for the default panel.
        hooks: functions your panel config names in ``hook = "..."`` (custom rows, your own markers).
        input_layer: the module whose INPUT is your observation ("" = the network itself); read for a critic and for
            "why this move".
        critic: ``"package.module:factory"`` (or ``("package.module:factory", {keyword arguments})``) -
            factory(**kwargs) returns obs [n, D] -> values [n], built in the worker process - for the outlook bar
            (without one the policy's critic or the pack's outlook probe is used). A critic that takes ``state`` (and
            ``cars``) gets the game too: one packet's Frames fields and whatever ``extras`` adds.
        worker: compute in a separate process (True, the safe default) or in the bot's process after the controls
            are sent (False: for debugging).
        title, draw_every, x, y: shortcuts that override the panel config's title, frames between draws and
            top-left corner.
        policy: ``"package.module:factory"`` (or with its keyword arguments, as ``critic``) - it returns a
            ``cds_interpret.Policy`` of the same network (with its obs function or input codec), built in the worker
            process - for the "why this move" and "looking at" rows (docs/howto/why-and-looking.md). With
            ``worker=False`` it may also be a Policy.
        error_log: a file the overlay also writes its errors to (with tracebacks), besides your bot's log.
        extras: ``extras(bot, packet) -> {name: array}``, run in your bot's process after each decision's controls
            are sent: more of the game, as your bot sees it, for the worker - added to the game state a critic with
            ``state`` and the policy's obs function receive (e.g. your bot's own state arrays). Keep it small and
            quick; names must not start with ``car_`` unless the array has a row per car.
    """
    pack_path = os.fspath(pack)
    from .._specs import is_spec
    if worker and policy is not None and not is_spec(policy):
        raise CDSInterpretError("with_thoughts: give policy as \"package.module:factory\" (or with its keyword "
                                "arguments: (\"package.module:factory\", {...})) - the overlay's worker builds its own "
                                "copy of your network from it (a Policy object works only with worker=False)")
    if critic is not None and not is_spec(critic):
        raise CDSInterpretError("with_thoughts: give critic as \"package.module:factory\" (or with its keyword "
                                "arguments), built in the overlay's worker")

    def init(bot: Any) -> None:
        pk = load_pack(pack_path)
        net = find_network(bot, network)
        out_layer = pk["meta"].get("output_layer", "")
        taps = {"h": pk["layer"], "y": out_layer}
        land = pk.get("land")
        if land is not None and land["module"] != pk["layer"]:
            taps["land_h"] = land["module"]
        bot._cds_th_taps = [LayerTap(net, {k: v}, keep="all").attach() for k, v in taps.items()]
        if critic or policy is not None:
            bot._cds_th_taps.append(LayerTap(net, {"x": input_layer}, inputs=True, keep="all").attach())
        th = Thoughts(pk)
        bot._cds_th = th
        notes: list[str] = []
        bot._cds_th_inline = None
        if worker:
            bot._cds_th_link = WorkerLink(serve, pack_path, critic, policy)
        else:
            bot._cds_th_link = None
            from .explain import Explainer, load_policy
            pol = load_policy(policy)
            ex = Explainer(th, pol, freeze=False) if pol is not None else None
            crit = load_critic(critic) if critic else (pol.value if pol is not None and pol.value_fn else None)
            bot._cds_th_inline = (crit, ex)
            if ex is not None:
                notes += ex.notes
                ex.notes.clear()
        # (loaded here, not at import: a mistake in the panel file switches the overlay off - the bot still plays)
        if isinstance(panel, PanelConfig):
            conf = panel
        elif panel is not None:
            conf = PanelConfig.load(panel)
        else:
            conf = PanelConfig()
        if conf.title is None:                          # (a panel file without a title: the pack's)
            conf = replace(conf, title=pk["meta"].get("title", "THOUGHTS"))
        changes: dict[str, Any] = {k: v for k, v in (("title", title), ("draw_every", draw_every), ("x", x), ("y", y))
                                   if v is not None}
        if changes:
            conf = replace(conf, **changes)
        bot._cds_th_panel = Panel(conf, hooks)
        wants = conf.wants
        for problem in check_panel(conf, pk) + notes:   # (things that would otherwise just not show)
            _log(bot, f"cds-interpret overlay: {problem}")
        if policy is None and (conf.rows_of("why") or conf.rows_of("looking")) and (th.fd is not None or th.donors):
            _log(bot, "cds-interpret overlay: tip - the 'why this move' and 'looking at' rows need your network in the "
                      "overlay's worker: pass policy=\"my_module:make_policy\" to with_thoughts "
                      "(docs/howto/why-and-looking.md)", "info")
        bot._cds_th_state = {"cars": {}, "drawn": -10 ** 9, "think": {}, "draw": {}, "sent": 0, "pending": None,
                             "follow": Follow(conf.follow), "seeded": False, "wants": wants, "cleared": True,
                             "warned_rows": False, "restarts": 0,
                             "guard": FrameGuard(bot, "thoughts overlay", MAX_FAILED_FRAMES, log_file=error_log),
                             "worker_guard": FrameGuard(bot, "thoughts overlay's worker", MAX_FAILED_READINGS,
                                                        log_file=error_log)}
        bot._cds_th_scale = float(pk["meta"].get("value_scale", 1.0))
        bot._cds_th_share = Neighbours()                # (the other team's overlay, if any: overlay.share)
        if hasattr(bot, "update_rendering_status"):
            bot.update_rendering_status(True)

    def before(bot: Any, packet: Any) -> None:
        for t in bot._cds_th_taps:
            t.clear()

    def available(bot: Any, n_cars: int) -> dict[str, bool]:
        link = bot._cds_th_link
        if link is not None:
            av = link.hello.get("available") or {}
        else:
            ex = bot._cds_th_inline[1]
            av = ex.available if ex is not None else {}
        return {"why": bool(av.get("why")), "looking": n_cars in (av.get("looking") or [])}

    def store(bot: Any, res: dict[str, Any], cars: list[int]) -> None:
        st = bot._cds_th_state
        for note in res.get("notes") or []:
            _log(bot, f"cds-interpret overlay: {note}")
        for car, r in zip(cars, res.get("cars", [])):
            st["cars"][car] = r

    def after(bot: Any, packet: Any) -> None:
        """One frame of the overlay; an error is logged and the next frame goes on (see the module's notes)."""
        st = bot._cds_th_state
        frame = getattr(getattr(packet, "match_info", None), "frame_num", "?")
        try:
            if _after(bot, packet):                     # (a run of failures ends with a frame drawn well)
                st["guard"].ok()
        except GiveUp:
            take_down(bot)
            raise
        except Exception as e:
            if st["guard"].failed(frame, e):
                take_down(bot)
                raise RuntimeError(f"{MAX_FAILED_FRAMES} frames in a row failed; the last: {e!r}") from e

    def take_down(bot: Any) -> None:
        """The overlay's drawing off the screen (when it switches itself off: a frozen panel would mislead)."""
        with contextlib.suppress(Exception):          # (nothing more to do: the overlay is going off anyway)
            bot.renderer.clear_render_group(GROUP)

    def check_worker(bot: Any) -> None:
        """A worker process that stopped is started again, a few times at most."""
        link, st = bot._cds_th_link, bot._cds_th_state
        if link is None or link.alive:
            return
        if st["restarts"] >= MAX_RESTARTS:
            raise GiveUp(f"the overlay's worker stopped {MAX_RESTARTS + 1} times")
        st["restarts"] += 1
        st["guard"].write(f"cds-interpret: thoughts overlay: its worker stopped (exit code {link.proc.exitcode}); "
                          f"starting it again ({st['restarts']} of {MAX_RESTARTS})")
        bot._cds_th_link = WorkerLink(serve, pack_path, critic, policy)
        st["pending"] = None
        st["worker_guard"].ok()                         # (a fresh worker, a fresh count)

    def _after(bot: Any, packet: Any) -> bool:
        """The frame's work; True when it drew."""
        share = bot._cds_th_share
        share.alive()                                   # (this overlay is running; what it shows: below)
        if not has_ball(packet):                        # (the replay after a goal: nothing to read)
            return False
        check_worker(bot)
        got: dict[str, np.ndarray] = {}
        for t in bot._cds_th_taps:
            for k in t.layers:
                r = _rows(t, k, flat=(k != "x"))
                if r is not None:
                    got[k] = r
        st = bot._cds_th_state
        frame = int(packet.match_info.frame_num)
        mine = our_indices(bot)                         # the cars this process drives, in its rows' order
        ours = our_cars(bot, packet)                    # every car this bot drives in the match, lowest index first
        if not st["seeded"]:
            st["follow"].seed = match_seed(packet, ours)
            st["seeded"] = True
        spec = spectated(packet)
        panel_car, rule = st["follow"].pick(ours, spec)
        conf = bot._cds_th_panel.config
        # (overlay.share) more overlays in the match - both teams' thoughts: one panel, the camera's rules over all their
        # cars; when another overlay's car has it, this one draws only its rings
        share.beat(ours, [int(bot.team)], spec, box=st.get("box"))
        shared_out = conf.share == "one" and not share.owns_panel(ours, spec)
        if shared_out:
            panel_car, rule = None, "hide"
        if panel_car != st.get("panel_car"):           # (a new car on the panel gets its "looking at" at once)
            st["panel_car"], st["sent"] = panel_car, 0
        wants = st["wants"]
        decided = "h" in got
        if decided:
            n_rows = len(got["h"])
            cars = mine[:n_rows]
            if n_rows != len(mine) and not st["warned_rows"]:
                st["warned_rows"] = True
                _log(bot, f"cds-interpret overlay: the network ran on {n_rows} rows for this bot's {len(mine)} cars; "
                          "rows are matched to its cars in order (a Hivemind should run its cars in `indices` order)")
            k = cars.index(panel_car) if panel_car in cars else None
            msg: dict[str, Any] = {"frame": frame, "layer": got["h"], "output": got.get("y"), "obs": got.get("x"),
                                   "land_h": got.get("land_h"), "panel": k, "cars": list(cars)}
            if policy is not None or extras is not None:   # (the game: for "looking at", a mirroring bot, a critic)
                msg["state"] = packet_row(packet)
                if extras is not None:
                    msg["state"].update({kk: np.asarray(v) for kk, v in extras(bot, packet).items()})
            if k is not None:
                msg["why"] = wants["why"]
                due = wants["looking"] and st["sent"] % max(1, int(wants["every"])) == 0
                if due:
                    msg.update(looking=True, me=int(panel_car), donors=wants["donors"])
                    msg.setdefault("state", packet_row(packet))
            link = bot._cds_th_link
            if link is None:
                crit, ex = bot._cds_th_inline
                store(bot, compute(bot._cds_th, msg, crit, ex), cars)
                st["sent"] += 1
            elif link.send(msg):
                st["sent"] += 1
                st["pending"] = cars
        if bot._cds_th_link is not None:
            r = bot._cds_th_link.poll()
            if r is not None and "cars" in r:
                store(bot, r, st["pending"] or mine)
                st["worker_guard"].ok()
            elif r is not None and "error" in r:                # (one bad reading: logged, the panel goes on)
                if st["worker_guard"].failed(frame, RuntimeError(f"the overlay's worker: {r['error']}")):
                    raise GiveUp(f"{MAX_FAILED_READINGS} of the worker's readings in a row failed; the last: "
                                 f"{r['error']}")
            if bot._cds_th_link.ready and bot._cds_th_link.hello.get("notes"):
                for note in bot._cds_th_link.hello.pop("notes"):
                    _log(bot, f"cds-interpret overlay: {note}")
        # not on the tick of a decision: the decision, its copy for the worker and the drawing would all land in one
        # 8.3 ms tick
        pnl = bot._cds_th_panel
        every = pnl.config.draw_every
        due = frame - st["drawn"] >= every
        if decided and frame - st["drawn"] < 3 * every:  # (a bot deciding every tick is drawn now and then)
            due = False
        if not st["cars"] or not due or not bot.renderer.can_render:
            return False
        dt = (frame - st["drawn"]) / 120.0
        st["drawn"] = frame
        markers = pnl.config.markers
        land_frame = bot._cds_th.land_frame if getattr(bot, "_cds_th", None) is not None else "team"
        marks = []
        for c in landing_cars(markers.landing_cars, ours, panel_car, spec):
            res_c = st["cars"].get(c)
            if res_c is None or res_c.get("land_xy") is None:
                continue
            phys = packet.players[c].physics if c < len(packet.players) else None
            new = land_point(land_frame, res_c["land_xy"], int(bot.team), phys)   # (the team's frame or the car's)
            if new is None:
                continue
            prev = st["think"].get(c)
            fresh = prev is None or frame - prev[1] > STALE * 120
            pt = smooth_point(None if fresh else prev[0], new, dt, markers.landing_smooth)
            st["think"][c] = (pt, frame)
            marks.append((c, pt))
        th = st["cars"].get(panel_car) if panel_car in mine else None
        first_of_team = bool(mine) and bool(ours) and ours[0] in mine
        from rlbot import flat
        crowded = conf.share != "none" and share.crowded([int(bot.team)])
        if conf.share == "one" and frame - st.get("measured", -10 ** 9) >= 60:
            # this overlay's own panel size (about twice a second), for the shared panel: as big as the biggest
            res_m = th if th is not None else next((st["cars"][c] for c in ours if c in st["cars"]), None)
            if res_m is not None:
                st["box"] = pnl.measure(flat, res_m, "-" if len(ours) > 1 and conf.subtitle else None,
                                        available(bot, len(packet.players)), int(bot.team), crowded)
                st["measured"] = frame
        if th is None and not marks:
            if not st["cleared"]:                       # (this copy no longer shows anything: take its drawing down)
                bot.renderer.clear_render_group(GROUP)
                st["cleared"] = True
            return True
        st["cleared"] = False
        cross = th is not None or (panel_car is None and first_of_team and not shared_out)
        true = landing_point(getattr(bot, "ball_prediction", None)) if markers.prediction and cross else None
        sub = None
        if th is not None and len(ours) > 1 and pnl.config.subtitle:
            name = getattr(packet.players[panel_car], "name", "") if panel_car < len(packet.players) else ""
            sub = subtitle_text(pnl.config.subtitle, panel_car, ours, rule, str(name))
        state = _fresh(st["draw"], panel_car, frame) if th is not None and panel_car is not None else None
        box = share.shared_box(st["box"]) if conf.share == "one" and st.get("box") is not None else None
        pnl.draw(bot.renderer, flat, th, car_index=panel_car, landing_true=true, state=state, frame=frame,
                 value_scale=bot._cds_th_scale, team=int(bot.team), marks=marks, subtitle=sub,
                 available=available(bot, len(packet.players)), crowded=crowded, box=box)
        return True

    def retire(bot: Any) -> None:
        for t in bot._cds_th_taps:
            t.detach_hooks()
        if getattr(bot, "_cds_th_share", None) is not None:
            bot._cds_th_share.close()
        if bot._cds_th_link is not None:
            bot._cds_th_link.close()

    return wrap_after_decision(bot_class, "thoughts overlay", after, before, init, retire)
