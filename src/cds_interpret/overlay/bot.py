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
4. any error in the overlay switches it off and is logged; your bot plays on.

Team modes: the panel shows one of your bot's cars, chosen by the panel's ``follow`` rules (the car the host's camera
follows, by default; see :mod:`cds_interpret.overlay.team`), and every car gets its own landing marker. A Hivemind's
network rows are matched to its ``indices`` in order. A plain Bot runs once per car: every copy works out the same
panel car from the same packet, that copy draws the panel and each copy draws its own marker.

The match must allow rendering (RLBot's "enable rendering" setting), and the pack must be made from the same network
your bot runs (its module paths name its layers).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from .._rlbot import (
    _log,
    find_network,
    landing_point,
    our_cars,
    our_indices,
    packet_row,
    spectated,
    wrap_after_decision,
)
from ..errors import CDSInterpretError
from ..hooks import LayerTap
from .pack import Thoughts, load_pack
from .panel import GROUP, Panel, PanelConfig, check_panel, smooth_point
from .team import Follow, landing_cars, match_seed, subtitle_text
from .worker import WorkerLink, compute, load_critic, serve

STALE = 0.5                             # seconds without a draw after which a car's smoothing starts afresh


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
                  input_layer: str = "", critic: str | None = None, worker: bool = True, title: str | None = None,
                  draw_every: int | None = None, x: float | None = None, y: float | None = None,
                  policy: Any = None) -> type:
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
        critic: ``"package.module:factory"`` - factory() returns obs [n, D] -> values [n], built in the worker
            process - for the outlook bar (without one the policy's critic or the pack's outlook probe is used).
        worker: compute in a separate process (True, the safe default) or in the bot's process after the controls
            are sent (False: for debugging).
        title, draw_every, x, y: shortcuts that override the panel config's title, frames between draws and
            top-left corner.
        policy: ``"package.module:factory"`` - factory() returns a ``cds_interpret.Policy`` of the same network
            (with its obs function), built in the worker process - for the "why this move" and "looking at" rows
            (docs/howto/why-and-looking.md). With ``worker=False`` it may also be a Policy.
    """
    pack_path = os.fspath(pack)
    if worker and policy is not None and not isinstance(policy, str):
        raise CDSInterpretError("with_thoughts: give policy as \"package.module:factory\" - the overlay's worker builds "
                                "its own copy of your network from it (a Policy object works only with worker=False)")

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
                             "warned_rows": False}
        bot._cds_th_scale = float(pk["meta"].get("value_scale", 1.0))
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
                                   "land_h": got.get("land_h"), "panel": k}
            if k is not None:
                msg["why"] = wants["why"]
                due = wants["looking"] and st["sent"] % max(1, int(wants["every"])) == 0
                if due:
                    msg.update(looking=True, state=packet_row(packet), me=int(panel_car), donors=wants["donors"])
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
            elif r is not None and "error" in r:
                raise RuntimeError(f"the overlay's worker failed: {r['error']}")
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
            return
        dt = (frame - st["drawn"]) / 120.0
        st["drawn"] = frame
        markers = pnl.config.markers
        sgn = -1.0 if bot.team == 1 else 1.0            # readouts are in the team's frame
        marks = []
        for c in landing_cars(markers.landing_cars, ours, panel_car, spec):
            res_c = st["cars"].get(c)
            if res_c is None or res_c.get("land_xy") is None:
                continue
            new = (sgn * res_c["land_xy"][0], sgn * res_c["land_xy"][1])
            prev = st["think"].get(c)
            fresh = prev is None or frame - prev[1] > STALE * 120
            pt = smooth_point(None if fresh else prev[0], new, dt, markers.landing_smooth)
            st["think"][c] = (pt, frame)
            marks.append((c, pt))
        th = st["cars"].get(panel_car) if panel_car in mine else None
        first_of_team = bool(mine) and bool(ours) and ours[0] in mine
        if th is None and not marks:
            if not st["cleared"]:                       # (this copy no longer shows anything: take its drawing down)
                bot.renderer.clear_render_group(GROUP)
                st["cleared"] = True
            return
        st["cleared"] = False
        from rlbot import flat
        cross = th is not None or (panel_car is None and first_of_team)
        true = landing_point(getattr(bot, "ball_prediction", None)) if markers.prediction and cross else None
        sub = None
        if th is not None and len(ours) > 1 and pnl.config.subtitle:
            name = getattr(packet.players[panel_car], "name", "") if panel_car < len(packet.players) else ""
            sub = subtitle_text(pnl.config.subtitle, panel_car, ours, rule, str(name))
        state = _fresh(st["draw"], panel_car, frame) if th is not None and panel_car is not None else None
        pnl.draw(bot.renderer, flat, th, car_index=panel_car, landing_true=true, state=state, frame=frame,
                 value_scale=bot._cds_th_scale, team=int(bot.team), marks=marks, subtitle=sub,
                 available=available(bot, len(packet.players)))

    def retire(bot: Any) -> None:
        for t in bot._cds_th_taps:
            t.detach_hooks()
        if bot._cds_th_link is not None:
            bot._cds_th_link.close()

    return wrap_after_decision(bot_class, "thoughts overlay", after, before, init, retire)
