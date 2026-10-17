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

from .._rlbot import _log, find_network, landing_point, our_indices, wrap_after_decision
from ..hooks import LayerTap
from .pack import Thoughts, load_pack
from .panel import Panel, PanelConfig, check_panel, smooth_point
from .worker import WorkerLink, compute, serve


def _rows(tap: LayerTap, key: str) -> np.ndarray | None:
    calls = tap.out.get(key)
    if not calls:
        return None
    return torch.cat([c.reshape(c.shape[0] if c.ndim > 1 else 1, -1) for c in calls]).float().cpu().numpy()


def with_thoughts(bot_class: type, pack: str | os.PathLike[str], network: str | Callable[[Any], torch.nn.Module] | None = None,
                  panel: str | os.PathLike[str] | PanelConfig | None = None, hooks: dict[str, Callable] | None = None,
                  input_layer: str = "", critic: str | None = None, worker: bool = True, title: str | None = None,
                  draw_every: int | None = None, x: float | None = None, y: float | None = None) -> type:
    """Your RLBot v5 bot class (a ``rlbot.managers.Bot`` or ``Hivemind`` subclass) with the thoughts overlay.

    Args:
        pack: the thoughts pack (overlay.save_pack) for this network.
        network: where your bot keeps its torch network - a dotted attribute name, a function bot -> module, or
            None to pick the largest nn.Module on the bot.
        panel: how the panel looks - a TOML file (``cds-interpret panel`` writes an example of every option) or a
            PanelConfig; None for the default panel.
        hooks: functions your panel config names in ``hook = "..."`` (custom rows, your own markers).
        input_layer: the module whose INPUT is your observation ("" = the network itself); read only for a critic.
        critic: ``"package.module:factory"`` - factory() returns obs [n, D] -> values [n], built in the worker
            process - for the outlook bar (without one the pack's outlook probe is used, if it has one).
        worker: compute in a separate process (True, the safe default) or in the bot's process after the controls
            are sent (False: for debugging).
        title, draw_every, x, y: shortcuts that override the panel config's title, frames between draws and
            top-left corner.
    """
    pack_path = os.fspath(pack)

    def init(bot: Any) -> None:
        pk = load_pack(pack_path)
        net = find_network(bot, network)
        out_layer = pk["meta"].get("output_layer", "")
        taps = {"h": pk["layer"], "y": out_layer}
        land = pk.get("land")
        if land is not None and land["module"] != pk["layer"]:
            taps["land_h"] = land["module"]
        bot._cds_th_taps = [LayerTap(net, {k: v}, keep="all").attach() for k, v in taps.items()]
        if critic:
            bot._cds_th_taps.append(LayerTap(net, {"x": input_layer}, inputs=True, keep="all").attach())
        bot._cds_th = Thoughts(pk)
        bot._cds_th_link = WorkerLink(serve, pack_path, critic) if worker else None
        bot._cds_th_state = {"result": None, "drawn": -10 ** 9, "think": None, "draw": {}}
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
        for problem in check_panel(conf, pk):           # (things that would otherwise just not show)
            _log(bot, f"cds-interpret overlay: {problem}")
        bot._cds_th_scale = float(pk["meta"].get("value_scale", 1.0))
        if hasattr(bot, "update_rendering_status"):
            bot.update_rendering_status(True)

    def before(bot: Any, packet: Any) -> None:
        for t in bot._cds_th_taps:
            t.clear()

    def after(bot: Any, packet: Any) -> None:
        got: dict[str, np.ndarray] = {}
        for t in bot._cds_th_taps:
            for k in t.layers:
                r = _rows(t, k)
                if r is not None:
                    got[k] = r
        st = bot._cds_th_state
        frame = int(packet.match_info.frame_num)
        decided = "h" in got
        if decided:
            msg = {"frame": frame, "layer": got["h"], "output": got.get("y"), "obs": got.get("x"),
                   "land_h": got.get("land_h")}
            link = bot._cds_th_link
            if link is None:
                st["result"] = compute(bot._cds_th, msg)
            else:
                link.send(msg)
        if bot._cds_th_link is not None:
            r = bot._cds_th_link.poll()
            if r is not None and "cars" in r:
                st["result"] = r
            elif r is not None and "error" in r:
                raise RuntimeError(f"the overlay's worker failed: {r['error']}")
        res = st["result"]
        # not on the tick of a decision: the decision, its copy for the worker and the drawing would all land in one
        # 8.3 ms tick
        every = bot._cds_th_panel.config.draw_every
        due = frame - st["drawn"] >= every
        if decided and frame - st["drawn"] < 3 * every:  # (a bot deciding every tick is drawn now and then)
            due = False
        if res is None or not due or not bot.renderer.can_render:
            return
        dt = (frame - st["drawn"]) / 120.0
        st["drawn"] = frame
        th = res["cars"][0]
        sgn = -1.0 if bot.team == 1 else 1.0                  # readouts are in the team's frame
        think = None
        markers = bot._cds_th_panel.config.markers
        if th.get("land_xy") is not None and markers.landing:
            think = smooth_point(st["think"], (sgn * th["land_xy"][0], sgn * th["land_xy"][1]), dt,
                                 markers.landing_smooth)
            st["think"] = think
        from rlbot import flat
        true = landing_point(getattr(bot, "ball_prediction", None)) if markers.prediction else None
        bot._cds_th_panel.draw(bot.renderer, flat, th, car_index=our_indices(bot)[0], landing_true=true,
                               landing_think=think, state=st["draw"], frame=frame, value_scale=bot._cds_th_scale,
                               team=int(bot.team))

    def retire(bot: Any) -> None:
        for t in bot._cds_th_taps:
            t.detach_hooks()
        if bot._cds_th_link is not None:
            bot._cds_th_link.close()

    return wrap_after_decision(bot_class, "thoughts overlay", after, before, init, retire)
