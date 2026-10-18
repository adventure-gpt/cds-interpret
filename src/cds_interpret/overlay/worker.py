"""The overlay's worker: a separate, low-priority process that turns what the bot's network computed anyway into
what the panel shows, so none of it runs in the bot's own process or thread pool.

``WorkerLink`` never makes the bot wait:

- at most one request is in flight, and a busy worker simply skips a decision;
- nothing is sent before the worker has said it is ready (a request bigger than the pipe's buffer would otherwise wait
  until a still-starting worker read it);
- requests are written by a thread of the link's own, on a one-way pipe (answers come back on another). A worker at
  low priority on a busy CPU can be slow to get back to reading, and a large write waits for it - only that thread
  ever waits, never the bot.
"""

from __future__ import annotations

import contextlib
import importlib
import multiprocessing as mp
import os
import queue
import sys
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from .pack import Thoughts, load_pack

if TYPE_CHECKING:
    from .explain import Explainer


class Ends:
    """The worker's side of a WorkerLink: requests in on one pipe, answers out on another."""

    def __init__(self, requests: Any, answers: Any):
        self.requests, self.answers = requests, answers

    def recv(self) -> Any:
        return self.requests.recv()

    def send(self, msg: Any) -> None:
        self.answers.send(msg)

    def poll(self, timeout: float = 0.0) -> bool:
        return bool(self.requests.poll(timeout))


class WorkerLink:
    """A child process answering requests (dicts) with results (dicts), never blocking the caller.

    ``target(ends, *args)`` runs in the child: it sends ``{"ready": True}`` once it has loaded, then answers each
    request ``ends.recv()`` with ``ends.send(result)`` until it receives None."""

    def __init__(self, target: Callable[..., None], *args: Any):
        ctx = mp.get_context("spawn")
        req_r, self._req_w = ctx.Pipe(duplex=False)
        self._rep_r, rep_w = ctx.Pipe(duplex=False)
        self.proc = ctx.Process(target=target, args=(Ends(req_r, rep_w), *args), daemon=True)
        self.proc.start()
        self.busy = True                # until the child says it is ready
        self.ready = False
        self.hello: dict[str, Any] = {}  # the child's "ready" message (what it can do, notes for the log)
        self._q: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._writer = threading.Thread(target=self._write, name="cds-interpret-overlay-writer", daemon=True)
        self._writer.start()

    def _write(self) -> None:
        while True:
            msg = self._q.get()
            try:
                self._req_w.send(msg)
            except Exception:                         # the child is gone
                return
            if msg is None:
                return

    def send(self, msg: dict[str, Any]) -> bool:
        """Hand a request to the writer thread if the worker is free; False (and nothing sent) when it is busy,
        starting or gone. Never waits."""
        if self.busy or not self.proc.is_alive():
            return False
        try:
            self._q.put_nowait(msg)
        except queue.Full:
            return False
        self.busy = True
        return True

    def poll(self) -> dict[str, Any] | None:
        """The answer to the last request, once it is there (None until then)."""
        if self.busy and self._rep_r.poll():
            self.busy = False
            msg = self._rep_r.recv()
            if not self.ready:
                self.ready = True
                if isinstance(msg, dict) and msg.get("ready"):
                    self.hello = msg
                    return None
            return msg
        return None

    @property
    def alive(self) -> bool:
        return self.proc.is_alive()

    def close(self) -> None:
        with contextlib.suppress(queue.Full):          # (a link that is already closing)
            self._q.put_nowait(None)


_WIN_RANK = {0x40: 0, 0x4000: 1, 0x20: 2, 0x8000: 3, 0x80: 4, 0x100: 5}   # idle < below normal < normal < ...


def below_normal() -> None:
    """This process at below-normal priority, or lower if it already is lower (it never raises its priority). A
    courtesy to the bot and the game; ignored where it fails."""
    try:
        if sys.platform == "win32":
            import ctypes
            k32 = ctypes.windll.kernel32
            me = ctypes.c_void_p(-1)
            if _WIN_RANK.get(int(k32.GetPriorityClass(me)), 2) > 1:
                k32.SetPriorityClass(me, 0x4000)     # BELOW_NORMAL_PRIORITY_CLASS
        else:
            os.nice(10)
    except Exception:                                 # (a platform without it: run as we are)
        pass


def load_critic(spec: str | None) -> Callable[[np.ndarray], Any] | None:
    """A critic from ``"package.module:factory"``: ``factory()`` returns a function obs [n, D] -> values [n]. It is
    imported and built in the worker process, never in the bot's."""
    if not spec:
        return None
    mod, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"a critic is given as 'module:factory', not {spec!r}")
    return getattr(importlib.import_module(mod), attr)()


def _action_of(th: Thoughts, out_row: Any) -> int | None:
    """The move the policy leans to: the argmax of its output row (the "Next move" row's move)."""
    if out_row is None:
        return None
    o = np.asarray(out_row, np.float64).reshape(-1)
    return int(o.argmax()) if len(o) else None


def compute(th: Thoughts, msg: dict[str, Any], critic: Callable[[np.ndarray], Any] | None = None,
            explainer: Explainer | None = None) -> dict[str, Any]:
    """One request -> one result per car. msg: "layer" [n, d] (the pack's layer), "output" [n, A] (the action
    output), optional "obs" [n, ...] (the network's input: for a critic and "why this move"), "land_h" [n, d'] (the
    landing layer when it is another one), "frame". For the panel's car (row "panel"): "why" (True: its "why this
    move"), "looking" (True: its "looking at", from "state" - the game as Frames fields of one packet - seen by car
    slot "me", averaged over "donors" donor moments). Returns {"frame", "cars": [...], "notes": [...]}."""
    layer = np.asarray(msg["layer"], np.float32)
    n = len(layer)
    out_rows = msg.get("output")
    obs = msg.get("obs")
    value: list[float | None] = [None] * n
    if critic is not None and obs is not None:
        v = critic(np.asarray(obs, np.float32))
        v = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        value = [float(x) for x in np.asarray(v, np.float64).reshape(-1)[:n]]
    land_h = msg.get("land_h")
    if land_h is None and th.land is not None and th.land_module == th.layer:
        land_h = layer
    panel = msg.get("panel")
    notes_before = len(explainer.notes) if explainer is not None else 0
    cars = []
    for k in range(n):
        out_k = None if out_rows is None else out_rows[k]
        r = th.read(layer[k], out_k)
        if value[k] is not None:
            r["value"] = value[k]
        if land_h is not None and th.land is not None:
            r["land_xy"] = th.land_xy(land_h[k])
        if explainer is not None and panel is not None and int(panel) == k:
            a = _action_of(th, out_k)
            if msg.get("why") and a is not None:
                acts = np.array([f["act"] for f in r["features"]], np.float32)
                r["why"] = explainer.why(None if obs is None else np.asarray(obs)[k], layer[k], a, acts)
            if msg.get("looking") and a is not None:
                r["looking"] = explainer.looking(msg.get("state"), int(msg.get("me", 0)), a,
                                                 int(msg.get("donors", 3)))
        cars.append(r)
    notes = explainer.notes[notes_before:] if explainer is not None else []
    return {"frame": msg.get("frame"), "cars": cars, "notes": list(notes)}


def serve(conn: Any, pack_path: str, critic_spec: str | None = None, policy_spec: str | None = None) -> None:
    """The worker process: below-normal priority, one torch thread if torch is used, answering each request with
    compute(); None ends it. ``policy_spec`` ("module:factory"): the bot's Policy, built here, for "why this move"
    and "looking at" (and the outlook, from its critic, when there is no ``critic_spec``). The ready message says
    what it can do ({"ready", "available", "notes"}); errors are sent back as {"error": ...} - the overlay must
    never stop the bot."""
    below_normal()
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:                                 # (torch is optional in the worker)
        pass
    th = Thoughts(load_pack(pack_path))
    notes: list[str] = []
    try:
        critic = load_critic(critic_spec)
    except Exception as e:
        critic = None
        notes.append(f"critic {critic_spec!r} not loaded ({e}); the outlook bar uses the pack")
    explainer = None
    if policy_spec:
        from .explain import Explainer, load_policy
        try:
            pol = load_policy(policy_spec)
            explainer = Explainer(th, pol)
            notes.extend(explainer.notes)
            explainer.notes.clear()
            if critic is None and pol is not None and pol.value_fn is not None:
                critic = pol.value
        except Exception as e:
            notes.append(f"policy {policy_spec!r} not loaded ({e}); 'why this move' and 'looking at' stay off")
    available = explainer.available if explainer is not None else {"why": False, "looking": []}
    conn.send({"ready": True, "available": available, "notes": notes})
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        if msg is None:
            break
        try:
            conn.send(compute(th, msg, critic, explainer))
        except Exception as e:                        # the overlay must never stop the bot: report it instead
            conn.send({"error": repr(e)})


def echo(conn: Any, delay: float = 0.0, read_delay: float = 0.0) -> None:
    """A worker for tests: a slow start of ``delay`` seconds, then ready, then each request answered as it is (after
    ``read_delay`` seconds of not reading, like a busy low-priority process)."""
    import time
    time.sleep(delay)
    conn.send({"ready": True})
    while True:
        time.sleep(read_delay)
        msg = conn.recv()
        if msg is None:
            break
        conn.send(msg)
