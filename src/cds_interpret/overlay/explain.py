"""The panel's two readings about a decision: "why this move" (which named features push the move the policy leans
to) and "looking at" (how much the move rests on the bot's own car, the ball, the opponents and the team-mates).

Both need your bot's network, so they run in the overlay's worker on a copy of it built from a factory you name:

    with_thoughts(MyBot, pack="thoughts.npz", policy="my_bot.interp:make_policy")

    # my_bot/interp.py - builds the same network as your bot, with its obs function and action table
    def make_policy():
        net = MyNet(); net.load_state_dict(torch.load("model.pt", weights_only=True))
        return ci.Policy(net, obs=my_obs, actions=my_table)

**Why this move.** Each named feature's push on the chosen move a* is

    push_j = z_j * (d_j . grad_h log p(a* | h))

z_j the feature's activation at the pack's layer h, d_j its decoder direction (in the layer's units), and the
gradient taken by running the network on the decision's own input with the layer's output replaced by a
gradient-carrying copy of the values the bot computed (attribution patching: a first-order estimate of what removing
the feature would do to log p(a*); on a 31M-weight self-play bot it ranked a decision's features like removing them
one by one, Spearman 0.96). In log-probability units: 0.7 is about a doubling of the move's chance.

**Looking at.** The live moment's game state (from the packet) is split into parts with
:class:`cds_interpret.attribution.FramesCodec`; each part is kept or swapped for the same part of a real recorded
moment (the pack's donors, with the same numbers of team-mates and opponents), the bot's input is rebuilt with its own
obs function for every combination - 2^G of them, all in ONE batched forward pass - and exact Shapley values split
log p(a*) between the parts. Shares are of the absolute values (a part pushing against the move counts as much as
one pushing for it), averaged over a few donors, and sum to 1. The heaviest of the panel's readings (2^G x donors
inputs), so it runs for the panel's car only and every few readings.
"""

from __future__ import annotations

import importlib
import time
from typing import Any

import numpy as np
import torch

from ..attribution import ROLE_MATE, ROLE_OPP, FramesCodec, hybrid, role_signature, shapley
from ..errors import CDSInterpretError
from ..frames import Frames
from ..hooks import get_module
from ..policy import Policy
from .pack import Thoughts

ENTITIES = ("own car", "ball", "opponents", "team-mates")


def load_policy(spec: Any) -> Policy | None:
    """A Policy from ``"package.module:factory"`` (``factory()`` returns a cds_interpret Policy, or a torch module that
    becomes one), or a Policy / module as it is. None for None."""
    if spec is None or spec == "":
        return None
    if isinstance(spec, Policy):
        return spec
    if isinstance(spec, torch.nn.Module):
        return Policy(spec)
    if not isinstance(spec, str):
        raise CDSInterpretError(f"policy should be \"package.module:factory\", not {type(spec).__name__}")
    mod, _, attr = spec.partition(":")
    if not attr:
        raise CDSInterpretError(f"policy {spec!r}: give it as \"package.module:factory\", where factory() returns a "
                                "cds_interpret.Policy of the same network your bot runs")
    try:
        made = getattr(importlib.import_module(mod), attr)()
    except (ImportError, AttributeError) as e:
        raise CDSInterpretError(f"policy {spec!r} could not be loaded: {e}. The module must be importable from where "
                                "RLBot starts your bot (usually its folder)") from None
    if isinstance(made, torch.nn.Module):
        made = Policy(made)
    if not isinstance(made, Policy):
        raise CDSInterpretError(f"policy {spec!r}: {attr}() returned a {type(made).__name__}, not a cds_interpret.Policy "
                                "(or a torch module)")
    return made


def _batch1(x: Any) -> torch.Tensor:
    """One car's network input (without the batch axis) as a batch of one."""
    return torch.as_tensor(np.asarray(x.detach().cpu() if torch.is_tensor(x) else x, np.float32))[None]


class Explainer:
    """"Why this move" and "looking at" for one decision of one car, from a pack (Thoughts) and the bot's Policy.

    ``available`` says what it can do: {"why": bool, "looking": [numbers of cars it has donors for]}; ``notes`` why
    not, in words. A reading that fails turns itself off (with a note) instead of failing again every decision.
    freeze: stop gradients to the network's own weights (only for a network of its own, as in the worker)."""

    def __init__(self, th: Thoughts, policy: Policy | None, seed: int = 0, freeze: bool = True):
        self.th, self.policy = th, policy
        self.rng = np.random.default_rng(seed)
        self.notes: list[str] = []
        self.why_ok = self.looking_ok = False
        self._module: torch.nn.Module | None = None
        self._donors: dict[int, tuple[dict, np.ndarray, bool]] = {}     # cars -> (parts, role codes, has controls)
        self.codec: FramesCodec | None = None
        if policy is None:
            return
        if not policy.discrete:
            self.notes.append("'why this move' and 'looking at' need a policy with discrete actions (this one outputs "
                              "controls), so they stay off")
            return
        if freeze:
            for p in policy.net.parameters():
                p.requires_grad_(False)
        if th.fd is None:
            if th.pack.get("features"):
                self.notes.append("'why this move' is off: the pack has no decoder directions for its features (add "
                                  "them with overlay.add_decoders(pack, sae); packs from make_pack have them)")
        else:
            try:
                self._module = get_module(policy.net, th.layer)
                self.why_ok = True
            except Exception as e:
                self.notes.append(f"'why this move' is off: the policy's network has no layer {th.layer!r} - it must "
                                  f"be the same network your bot runs ({e})")
        if not th.donors:
            self.notes.append("'looking at' is off: the pack has no recorded moments to compare with (add them with "
                              "overlay.add_donors(pack, moments); Study.make_pack adds them)")
        elif policy.obs_fn is None:
            self.notes.append("'looking at' is off: the policy has no obs function, so it can't rebuild your bot's "
                              "input from the game (ci.Policy(net, obs=my_obs, ...))")
        else:
            self.codec = FramesCodec(policy)
            for fr, me in th.donors:
                parts = self.codec.decompose(fr, me)
                self._donors[fr.n_cars] = (parts, role_signature(parts), fr.has("prev_controls"))
            self.looking_ok = True

    @property
    def available(self) -> dict[str, Any]:
        return {"why": self.why_ok, "looking": sorted(self._donors) if self.looking_ok else []}

    def _off(self, what: str, e: Exception) -> None:
        if what == "why":
            self.why_ok = False
        else:
            self.looking_ok = False
        self.notes.append(f"'{'why this move' if what == 'why' else 'looking at'}' stopped: {e!r}")

    # ---------------------------------------------------------------------------------------- why this move
    def why(self, x: Any, h: Any, action: int, acts: Any = None) -> list[dict[str, Any]] | None:
        """Each named feature's push on ``action`` [{"name", "push"}] in the pack's order, or None when it can't.
        x: this decision's network input (one car), h: the pack layer's values the bot computed, acts: the features'
        activations at h (default: read from the pack)."""
        if not self.why_ok or x is None or self.policy is None or self._module is None or self.th.fd is None:
            return None
        try:
            push = why_pushes(self.policy, self._module, x, h, int(action),
                              self._acts(h) if acts is None else np.asarray(acts, np.float32), self.th.fd)
        except Exception as e:                        # the overlay must never stop the bot: switch this reading off
            self._off("why", e)
            return None
        return [{"name": n, "push": float(p)} for n, p in zip(self.th.names, push)]

    def _acts(self, h: Any) -> np.ndarray:
        return np.array([f["act"] for f in self.th.read(h)["features"]], np.float32)

    # ------------------------------------------------------------------------------------------- looking at
    def looking(self, state: dict[str, Any] | None, me: int, action: int, donors: int = 3) -> dict[str, float] | None:
        """{part: share} of what ``action`` rests on for car slot ``me`` of the live game ``state`` (one packet's
        Frames fields, without the moment axis, as overlay sends them), or None when it can't (no donors with the
        same numbers of cars, team-mates and opponents)."""
        if not self.looking_ok or self.policy is None or self.codec is None or state is None:
            return None
        try:
            fields = {k: np.asarray(v)[None] for k, v in state.items()}
            got = self._donors.get(int(fields["car_pos"].shape[1]))
            if got is None:
                return None
            dparts, dsig, donors_prev = got
            if donors_prev and "prev_controls" not in fields:      # (the parts must line up with the donors')
                fields["prev_controls"] = np.zeros(fields["car_pos"].shape[:2] + (8,), np.float32)
            elif not donors_prev:
                fields.pop("prev_controls", None)
            return looking_shares(self.policy, self.codec, Frames(fields), int(me), int(action), dparts, dsig,
                                  int(donors), self.rng)
        except Exception as e:                        # (as above)
            self._off("looking", e)
            return None


def why_pushes(policy: Policy, module: torch.nn.Module, x: Any, h: Any, action: int, acts: np.ndarray,
               directions: np.ndarray) -> np.ndarray:
    """[F] each feature's push on log p(action): acts [F] x (directions [F, D] . the gradient of log p(action) at the
    module's output), the network run on x (one car's input) with that output replaced by a gradient-carrying copy of
    h (the values the bot computed)."""
    leaf = torch.as_tensor(np.asarray(h, np.float32)).reshape(1, -1).clone().requires_grad_(True)

    def swap_in(mod: torch.nn.Module, args: tuple, out: Any) -> Any:
        first = out[0] if isinstance(out, tuple) else out
        new = leaf.reshape(first.shape).to(first.dtype)
        return (new, *out[1:]) if isinstance(out, tuple) else new
    handle = module.register_forward_hook(swap_in)
    try:
        with torch.enable_grad():
            lp = policy.log_probs(_batch1(x))
            (g,) = torch.autograd.grad(lp[0, int(action)], leaf)
    finally:
        handle.remove()
    return np.asarray(acts, np.float32) * (np.asarray(directions, np.float32) @ g.reshape(-1).numpy())


def looking_shares(policy: Policy, codec: FramesCodec, live: Frames, me: int, action: int, dparts: dict,
                   dsig: np.ndarray, donors: int = 3, rng: np.random.Generator | None = None) -> dict[str, float] | None:
    """{part: share} for one live moment (Frames of one row, the bot's car in slot ``me``): its own car, the ball, the
    opponents and - when it has any - the team-mates, each kept or swapped for the same part of a donor (rows of
    ``dparts``, decomposed by ``codec``, whose role signature ``dsig`` matches), every coalition of every donor in one
    batch, exact Shapley values of log p(action) averaged over the donors, as shares of their absolute values."""
    rng = rng or np.random.default_rng(0)
    p = codec.decompose(live, np.array([me]))
    sig = int(role_signature(p)[0])
    match = np.nonzero(np.asarray(dsig) == sig)[0]
    if not len(match):
        return None
    all_groups = codec.groups(boost_apart=False)
    roles = np.asarray(p["cars:role"][0])
    names = ["own car", "ball"] + (["opponents"] if (roles == ROLE_OPP).any() else []) + \
        (["team-mates"] if (roles == ROLE_MATE).any() else [])
    grp = [all_groups[n] for n in names]
    G = len(grp)
    pick = rng.choice(match, size=min(int(donors), len(match)), replace=False)
    rows = []
    for d in pick:
        q = {k: v[d:d + 1] for k, v in dparts.items()}
        for s in range(2 ** G):
            rows.append(hybrid(p, q, grp, s))
    batch = {k: np.concatenate([np.asarray(r[k]) for r in rows]) for k in rows[0]}
    with torch.no_grad():
        lp = policy.log_probs(codec.compose(batch))[:, int(action)].float().cpu().numpy()
    phi = np.abs(shapley(lp.reshape(len(pick), 1, 2 ** G, 1).astype(np.float64))[0, :, 0])
    tot = float(phi.sum())
    if not np.isfinite(tot) or tot <= 1e-9:
        return None
    return {n: float(v / tot) for n, v in zip(names, phi)}


def try_readings(pack: Any, policy: Any, donors: int = 3, repeats: int = 5) -> dict[str, Any]:
    """Run "why this move" and "looking at" on the pack's own donor moments (of the largest mode it has: the
    costliest), as the overlay would in a match:
    whether they work with this policy, and what they cost on one thread. {"why_ms", "looking_ms" (None when off),
    "problems": [...]}. pack: a pack, its file or a Thoughts; policy: a Policy or "module:factory"."""
    from .pack import load_pack
    th = pack if isinstance(pack, Thoughts) else Thoughts(pack if isinstance(pack, dict) else load_pack(pack))
    pol = load_policy(policy)
    ex = Explainer(th, pol, freeze=False)
    out: dict[str, Any] = {"why_ms": None, "looking_ms": None, "problems": list(ex.notes)}
    if pol is None or not th.donors:
        if pol is not None and (ex.why_ok or ex.looking_ok) and not th.donors:
            out["problems"].append("(the timings need the pack's donor moments to try the readings on)")
        return out
    fr, me = th.donors[-1]
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        x = pol.observe(fr.take([0]), me[:1])
        acts = None
        tap_out: dict[str, torch.Tensor] = {}
        if ex.why_ok and ex._module is not None:
            hd = ex._module.register_forward_hook(lambda m, a, o: tap_out.__setitem__("h", o[0] if isinstance(o, tuple)
                                                                                      else o))
            try:
                with torch.no_grad():
                    lp = pol.log_probs(x)
            finally:
                hd.remove()
            a = int(lp[0].argmax())
            h = tap_out["h"].reshape(1, -1)[0].numpy()
            t = time.perf_counter()
            for _ in range(repeats):
                r = ex.why(x[0], h, a, acts)
            out["why_ms"] = 1000 * (time.perf_counter() - t) / repeats if r is not None else None
        else:
            with torch.no_grad():
                a = int(pol.log_probs(x)[0].argmax())
        if ex.looking_ok:
            state = {k: v[0] for k, v in fr.take([0]).data.items()}
            t = time.perf_counter()
            for _ in range(repeats):
                look = ex.looking(state, int(me[0]), a, donors)
            out["looking_ms"] = 1000 * (time.perf_counter() - t) / repeats if look is not None else None
        out["problems"] = list(ex.notes)
    finally:
        torch.set_num_threads(threads)
    return out
