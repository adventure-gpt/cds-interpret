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
log p(a*) between the parts. A bot whose input can't be rebuilt from the game state alone can split its INPUT itself:
give its Policy a ``codec`` (decompose / compose / groups over the network's input) and the pack recorded inputs
(``add_input_donors``); the same Shapley split then runs on the input's own parts. Shares are of the absolute values (a part pushing against the move counts as much as
one pushing for it), averaged over a few donors, and sum to 1. The heaviest of the panel's readings (2^G x donors
inputs), so it runs for the panel's car only and every few readings.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from .._specs import build, is_spec
from ..attribution import ROLE_MATE, ROLE_OPP, FramesCodec, hybrid, role_signature, shapley
from ..errors import CDSInterpretError
from ..frames import Frames
from ..hooks import get_module
from ..policy import Policy
from .pack import Thoughts

ENTITIES = ("own car", "ball", "opponents", "team-mates")


def load_policy(spec: Any) -> Policy | None:
    """A Policy from ``"package.module:factory"`` or ``("package.module:factory", {keyword arguments})``
    (``factory(**kwargs)`` returns a cds_interpret Policy, or a torch module that becomes one), or a Policy / module as
    it is. None for None."""
    if spec is None or spec == "":
        return None
    if isinstance(spec, Policy):
        return spec
    if isinstance(spec, torch.nn.Module):
        return Policy(spec)
    if not is_spec(spec):
        raise CDSInterpretError(f"policy should be \"package.module:factory\" (or with its keyword arguments: "
                                f"(\"package.module:factory\", {{...}})), not {type(spec).__name__}")
    made = build(spec, "policy")
    if isinstance(made, torch.nn.Module):
        made = Policy(made)
    if not isinstance(made, Policy):
        raise CDSInterpretError(f"policy {spec!r} returned a {type(made).__name__}, not a cds_interpret.Policy (or a "
                                "torch module)")
    return made


def _signature(parts: dict) -> np.ndarray:
    """[N] a code of each row's numbers of team-mates and opponents (role_signature on the first ``<key>:role`` array
    of the parts; all zero when there is none), so donors line up with the live moment."""
    key = next((k for k in parts if k.endswith(":role")), None)
    if key is None:
        first = next(iter(parts.values()))
        return np.zeros(len(first), np.int64)
    r = np.asarray(parts[key].cpu() if torch.is_tensor(parts[key]) else parts[key])
    return (r == ROLE_MATE).sum(1) * 8 + (r == ROLE_OPP).sum(1)


def _cat(rows: list[dict]) -> dict:
    """The parts of several rows as one batch, each key in its own kind (torch or numpy)."""
    out = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        out[k] = torch.cat([torch.as_tensor(v) for v in vals]) if torch.is_tensor(vals[0]) else \
            np.concatenate([np.asarray(v) for v in vals])
    return out


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
        self._idonors: dict[int, tuple[dict, np.ndarray]] = {}           # cars -> (input parts, role codes)
        self.codec: FramesCodec | None = None
        self.input_codec: Any = None                  # the policy's own codec of its input (Policy(codec=...))
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
        codec = getattr(policy, "codec", None)
        if codec is not None and th.input_donors:
            self.input_codec = codec
            for cars, x in th.input_donors:
                parts = codec.decompose(x)
                self._idonors[int(cars)] = (parts, _signature(parts))
            self.looking_ok = True
        elif codec is not None and not th.donors:
            self.notes.append("'looking at' is off: the policy splits its input itself (codec), but the pack has no "
                              "recorded inputs to compare with (overlay.add_input_donors(pack, {cars: inputs}))")
        elif th.input_donors and codec is None and not th.donors:
            self.notes.append("'looking at' is off: the pack has recorded inputs, but the policy has no codec to split "
                              "them (ci.Policy(..., codec=...))")
        elif not th.donors:
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
        groups = self._idonors if self.input_codec is not None else self._donors
        return {"why": self.why_ok, "looking": sorted(groups) if self.looking_ok else []}

    def mirrored(self, state: dict[str, Any] | None, cars: Any) -> np.ndarray | None:
        """[n] bool: which of these cars (player indices) see the live moment ``state`` mirrored, or None when the
        policy has no mirror function (or there is no state)."""
        if self.policy is None or self.policy.mirror_fn is None or state is None or cars is None:
            return None
        cars = np.asarray(cars, np.int64).reshape(-1)
        fr = Frames({k: np.repeat(np.asarray(v)[None], len(cars), 0) for k, v in state.items()})
        return self.policy.mirrored(fr, cars)

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
    def looking(self, state: dict[str, Any] | None, me: int, action: int, donors: int = 3, x: Any = None,
                n_cars: int | None = None) -> dict[str, float] | None:
        """{part: share} of what ``action`` rests on for car slot ``me`` of the live game ``state`` (one packet's
        Frames fields, without the moment axis, as overlay sends them), or None when it can't (no donors with the
        same numbers of cars, team-mates and opponents). With the policy's own input codec: ``x``, this car's network
        input, is split instead (and ``n_cars``, or the state's, picks the donors)."""
        if not self.looking_ok or self.policy is None:
            return None
        if self.input_codec is not None:
            if x is None:
                return None
            try:
                cars = n_cars if n_cars is not None else (np.asarray(state["car_pos"]).shape[0] if state else None)
                igot = self._idonors.get(int(cars)) if cars is not None else (
                    next(iter(self._idonors.values())) if len(self._idonors) == 1 else None)
                if igot is None:
                    return None
                return looking_shares_inputs(self.policy, self.input_codec, x, int(action), igot[0], igot[1],
                                             int(donors), self.rng)
            except Exception as e:                    # (as below)
                self._off("looking", e)
                return None
        if self.codec is None or state is None:
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


def looking_shares_inputs(policy: Policy, codec: Any, x: Any, action: int, dparts: dict, dsig: np.ndarray,
                          donors: int = 3, rng: np.random.Generator | None = None) -> dict[str, float] | None:
    """looking_shares on the network's INPUT, split by the policy's own codec: x (one car's input) is decomposed,
    each of the codec's groups for it (``codec.groups(parts)``: name -> Group) is kept or swapped for the same group of
    a donor input (rows of ``dparts`` with a matching role signature ``dsig``), every coalition of every donor composed
    back into inputs and run in one batch; exact Shapley shares of log p(action), averaged over the donors."""
    rng = rng or np.random.default_rng(0)
    p = codec.decompose(_batch1(x))
    match = np.nonzero(np.asarray(dsig) == int(_signature(p)[0]))[0]
    if not len(match):
        return None
    named = dict(codec.groups(p))
    names, grp = list(named), list(named.values())
    G = len(grp)
    pick = rng.choice(match, size=min(int(donors), len(match)), replace=False)
    rows = []
    for d in pick:
        q = {k: v[d:d + 1] for k, v in dparts.items()}
        for s_ in range(2 ** G):
            rows.append(hybrid(p, q, grp, s_))
    with torch.no_grad():
        lp = policy.log_probs(codec.compose(_cat(rows)))[:, int(action)].float().cpu().numpy()
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
    inputs = ex.input_codec is not None
    if pol is None or not (th.input_donors if inputs else th.donors):
        if pol is not None and (ex.why_ok or ex.looking_ok):
            out["problems"].append("(the timings need the pack's recorded moments to try the readings on)")
        return out
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        if inputs:
            cars, xs = th.input_donors[-1]
            x = torch.as_tensor(np.asarray(xs[:1], np.float32))
            state, me0 = None, 0
        else:
            fr, me = th.donors[-1]
            cars = fr.n_cars
            x = pol.observe(fr.take([0]), me[:1])
            state, me0 = {k: v[0] for k, v in fr.take([0]).data.items()}, int(me[0])
        tap_out: dict[str, torch.Tensor] = {}
        hd = ex._module.register_forward_hook(lambda m, a, o: tap_out.__setitem__("h", o[0] if isinstance(o, tuple)
                                                                                  else o)) if ex._module else None
        try:
            with torch.no_grad():
                a = int(pol.log_probs(x)[0].argmax())
        finally:
            if hd is not None:
                hd.remove()
        if ex.why_ok and "h" in tap_out:
            h = tap_out["h"].reshape(1, -1)[0].numpy()
            t = time.perf_counter()
            for _ in range(repeats):
                r = ex.why(x[0], h, a)
            out["why_ms"] = 1000 * (time.perf_counter() - t) / repeats if r is not None else None
        if ex.looking_ok:
            t = time.perf_counter()
            for _ in range(repeats):
                look = ex.looking(state, me0, a, donors, x=x[0], n_cars=int(cars))
            out["looking_ms"] = 1000 * (time.perf_counter() - t) / repeats if look is not None else None
        out["problems"] = list(ex.notes)
    finally:
        torch.set_num_threads(threads)
    return out
