"""Bring your own policy: any PyTorch network, and how to build its input from the game state.

    import cds_interpret as ci

    policy = ci.Policy(
        net,                                  # your torch.nn.Module (eager; TorchScript: eager_copy(it) first)
        obs=my_obs,                           # my_obs(frames, me) -> [N, D]: its input for car slot me[i] of moment i
        actions=ci.lookup_table(),            # [A, 8] the controls of each discrete action (optional: names actions)
        output="logits",                      # what net(x) returns: "logits", "probs", "log_probs" or "controls"
    )
    policy.layers(example_x)                  # every module, its type and output shape: what can be probed
    policy.act(x)                             # the action each moment's argmax picks

That is the whole contract. ``obs`` is needed only where the toolkit builds inputs itself (recording in RocketSim,
synthetic or packet-built moments); recordings made from your own bot (RLBot, your RLGym env) store the exact
inputs your network saw. An RLGym v2 ObsBuilder becomes an ``obs`` function with
:func:`cds_interpret.rlgym.obs_fn`.

Controls are the 8 RLBot / RLGym controls in this order: throttle, steer, pitch, yaw, roll, jump, boost, handbrake.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from itertools import product
from typing import Any

import numpy as np
import torch

from .errors import CDSInterpretError, ShapeError
from .frames import Frames
from .hooks import LayerInfo, default_layers, list_layers, run_layers

ObsFn = Callable[[Frames, np.ndarray], Any]
OUTPUTS = ("logits", "probs", "log_probs", "controls")


# --------------------------------------------------------------------------------------------- the action table
def lookup_table() -> np.ndarray:
    """A discrete action table [90, 8] in the layout most Rocket League RL bots use: 24 ground rows, then 66 air
    rows. Ground: throttle and steer in {-1, 0, 1}, handbrake off / on, boost only at full throttle. Air: pitch, yaw
    and roll in {-1, 0, 1} with jump and boost, leaving out a jump with yaw (roll makes the side flip) and rows with
    no pitch, roll or jump (those are ground rows). In an air row steer equals yaw and throttle equals boost.

    Bots trained with another table must pass that table instead (e.g. RLGym's parser's own)."""
    tri = (-1.0, 0.0, 1.0)
    ground = [(t, s, b, h) for t, s, b, h in product(tri, tri, (0.0, 1.0), (0.0, 1.0)) if not (b and t < 1)]
    air = [(p, y, r, j, b) for p, y, r, j, b in product(tri, tri, tri, (0.0, 1.0), (0.0, 1.0))
           if not (j and y) and (p or r or j)]
    rows = [[t if t else b, s, 0.0, s, 0.0, 0.0, b, h] for t, s, b, h in ground]
    rows += [[b, y, p, y, r, j, b, 0.0] for p, y, r, j, b in air]
    return np.asarray(rows, dtype=np.float32)


MIRROR_COLS = (1, 3, 4)                 # steer, yaw, roll: what changes sign when the field is seen mirrored


def unmirror_controls(row: Any) -> np.ndarray:
    """Controls [..., 8] as a bot that saw the field mirrored left to right really sends them: steer, yaw and roll
    with their signs flipped."""
    out = np.array(row, np.float32, copy=True)
    out[..., list(MIRROR_COLS)] *= -1.0
    return out


def describe_action(row: Any) -> str:
    """Controls [8] in words: ``"throttle, steer left, boost"``."""
    t, s, p, y, r, j, b, h = (float(v) for v in np.asarray(row, dtype=np.float64).reshape(8))
    words = []
    if j > 0.5:
        if p == 0 and r == 0 and y == 0:
            words.append("jump")
        else:
            dirs = [w for w, on in (("forward", p < 0), ("back", p > 0), ("left", r < 0 or y < 0),
                                    ("right", r > 0 or y > 0)) if on]
            words.append("flip " + " ".join(dirs))
    else:
        if t > 0:
            words.append("throttle")
        elif t < 0:
            words.append("reverse")
        if s < 0 or y < 0:
            words.append("steer left")
        elif s > 0 or y > 0:
            words.append("steer right")
        if p < 0:
            words.append("nose down")
        elif p > 0:
            words.append("nose up")
        if r != 0:
            words.append("roll " + ("left" if r < 0 else "right"))
    if b > 0.5:
        words.append("boost")
    if h > 0.5 and j < 0.5:
        words.append("handbrake")
    return ", ".join(words) or "coast"


# ------------------------------------------------------------------------------------------------------ the policy
class Policy:
    """Your bot as the toolkit sees it: a network, how to build its input, and what its output means.

    Args:
        net: any eager ``torch.nn.Module`` (put it in eval mode; it is never trained here).
        obs: ``obs(frames, me) -> [N, D]``, the network's input for car slot ``me[i]`` of moment ``i`` (numpy or
            torch). Optional: needed only where the toolkit builds inputs itself.
        forward: how to run the network on a batch of inputs (default ``net(x)``; a tuple output's first element
            is used). E.g. ``lambda x: net.actor(x)`` for an actor-critic, ``net.get_output`` for some trainers.
        output: what ``forward`` returns - ``"logits"`` (the default), ``"probs"`` (a softmax at the end),
            ``"log_probs"``, or ``"controls"`` (a continuous policy: [N, 8] controls).
        actions: [A, 8] the controls of each discrete action, in the order of the network's outputs (optional;
            needed to play the policy in RocketSim and to name actions in reports and the overlay).
        value: an optional critic, ``value(x) -> [N]`` (the overlay's outlook bar; reports).
        name: what reports and the overlay call it.
        device: where the network runs ("cpu" is the default and the safe choice for everything here).
        mirror: for a bot whose observation shows the field mirrored left to right at some moments (x negated, so
            one side looks like the other - some bots do this to learn both sides at once): ``mirror(frames, me) ->
            [N] bool``, True where moment i is seen mirrored by car slot me[i]. Its network then can't tell the two
            sides apart, so the toolkit fits place readouts (the landing ring) in the bot's own frame and turns them
            back with this, and flips steer, yaw and roll when it says a move in words. See docs/howto/probes.md.
    """

    def __init__(self, net: torch.nn.Module, obs: ObsFn | None = None, forward: Callable[[torch.Tensor], Any] | None = None,
                 output: str = "logits", actions: Any = None, value: Callable[[torch.Tensor], Any] | None = None,
                 name: str = "policy", device: str | torch.device = "cpu",
                 mirror: Callable[[Frames, np.ndarray], Any] | None = None):
        if not isinstance(net, torch.nn.Module):
            raise TypeError(f"Policy needs a torch.nn.Module, got {type(net).__name__}")
        if isinstance(net, torch.jit.ScriptModule):
            raise CDSInterpretError(
                "This network is a TorchScript module. PyTorch can't attach listeners (hooks) to those, so its layers "
                "can't be read. Convert it first: net = ci.eager_copy(net) works for networks built from standard "
                "layers. Otherwise rebuild your network class in Python and load the weights into it: "
                "net.load_state_dict(scripted.state_dict())."
            )
        if output not in OUTPUTS:
            raise ValueError(f"output must be one of {OUTPUTS}, not {output!r}")
        self.net = net.to(device).eval()
        self.obs_fn = obs
        self._forward = forward
        self.output = output
        self.actions = None if actions is None else np.asarray(actions, dtype=np.float32)
        if self.actions is not None and (self.actions.ndim != 2 or self.actions.shape[1] != 8):
            raise ShapeError(f"actions must be [A, 8] (one row of 8 controls per action), not {self.actions.shape}")
        self.value_fn = value
        self.name = name
        self.device = torch.device(device)
        self.mirror_fn = mirror

    def __repr__(self) -> str:
        n = sum(p.numel() for p in self.net.parameters())
        acts = f", {len(self.actions)} actions" if self.actions is not None else ""
        return f"Policy({self.name!r}, {type(self.net).__name__}, {n:,} parameters, output={self.output!r}{acts})"

    # ---- running it
    @property
    def discrete(self) -> bool:
        return self.output != "controls"

    def as_input(self, x: Any) -> torch.Tensor:
        t = torch.as_tensor(np.asarray(x, dtype=np.float32) if not torch.is_tensor(x) else x)
        return t.to(self.device, torch.float32)

    def forward(self, x: Any) -> torch.Tensor:
        """The network's output for inputs x [N, D] (a tuple's first element)."""
        xt = self.as_input(x)
        try:
            out = (self._forward or self.net)(xt)
        except RuntimeError as e:
            msg = str(e)
            if any(w in msg for w in ("shape", "size", "mat1", "dimension")):
                raise ShapeError(
                    f"Your network couldn't run on these inputs ({tuple(xt.shape[1:])} per moment). PyTorch said: "
                    f"{msg.splitlines()[0]}. Check that your obs function builds what your network expects - same "
                    "numbers, same order.") from e
            raise
        if isinstance(out, (tuple, list)):
            out = out[0]
        if not torch.is_tensor(out):
            raise CDSInterpretError(f"the policy's forward returned {type(out).__name__}, not a tensor: pass "
                                    "forward= to say how to get the action output from your network")
        return out

    def __call__(self, x: Any) -> torch.Tensor:
        return self.forward(x)

    def log_probs(self, x: Any) -> torch.Tensor:
        """[N, A] log-probabilities of the discrete actions."""
        out = self.forward(x).float()
        if self.output == "logits":
            return torch.log_softmax(out, -1)
        if self.output == "probs":
            return torch.log(out.clamp_min(1e-12))
        if self.output == "log_probs":
            return out
        raise CDSInterpretError("this policy outputs controls (output='controls'): it has no action distribution")

    def log_probs_from_output(self, out: torch.Tensor) -> torch.Tensor:
        out = out.float()
        if self.output == "logits":
            return torch.log_softmax(out, -1)
        if self.output == "probs":
            return torch.log(out.clamp_min(1e-12))
        return out

    @torch.no_grad()
    def act(self, x: Any, deterministic: bool = True, generator: torch.Generator | None = None) -> np.ndarray:
        """The action of each moment: [N] action indices for a discrete policy (argmax, or sampled), [N, 8] controls
        for a continuous one."""
        if not self.discrete:
            return self.forward(x).float().clamp(-1, 1).cpu().numpy()
        lp = self.log_probs(x)
        if deterministic:
            return lp.argmax(-1).cpu().numpy()
        return torch.multinomial(lp.exp(), 1, generator=generator)[:, 0].cpu().numpy()

    @torch.no_grad()
    def controls(self, x: Any, deterministic: bool = True, generator: torch.Generator | None = None) -> np.ndarray:
        """[N, 8] controls (through the action table for a discrete policy)."""
        a = self.act(x, deterministic, generator)
        if not self.discrete:
            return np.asarray(a, np.float32)
        if self.actions is None:
            raise CDSInterpretError("This policy has no action table, so its choices can't be turned into controls. "
                                    "Pass actions= to ci.Policy: one row of 8 controls per action "
                                    "(ci.lookup_table() is the common 90-action table).")
        self.check_actions(int(a.max()) + 1 if a.size else 0)
        return self.actions[a]

    def check_actions(self, n_out: int, exact: bool = False) -> None:
        """Raise a friendly error when the action table cannot cover n_out actions (exact: must have that many)."""
        if self.actions is None:
            return
        if (exact and n_out != len(self.actions)) or n_out > len(self.actions):
            raise ShapeError(f"Your network has {n_out} outputs, but the action table has {len(self.actions)} rows. "
                             "Pass the table your bot was trained with: one row of 8 controls per output.")

    @torch.no_grad()
    def value(self, x: Any) -> np.ndarray | None:
        """The critic's value of each moment, or None without one."""
        if self.value_fn is None:
            return None
        v = self.value_fn(self.as_input(x))
        v = v[0] if isinstance(v, (tuple, list)) else v
        return np.asarray(v.detach().cpu() if torch.is_tensor(v) else v, np.float32).reshape(-1)

    def mirrored(self, frames: Frames, me: Any) -> np.ndarray:
        """[N] bool: the moments this bot sees mirrored left to right (all False without a ``mirror`` function)."""
        me = np.broadcast_to(np.asarray(me, dtype=np.int64), (len(frames),))
        if self.mirror_fn is None:
            return np.zeros(len(frames), bool)
        m = self.mirror_fn(frames, me)
        m = m.detach().cpu().numpy() if torch.is_tensor(m) else np.asarray(m)
        m = np.broadcast_to(m.astype(bool).reshape(-1), (len(frames),))
        return m.copy()

    def require_obs(self) -> None:
        """Raise a friendly error when the policy has no obs function."""
        if self.obs_fn is None:
            raise CDSInterpretError(
                "This policy doesn't know how to build its input from the game state yet. Give it a function: "
                "ci.Policy(net, obs=my_obs), where my_obs(frames, me) returns your bot's input for each moment. "
                "Or skip it: moments recorded from your bot in RLBot or in your RLGym env already carry its inputs. "
                "(An RLGym ObsBuilder works too: obs=ci.rlgym.obs_fn(MyObsBuilder()).)"
            )

    def observe(self, frames: Frames, me: Any) -> torch.Tensor:
        """The network's input [N, D] for the moments ``frames`` seen by car slots ``me``."""
        self.require_obs()
        assert self.obs_fn is not None
        me = np.broadcast_to(np.asarray(me, dtype=np.int64), (len(frames),))
        x = self.obs_fn(frames, me)
        x = x.detach().cpu() if torch.is_tensor(x) else torch.as_tensor(np.asarray(x, dtype=np.float32))
        x = x.float()
        if x.ndim < 2 or len(x) != len(frames):
            raise ShapeError(f"obs(frames, me) returned shape {tuple(x.shape)}; expected [N={len(frames)}, D] (or "
                             "[N, ...] for a network that takes per-car rows)")
        return x

    # ---- its layers
    def layers(self, example: Any = None) -> list[LayerInfo]:
        """Every module of the network with its type and output shape on ``example`` (a batch of inputs)."""
        ex = None if example is None else self.as_input(example)
        return list_layers(self.net, ex, forward=None if ex is None else (lambda e: self.forward(e)))

    def default_layers(self, example: Any) -> list[str]:
        """The layers worth probing first: each hidden layer after its nonlinearity."""
        return default_layers(self.layers(example))

    def activations(self, x: Any, layers: list[str] | dict[str, str], batch: int = 4096) -> dict[str, torch.Tensor]:
        """name -> [N, D] the layers' outputs for inputs x (on the CPU, float32)."""
        return run_layers(self.net, self.forward, self.as_input(x).cpu(), layers, batch=batch)

    # ---- baselines
    def random_like(self, seed: int = 0) -> Policy:
        """The same architecture with fresh random weights: the 'random network' baseline for probes. (With a custom
        ``forward`` that is not a method of the network, the random weights are swapped into this very network
        for each call and swapped back after.)"""
        net = copy.deepcopy(self.net)
        gen_state = torch.random.get_rng_state()
        torch.manual_seed(seed)
        try:
            for m in net.modules():
                reset = getattr(m, "reset_parameters", None)
                if callable(reset):
                    reset()
        finally:
            torch.random.set_rng_state(gen_state)
        name = f"{self.name} (random weights)"
        forward = self._forward
        if forward is None:
            return Policy(net, self.obs_fn, None, self.output, self.actions, None, name, self.device, self.mirror_fn)
        owner = getattr(forward, "__self__", None)
        if isinstance(owner, torch.nn.Module):
            for m_old, m_new in zip(self.net.modules(), net.modules()):
                if m_old is owner:
                    return Policy(net, self.obs_fn, getattr(m_new, forward.__name__), self.output, self.actions, None,
                                  name, self.device, self.mirror_fn)
        return _SwappedWeights(self, {k: v.detach().clone() for k, v in net.state_dict().items()}, name)


class _SwappedWeights(Policy):
    """A policy that runs ``base``'s network with other weights swapped in for each call (for a custom forward
    function that refers to the network itself)."""

    def __init__(self, base: Policy, state: dict[str, torch.Tensor], name: str):
        super().__init__(base.net, base.obs_fn, base._forward, base.output, base.actions, None, name, base.device,
                         base.mirror_fn)
        self._state = state

    @contextmanager
    def _swapped(self) -> Iterator[None]:
        keep = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
        self.net.load_state_dict(self._state)
        try:
            yield
        finally:
            self.net.load_state_dict(keep)

    def forward(self, x: Any) -> torch.Tensor:
        with self._swapped():
            return super().forward(x)

    def activations(self, x: Any, layers: list[str] | dict[str, str], batch: int = 4096) -> dict[str, torch.Tensor]:
        with self._swapped():
            return run_layers(self.net, super().forward, self.as_input(x).cpu(), layers, batch=batch)


# ------------------------------------------------------------------------------------------ TorchScript -> eager
_SIMPLE = {"ReLU": torch.nn.ReLU, "Tanh": torch.nn.Tanh, "Sigmoid": torch.nn.Sigmoid, "GELU": torch.nn.GELU,
           "SiLU": torch.nn.SiLU, "Identity": torch.nn.Identity, "Mish": torch.nn.Mish, "ReLU6": torch.nn.ReLU6,
           "Hardswish": torch.nn.Hardswish}


def eager_copy(module: torch.nn.Module) -> torch.nn.Module:
    """An eager copy of a TorchScript network built from the usual MLP blocks (Sequential, Linear, LayerNorm,
    ReLU / LeakyReLU / ELU / Tanh / Sigmoid / GELU / SiLU / Mish, Softmax / LogSoftmax, Dropout, Identity), with the
    same weights - so forward hooks can read its layers. Other module types raise an error that says which one: for
    those, rebuild your network class in Python and ``load_state_dict(module.state_dict())``."""
    if not isinstance(module, torch.jit.ScriptModule):
        return module
    kind = getattr(module, "original_name", type(module).__name__)
    if kind == "Sequential":
        return torch.nn.Sequential(*[eager_copy(m) for _, m in module.named_children()])
    if kind == "Linear":
        w = module.weight.detach()
        b = getattr(module, "bias", None)
        lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=b is not None)
        with torch.no_grad():
            lin.weight.copy_(w)
            if b is not None:
                lin.bias.copy_(b.detach())
        return lin
    if kind == "LayerNorm":
        w = getattr(module, "weight", None)
        if hasattr(module, "normalized_shape"):
            shape = tuple(module.normalized_shape)
        elif w is not None:
            shape = (int(w.shape[0]),)
        else:
            raise CDSInterpretError("eager_copy: a LayerNorm without weights or a normalized_shape")
        ln = torch.nn.LayerNorm(shape, eps=float(getattr(module, "eps", 1e-5)),
                                elementwise_affine=w is not None)
        if w is not None:
            with torch.no_grad():
                ln.weight.copy_(w.detach())
                ln.bias.copy_(module.bias.detach())
        return ln
    if kind in _SIMPLE:
        return _SIMPLE[kind]()
    if kind == "LeakyReLU":
        return torch.nn.LeakyReLU(float(getattr(module, "negative_slope", 0.01)))
    if kind == "ELU":
        return torch.nn.ELU(float(getattr(module, "alpha", 1.0)))
    if kind in ("Softmax", "LogSoftmax"):
        dim = getattr(module, "dim", -1)
        return (torch.nn.Softmax if kind == "Softmax" else torch.nn.LogSoftmax)(dim=-1 if dim is None else int(dim))
    if kind == "Dropout":
        return torch.nn.Dropout(float(getattr(module, "p", 0.0)))
    children = list(module.named_children())
    if children:
        parts = ", ".join(f"{n}: {getattr(m, 'original_name', type(m).__name__)}" for n, m in children)
        raise CDSInterpretError(
            f"eager_copy cannot rebuild a TorchScript '{kind}' module: its forward is custom code. Rebuild your "
            "network class in Python and load the weights: net = MyNet(...); "
            f"net.load_state_dict(scripted.state_dict()). (Its parts: {parts})"
        )
    raise CDSInterpretError(f"eager_copy does not know the TorchScript module type '{kind}': rebuild it in Python "
                            "and load its state_dict()")
