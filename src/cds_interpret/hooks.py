"""Read and change what named layers of any PyTorch network compute, through forward hooks: the bot's own code is
not touched.

    tap = LayerTap(model, {"h2": "mlp.3"})                 # short name -> module path (as in model.named_modules())
    with tap:
        out = model(obs)
    tap.out["h2"]                                          # that module's output on the last call [N, D], detached

    tap = LayerTap(model, {"obs": ""}, inputs=True)        # a module's first positional INPUT ("" = the whole model)

    with LayerTap(model, {"h2": "mlp.3"}, edits={"h2": fn}):   # fn(output) -> the output used from there on
        out = model(obs)

    list_layers(model, example_obs)                        # every module, its type and output shape: what to tap

Notes

- Works with eager ``nn.Module`` networks. A TorchScript module runs no Python hooks: rebuild it as an eager module
  first (:func:`cds_interpret.policy.eager_copy` does that for the usual MLP building blocks).
- Several calls inside one ``with`` keep the last output unless ``keep="all"`` (then a list per name).
- A module whose output is a tuple: the first element is read / replaced.
- A read-only tap (no edits) never changes what the network computes: its outputs stay bit for bit the same.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from .errors import LayerNotFound

Edit = Callable[[torch.Tensor], torch.Tensor]


def get_module(model: torch.nn.Module, path: str) -> torch.nn.Module:
    """``model.get_submodule(path)`` with a friendly error that lists the closest layer names."""
    try:
        return model.get_submodule(path)
    except AttributeError:
        names = [n for n, _ in model.named_modules() if n]
        close = difflib.get_close_matches(path, names, n=5, cutoff=0.4)
        hint = f" Did you mean: {', '.join(repr(c) for c in close)}?" if close else ""
        listing = ", ".join(repr(n) for n in names[:25]) + (" ..." if len(names) > 25 else "")
        raise LayerNotFound(
            f"There's no layer called {path!r} in your {type(model).__name__}.{hint}\nIts layers are: {listing}\n"
            "(policy.layers(x) lists them with their types and output shapes.)"
        ) from None


class LayerTap:
    """Named submodules of a network read - their outputs, or with ``inputs=True`` their first input - and
    optionally replaced through ``edits`` {name: fn(output) -> new output}, while attached (as a context manager, or
    ``attach()`` / ``detach_hooks()``). ``out[name]`` holds what was read on the last call (``keep="all"``: every
    call, as a list)."""

    def __init__(self, model: torch.nn.Module, layers: dict[str, str] | list[str], inputs: bool = False,
                 edits: dict[str, Edit] | None = None, keep: str = "last", detach: bool = True, read: bool = True):
        if keep not in ("last", "all"):
            raise ValueError(f"keep must be 'last' or 'all', not {keep!r}")
        self.model = model
        self.layers = dict(layers) if isinstance(layers, dict) else {n: n for n in layers}
        self.inputs, self.keep, self.detach, self.read = inputs, keep, detach, read
        self.edits = dict(edits or {})
        unknown = set(self.edits) - set(self.layers)
        if unknown:
            raise LayerNotFound(f"edits given for layers that are not tapped: {sorted(unknown)}")
        self.out: dict[str, Any] = {}
        self._handles: list[Any] = []
        self.modules = {n: get_module(model, p) for n, p in self.layers.items()}

    def _store(self, name: str, x: torch.Tensor) -> None:
        if not self.read:
            return
        x = x.detach() if self.detach else x
        if self.keep == "all":
            self.out.setdefault(name, []).append(x)
        else:
            self.out[name] = x

    def _make_hook(self, name: str) -> Callable:
        edit = self.edits.get(name)
        if self.inputs:
            def pre(module: torch.nn.Module, args: tuple) -> tuple | None:
                x = args[0]
                if edit is not None:
                    x = edit(x)
                    self._store(name, x)
                    return (x, *args[1:])
                self._store(name, x)
                return None
            return pre

        def post(module: torch.nn.Module, args: tuple, output: Any) -> Any:
            first = output[0] if isinstance(output, tuple) else output
            if edit is not None:
                new = edit(first)
                self._store(name, new)
                return (new, *output[1:]) if isinstance(output, tuple) else new
            self._store(name, first)
            return None
        return post

    def attach(self) -> LayerTap:
        if self._handles:
            return self
        for name, mod in self.modules.items():
            hook = self._make_hook(name)
            h = mod.register_forward_pre_hook(hook) if self.inputs else mod.register_forward_hook(hook)
            self._handles.append(h)
        return self

    def detach_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @property
    def attached(self) -> bool:
        return bool(self._handles)

    def clear(self) -> None:
        self.out.clear()

    def __enter__(self) -> LayerTap:
        self.clear()
        return self.attach()

    def __exit__(self, *exc: object) -> None:
        self.detach_hooks()


ACTIVATIONS = ("ReLU", "NormReLU", "GELU", "SiLU", "Tanh", "LeakyReLU", "ELU", "Sigmoid", "Mish", "Softplus", "PReLU",
               "SELU", "CELU", "Hardtanh", "ReLU6", "Hardswish")


def sequential_layers(seq: torch.nn.Sequential, prefix: str = "", tag: str = "") -> tuple[dict[str, str], dict[str, str]]:
    """Short layer names for any ``nn.Sequential`` stack (an MLP, or a network's trunk): (outputs, inputs), the maps
    LayerTap and run_layers take. ``in`` is the first module's input, ``h1`` ... ``hk`` the output of each
    activation (the hidden layers, after their nonlinearity), ``out`` the last module's output. ``prefix``: the
    Sequential's path inside the model (e.g. ``"actor.layers"``); ``tag``: put before each short name (``"a"`` gives
    ``"a.h1"``).

        outs, ins = sequential_layers(policy.mlp, "mlp")   # {"h1": "mlp.1", "h2": "mlp.3", "out": "mlp.4"}, {"in": "mlp.0"}
    """
    p = f"{prefix}." if prefix else ""
    t = f"{tag}." if tag else ""
    outs, ins = {}, {f"{t}in": f"{p}0"}
    h = 0
    for i, m in enumerate(seq):
        if type(m).__name__ in ACTIVATIONS:
            h += 1
            outs[f"{t}h{h}"] = f"{p}{i}"
    outs[f"{t}out"] = f"{p}{len(seq) - 1}"
    return outs, ins


@dataclass
class LayerInfo:
    """One module of a network as list_layers found it."""

    path: str
    type: str
    shape: tuple[int, ...] | None          # its output's shape on the example (None: not called, or not a tensor)
    leaf: bool                             # it has no children
    activation: bool                       # it is a nonlinearity (its output is a hidden layer after activation)

    @property
    def width(self) -> int | None:
        """Features per moment (everything after the batch axis multiplied), or None."""
        if not self.shape:
            return None
        w = 1
        for s in self.shape[1:]:
            w *= int(s)
        return w

    def __str__(self) -> str:
        shape = "x".join(str(s) for s in self.shape) if self.shape else "-"
        return f"{self.path or '(model)':32s} {self.type:18s} {shape}"


@torch.no_grad()
def list_layers(model: torch.nn.Module, example: Any = None, forward: Callable[[Any], Any] | None = None) -> list[LayerInfo]:
    """Every module of ``model`` in call order (the ones the example reached) with its type and output shape, run
    on ``example`` (one batch of the network's input); without an example, in definition order with no shapes.
    ``forward``: how to run the network on the example (default ``model(example)``)."""
    mods = dict(model.named_modules())
    shapes: dict[str, tuple[int, ...] | None] = {}
    order: list[str] = []
    if example is not None:
        handles = []

        def make(name: str) -> Callable:
            def hook(module: torch.nn.Module, args: tuple, output: Any) -> None:
                first = output[0] if isinstance(output, tuple) else output
                shapes[name] = tuple(first.shape) if torch.is_tensor(first) else None
            return hook

        def make_pre(name: str) -> Callable:
            def pre(module: torch.nn.Module, args: tuple) -> None:
                if name not in order:                  # (in the order the modules start: outer before inner)
                    order.append(name)
            return pre
        for name, m in mods.items():
            handles.append(m.register_forward_pre_hook(make_pre(name)))
            handles.append(m.register_forward_hook(make(name)))
        try:
            (forward or model)(example)
        finally:
            for h in handles:
                h.remove()
    names = order + [n for n in mods if n not in order]
    out = []
    for n in names:
        m = mods[n]
        out.append(LayerInfo(n, type(m).__name__, shapes.get(n), not any(True for _ in m.children()),
                             type(m).__name__ in ACTIVATIONS))
    return out


def default_layers(infos: list[LayerInfo]) -> list[str]:
    """The layers worth probing first: every activation's output (a hidden layer after its nonlinearity); without
    activation modules (a network that calls F.relu in forward), every Linear layer but the last. Only layers with
    a per-moment vector output ([N, D])."""
    flat = [i for i in infos if i.shape is not None and len(i.shape) == 2]
    acts = [i.path for i in flat if i.activation]
    if acts:
        return acts
    lin = [i.path for i in flat if i.type == "Linear"]
    return lin[:-1] if len(lin) > 1 else lin


@torch.no_grad()
def run_layers(model: torch.nn.Module, forward: Callable[[torch.Tensor], Any], x: torch.Tensor,
               layers: dict[str, str] | list[str], inputs: dict[str, str] | None = None, batch: int = 4096,
               dtype: torch.dtype = torch.float32, device: str | torch.device | None = None) -> dict[str, torch.Tensor]:
    """Every layer of ``layers`` (outputs) and ``inputs`` (module inputs) for the rows of x, in batches:
    ``forward(xb)`` runs the network (e.g. ``model``, or ``lambda o: model.actor(o)``). Returns name -> [N, D]
    tensors on the CPU in ``dtype`` (per-entity layers [N, E, D] are flattened to [N, E*D])."""
    layers = dict(layers) if isinstance(layers, dict) else {n: n for n in layers}
    parts: dict[str, list[torch.Tensor]] = {}
    for i in range(0, len(x), batch):
        xb = x[i:i + batch]
        if device is not None:
            xb = xb.to(device)
        taps = [LayerTap(model, layers)] + ([LayerTap(model, inputs, inputs=True)] if inputs else [])
        for t in taps:
            t.attach()
        try:
            forward(xb)
        finally:
            for t in taps:
                t.detach_hooks()
        for t in taps:
            for n, v in t.out.items():
                parts.setdefault(n, []).append(v.reshape(len(xb), -1).to("cpu", dtype))
    missing = [n for n in list(layers) + list(inputs or {}) if n not in parts]
    if missing and len(x):
        raise LayerNotFound(f"the layers {missing} were never called by forward(x): is it the right module path, "
                            "and does the network's forward pass go through it?")
    return {n: torch.cat(v) for n, v in parts.items()}


def tail(seq: torch.nn.Sequential, after: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """The rest of a Sequential after its module number ``after``: a function from that module's output to the
    Sequential's output - e.g. a policy's hidden layer -> logits, for exact feature effects (effects.py) and
    policy-aware dictionaries (sae.train_sae(downstream=...)) straight from stored activations."""
    rest = torch.nn.Sequential(*list(seq.children())[after + 1:])

    def f(x: torch.Tensor) -> torch.Tensor:
        return rest(x)
    return f
