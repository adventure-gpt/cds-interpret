"""What each attention head looks at: per-head attention maps over a network's tokens, by name, averaged over recorded
moments - and how they shift with the situation.

    import cds_interpret as ci

    maps = study.attention(tokens=["own car", "ball", "opponent"])    # every attention layer the policy runs
    print(maps.summary())                    # per head: what it watches, how spread out, where it shifts, in words
    study.report("report.html")              # a heatmap per layer x head, and the shifts by situation

    probs = ci.attention.probs(net, x)       # any network, one batch: {layer: [N, heads, queries, keys]}

**Where the numbers come from.** Fused attention kernels (``scaled_dot_product_attention``, flash attention, a
transformer layer's fast path) never hand their weights out. So the toolkit doesn't ask for them: a hook captures
each attention module's *input* and recomputes softmax(Q K^T / sqrt(d)) per head, in float32, from the module's own
projections. The network runs exactly as it always does.

- ``nn.MultiheadAttention``, ``nn.TransformerEncoderLayer`` (fast path or not) and the attention modules inside
  ``nn.TransformerDecoderLayer``: built in, found by :func:`find_sites`.
- Your own module: give it a method ``attention_probs(*args, **kwargs) -> [B, H, Q, K]`` that takes what its
  ``forward`` takes; or describe its projections with :func:`recompute` (a packed qkv layer, or separate q and k
  layers, an optional norm first, a mask argument); or pass any function ``fn(module, args, kwargs) -> [B, H, Q, K]``
  as a :class:`Site`. :func:`register_attention` makes a function the default for a module class.

**Token names.** A list of names, one per token position (``["own car", "ball", "opponent"]``), or, for tokens whose
order changes from moment to moment (a set of cars), :class:`Tokens` with a ``roles`` function that says which named
role each token plays at each moment: the attention to every token of a role is added up ("opponents"), and the rows
of every query token of a role averaged.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .errors import CDSInterpretError, LayerNotFound, ShapeError
from .hooks import get_module

ProbsFn = Callable[[torch.nn.Module, tuple, dict], torch.Tensor]
"""``fn(module, args, kwargs) -> [B, H, Q, K]``: a module's per-head attention probabilities from the positional and
keyword arguments it was called with."""


# ------------------------------------------------------------------------------- probabilities from queries and keys
def _expand(m: torch.Tensor, ndim: int = 4) -> torch.Tensor:
    """A mask or bias as [B, H, Q, K]-broadcastable: [B, K] (padding) -> [B, 1, 1, K]; [B, Q, K] -> [B, 1, Q, K]."""
    if m.dim() == 2:
        return m[:, None, None, :]
    if m.dim() == 3:
        return m[:, None]
    return m


def softmax_attention(q: torch.Tensor, k: torch.Tensor, heads: int | None = None, keep: torch.Tensor | None = None,
                      bias: torch.Tensor | None = None, scale: float | None = None) -> torch.Tensor:
    """softmax(q k^T * scale) per head, in float32: [B, H, Q, K].

    q [B, Q, D] and k [B, K, D] are split into ``heads`` heads of D / heads numbers each (the usual layout: head h
    is numbers h*d .. (h+1)*d - 1), or come per head already: [B, H, Q, d] and [B, H, K, d]. ``keep``: which keys a
    query may attend to, True = may (a [B, K] padding mask, [B, Q, K], or anything broadcasting to [B, H, Q, K]);
    ``bias``: added to the scores first (an additive float mask, broadcasting the same way); ``scale``: default
    1 / sqrt(d). A query with no key left to attend to gets a row of zeros."""
    q, k = q.float(), k.float()
    if q.dim() == 3:
        if heads is None:
            raise ShapeError("softmax_attention: q is [B, Q, D] - say how many heads it splits into (heads=...)")
        b, nq, d = q.shape
        if d % heads or k.shape[-1] != d:
            raise ShapeError(f"softmax_attention: q has {d} numbers per token and k {k.shape[-1]}: they must be the "
                             f"same, and split evenly into {heads} heads")
        q = q.reshape(b, nq, heads, d // heads).transpose(1, 2)
        k = k.reshape(k.shape[0], k.shape[1], heads, d // heads).transpose(1, 2)
    elif q.dim() != 4 or k.dim() != 4:
        raise ShapeError(f"softmax_attention: q and k must be [B, Q, D] / [B, K, D] or [B, H, Q, d] / [B, H, K, d], "
                         f"not {tuple(q.shape)} and {tuple(k.shape)}")
    s = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(q.shape[-1]) if scale is None else float(scale))
    if bias is not None:
        s = s + _expand(bias.to(s.device)).float()
    if keep is not None:
        s = s.masked_fill(~_expand(keep.to(s.device)).bool(), float("-inf"))
    return torch.nan_to_num(torch.softmax(s, -1), nan=0.0)


# ---------------------------------------------------------------------------------------------- argument binding
_SIGS: dict[type, inspect.Signature] = {}


def bind(module: torch.nn.Module, args: tuple, kwargs: dict) -> dict[str, Any]:
    """The arguments a module was called with, by the names of its ``forward`` (defaults filled in)."""
    t = type(module)
    if t not in _SIGS:
        _SIGS[t] = inspect.signature(t.forward)
    try:
        b = _SIGS[t].bind(module, *args, **kwargs)
    except TypeError:                                     # (an odd signature: positional names only)
        out = {f"arg{i}": a for i, a in enumerate(args)}
        out.update(kwargs)
        return out
    b.apply_defaults()
    out = dict(b.arguments)
    out.pop(next(iter(_SIGS[t].parameters)), None)        # (self)
    return out


# --------------------------------------------------------------------------------------- the built-in modules
def mha_probs(module: Any, args: tuple, kwargs: dict) -> torch.Tensor:
    """``nn.MultiheadAttention``'s per-head probabilities, recomputed from its input projection, with its key padding
    mask and attention mask (bool: True = not allowed; or additive floats) - what ``need_weights=True,
    average_attn_weights=False`` would return, whatever path its forward takes."""
    a = bind(module, args, kwargs)
    query, key = a["query"], a["key"]
    kpm, mask = a.get("key_padding_mask"), a.get("attn_mask")
    if getattr(module, "bias_k", None) is not None or getattr(module, "add_zero_attn", False):
        raise CDSInterpretError("this nn.MultiheadAttention has add_bias_kv or add_zero_attn: its extra key isn't one "
                                "of your tokens, so its maps aren't supported - give it your own function as a Site")
    if query.dim() == 2:                                  # unbatched
        query, key = query[None], key[None]
        kpm = None if kpm is None else kpm[None]
    elif not module.batch_first:
        query, key = query.transpose(0, 1), key.transpose(0, 1)
    e, h = module.embed_dim, module.num_heads
    if module._qkv_same_embed_dim:
        wq, wk = module.in_proj_weight[:e], module.in_proj_weight[e:2 * e]
    else:
        wq, wk = module.q_proj_weight, module.k_proj_weight
    b = module.in_proj_bias
    bq, bk = (None, None) if b is None else (b[:e].float(), b[e:2 * e].float())
    q = F.linear(query.float(), wq.float(), bq)
    k = F.linear(key.float(), wk.float(), bk)
    nb, nq, nk = q.shape[0], q.shape[1], k.shape[1]
    keep: torch.Tensor | None = None
    bias: torch.Tensor | None = None
    if kpm is not None:
        if kpm.dtype == torch.bool:
            keep = ~kpm[:, None, None, :]
        else:
            bias = kpm.float()[:, None, None, :]
    if mask is not None:
        m = mask if mask.dim() == 3 else mask[None]
        m = m.reshape(nb, h, nq, nk) if m.shape[0] == nb * h and nb * h > 1 else m.reshape(1, 1, nq, nk)
        if m.dtype == torch.bool:
            keep = ~m if keep is None else keep & ~m
        else:
            bias = m.float() if bias is None else bias + m.float()
    elif a.get("is_causal"):
        causal = torch.ones(nq, nk, dtype=torch.bool, device=q.device).tril()
        keep = causal[None, None] if keep is None else keep & causal[None, None]
    return softmax_attention(q, k, h, keep=keep, bias=bias)


def encoder_layer_probs(layer: Any, args: tuple, kwargs: dict) -> torch.Tensor:
    """``nn.TransformerEncoderLayer``'s self-attention, from the layer's own input (its fast path skips its
    ``self_attn`` module, so that module's hooks may never run): the input normalised first when ``norm_first``."""
    a = bind(layer, args, kwargs)
    x, kpm, mask = a["src"], a.get("src_key_padding_mask"), a.get("src_mask")
    if getattr(x, "is_nested", False):                   # (nn.TransformerEncoder's fast path: padding removed)
        lens = [int(t.shape[0]) for t in x.unbind()]
        x = torch.nested.to_padded_tensor(x, 0.0)
        kpm = torch.arange(x.shape[1], device=x.device)[None] >= torch.tensor(lens, device=x.device)[:, None]
    h = layer.norm1(x) if getattr(layer, "norm_first", False) else x
    return mha_probs(layer.self_attn, (h, h, h), {"key_padding_mask": kpm, "attn_mask": mask,
                                                  "is_causal": bool(a.get("is_causal", False))})


def _protocol_probs(module: Any, args: tuple, kwargs: dict) -> torch.Tensor:
    return module.attention_probs(*args, **kwargs)


_REGISTRY: dict[type, ProbsFn] = {
    torch.nn.MultiheadAttention: mha_probs,
    torch.nn.TransformerEncoderLayer: encoder_layer_probs,
}


def register_attention(cls: type, fn: ProbsFn | None = None) -> Any:
    """Make ``fn(module, args, kwargs) -> [B, H, Q, K]`` how the toolkit reads every module of class ``cls`` (and
    :func:`find_sites` find them). Works as a decorator too: ``@register_attention(MyBlock)``."""
    if fn is not None:
        _REGISTRY[cls] = fn
        return fn

    def deco(f: ProbsFn) -> ProbsFn:
        _REGISTRY[cls] = f
        return f
    return deco


def _sub(module: torch.nn.Module, part: Any) -> Any:
    if part is None or (callable(part) and not isinstance(part, str)):
        return part
    try:
        return module.get_submodule(part)
    except AttributeError:
        names = [n for n, _ in module.named_children()]
        raise LayerNotFound(f"{type(module).__name__} has no part called {part!r} (its parts: {', '.join(names)})") \
            from None


def _heads(module: torch.nn.Module, heads: Any) -> int:
    if isinstance(heads, int):
        return heads
    for name in ([heads] if isinstance(heads, str) else []) + ["num_heads", "heads", "n_heads", "nhead", "n_head"]:
        v = getattr(module, name, None)
        if isinstance(v, int):
            return v
    raise CDSInterpretError(f"How many heads does this {type(module).__name__} have? Pass heads=4 (or the name of "
                            "the attribute that holds it) to recompute(...)")


def _arg(a: dict[str, Any], module: torch.nn.Module, spec: Any) -> Any:
    if spec is None:
        return None
    if callable(spec) and not isinstance(spec, str):
        return spec(module, a)
    if spec not in a:
        raise CDSInterpretError(f"{type(module).__name__}.forward has no argument {spec!r} (it has: "
                                f"{', '.join(a)})")
    return a[spec]


def recompute(qkv: str | None = None, q: str | None = None, k: str | None = None, heads: int | str | None = None,
              norm: str | Callable | None = None, x: str | None = None, context: str | None = None,
              keep: str | Callable | None = None, masked: str | Callable | None = None, layout: str = "qkv",
              scale: float | None = None) -> ProbsFn:
    """A probabilities function for your own attention module, from the names of its parts and arguments:

        recompute(qkv="qkv", heads=4, norm="ln1", keep="valid")    # a pre-norm block with one packed q,k,v Linear
        recompute(q="q_proj", k="k_proj", heads="n_heads")         # separate projections

    qkv: a part producing [.., 3 D] (q, k, v one after the other - ``layout="qkv"``, as nn.MultiheadAttention packs
    them - or per head, [heads, 3, d]: ``layout="heads"``); or q and k: parts producing [.., D]. heads: a number, or
    the attribute holding it (found by itself when it is num_heads, heads, n_heads or nhead). norm: a part (or
    function) the input goes through first. x: the forward argument holding the tokens (default: the first);
    context: the one holding the keys' tokens, for cross-attention (default: x, after norm). keep / masked: the
    forward argument holding a mask - ``keep`` True where a key may be attended to, ``masked`` True where it may not
    (padding) - either [B, K] or [B, Q, K] (or a function (module, arguments) -> mask). scale: default 1 / sqrt(d)."""
    if (qkv is None) == (q is None or k is None) or (qkv is not None and (q is not None or k is not None)):
        raise CDSInterpretError("recompute: name either qkv (one packed projection) or both q and k")
    if layout not in ("qkv", "heads"):
        raise CDSInterpretError(f"recompute: layout is \"qkv\" or \"heads\", not {layout!r}")

    def fn(module: torch.nn.Module, args: tuple, kwargs: dict) -> torch.Tensor:
        a = bind(module, args, kwargs)
        if x is not None and x not in a:
            raise CDSInterpretError(f"{type(module).__name__}.forward has no argument {x!r} (it has: {', '.join(a)})")
        tokens = a[x] if x is not None else next(iter(a.values()))
        nrm = _sub(module, norm)
        hx = nrm(tokens) if nrm is not None else tokens
        ctx = _arg(a, module, context) if context is not None else hx
        h = _heads(module, heads)
        if qkv is not None:
            out = _sub(module, qkv)(hx).float()
            d = out.shape[-1] // 3
            if layout == "qkv":
                qq, kk = out[..., :d], out[..., d:2 * d]
            else:
                per = out.reshape(*out.shape[:-1], h, 3, d // h)
                qq, kk = per[..., 0, :].flatten(-2), per[..., 1, :].flatten(-2)
        else:
            qq, kk = _sub(module, q)(hx), _sub(module, k)(ctx)
        m_keep, m_masked = _arg(a, module, keep), _arg(a, module, masked)
        km = None if m_keep is None else m_keep.bool()
        if m_masked is not None:
            km = ~m_masked.bool() if km is None else km & ~m_masked.bool()
        if qq.dim() == 2:                                # unbatched [T, D]
            qq, kk = qq[None], kk[None]
            km = None if km is None else km[None]
        return softmax_attention(qq, kk, h, keep=km, scale=scale)
    return fn


# ---------------------------------------------------------------------------------------------- sites and taps
@dataclass
class Site:
    """One attention layer to read: the module's path in the network ("" = the network itself) and, optionally, how
    to get its probabilities (``fn(module, args, kwargs) -> [B, H, Q, K]``; default: built in for its class, or its
    ``attention_probs`` method). Several sites may share a module (each recomputing a different layer of it)."""

    path: str
    probs: ProbsFn | None = None


def probs_fn(module: torch.nn.Module, site: Site | None = None) -> ProbsFn:
    """How a module's probabilities are read: the site's function, a registered one for its class, or its
    ``attention_probs`` method."""
    if site is not None and site.probs is not None:
        return site.probs
    for cls in type(module).__mro__:
        if cls in _REGISTRY:
            return _REGISTRY[cls]
    if callable(getattr(module, "attention_probs", None)):
        return _protocol_probs
    where = f" at {site.path!r}" if site is not None and site.path else ""
    raise CDSInterpretError(
        f"The toolkit doesn't know how to read the attention of a {type(module).__name__}{where}. Built in: "
        "nn.MultiheadAttention and nn.TransformerEncoderLayer. For your own module, either give it a method "
        "attention_probs(*args, **kwargs) returning [batch, heads, queries, keys], or describe its projections: "
        "Site(path, probs=ci.attention.recompute(qkv=\"qkv\", heads=4)) - see docs/howto/attention-heads.md.")


def find_sites(model: torch.nn.Module) -> dict[str, Site]:
    """Every attention module of a network, by path, in definition order: nn.MultiheadAttention (except the one
    inside an nn.TransformerEncoderLayer, which is read from the layer), nn.TransformerEncoderLayer, registered
    classes, and modules with an ``attention_probs`` method."""
    out: dict[str, Site] = {}
    covered: set[str] = set()
    for path, m in model.named_modules():
        if path in covered:
            continue
        known = any(c in _REGISTRY for c in type(m).__mro__) or callable(getattr(m, "attention_probs", None))
        if known:
            out[path] = Site(path)
            if isinstance(m, torch.nn.TransformerEncoderLayer):
                covered.add(f"{path}.self_attn" if path else "self_attn")
    return out


def as_sites(model: torch.nn.Module, sites: Any) -> dict[str, Site]:
    """{name: Site} from None (every attention module: find_sites), {name: path or Site}, or [path or Site, ...]."""
    if sites is None:
        found = find_sites(model)
        if not found:
            raise CDSInterpretError(
                f"No attention modules were found in this {type(model).__name__}. Built in: nn.MultiheadAttention "
                "and nn.TransformerEncoderLayer. For your own attention module, name it as a Site with a way to "
                "recompute its probabilities: sites={\"L1\": ci.attention.Site(\"blocks.0\", "
                "probs=ci.attention.recompute(qkv=\"qkv\", heads=4))} (docs/howto/attention-heads.md).")
        return found
    if isinstance(sites, dict):
        return {str(n): s if isinstance(s, Site) else Site(str(s)) for n, s in sites.items()}
    if isinstance(sites, (str, Site)):
        sites = [sites]
    out = {}
    for s in sites:
        site = s if isinstance(s, Site) else Site(str(s))
        out[site.path] = site
    return out


class AttentionTap:
    """Hooks that read the per-head attention of named sites while attached (a context manager, or ``attach()`` /
    ``detach_hooks()``): ``probs[name]`` is [B, H, Q, K] (float32) from the last call of that module; ``order`` the
    names in the order the network first ran them. The network's own computation is not changed."""

    def __init__(self, model: torch.nn.Module, sites: Any = None):
        self.model = model
        self.sites = as_sites(model, sites)
        self.modules = {n: get_module(model, s.path) for n, s in self.sites.items()}
        self.fns = {n: probs_fn(self.modules[n], s) for n, s in self.sites.items()}
        self.probs: dict[str, torch.Tensor] = {}
        self.order: list[str] = []
        self._handles: list[Any] = []

    def _make_hook(self, name: str) -> Callable:
        fn = self.fns[name]

        def pre(module: torch.nn.Module, args: tuple, kwargs: dict) -> None:
            with torch.no_grad():
                try:
                    p = fn(module, args, kwargs)
                except CDSInterpretError:
                    raise
                except Exception as e:
                    raise CDSInterpretError(f"reading the attention of {name!r} ({type(module).__name__}) failed: "
                                            f"{type(e).__name__}: {e}") from e
            if not torch.is_tensor(p) or p.dim() != 4:
                raise ShapeError(f"the attention function of {name!r} returned "
                                 f"{tuple(p.shape) if torch.is_tensor(p) else type(p).__name__}: it must be "
                                 "[batch, heads, queries, keys]")
            self.probs[name] = p.detach().float()
            if name not in self.order:
                self.order.append(name)
        return pre

    def attach(self) -> AttentionTap:
        if not self._handles:
            for n, m in self.modules.items():
                self._handles.append(m.register_forward_pre_hook(self._make_hook(n), with_kwargs=True))
        return self

    def detach_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def __enter__(self) -> AttentionTap:
        self.probs.clear()
        return self.attach()

    def __exit__(self, *exc: object) -> None:
        self.detach_hooks()


@torch.no_grad()
def probs(model: torch.nn.Module, x: Any, sites: Any = None, forward: Callable[[Any], Any] | None = None
          ) -> dict[str, torch.Tensor]:
    """The per-head attention probabilities of every site for one batch: {name: [N, heads, queries, keys]} (float32,
    on the device the network ran on), in the order the network runs them. ``sites``: None (every attention module:
    find_sites), {name: path or Site} or a list; ``forward``: how to run the network on x (default ``model(x)``)."""
    tap = AttentionTap(model, sites)
    with tap:
        (forward or model)(x)
    missing = [n for n in tap.sites if n not in tap.probs]
    if missing and sites is not None:
        raise LayerNotFound(f"the attention sites {missing} were never run by the forward pass")
    return {n: tap.probs[n] for n in tap.order}


def iter_probs(model: torch.nn.Module, forward: Callable[[Any], Any], x: torch.Tensor, sites: Any = None,
               batch: int = 4096, strict: bool | None = None) -> Iterator[tuple[int, dict[str, torch.Tensor]]]:
    """(start row, {name: [b, H, Q, K]}) for x in batches, every site in the order the network runs them. Sites the
    forward pass never runs are left out - an error when ``strict`` (default: when the sites were named)."""
    strict = sites is not None if strict is None else strict
    tap = AttentionTap(model, sites)
    tap.attach()
    try:
        for i in range(0, len(x), batch):
            tap.probs.clear()
            with torch.no_grad():
                forward(x[i:i + batch])
            if i == 0:
                missing = [n for n in tap.sites if n not in tap.probs]
                if missing and strict:
                    raise LayerNotFound(f"the attention sites {missing} were never run by the forward pass: is it the "
                                        "right module, and does this forward go through it? (A critic's attention "
                                        "runs in the critic: pass its function as forward.)")
                if not tap.order:
                    raise LayerNotFound("the forward pass ran none of the attention modules")
            yield i, {n: tap.probs[n] for n in tap.order if n in tap.probs}
    finally:
        tap.detach_hooks()


# ------------------------------------------------------------------------------------------------- token names
@dataclass
class Tokens:
    """Names for a network's tokens. ``names``: one per token position - or, with ``roles``, the names of the roles
    tokens play, and ``roles(x) -> [N, K]`` ints: the role (an index into names) of token k at moment i, -1 when the
    token is absent. Every token of a role is added up as keys and averaged as queries, so "opponents" is the
    attention to all opponents together. ``queries``: names of the query tokens when they differ from the keys
    (cross-attention)."""

    names: list[str]
    roles: Callable[[torch.Tensor], Any] | None = None
    queries: list[str] | None = None

    def __post_init__(self) -> None:
        self.names = [str(n) for n in self.names]
        if len(set(self.names)) != len(self.names):
            raise CDSInterpretError(f"token names must differ: {self.names}")


def as_tokens(tokens: Any) -> Tokens | None:
    if tokens is None or isinstance(tokens, Tokens):
        return tokens
    if isinstance(tokens, str):
        return Tokens([t.strip() for t in tokens.split(",") if t.strip()])
    if isinstance(tokens, Sequence):
        return Tokens(list(tokens))
    raise CDSInterpretError("tokens: a list of names (one per token), or ci.attention.Tokens(names, roles=fn)")


def aggregate(p: torch.Tensor, roles: torch.Tensor, n_roles: int, qroles: torch.Tensor | None = None,
              n_qroles: int | None = None) -> dict[str, torch.Tensor]:
    """One site's probabilities p [B, H, Q, K] by role. roles [B, K]: each key token's role (-1: absent); qroles
    [B, Q] the query tokens' (default: roles, for self-attention). Returns "full" [B, H, Rq, R] (the mean row of each
    query role, the attention to each key role summed; NaN where the query role is absent), "focus" [B, H, R] (the
    mean row of every present query token), "entropy" [B, H] (nats, over the present keys, averaged over present
    queries), "spread" [B, H] (entropy / log of the present keys: 0 = one token, 1 = even), "self" [B, H] (the
    attention a token keeps on itself, for self-attention; NaN otherwise), "present" [B, R] (the role is there)."""
    b, h, nq, nk = p.shape
    dev = p.device
    roles = roles.to(dev).long()
    self_attn = qroles is None and nq == nk
    if qroles is None:
        qroles = roles if nq == nk else torch.arange(nq, device=dev).expand(b, nq)
        n_qroles = n_roles if nq == nk else nq
    qroles = qroles.to(dev).long()
    assert n_qroles is not None
    mk = F.one_hot(roles.clamp_min(0), n_roles).float() * (roles >= 0)[..., None].float()       # [B, K, R]
    mq = F.one_hot(qroles.clamp_min(0), n_qroles).float() * (qroles >= 0)[..., None].float()    # [B, Q, Rq]
    ak = p @ mk[:, None]                                                                       # [B, H, Q, R]
    cnt = mq.sum(1)                                                                            # [B, Rq]
    full = torch.einsum("bhqr,bqs->bhsr", ak, mq) / cnt[:, None, :, None]
    full = torch.where(cnt[:, None, :, None] > 0, full, torch.full_like(full, float("nan")))
    pq = (qroles >= 0).float()                                                                 # [B, Q]
    nqp = pq.sum(1).clamp_min(1.0)
    focus = (ak * pq[:, None, :, None]).sum(2) / nqp[:, None, None]
    ent = -(p * torch.log(p.clamp_min(1e-30))).sum(-1)                                         # [B, H, Q]
    ent = (ent * pq[:, None]).sum(-1) / nqp[:, None]
    nkp = (roles >= 0).sum(1).float()
    spread = ent / torch.log(nkp.clamp_min(2.0))[:, None]
    spread = torch.where(nkp[:, None] > 1, spread, torch.full_like(spread, float("nan")))
    if self_attn:
        diag = p.diagonal(dim1=-2, dim2=-1)                                                    # [B, H, Q]
        selfish = (diag * pq[:, None]).sum(-1) / nqp[:, None]
    else:
        selfish = torch.full((b, h), float("nan"), device=dev)
    return {"full": full, "focus": focus, "entropy": ent, "spread": spread, "self": selfish,
            "present": mk.sum(1) > 0}


# ------------------------------------------------------------------------------------------------- situations
@dataclass
class Situation:
    """A situation the maps are split by: ``on`` [N] where it holds, ``valid`` [N] where it is known (and compared:
    other situations leave kickoffs out), and ``when`` - how a sentence says it ("when the ball is in the air")."""

    on: np.ndarray
    valid: np.ndarray
    when: str


# label -> (situation name, how a sentence says it)
LABEL_SITUATIONS: dict[str, tuple[str, str]] = {
    "kickoff": ("kickoff", "at kickoffs"),
    "in_air": ("in the air", "when it is in the air itself"),
    "on_wall": ("on a wall", "when it drives on a wall"),
    "on_ceiling": ("on the ceiling", "when it drives on the ceiling"),
    "has_flip": ("flip ready", "when it has a flip ready"),
    "threat": ("threat on goal", "when the ball is heading into its goal"),
    "on_target": ("shot on target", "when the ball is heading into their goal"),
    "demo_chance": ("demo chance", "when it has a demo chance"),
    "first_to_ball": ("first to the ball", "when it is first to the ball"),
    "goal_side": ("goal side", "when it is goal side of the ball"),
    "last_back": ("last back", "when it is its team's last back"),
    "supersonic": ("supersonic", "when it is supersonic"),
    "touch_soon": ("about to touch", "when it is about to touch the ball"),
    "demo_soon": ("about to demo", "when it is about to demolish an opponent"),
    "shot_soon": ("about to shoot", "when it is about to shoot"),
    "demoed_soon": ("about to be demoed", "when it is about to be demolished"),
    "concede_soon": ("about to concede", "when the other team is about to score"),
    "goal_soon": ("about to score", "when its team is about to score"),
}


def situations(moments: Any, labels: dict[str, np.ndarray] | None = None, open_play: bool = True,
               opponent_close: float = 1000.0, ball_close: float = 800.0, ball_air: float = 300.0,
               low_boost: float = 20.0) -> dict[str, Situation]:
    """The situations attention maps are split by, from the moments' game state and labels: kickoff; the ball in the
    air (above ``ball_air`` uu); the ball in its own half (defending) or not (attacking); an opponent within
    ``opponent_close`` uu; the ball within ``ball_close`` uu; low boost; and every yes / no label in
    LABEL_SITUATIONS (in the air, threat on goal, demo chance, ...) plus your registered ones. With ``open_play``,
    every situation but kickoff is compared within open play (kickoffs left out of both sides), so that a kickoff's
    own pattern doesn't pass for, say, "the ball on the ground"."""
    from .frames import canon, team_sign
    from .labels import LABEL_KIND, short_label
    labels = dict(moments.labels if labels is None else labels)
    fr, me = moments.frames, np.asarray(moments.me, np.int64)
    n = len(me)
    rows = np.arange(n)
    ok = np.ones(n, bool)
    kick = np.nan_to_num(np.asarray(labels.get("kickoff", np.zeros(n)), np.float64)) > 0.5
    base = ~kick if open_play else ok
    out: dict[str, Situation] = {}
    if kick.any():
        out["kickoff"] = Situation(kick, ok.copy(), "at kickoffs")
    s = team_sign(fr.car_team[rows, me])
    ball = canon(fr.ball_pos, s)
    out["ball in the air"] = Situation(ball[:, 2] > ball_air, base.copy(), "when the ball is in the air")
    out["ball in our half"] = Situation(ball[:, 1] < 0, base.copy(), "when the ball is in its own half")
    pos = fr.car_pos[rows, me]
    opp = (fr.car_team != fr.car_team[rows, me][:, None]) & (fr.car_present > 0.5)
    if fr.has("car_demoed"):
        opp &= fr.car_demoed < 0.5
    if opp.any():
        d = np.where(opp, np.linalg.norm(fr.car_pos - pos[:, None], axis=-1), np.inf).min(1)
        out["opponent close"] = Situation(d < opponent_close, base.copy(),
                                          f"when an opponent is within {opponent_close:,.0f} uu")
    out["ball close"] = Situation(np.linalg.norm(ball - canon(pos, s), axis=-1) < ball_close, base.copy(),
                                  "when the ball is close")
    out["low boost"] = Situation(fr.car_boost[rows, me] < low_boost, base.copy(), f"when it has under "
                                                                                 f"{low_boost:g} boost")
    for k, v in labels.items():
        if k == "kickoff" or LABEL_KIND.get(k) != "bin":
            continue
        arr = np.asarray(v, np.float64)
        name, when = LABEL_SITUATIONS.get(k, (short_label(k), f"when '{short_label(k)}' holds"))
        out[name] = Situation(np.nan_to_num(arr) > 0.5, base & np.isfinite(arr), when)
    return out


def _group_sums(f: np.ndarray, g: np.ndarray, m: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-group sums of the rows of f where m holds, and their counts (g: group ids, sorted)."""
    rows, gs = f[m], g[m]
    out = np.zeros((n, f.shape[1]))
    if len(rows):
        starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
        out[gs[starts]] = np.add.reduceat(rows, starts, axis=0)
    return out, np.bincount(gs, minlength=n).astype(np.float64)


def shift_stats(focus: np.ndarray, games: np.ndarray, sits: dict[str, Situation], boot: int = 200, seed: int = 0,
                min_moments: int = 50, min_games: int = 3) -> dict[str, dict[str, Any]]:
    """For each situation: how each head's focus [N, ...] differs where it holds from where it doesn't (among the
    valid moments): "shift" (mean in - mean out), "lo" / "hi" (a 95% interval from resampling whole games),
    "tv" (half the summed |shift| over the last axis: how far the head's attention moved, 0..1) with "tv_lo" /
    "tv_hi", and the counts. A situation with fewer than ``min_moments`` moments or ``min_games`` games on a side
    is skipped (its entry says why)."""
    shape = focus.shape[1:]
    gid, ginv = np.unique(games, return_inverse=True)
    order = np.argsort(ginv, kind="stable")              # (by game: group sums with reduceat)
    ginv = ginv[order]
    flat = focus.reshape(len(focus), -1)[order]
    used = np.isfinite(flat).any(0)                      # (columns of padded heads are NaN everywhere: not counted)
    fin = np.isfinite(flat[:, used]).all(1)
    f = np.nan_to_num(flat.astype(np.float64))
    g = len(gid)
    rng = np.random.default_rng(seed)
    w = rng.poisson(1.0, (boot, g)).astype(np.float64)
    out: dict[str, dict[str, Any]] = {}
    for name, st in sits.items():
        valid = np.asarray(st.valid, bool)[order] & fin
        sit_on = np.asarray(st.on, bool)[order]
        on, off = valid & sit_on, valid & ~sit_on
        n_in, n_out = int(on.sum()), int(off.sum())
        g_in, g_out = len(np.unique(ginv[on])), len(np.unique(ginv[off]))
        res: dict[str, Any] = {"when": st.when, "n_in": n_in, "n_out": n_out, "games_in": g_in, "games_out": g_out}
        if min(n_in, n_out) < min_moments or min(g_in, g_out) < min_games:
            res["skipped"] = (f"too few moments ({n_in} in, {n_out} out; at least {min_moments} each)"
                              if min(n_in, n_out) < min_moments else
                              f"too few games ({g_in} in, {g_out} out; at least {min_games} each)")
            out[name] = res
            continue
        sums_in, c_in = _group_sums(f, ginv, on, g)
        sums_out, c_out = _group_sums(f, ginv, off, g)
        shift = sums_in.sum(0) / c_in.sum() - sums_out.sum(0) / c_out.sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            bs = (w @ sums_in) / (w @ c_in)[:, None] - (w @ sums_out) / (w @ c_out)[:, None]
        bs = bs[np.isfinite(bs).all(1)]
        sh = shift.reshape(shape)
        res["shift"] = sh
        res["lo"] = np.quantile(bs, 0.025, axis=0).reshape(shape) if len(bs) else np.full(shape, np.nan)
        res["hi"] = np.quantile(bs, 0.975, axis=0).reshape(shape) if len(bs) else np.full(shape, np.nan)
        tv_b = 0.5 * np.abs(bs.reshape((len(bs), *shape))).sum(-1)
        res["tv"] = 0.5 * np.abs(sh).sum(-1)
        res["tv_lo"] = np.quantile(tv_b, 0.025, axis=0) if len(bs) else np.full(shape[:-1], np.nan)
        res["tv_hi"] = np.quantile(tv_b, 0.975, axis=0) if len(bs) else np.full(shape[:-1], np.nan)
        out[name] = res
    return out


# ---------------------------------------------------------------------------------------------------- the maps
def _the(name: str) -> str:
    """A token name as a sentence says it: "ball" -> "the ball", "own car" -> "its own car"."""
    low = name.lower()
    if low.startswith(("own ", "self")):
        return f"its {name}" if low.startswith("own ") else "itself"
    if low.startswith(("the ", "its ", "their ", "a ", "an ", "token ", "query ")):
        return name
    return f"the {name}"


@dataclass
class AttentionMaps:
    """Per-head attention over recorded moments (Study.attention). Arrays per moment (N) and attention layer (S),
    heads padded to the widest layer with NaN: ``full`` [N, S, H, Rq, R] (each query role's row, the attention to
    each key role), ``focus`` [N, S, H, R] (what the head looks at: the mean row of every present query token, or of
    the query role ``query``), ``entropy`` / ``spread`` / ``selfish`` [N, S, H], ``present`` [N, R]. ``shifts``:
    shift_stats by situation, of ``focus``."""

    layers: list[str]
    paths: dict[str, str]
    heads: dict[str, int]
    queries: list[str]
    keys: list[str]
    full: np.ndarray
    focus: np.ndarray
    entropy: np.ndarray
    spread: np.ndarray
    selfish: np.ndarray
    present: np.ndarray
    games: np.ndarray
    query: str | None = None
    shifts: dict[str, dict[str, Any]] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"AttentionMaps({len(self.focus):,} moments, layers {self.layers}, "
                f"{sum(self.heads.values())} heads, tokens {self.keys})")

    @property
    def head_names(self) -> list[str]:
        """"L1 H1", "L1 H2", ... - every head, layer by layer."""
        return [f"{la} H{h + 1}" for la in self.layers for h in range(self.heads[la])]

    def _pairs(self) -> list[tuple[int, int]]:
        return [(s, h) for s, la in enumerate(self.layers) for h in range(self.heads[la])]

    def mean(self, where: Any = None) -> np.ndarray:
        """[S, H, Rq, R] the mean map of each head over the moments (or those where ``where`` [N] holds)."""
        x = self.full if where is None else self.full[np.asarray(where, bool)]
        with np.errstate(invalid="ignore"), _quiet():
            return np.nanmean(x, 0)

    def focus_mean(self, where: Any = None) -> np.ndarray:
        """[S, H, R] what each head looks at on average (its ``focus``)."""
        x = self.focus if where is None else self.focus[np.asarray(where, bool)]
        with _quiet():
            return np.nanmean(x, 0)

    def table(self) -> list[dict[str, Any]]:
        """One row per head: name, layer, head, focus {token: share}, entropy, spread, selfish, top shifts."""
        fm = self.focus_mean()
        with _quiet():
            ent, spr, sel = (np.nanmean(a, 0) for a in (self.entropy, self.spread, self.selfish))
        rows = []
        for s, h in self._pairs():
            rows.append({"head": f"{self.layers[s]} H{h + 1}", "layer": self.layers[s], "index": h,
                         "focus": {k: float(fm[s, h, j]) for j, k in enumerate(self.keys)},
                         "entropy": float(ent[s, h]), "spread": float(spr[s, h]), "selfish": float(sel[s, h]),
                         "shifts": self.top_shifts(s, h)})
        return rows

    def top_shifts(self, s: int, h: int, n: int = 3, min_points: float = 0.03) -> list[dict[str, Any]]:
        """The situations where head h of layer s shifts most (by how far its attention moves, ``tv``), keeping only
        those whose biggest single change is steady across games (its interval leaves 0 out) and at least
        ``min_points`` (0.03 = 3 points of attention)."""
        out = []
        for name, r in self.shifts.items():
            if "shift" not in r:
                continue
            sh, lo, hi = r["shift"][s, h], r["lo"][s, h], r["hi"][s, h]
            j = _steadiest(sh, lo, hi, min_points)
            if j is None:
                continue
            out.append({"situation": name, "when": r["when"], "tv": float(r["tv"][s, h]),
                        "tv_lo": float(r["tv_lo"][s, h]), "tv_hi": float(r["tv_hi"][s, h]), "token": self.keys[j],
                        "change": float(sh[j]), "lo": float(lo[j]), "hi": float(hi[j]), "n": int(r["n_in"])})
        out.sort(key=lambda t: -t["tv"])
        return out[:n]

    def describe(self, n_shifts: int = 2) -> dict[str, str]:
        """{head: a sentence}: what it watches and where that shifts most - "L1 H2 watches the ball (58% of its
        attention); more so when the ball is in the air (+12 points)." A starting point to check against the maps,
        like a feature's suggested name."""
        out = {}
        for row in self.table():
            items = sorted(row["focus"].items(), key=lambda kv: -kv[1])
            (k0, p0), (k1, p1) = items[0], items[1] if len(items) > 1 else ("", 0.0)
            sel, spr = row["selfish"], row["spread"]
            rest = (1.0 - p0 - p1) / max(len(items) - 2, 1)          # (what each other token gets, on average)
            if np.isfinite(sel) and sel >= 0.5:
                lead = f"mostly keeps each token on itself ({sel:.0%} of a token's attention goes to itself)"
            elif p0 >= 0.5:
                lead = f"watches {_the(k0)} ({p0:.0%} of its attention)"
            elif np.isfinite(spr) and spr >= 0.85:
                lead = f"spreads its attention almost evenly ({_the(k0)} gets the most, {p0:.0%})"
            elif np.isfinite(spr) and spr < 0.6:
                lead = (f"looks at one token at a time and switches between them: {_the(k0)} most ({p0:.0%}), then "
                        f"{_the(k1)} ({p1:.0%})")
            elif p1 >= 0.2 and (len(items) == 2 or rest < p1 / 2):
                lead = f"splits its attention between {_the(k0)} ({p0:.0%}) and {_the(k1)} ({p1:.0%})"
            else:
                lead = f"looks most at {_the(k0)} ({p0:.0%}), then {_the(k1)} ({p1:.0%})"
            parts = []
            for t in row["shifts"][:n_shifts]:
                pts = round(100 * t["change"])
                if t["token"] == k0 and not (np.isfinite(sel) and sel >= 0.5):
                    parts.append(f"{'more' if pts > 0 else 'less'} so {t['when']} ({pts:+d} points)")
                elif pts > 0:
                    parts.append(f"turns to {_the(t['token'])} {t['when']} ({pts:+d} points)")
                else:
                    parts.append(f"looks less at {_the(t['token'])} {t['when']} ({pts:+d} points)")
            text = f"{row['head']} {lead}"
            if parts:
                text += "; " + ", and ".join(parts)
            out[row["head"]] = text + "."
        return out

    def summary(self, width: int = 3) -> str:
        """A table per head - its focus, how spread out it is, where it shifts most - then a sentence for each."""
        who = f"the query token {self.query!r}" if self.query else "every query token"
        lines = [f"What each attention head looks at, over {len(self.focus):,} moments ({who}'s attention, averaged).",
                 "spread: 0 = all on one token, 1 = even over the tokens there. Shifts: the situations where a head's",
                 "attention moves most, steady across games (points = percentage points of its attention).", "",
                 f"  {'head':8s} {'looks at':44s} {'spread':>6s}  where it shifts most",
                 "  " + "-" * 104]
        for row in self.table():
            items = sorted(row["focus"].items(), key=lambda kv: -kv[1])[:width]
            look = ", ".join(f"{k} {v:.0%}" for k, v in items)
            sh = ", ".join(f"{t['situation']} ({t['change'] * 100:+.0f} {t['token']})" for t in row["shifts"][:2])
            spr = f"{row['spread']:.2f}" if np.isfinite(row["spread"]) else "-"
            lines.append(f"  {row['head']:8s} {look[:44]:44s} {spr:>6s}  {sh or '-'}")
        lines.append("")
        lines.append("In words (a starting point - check it against the maps):")
        lines.extend(f"  {t}" for t in self.describe().values())
        skipped = {k: r["skipped"] for k, r in self.shifts.items() if "skipped" in r and r["n_in"] and r["n_out"]}
        if skipped:
            lines.append("")
            lines.append("Situations not compared: " + "; ".join(f"{k} ({v})" for k, v in skipped.items()))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Everything but the per-moment arrays, as plain numbers (for JSON)."""
        mean = self.mean()
        heads = []
        for row, (s, h) in zip(self.table(), self._pairs()):
            row = dict(row)
            row["map"] = {q: {k: _num(mean[s, h, i, j]) for j, k in enumerate(self.keys)}
                          for i, q in enumerate(self.queries)}
            heads.append(row)
        sits = {}
        for name, r in self.shifts.items():
            sits[name] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
        desc = self.describe()
        for row in heads:
            row["description"] = desc[row["head"]]
        with _quiet():
            present = np.nanmean(self.present, 0)
        return {"moments": int(len(self.focus)), "layers": self.layers, "paths": self.paths, "heads": self.heads,
                "queries": self.queries, "keys": self.keys, "query": self.query,
                "present": {k: float(present[j]) for j, k in enumerate(self.keys)},
                "per_head": heads, "situations": sits, "meta": self.meta}

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Write to_dict() as JSON."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_clean(self.to_dict()), indent=1), encoding="utf-8")
        return p


def _num(v: Any) -> Any:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f if math.isfinite(f) else None


def _clean(v: Any) -> Any:
    """Plain JSON values: numpy to Python, NaN and infinities to None."""
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return _clean(v.tolist())
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(float(v)) else None
    return v


@contextmanager
def _quiet() -> Iterator[None]:
    """Silence numpy's "mean of empty slice" while averaging columns that may be all NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        yield


def measure(model: torch.nn.Module, forward: Callable[[Any], Any], x: torch.Tensor, sites: Any = None,
            tokens: Any = None, query: str | None = None, batch: int = 4096,
            log: Callable[[str], Any] | None = None) -> AttentionMaps:
    """Run the network over x [N, ...] in batches and keep, per moment, each head's attention by token name (see
    AttentionMaps). Without ``sites``, every attention module the forward pass runs is read and named L1, L2, ... in
    the order it runs them. ``tokens``: names (a list, one per token), Tokens, or None ("token 1", ...). ``query``:
    a query token (role) name whose row is the head's ``focus`` (default: the mean row of every query token)."""
    auto = sites is None
    tok = as_tokens(tokens)
    res: dict[str, list[np.ndarray]] = {}
    layers: list[str] = []
    paths: dict[str, str] = {}
    heads: dict[str, int] = {}
    keys: list[str] = []
    queries: list[str] = []
    named = as_sites(model, sites)
    for start, pb in iter_probs(model, forward, x, named, batch, strict=not auto):
        if not layers:
            layers = list(pb)
            shapes = {n: tuple(p.shape) for n, p in pb.items()}
            nk = {s[3] for s in shapes.values()}
            nq = {s[2] for s in shapes.values()}
            if len(nk) != 1 or len(nq) != 1:
                raise ShapeError(f"the attention layers have different numbers of tokens ({shapes}): study them one "
                                 "set at a time (sites=...)")
            nk_, nq_ = nk.pop(), nq.pop()
            if tok is None:
                tok = Tokens([f"token {i + 1}" for i in range(nk_)])
            if tok.roles is None and len(tok.names) != nk_:
                raise ShapeError(f"{len(tok.names)} token names for {nk_} tokens: give one name per token "
                                 f"({tok.names})")
            keys = list(tok.names)
            if tok.queries is not None:
                queries = list(tok.queries)
            elif nq_ == nk_:
                queries = list(keys)
            else:
                queries = [f"query {i + 1}" for i in range(nq_)]
            if query is not None and query not in queries:
                raise CDSInterpretError(f"query {query!r} isn't one of the query tokens: {', '.join(queries)}")
            heads = {n: int(shapes[n][1]) for n in layers}
            paths = {n: named[n].path for n in layers}
            if log is not None:
                log(f"Attention: {len(layers)} layers ({', '.join(f'{named[n].path or n}' for n in layers)}), "
                    f"{max(heads.values())} heads, {nq_} queries x {nk_} tokens")
        b = len(next(iter(pb.values())))
        first = pb[layers[0]]
        dev = first.device
        if tok is not None and tok.roles is not None:
            r = tok.roles(x[start:start + b])
            r = torch.as_tensor(np.asarray(r.cpu() if torch.is_tensor(r) else r), dtype=torch.long, device=dev)
            if r.shape != (b, first.shape[3]):
                raise ShapeError(f"Tokens.roles returned {tuple(r.shape)}; expected [{b}, {first.shape[3]}]")
        else:
            r = torch.arange(first.shape[3], device=dev).expand(b, -1).clone()
        absent = torch.stack([(p <= 0).all(2).all(1) for p in pb.values()]).all(0)       # masked keys: absent
        r = torch.where(absent, torch.full_like(r, -1), r)
        qr = None
        if first.shape[2] != first.shape[3] or (tok is not None and tok.queries is not None):
            qr = torch.arange(first.shape[2], device=dev).expand(b, -1)
        hmax = max(heads.values())
        parts: dict[str, list[torch.Tensor]] = {k: [] for k in ("full", "focus", "entropy", "spread", "self")}
        present = None
        for n in layers:
            ag = aggregate(pb[n], r, len(keys), qr, len(queries) if qr is not None else None)
            if query is not None:
                ag["focus"] = ag["full"][:, :, queries.index(query)]
            for k in parts:
                v = ag[k]
                if v.shape[1] < hmax:                     # (pad heads)
                    pad = torch.full((v.shape[0], hmax - v.shape[1], *v.shape[2:]), float("nan"), device=v.device)
                    v = torch.cat([v, pad], 1)
                parts[k].append(v)
            present = ag["present"]
        for k, per_layer in parts.items():
            res.setdefault(k, []).append(torch.stack(per_layer, 1).cpu().numpy().astype(np.float32))
        assert present is not None
        res.setdefault("present", []).append(present.cpu().numpy())
    if not res:
        raise CDSInterpretError("no moments to measure")
    if auto:
        ren = {n: f"L{i + 1}" for i, n in enumerate(layers)}
        paths = {ren[n]: p for n, p in paths.items()}
        heads = {ren[n]: h for n, h in heads.items()}
        layers = [ren[n] for n in layers]
    cat = {k: np.concatenate(v) for k, v in res.items()}
    return AttentionMaps(layers, paths, heads, queries, keys, cat["full"], cat["focus"], cat["entropy"],
                         cat["spread"], cat["self"], cat["present"], np.zeros(len(cat["focus"]), np.int64), query)


# ------------------------------------------------------------------------------------------------------ figures
def _steadiest(sh: np.ndarray, lo: np.ndarray, hi: np.ndarray, min_points: float = 0.03) -> int | None:
    """The token whose change is biggest among those steady across games and at least min_points, or None."""
    ok = ((lo > 0) | (hi < 0)) & (np.abs(sh) >= min_points)
    if not ok.any():
        return None
    return int(np.argmax(np.where(ok, np.abs(sh), -1.0)))


def steady_tv(r: dict[str, Any], min_points: float = 0.03) -> np.ndarray:
    """[S, H] how far each head's attention moves in a situation (shift_stats' tv), NaN where no single token's change
    is steady across games (its interval leaves 0 out) and at least ``min_points``."""
    sh, lo, hi = r["shift"], r["lo"], r["hi"]
    steady = ((lo > 0) | (hi < 0)) & (np.abs(sh) >= min_points)
    return np.where(steady.any(-1), r["tv"], np.nan)


def shown_situations(maps: AttentionMaps, n: int = 10) -> list[str]:
    """The compared situations where some head's attention moves most, steadily (steady_tv), up to n."""
    have = []
    for k, r in maps.shifts.items():
        if "tv" in r:
            t = steady_tv(r)
            if np.isfinite(t).any():
                have.append((float(np.nanmax(t)), k))
    return [k for _, k in sorted(have, reverse=True)[:n]]


def figures(maps: AttentionMaps, situations_shown: int = 10, details: int = 3) -> dict[str, str]:
    """The figures of a study of attention heads, as SVG text: "focus" (what each head looks at: a row per head, a
    column per token), "heads" (one small query x token map per layer and head), "shifts" (how far each head's
    attention moves in each situation, with the biggest steady change printed) and "shift: <situation>" for the
    ``details`` situations with the biggest moves (each token's change, red less / blue more)."""
    from .viz import plots
    out: dict[str, str] = {}
    names = maps.head_names
    pairs = maps._pairs()
    fm = maps.focus_mean()
    vals = np.array([[fm[s, h, j] for j in range(len(maps.keys))] for s, h in pairs])
    who = f"the {maps.query!r} token's" if maps.query else "each query token's, averaged"
    # (shares on a fixed 0 - 100% scale: a head that spreads its attention evenly looks pale, as it should)
    out["focus"] = plots.heatmap(names, maps.keys, vals, "What each attention head looks at", vmin=0.0, vmax=1.0,
                                 fmt="{:.0%}", row_w=70, cell=64,
                                 note=f"share of the head's attention ({who}) over {len(maps.focus):,} moments")
    out["heads"] = plots.head_grid(maps.mean(), maps.layers, [maps.heads[la] for la in maps.layers], maps.queries,
                                   maps.keys, "Each head's map: who looks at whom", vmax=1.0,
                                   note="rows: the token looking (query); columns: the token looked at; percent of "
                                        "its attention")
    shown = shown_situations(maps, situations_shown)
    if shown:
        tv = np.full((len(pairs), len(shown)), np.nan)
        text = [["" for _ in shown] for _ in pairs]
        tips = [["" for _ in shown] for _ in pairs]
        for c, k in enumerate(shown):
            r = maps.shifts[k]
            st = steady_tv(r)
            for i, (s, h) in enumerate(pairs):
                tv[i, c] = 100 * st[s, h]
                sh, lo, hi = r["shift"][s, h], r["lo"][s, h], r["hi"][s, h]
                js = _steadiest(sh, lo, hi)
                j = int(np.nanargmax(np.abs(sh))) if js is None else js
                if js is not None:
                    text[i][c] = f"{100 * sh[j]:+.0f} {maps.keys[j]}"
                tips[i][c] = (f"{names[i]} {r['when']}: its attention moves {100 * r['tv'][s, h]:.0f} points; most on "
                              f"{maps.keys[j]} {100 * sh[j]:+.0f} (95% interval {100 * lo[j]:+.0f} to "
                              f"{100 * hi[j]:+.0f}); {r['n_in']:,} moments")
        vmax = max(10.0, math.ceil(float(np.nanmax(tv)) / 5) * 5) if np.isfinite(tv).any() else 10.0
        out["shifts"] = plots.heatmap(names, shown, tv, "Where each head's attention shifts", vmin=0.0, vmax=vmax,
                                      fmt="{:.0f}", row_w=70, cell=64, text=text, tips=tips,
                                      note="how far the head's attention moves (points), and the token that moves "
                                           "most; blank: no change steady across games")
        for k in shown[:details]:
            r = maps.shifts[k]
            ch = np.array([[100 * r["shift"][s, h, j] for j in range(len(maps.keys))] for s, h in pairs])
            tips_k = [[f"{names[i]} {r['when']}: {maps.keys[j]} {ch[i, j]:+.1f} points (95% interval "
                       f"{100 * r['lo'][s, h, j]:+.1f} to {100 * r['hi'][s, h, j]:+.1f})"
                       for j in range(len(maps.keys))] for i, (s, h) in enumerate(pairs)]
            v = max(5.0, math.ceil(float(np.nanmax(np.abs(ch))) / 5) * 5) if np.isfinite(ch).any() else 5.0
            out[f"shift: {k}"] = plots.heatmap(names, maps.keys, ch, f"{k[0].upper()}{k[1:]}: the change in each "
                                               "head's attention", vmax=v, fmt="{:+.0f}", row_w=70, cell=64,
                                               diverging=True, tips=tips_k,
                                               note=f"points of attention {r['when']} against the rest "
                                                    f"({r['n_in']:,} against {r['n_out']:,} moments)")
    return out
