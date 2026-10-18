"""Factories named by text, so a separate process (the overlay's worker, a watch script) can build its own copy of
something - a policy, a critic, an observation function - from a name it can be sent:

    "package.module:factory"                         factory()
    ("package.module:factory", {"model": "a.pt"})    factory(model="a.pt")     (a list of the two works too)

The keyword arguments must be plain values (text, numbers, lists, dicts): they travel to the other process.
"""
from __future__ import annotations

import importlib
from typing import Any

from .errors import CDSInterpretError


def is_spec(x: Any) -> bool:
    """Is ``x`` a factory spec (text, or text with a dict of keyword arguments)?"""
    if isinstance(x, str):
        return True
    return (isinstance(x, (tuple, list)) and len(x) == 2 and isinstance(x[0], str)
            and isinstance(x[1], dict))


def build(spec: Any, what: str = "factory") -> Any:
    """What ``spec`` names, built: ``factory(**kwargs)``. ``what`` names it in errors ("policy", "critic", ...)."""
    if isinstance(spec, str):
        name, kwargs = spec, {}
    elif is_spec(spec):
        name, kwargs = spec[0], dict(spec[1])
    else:
        raise CDSInterpretError(f"{what} should be \"package.module:factory\" or (\"package.module:factory\", "
                                f"{{keyword arguments}}), not {type(spec).__name__}")
    mod, _, attr = name.partition(":")
    if not attr:
        raise CDSInterpretError(f"{what} {name!r}: give it as \"package.module:factory\" (a module you can import and "
                                "a function in it)")
    try:
        factory = getattr(importlib.import_module(mod), attr)
    except (ImportError, AttributeError) as e:
        raise CDSInterpretError(f"{what} {name!r} could not be loaded: {e}. The module must be importable from where "
                                "RLBot starts your bot (usually its folder)") from None
    try:
        return factory(**kwargs)
    except TypeError as e:
        if kwargs:
            raise CDSInterpretError(f"{what} {name!r} doesn't take the arguments {sorted(kwargs)}: {e}") from None
        raise
