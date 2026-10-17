"""The toolkit's errors: each says what went wrong and what to do about it."""

from __future__ import annotations

import importlib
from types import ModuleType


class CDSInterpretError(Exception):
    """Base class of every error this package raises on purpose."""


class MissingDependency(CDSInterpretError, ImportError):
    """An optional package is needed for this feature and is not installed."""


class MeshesNotFound(CDSInterpretError, FileNotFoundError):
    """RocketSim needs the arena's collision meshes and none were found."""


class ShapeError(CDSInterpretError, ValueError):
    """An array or tensor does not have the shape a function expects."""


class LayerNotFound(CDSInterpretError, KeyError):
    """A layer name does not name a module of the network."""

    def __str__(self) -> str:                     # KeyError would print the message in quotes
        return str(self.args[0]) if self.args else ""


_EXTRA_OF = {
    "RocketSim": "sim",
    "rlbot": "rlbot",
    "rlgym": "rlgym",
    "rlgym.rocket_league": "rlgym",
}


def require(module: str, feature: str, extra: str | None = None, pip_name: str | None = None) -> ModuleType:
    """Import an optional module, or raise MissingDependency with the exact install line.

    >>> rs = require("RocketSim", "recording games in RocketSim")
    """
    try:
        return importlib.import_module(module)
    except ImportError as e:
        extra = extra or _EXTRA_OF.get(module)
        how = f'pip install "cds-interpret[{extra}]"' if extra else f"pip install {pip_name or module}"
        raise MissingDependency(
            f"{feature[0].upper() + feature[1:]} needs the '{pip_name or module}' package, which isn't installed yet. "
            f"Install it with:\n    {how}\n(Python said: {e})"
        ) from e
