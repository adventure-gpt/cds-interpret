"""The live "thoughts" overlay: what your bot's network is noticing, drawn in the real game while it plays.

=================================  ==============================================================================
make_pack / save_pack / load_pack  what to show from one layer: named SAE features, probes, the action table
add_land                           a landing marker read from its own (often earlier) layer
Thoughts                           reads a pack from a layer's values (pure numpy)
with_thoughts(BotClass, pack)      your RLBot v5 bot class with the overlay (needs ``cds-interpret[rlbot]``)
PanelConfig / Panel                how the panel looks: rows, names, colours, markers (a TOML file), and hooks
draw_thoughts                      the panel itself, for your own drawing code
=================================  ==============================================================================

Display only: read-only hooks, a low-priority worker process, everything after your controls are sent. See
docs/howto/overlay.md.
"""

from .bot import with_thoughts
from .pack import Thoughts, add_land, load_pack, make_pack, save_pack
from .panel import (
    FeatureStyle,
    Markers,
    Panel,
    PanelConfig,
    RowDrawer,
    RowSpec,
    check_panel,
    draw_thoughts,
    smooth_point,
)
from .worker import WorkerLink, compute

__all__ = [
    "FeatureStyle",
    "Markers",
    "Panel",
    "PanelConfig",
    "RowDrawer",
    "RowSpec",
    "Thoughts",
    "WorkerLink",
    "add_land",
    "check_panel",
    "compute",
    "draw_thoughts",
    "load_pack",
    "make_pack",
    "save_pack",
    "smooth_point",
    "with_thoughts",
]
