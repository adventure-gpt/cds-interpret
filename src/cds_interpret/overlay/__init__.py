"""The live "thoughts" overlay: what your bot's network is noticing, drawn in the real game while it plays.

=================================  ==============================================================================
make_pack / save_pack / load_pack  what to show from one layer: named SAE features, probes, the action table
add_land                           a landing marker read from its own (often earlier) layer (team, mirrored or car frame)
car_frame_offsets / land_point     the targets of a car-frame landing readout; a reading put back on the field
add_decoders / add_donors          what "why this move" and "looking at" need (decoder directions, real moments)
add_input_donors                   ... recorded network inputs, for a policy that splits its own input (codec)
Thoughts                           reads a pack from a layer's values (pure numpy)
with_thoughts(BotClass, pack)      your RLBot v5 bot class with the overlay (needs ``cds-interpret[rlbot]``)
PanelConfig / Panel                how the panel looks: rows, names, colours, markers, team modes (a TOML file)
Follow / landing_cars              team modes: whose thoughts the panel shows, which cars get a landing marker
Explainer / try_readings           "why this move" and "looking at" for one decision; trying them on a pack
watch_thoughts(policy, pack, cars) a display-only RLBot script: the thoughts of a bot you can't modify
draw_thoughts                      the panel itself, for your own drawing code
fix_socket_reads                   rlbot before 2.0.0b56: read each message from RLBotServer whole (the overlay,
                                   the watch script and the recorder call it themselves)
=================================  ==============================================================================

Display only: read-only hooks, a low-priority worker process, everything after your controls are sent. See
docs/howto/overlay.md.
"""

from .._rlbot import fix_socket_reads
from .bot import with_thoughts
from .explain import Explainer, load_policy, try_readings
from .pack import (
                   Thoughts,
                   add_decoders,
                   add_donors,
                   add_input_donors,
                   add_land,
                   car_frame_offsets,
                   car_frame_point,
                   heading,
                   land_point,
                   load_pack,
                   make_pack,
                   save_pack,
)
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
from .team import Follow, landing_cars
from .watch import match_cars, watch_thoughts
from .worker import WorkerLink, compute

__all__ = [
    "Explainer",
    "FeatureStyle",
    "Follow",
    "Markers",
    "Panel",
    "PanelConfig",
    "RowDrawer",
    "RowSpec",
    "Thoughts",
    "WorkerLink",
    "add_decoders",
    "add_donors",
    "add_input_donors",
    "add_land",
    "car_frame_offsets",
    "car_frame_point",
    "check_panel",
    "compute",
    "draw_thoughts",
    "fix_socket_reads",
    "heading",
    "land_point",
    "landing_cars",
    "load_pack",
    "load_policy",
    "make_pack",
    "match_cars",
    "save_pack",
    "smooth_point",
    "try_readings",
    "watch_thoughts",
    "with_thoughts",
]
