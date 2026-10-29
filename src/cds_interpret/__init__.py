"""cds-interpret: look inside any PyTorch Rocket League bot.

The interpretability toolkit from the Cat Distribution System, on its own. It answers three questions about a bot:
what does it *know* (can "the ball is about to go in my goal" be read from its network?), what *features* has its
network learned, and what *happens* when you change one of them - and it can show the answers live, in the real
game, while the bot plays.

    import cds_interpret as ci

    policy = ci.Policy(net, obs=my_obs, actions=my_table)     # bring your own network (or ci.demo.policy())
    moments = ci.record.rocketsim(policy, games=20)           # or record the real game, or your RLGym env
    study = ci.Study(policy, moments)
    print(study.probes().summary())                           # what each layer knows, against baselines
    study.features("mlp.3")                                   # a sparse autoencoder's features of a layer
    study.attention(tokens=["own car", "ball", "opponent"])   # what each attention head looks at
    study.messages(logits=msg_logits, channel=channel)        # a team-mate message channel: meaning, listening
    study.report("report.html")

Three tiers, by what you have of the bot - every tier works for any bot:

1. **Watch** (game state only): concept labels, habits, drawings - for any bot you can watch.
2. **Open** (its PyTorch network): probes, sparse-autoencoder features, edits, reports, the live overlay.
3. **Run** (it plays in RocketSim or your RLGym env): record its own play at will, causal tests in whole games,
   concepts followed across training snapshots.

Start with docs/quickstart.md.
"""

from . import (
    analysis,
    attention,
    behavior,
    curriculum,
    demo,
    emergence,
    frames,
    hooks,
    intervene,
    labels,
    messages,
    names,
    overlay,
    probes,
    record,
    refit,
    report,
    rlgym,
    sae,
    store,
    tracking,
    viz,
)
from ._version import __version__
from .errors import CDSInterpretError, LayerNotFound, MeshesNotFound, MissingDependency, ShapeError
from .frames import Frames, from_arrays, from_rlbot_packet, from_rlgym
from .hooks import LayerTap, list_layers, run_layers, sequential_layers
from .labels import LABEL_KIND, concept_labels, describe_label, future_labels, register_label, short_label
from .moments import Moments, Tracks
from .policy import Policy, describe_action, eager_copy, lookup_table
from .study import FeatureSet, ProbeResults, Study

__all__ = [
    "LABEL_KIND",
    "CDSInterpretError",
    "FeatureSet",
    "Frames",
    "LayerNotFound",
    "LayerTap",
    "MeshesNotFound",
    "MissingDependency",
    "Moments",
    "Policy",
    "ProbeResults",
    "ShapeError",
    "Study",
    "Tracks",
    "__version__",
    "analysis",
    "attention",
    "behavior",
    "concept_labels",
    "curriculum",
    "demo",
    "describe_action",
    "describe_label",
    "eager_copy",
    "emergence",
    "frames",
    "from_arrays",
    "from_rlbot_packet",
    "from_rlgym",
    "future_labels",
    "hooks",
    "intervene",
    "labels",
    "list_layers",
    "lookup_table",
    "messages",
    "names",
    "overlay",
    "probes",
    "record",
    "refit",
    "register_label",
    "report",
    "rlgym",
    "run_layers",
    "sae",
    "sequential_layers",
    "short_label",
    "store",
    "tracking",
    "viz",
]
