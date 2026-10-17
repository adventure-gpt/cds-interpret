"""Recorders: games of your bot turned into :class:`~cds_interpret.Moments`.

==================================  ===========================================================================
record.rocketsim(policy, ...)       play your Policy in RocketSim against itself, another Policy or a scripted
                                    player (module :mod:`cds_interpret.record.sim`)
record.with_recorder(BotClass)      wrap your RLBot v5 bot class: it records the real game while it plays,
                                    unchanged (module :mod:`cds_interpret.record.rlbot`)
record.rlgym(env, policy, ...)      run your own RLGym v2 env (your obs builder, your action parser) and record it
                                    (module :mod:`cds_interpret.record.rlgym_env`)
==================================  ===========================================================================

Synthetic moments for quick tries: :func:`cds_interpret.demo.synthetic_moments`.
"""

from . import scripted
from .rlbot import with_recorder
from .rlgym_env import rlgym
from .sim import find_meshes, init_rocketsim, rocketsim

__all__ = ["find_meshes", "init_rocketsim", "rlgym", "rocketsim", "scripted", "with_recorder"]
