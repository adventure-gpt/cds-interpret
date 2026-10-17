"""Record your own RLGym v2 environment: your state mutator, your obs builder, your action parser - so the moments
carry exactly the inputs your bot saw in training.

    from rlgym.api import RLGym
    env = RLGym(...)                                       # the env you train in (any reward; it is ignored)
    moments = ci.record.rlgym(env, policy, episodes=20, out="recordings/rlgym")

Every agent of the env is played by ``policy`` (self-play). ``to_action`` turns the policy's choice into what your
action parser expects: by default an array holding the action index for a discrete policy (what RLGym's
LookupTableAction takes), or the 8 controls for a continuous one. Episodes end when your termination or truncation
condition says so; each is one game. Events come from the states: touches from ``ball_touches``, demolitions from
``is_demoed`` (credited to the car that bumped the victim, when known), goals from ``goal_scored``.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from ..frames import from_rlgym
from ..moments import Moments, check_out
from ..policy import Policy
from ._builder import Recording, empty_events


def _default_action(policy: Policy) -> Callable[[Any], Any]:
    if policy.discrete:
        return lambda a: np.asarray([int(a)])
    return lambda a: np.asarray(a, np.float32)


def rlgym(env: Any, policy: Policy, episodes: int = 10, every: int = 1, max_steps: int | None = None,
          deterministic: bool = True, tick_skip: int = 8, to_action: Callable[[Any], Any] | None = None,
          out: str | os.PathLike[str] | None = None, if_exists: str = "error",
          log: Callable[[str], Any] | None = print) -> Moments:
    """Run ``episodes`` episodes of your RLGym v2 env with ``policy`` playing every agent, and record them (a moment
    at every ``every``-th step of every agent). ``tick_skip``: the env's ticks per step (for the time labels).
    ``max_steps``: stop an episode after this many steps even if the env would go on."""
    import torch
    if out is not None and check_out(out, if_exists, recorder=True) and if_exists == "reuse":
        if log:
            log(f"  {out} already holds a recording: using it (if_exists=\"overwrite\" plays new episodes)")
        return Moments.load(out)
    to_action = to_action or _default_action(policy)
    step_seconds = tick_skip / 120
    rec: Recording | None = None
    t0 = time.time()
    gen = torch.Generator().manual_seed(0)
    for ep in range(episodes):
        obs = env.reset()
        step = 0
        while True:
            state = env.state
            agents = list(env.agents)
            car_ids = list(state.cars)
            fr = from_rlgym([state])
            if rec is None:
                rec = Recording(fr.n_cars, step_seconds, {"source": "rlgym", "policy": policy.name, "every": every,
                                                          "episodes": episodes, "step_seconds": step_seconds})
            row = {k: v[0] for k, v in fr.data.items() if k != "ev_touch"}
            ti = rec.add_step(ep, step, {**row, **empty_events(rec.c), "goal": -1, "time": step * step_seconds})
            x = np.stack([np.asarray(obs[a], np.float32).reshape(-1) for a in agents])
            with torch.no_grad():
                out_t = policy.forward(x)
                if policy.discrete:
                    lp = policy.log_probs_from_output(out_t)
                    act = lp.argmax(-1) if deterministic else torch.multinomial(lp.exp(), 1, generator=gen)[:, 0]
                    choice = act.cpu().numpy()
                else:
                    choice = out_t.float().clamp(-1, 1).cpu().numpy()
            if step % every == 0:
                for k, a in enumerate(agents):
                    rec.add_moment(ti, car_ids.index(a), x[k], out_t[k].float().cpu().numpy())
            obs, _, terminated, truncated = env.step({a: to_action(choice[k]) for k, a in enumerate(agents)})
            new = env.state
            ev = empty_events(rec.c)
            for j, cid in enumerate(car_ids):
                car0, car1 = state.cars[cid], new.cars.get(cid)
                if car1 is None:
                    continue
                ev["ev_touch"][j] = float(getattr(car1, "ball_touches", 0) > 0)
                if car1.is_demoed and not car0.is_demoed:
                    ev["ev_demoed"][j] = 1.0
                    for k2, cid2 in enumerate(car_ids):
                        if getattr(new.cars[cid2], "bump_victim_id", None) == cid:
                            ev["ev_demo_made"][k2] = 1.0
            goal = -1
            if getattr(new, "goal_scored", False):
                goal = int(new.scoring_team)
            rec.set_events(ti, ev["ev_touch"], ev["ev_demo_made"], ev["ev_demoed"], goal)
            step += 1
            if any(terminated.values()) or any(truncated.values()) or (max_steps is not None and step >= max_steps):
                break
        if log is not None:
            log(f"  episode {ep + 1}/{episodes}: {step} steps ({len(rec) if rec else 0} moments, "
                f"{time.time() - t0:.0f} s)")
    assert rec is not None
    m = rec.finish()
    if out is not None:
        m.save(out, if_exists="error" if if_exists == "reuse" else if_exists, renumber=True)
    return m
