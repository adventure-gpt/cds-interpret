"""A tiny demo bot, so everything here can be tried end to end without anyone's weights.

    import cds_interpret as ci
    policy = ci.demo.policy()                    # a small MLP (30 -> 64 -> 64 -> 90 actions), weights shipped here
    policy = ci.demo.attention_policy()          # the same bot with an entity-attention block (3 tokens, 2 x 2 heads)
    moments = ci.demo.synthetic_moments()        # random game moments (no simulator needed)
    team = ci.demo.message_team()                # a toy team that talks (4 symbols) and listens - or listens=False
    ci.demo.run("cds-demo")                      # the whole tour: probes, features, an edit, a report

The demo bot was taught (by imitation, ``scripts/train_demo_policy.py``) to copy a simple scripted player,
:func:`teacher`: when the ball is heading into its goal, or it is on the wrong side of the ball, it drives back
toward its own goal; otherwise it drives at a point just behind the ball, lined up with the opponent's goal,
boosting on straight lines and jumping at a ball in the air nearby. So its network should know about threats to its
goal, which side of the ball it is on, and where the ball is - and the tour checks that it does.

Its observation, :func:`obs`, is built from the game state in the bot's team frame (it attacks +y): the ball (position
/ 4096, velocity / 2300), its own car (position, velocity, forward and up axes, boost / 100, on the ground, a flip
left), the nearest opponent (position, velocity) and the ball relative to its car. 30 numbers.

The attention demo bot, :func:`attention_policy`, learned the same scripted player from the same 30 numbers, cut into
three tokens - its own car, the ball, the opponent - that two small transformer layers let look at each other
(:class:`DemoAttentionNet`). It is there to try the attention-heads tools on (``cds-interpret attention --demo``).
"""

from __future__ import annotations

import importlib.resources
import os
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from ..frames import Frames, canon, from_arrays, team_sign
from ..labels import FIELD_Y, ball_path, goal_entry
from ..moments import Moments
from ..policy import Policy, lookup_table

OBS_DIM = 30
WEIGHTS = "demo_policy.pt"
ATTENTION_WEIGHTS = "demo_attention.pt"
TOKENS = ("own car", "ball", "opponent")


class DemoNet(torch.nn.Module):
    """The demo bot's network: ``body`` (Linear, ReLU, Linear, ReLU) and ``head`` (Linear to 90 action logits).
    Its hidden layers are ``body.1`` and ``body.3`` (after each ReLU)."""

    def __init__(self, obs_dim: int = OBS_DIM, hidden: int = 64, n_actions: int = 90):
        super().__init__()
        self.body = torch.nn.Sequential(torch.nn.Linear(obs_dim, hidden), torch.nn.ReLU(),
                                        torch.nn.Linear(hidden, hidden), torch.nn.ReLU())
        self.head = torch.nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


class DemoAttentionNet(torch.nn.Module):
    """The demo bot with an entity-attention block. The 30 inputs are cut into three tokens - ``own car`` (position,
    velocity, forward and up axes, boost, on the ground, a flip left: 15 numbers), ``ball`` (position, velocity, and
    where it is relative to the car: 9) and ``opponent`` (position, velocity: 6) - each embedded to ``d`` numbers by its
    own Linear; ``layers`` transformer layers (``nn.TransformerEncoderLayer``, pre-norm, ``heads`` heads) let the
    tokens look at each other, and the three tokens' outputs are read together into 90 action logits. Its attention
    layers are ``layers.0`` and ``layers.1``."""

    def __init__(self, d: int = 32, heads: int = 2, layers: int = 2, n_actions: int = 90):
        super().__init__()
        self.embed = torch.nn.ModuleList([torch.nn.Linear(15, d), torch.nn.Linear(9, d), torch.nn.Linear(6, d)])
        self.layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
            for _ in range(layers)])
        self.norm = torch.nn.LayerNorm(d)
        self.head = torch.nn.Sequential(torch.nn.Linear(3 * d, 64), torch.nn.ReLU(), torch.nn.Linear(64, n_actions))

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """The three tokens [N, 3, d] of inputs x [N, 30], in the order of TOKENS."""
        parts = (x[:, 6:21], torch.cat([x[:, 0:6], x[:, 27:30]], 1), x[:, 21:27])
        return torch.stack([e(p) for e, p in zip(self.embed, parts)], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.tokens(x)
        for layer in self.layers:
            t = layer(t)
        return self.head(self.norm(t).flatten(1))


def obs(frames: Frames, me: np.ndarray) -> np.ndarray:
    """The demo bot's input [N, 30] for car slot me[i] of moment i (see the module's notes)."""
    me = np.asarray(me, np.int64)
    rows = np.arange(len(frames))
    s = team_sign(frames.car_team[rows, me])
    ball = canon(frames.ball_pos, s)
    bvel = canon(frames.ball_vel, s)
    pos = canon(frames.car_pos[rows, me], s)
    vel = canon(frames.car_vel[rows, me], s)
    fwd = canon(frames.car_fwd[rows, me], s)
    up = canon(frames.car_up[rows, me], s)
    # the nearest opponent (zeros when there is none)
    opp_mask = (frames.car_team != frames.car_team[rows, me][:, None]) & (frames.car_present > 0.5)
    d = np.linalg.norm(frames.car_pos - frames.car_pos[rows, me][:, None], axis=-1)
    d = np.where(opp_mask, d, np.inf)
    j = np.argmin(d, axis=1)
    has = np.isfinite(d[rows, j])
    opos = np.where(has[:, None], canon(frames.car_pos[rows, j], s), 0.0)
    ovel = np.where(has[:, None], canon(frames.car_vel[rows, j], s), 0.0)
    parts = [ball / 4096, bvel / 2300, pos / 4096, vel / 2300, fwd, up,
             frames.car_boost[rows, me][:, None] / 100, frames.car_on_ground[rows, me][:, None],
             frames.car_has_flip[rows, me][:, None], opos / 4096, ovel / 2300, (ball - pos) / 4096]
    return np.concatenate(parts, axis=1).astype(np.float32)


def teacher(frames: Frames, me: np.ndarray) -> np.ndarray:
    """The scripted player the demo bot learned from: controls [N, 8] (see the module's notes)."""
    me = np.asarray(me, np.int64)
    rows = np.arange(len(frames))
    s = team_sign(frames.car_team[rows, me])
    ball = canon(frames.ball_pos, s)
    pos = canon(frames.car_pos[rows, me], s)
    fwd = canon(frames.car_fwd[rows, me], s)
    times = np.arange(1, 46) / 15.0                       # every 8 ticks out to 3 s: enough for a threat
    threat, _, _, _ = goal_entry(ball_path(frames.ball_pos, frames.ball_vel, times, dt=1 / 60), times, s)
    wrong_side = pos[:, 1] > ball[:, 1] + 200
    defend = threat | wrong_side
    # where to drive: back toward our goal, or just behind the ball, lined up with their goal
    to_goal = np.stack([-ball[:, 0], FIELD_Y - ball[:, 1]], 1)
    to_goal /= np.maximum(np.linalg.norm(to_goal, axis=1, keepdims=True), 1e-6)
    behind = ball[:, :2] - 250.0 * to_goal
    home = np.stack([0.4 * ball[:, 0], np.full(len(rows), -FIELD_Y + 600.0)], 1)
    target = np.where(defend[:, None], home, behind)
    d = target - pos[:, :2]
    err = np.arctan2(d[:, 1], d[:, 0]) - np.arctan2(fwd[:, 1], fwd[:, 0])
    err = (err + np.pi) % (2 * np.pi) - np.pi
    dist = np.linalg.norm(d, axis=1)
    on_ground = frames.car_on_ground[rows, me] > 0.5
    out = np.zeros((len(rows), 8), np.float32)
    out[:, 0] = 1.0
    out[:, 1] = np.clip(3.0 * err, -1, 1)
    out[:, 3] = out[:, 1]
    out[:, 6] = ((np.abs(err) < 0.35) & (dist > 900) & on_ground & (frames.car_boost[rows, me] > 0)).astype(np.float32)
    near_high = (ball[:, 2] > 220) & (np.linalg.norm(ball[:, :2] - pos[:, :2], axis=1) < 350) & on_ground
    out[:, 5] = (near_high & ~defend).astype(np.float32)
    out[:, 7] = (np.abs(err) > 1.9).astype(np.float32)
    return out


def nearest_actions(controls: np.ndarray, table: np.ndarray) -> np.ndarray:
    """The index of the table row closest to each row of controls (jump and boost weighted up)."""
    wts = np.array([1.0, 1.0, 0.5, 0.5, 0.5, 3.0, 2.0, 1.0], np.float32)
    d = (((controls[:, None, :] - table[None]) * wts) ** 2).sum(-1)
    return d.argmin(1)


def random_frames(n: int, rng: np.random.Generator, n_cars: int = 2) -> Frames:
    """n random 1v1 moments that look roughly like play: cars mostly on the ground, the ball rolling or flying."""
    half = n_cars // 2
    team = np.tile(np.r_[np.zeros(half), np.ones(n_cars - half)], (n, 1)).astype(np.float32)
    car_pos = rng.uniform([-3800, -4900, 17], [3800, 4900, 17], (n, n_cars, 3))
    air = rng.random((n, n_cars)) < 0.12
    car_pos[..., 2] = np.where(air, rng.uniform(100, 900, (n, n_cars)), 17.0)
    yaw = rng.uniform(-np.pi, np.pi, (n, n_cars))
    car_fwd = np.stack([np.cos(yaw), np.sin(yaw), np.zeros_like(yaw)], -1)
    speed = rng.uniform(0, 2300, (n, n_cars))
    drift = rng.normal(0, 0.25, (n, n_cars))
    car_vel = np.stack([np.cos(yaw + drift), np.sin(yaw + drift), np.zeros_like(yaw)], -1) * speed[..., None]
    ball_pos = rng.uniform([-3800, -4800, 93], [3800, 4800, 93], (n, 3))
    flying = rng.random(n) < 0.4
    ball_pos[:, 2] = np.where(flying, rng.uniform(150, 1600, n), 93.0)
    ball_vel = rng.normal(0, 1200, (n, 3))
    ball_vel[:, 2] = np.where(flying, rng.normal(0, 600, n), 0.0)
    kick = rng.random(n) < 0.03
    ball_pos[kick] = [0.0, 0.0, 93.15]
    ball_vel[kick] = 0.0
    return from_arrays(ball_pos=ball_pos, ball_vel=ball_vel, car_pos=car_pos, car_vel=car_vel, car_fwd=car_fwd,
                       car_up=np.tile([0.0, 0.0, 1.0], (n, n_cars, 1)), car_boost=rng.uniform(0, 100, (n, n_cars)),
                       car_on_ground=(~air).astype(np.float32), car_team=team,
                       car_has_flip=np.ones((n, n_cars), np.float32))


def train(steps: int = 4000, n: int = 60_000, seed: int = 0, log: Callable[[str], Any] | None = None,
          attention: bool = False) -> torch.nn.Module:
    """Teach a fresh DemoNet (``attention``: a DemoAttentionNet) to copy :func:`teacher` on random moments
    (cross-entropy on the nearest action)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    fr = random_frames(n, rng)
    me = rng.integers(0, 2, n)
    x = torch.from_numpy(obs(fr, me))
    table = lookup_table()
    y = torch.from_numpy(nearest_actions(teacher(fr, me), table))
    net: torch.nn.Module = DemoAttentionNet() if attention else DemoNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3 if attention else 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = torch.from_numpy(rng.integers(0, n, 1024))
        loss = torch.nn.functional.cross_entropy(net(x[idx]), y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if log is not None and (step % 500 == 0 or step == 1):
            with torch.no_grad():
                acc = float((net(x[:20000]).argmax(-1) == y[:20000]).float().mean())
            log(f"  step {step}/{steps}: loss {loss.item():.3f}, copies the teacher on {acc:.0%} ({time.time() - t0:.0f} s)")
    return net.eval()


def policy(weights: str | os.PathLike[str] | None = None) -> Policy:
    """The demo bot as a Policy (its shipped weights, or ``weights`` from train())."""
    net = DemoNet()
    if weights is None:
        ref = importlib.resources.files(__package__).joinpath(WEIGHTS)
        with importlib.resources.as_file(ref) as p:
            state = torch.load(p, map_location="cpu", weights_only=True)
    else:
        state = torch.load(weights, map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    return Policy(net.eval(), obs=obs, actions=lookup_table(), name="demo bot")


def attention_policy(weights: str | os.PathLike[str] | None = None) -> Policy:
    """The attention demo bot as a Policy (its shipped weights, or ``weights`` from train(attention=True)). Its tokens
    are named by TOKENS: ``study.attention(tokens=ci.demo.TOKENS)``."""
    net = DemoAttentionNet()
    if weights is None:
        ref = importlib.resources.files(__package__).joinpath(ATTENTION_WEIGHTS)
        with importlib.resources.as_file(ref) as p:
            state = torch.load(p, map_location="cpu", weights_only=True)
    else:
        state = torch.load(weights, map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    return Policy(net.eval(), obs=obs, actions=lookup_table(), name="demo bot with attention")


def demo_moments(policy: Policy, source: str = "auto", games: int = 12, seconds: float = 60.0,
                 log: Callable[[str], Any] | None = print) -> Moments:
    """Moments of a demo bot's play: recorded in RocketSim against a ball chaser when it and its meshes are there
    (source "auto" or "rocketsim"), else random game moments (source "synthetic", or "auto" without RocketSim)."""
    say = log or (lambda *_: None)
    if source not in ("auto", "rocketsim", "synthetic"):
        raise ValueError(f"source must be 'auto', 'rocketsim' or 'synthetic', not {source!r}")
    if source in ("auto", "rocketsim"):
        try:
            from ..record import init_rocketsim, rocketsim
            init_rocketsim()
            say(f"Recording {games} games ({seconds:.0f} s each) against a ball chaser in RocketSim")
            return rocketsim(policy, games=games, seconds=seconds, opponent="chaser", log=say)
        except Exception as e:
            if source == "rocketsim":
                raise
            say(f"RocketSim isn't ready here ({str(e).splitlines()[0]}) - using random game moments instead. To "
                "record real games: pip install \"cds-interpret[rlgym]\" (see docs/howto/record-moments.md)")
    m = synthetic_moments()
    say(f"  {len(m)} random moments in {m.n_games} groups (good for a tour, not for findings)")
    return m


def synthetic_moments(n_games: int = 60, per_game: int = 60, seed: int = 1) -> Moments:
    """Random 1v1 moments in "games" of ``per_game`` moments each, seen by the blue car, with their labels. The
    moments of a game are not consecutive in time (they are independent draws): fine for probes and features, not
    for anything that reads time (jitter, future labels, clips)."""
    rng = np.random.default_rng(seed)
    n = n_games * per_game
    fr = random_frames(n, rng)
    me = np.zeros(n, np.int64)
    m = Moments(fr, me, np.repeat(np.arange(n_games), per_game), np.tile(np.arange(per_game), n_games),
                meta={"source": "synthetic", "seed": seed})
    return m.with_labels()


# ------------------------------------------------------------------------------------------- the talking toy team
MESSAGE_RULE = ("closest, ball in their half", "closest, ball in our half", "a team-mate is closer, I'm last back",
                "a team-mate is closer, I'm up-field")
"""What each symbol of the toy team's message means (message_team): the sender's situation."""


class MessageNet(torch.nn.Module):
    """The toy team's receiver: 8 numbers about its own situation and its team-mate's symbol (a one-hot of 4) -> 6
    action logits. With ``listens`` each symbol pushes one action up (symbol s: action s + 2); without, the symbol's
    weights are zero and the team-mate's message changes nothing."""

    def __init__(self, listens: bool = True, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.own = torch.nn.Linear(8, 6)
        self.msg = torch.nn.Linear(4, 6, bias=False)
        with torch.no_grad():
            self.own.weight.copy_(torch.randn(6, 8, generator=g))
            self.own.bias.zero_()
            self.msg.weight.zero_()
            if listens:
                for sym in range(4):
                    self.msg.weight[sym + 2, sym] = 3.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.own(x[:, :8]) + self.msg(x[:, 8:12])


def _message_rule(fr: Frames, me: np.ndarray) -> np.ndarray:
    """The toy sender's symbol (MESSAGE_RULE) for car slot me[i] of each moment."""
    from ..messages import situations
    sit = situations(fr, me)
    rows = np.arange(len(fr))
    team = fr.car_team[rows, me]
    y = canon(fr.car_pos, team_sign(team)[:, None])[..., 1]
    mates = fr.car_team == team[:, None]
    back = y[rows, me] <= np.where(mates, y, np.inf).min(1) + 1e-6
    return np.where(sit["closest"] > 0.5, np.where(sit["ball_half"] > 0.5, 0, 1), np.where(back, 2, 3))


def message_team(n_games: int = 40, per_game: int = 150, listens: bool = True, noise: float = 1.0,
                 seed: int = 0) -> Any:
    """A toy 2v2 team that talks, for trying the message-channel tools (``study.messages``) in seconds.

    Every car sends one of 4 symbols that says where it stands (MESSAGE_RULE): its message logits are 3 for the
    rule's symbol plus noise (``noise``), so most decisions send it. Its team-mate's network, :class:`MessageNet`,
    reads the symbol (the input's last 4 numbers) and, with ``listens``, plays differently for each one. The moments
    are independent random 2v2 states (synthetic_moments' kind), seen by a blue car.

    Returns a namespace: ``moments`` (with ``obs``, the 12 inputs), ``logits`` [N, 4] (what each seen car sends),
    ``policy`` (the receiver as a Policy), ``channel`` (messages.Channel: the symbol's 4 columns) and ``rule``."""
    from types import SimpleNamespace

    from ..messages import Channel, situations
    rng = np.random.default_rng(seed)
    n = n_games * per_game
    fr = random_frames(n, rng, n_cars=4)
    me = rng.integers(0, 2, n)
    mate = 1 - me
    sent = _message_rule(fr, me)
    heard = _message_rule(fr, mate)
    logits = 3.0 * np.eye(4)[sent] + rng.normal(0.0, noise, (n, 4))
    sit = situations(fr, me)
    rows = np.arange(n)
    team = fr.car_team[rows, me]
    sign = team_sign(team)
    bpos, cpos = canon(fr.ball_pos, sign), canon(fr.car_pos[rows, me], sign)
    own = np.stack([bpos[:, 0] / 4096, bpos[:, 1] / 5120, cpos[:, 0] / 4096, cpos[:, 1] / 5120,
                    fr.car_boost[rows, me] / 100, sit["closest"], sit["ball_half"], np.ones(n)], 1)
    x = np.concatenate([own, np.eye(4)[heard]], 1).astype(np.float32)
    m = Moments(fr, me, np.repeat(np.arange(n_games), per_game), np.tile(np.arange(per_game), n_games), obs=x,
                meta={"source": "synthetic", "seed": seed, "demo": "message_team"})
    m.with_labels()
    pol = Policy(MessageNet(listens).eval(), name="toy team" + ("" if listens else " (deaf)"))
    return SimpleNamespace(moments=m, logits=logits.astype(np.float32), policy=pol,
                           channel=Channel(symbols=4, columns=[range(8, 12)]), rule=MESSAGE_RULE, sent=sent)


def run(out: str | os.PathLike[str] = "cds-demo", source: str = "auto", games: int = 12, seconds: float = 60.0,
        log: Callable[[str], Any] | None = print) -> dict[str, Any]:
    """The whole tour on the demo bot: moments (recorded in RocketSim when it and its meshes are there, else
    synthetic), probes on every layer against the baselines, a sparse autoencoder and its features, edits, and an
    HTML report. Returns a summary. source: "auto", "rocketsim" or "synthetic"."""
    from .. import intervene
    from ..study import Study
    say = log or (lambda *_: None)
    pol = policy()
    if source not in ("auto", "rocketsim", "synthetic"):
        raise ValueError(f"source must be 'auto', 'rocketsim' or 'synthetic', not {source!r}")
    say("The demo bot is a tiny network that learned to copy a simple scripted player. Let's look inside it.\n")
    moments = None
    if source in ("auto", "rocketsim"):
        try:
            from ..record import init_rocketsim, rocketsim
            init_rocketsim()
            say(f"Step 1 of 4 - recording {games} games ({seconds:.0f} s each) against a ball chaser in RocketSim")
            moments = rocketsim(pol, games=games, seconds=seconds, opponent="chaser", log=say)
        except Exception as e:
            if source == "rocketsim":
                raise
            say(f"Step 1 of 4 - RocketSim isn't ready here ({str(e).splitlines()[0]})\n"
                "  No problem: using random game moments instead. To record real games, install the rlgym extra:\n"
                "  pip install \"cds-interpret[rlgym]\"  (see docs/howto/record-moments.md)")
    if moments is None:
        moments = synthetic_moments()
        say(f"  {len(moments)} random moments in {moments.n_games} groups (good for a tour, not for findings)")
    say("\nStep 2 of 4 - asking every layer questions about the game (probes)")
    study = Study(pol, moments, log=lambda m: say("  " + m.strip()))
    probes = study.probes()
    say(probes.summary(top=8))
    say("\nStep 3 of 4 - finding the separate ideas in its last hidden layer (a sparse autoencoder)")
    feats = study.features(study.layers[-1], steps=1500)
    say(feats.summary(top=6))
    say("\nStep 4 of 4 - switching ideas off to see whether the bot uses them (edits)")
    top = feats.top(12)
    pick = next((j for j in top if feats.guess(j, 1) and feats.guess(j, 1)[0][0] == "threat"), top[0] if top else None)
    if pick is not None:
        c = study.check_feature(feats.layer, pick, scales=(0.0,))
        say(f"  feature {pick} ({feats.name(pick)}) switched off: the bot's choice changes on "
            f"{c.results[0.0]['changed']:.0%} of the {c.n} held-out moments it fires on")
    best = probes.best("goal_side")
    at = best[0] if best is not None else feats.layer
    edits = {}
    if best is not None:
        edits[f"'goal_side' probe direction removed at {at}"] = (at, [study.probe_edit(at, "goal_side", 0.0)])
    gen = torch.Generator().manual_seed(0)
    width = study.activations(at).shape[1]
    edits[f"a random direction removed at {at} (the control; 3 tries)"] = (
        at, [intervene.project_out(torch.randn(width, generator=gen)) for _ in range(3)])
    edits[f"the autoencoder's rebuild in place of {feats.layer}"] = (feats.layer, [intervene.replace_with(feats.sae)])
    for name, (layer, tries) in edits.items():
        effs = [study.edits(layer, {f"{name} #{i + 1}" if len(tries) > 1 else name: e})
                for i, e in enumerate(tries)]
        changed = float(np.mean([next(iter(r.values()))["argmax_changed"] for r in effs]))
        kl = float(np.mean([next(iter(r.values()))["kl"] for r in effs]))
        say(f"  {name}: the bot's choice changes on {changed:.0%} of moments (KL {kl:.3f})")
    path = study.report(os.path.join(os.fspath(out), "report.html"), title="The demo bot, inside")
    say("\nAll done. The full picture - every probe, the features with their moments drawn on the field, the edits - "
        f"is in\n{path}")
    return {"moments": len(moments), "source": moments.meta.get("source"), "report": str(path),
            "layers": study.layers, "learned": probes.learned(), "edits": study.edit_results}
