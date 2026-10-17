# See what your bot learns between checkpoints

Looking inside a bot is most useful while it's still learning. Every new checkpoint raises the same questions: what
did it pick up since the last one? Is its network still healthy? And what should it practise next? Two modules answer
them, for any bot, from saved checkpoints and recorded games.

## What did it pick up? Ask every checkpoint the same questions

Keep **one** fixed set of recorded moments (a "bank" of a few tens of thousands, with their labels) and ask every
checkpoint what it would do there. Because the moments never change, any change in the answers comes from the bot, not
from a different sample of games.

```python
import torch
import cds_interpret as ci
from cds_interpret import emergence as E

bank = ci.Moments.load("recordings/bank")        # one fixed set of recorded moments, kept for every checkpoint
masks = E.situations(bank.labels)                # which moments are which situation


def choices(policy):
    """A checkpoint's action probabilities on the bank [N, A]."""
    with torch.no_grad():
        return policy.log_probs(policy.observe(bank.frames, bank.me)).exp().numpy()


policy = ci.Policy(net_now, obs=my_obs, actions=my_actions)            # this checkpoint
previous = ci.Policy(net_before, obs=my_obs, actions=my_actions)       # the one before it
probs = choices(policy)
d = E.dispositions(probs, policy.actions, masks)
d["demo_chance"]["p_boost"]                      # how often it boosts at an opponent in line for a demo
d["off_aerial"]["p_jump"]                        # how readily it jumps for an attacking ball in the air
E.kl_rows(probs, choices(previous)).mean()       # how far its choices moved since the previous checkpoint
```

(`net_now` and `net_before` are two checkpoints of your network loaded into fresh instances, and `my_obs` /
`my_actions` your input function and action table, as in [bring your own bot](bring-your-own-policy.md). A bank
recorded with RLBot's recorder doesn't need `my_obs`: use `bank.obs` in place of `policy.observe(...)`.)

The situations (`E.SITUATIONS` has every rule): a demo chance (and, for comparison, being fast with nobody to demo),
attacking and defending aerial chances, on a wall with the ball above, in the air with a flip near the ball, a 50/50,
carrying the ball, beaten to the ball, low on boost with nothing happening, the kickoff, and a ball heading into its
goal. The kinds of action come from your action table (`E.action_classes`): boost, jump, dodge, plain jump, powerslide,
reverse, coast.

## What does it do in games? Count its habits

```python
import numpy as np

games = ci.Moments.load("recordings/now").tracks         # every step of every game (RocketSim and RLBot recordings)
side_a = np.array([True, False])                          # which car slots are "A": slot 0, your bot in 1v1
habits = [E.game_habits(games.game_frames(g), side_a, step_seconds=games.step_seconds) for g in games.games]
print(E.summarize_habits(habits, "A", step_seconds=games.step_seconds))
```

```text
{'goal_share': {'v': 0.676, 'lo': ..., 'hi': ...}, 'open_goal_share': {'v': 0.92, ...}, ...}
```

For each game, `game_habits` counts what each side did: goals, demos made and taken, time supersonic, in the air, on
walls and the ceiling, touches in the air (split by half), touches of high balls, hard hits, aerial goals, flip
resets, ceiling drops, air dribbles, wavedash-like landings, kickoffs (and the goals scored within 5 seconds of one),
and ball contact. With labels it also counts demo chances and how many it took, and whether it gets between the ball
and its goal when a shot is coming.

`summarize_habits` turns a list of games into rates per game with error bars (from resampling the games).
`open_goal_share` leaves kickoff goals out: a kickoff routine that happens to work on one opponent can carry a goal
share on its own.

**What changed since last time?** `interval_changes` lists the habits whose error bars no longer overlap, biggest
change first - the short answer to "what did it pick up this time?":

```python
before = E.summarize_habits(habits_before, "A")      # game_habits of the previous checkpoint's games
after = E.summarize_habits(habits, "A")              # ... and of this one's
for c in E.interval_changes(before, after):
    print(f"{c['name']}: {c['old']:.3g} -> {c['new']:.3g} ({c['dir']})")
```

```text
demos_suffered_a_game: 0 -> 0.3 (up)
boost_used_a_car_min: 211 -> 271 (up)
```

(The outputs on this page are from the demo bot. Here "before" is its first ten games and "after" its last ten; below,
"previous" is a copy of it with its weights nudged a little.)

## Is its network still healthy?

On the same bank, for each hidden layer:

```python
acts = ci.Study(policy, bank).activations("body.3")             # the layer's values on the bank [N, D]
acts_before = ci.Study(previous, bank).activations("body.3")
E.layer_health(acts)                    # {'dead': 0.11, 'rare': 0.12, 'dormant': 0.12, 'erank': 19.0, ...}
E.linear_cka(acts, acts_before, standardize=True)               # 0.97: the layer barely changed
E.weight_norms(policy.net)              # {'body.0': 13.4, 'body.2': 20.0, 'head': 11.7, 'total': 26.8}
```

- `layer_health` - the share of units that are dead (never active), rarely active (under 1% of moments) or dormant
  (barely doing anything), how much of the layer a handful of units carry, and its **effective rank** ("erank":
  roughly, how many independent things the layer is using);
- `linear_cka` - a standard measure of how alike two sets of a layer's values are, from 0 (nothing alike) to 1 (the
  same up to rotation and scale): how much the layer changed since the last checkpoint. `standardize=True` keeps a few
  loud units from deciding it;
- `weight_norms` - how big the weights are getting, layer by layer.

A critic whose layers go dormant and lose rank while its weights keep growing is losing its ability to learn new
things. That's worth catching early.

**Is the critic still honest?** `E.first_goal_returns(goal, team, steps, gamma)` works out what actually came next
for every decision - who scored the next goal, counted for less the further away it was (the "discounted" outcome
your critic is trained to predict: `gamma` is your discount factor); `E.calibration(value, returns, groups=masks)` says how
well the critic's value tracks it, and in which situations it's too hopeful or too gloomy.

## What should it practise? `cds_interpret.curriculum`

**Where it wins and where it loses.** For every car of a recorded game, `curriculum.car_situation_rows(frames, car)`
lists the situations it's in every few steps - the ones above, plus ones that need the other cars (being dribbled at,
an opponent going up for an aerial, an opponent nearer the ball, being outnumbered) - and whether its team scored or
conceded soon after. `curriculum.situation_map(rows, games)` then gives, for each situation, how often the bot is in
it and its net result (chance of scoring soon minus chance of conceding soon), with error bars.

Do the same for its opponent. A situation the opponent creates often, where your bot does much worse than the
opponent does, is a good thing to practise.

**What happens just before it concedes.** `curriculum.enrichment(pre_concede_mask, columns)` says which labels,
situations or features show up more than usual in the seconds before a goal against. `curriculum.kmeans` groups those
moments into kinds of mistakes, and `curriculum.eta_squared` measures how much of a number (the critic's surprise, say)
a grouping explains - a quick way to compare two ways of describing a situation.

## Make it a habit

All of these are small, CPU-only functions. A tracker that runs them on every new checkpoint is a short script, and
the [API reference](../api.md) lists each one's arguments. To see *when* a probe's concept or an
autoencoder's feature appears during training, see `ci.tracking.track` ([probes](probes.md#across-training)) and
`cds_interpret.crosscoder`, one dictionary over the same layer of several checkpoints
([API reference](../api.md#sparse-autoencoders)).
