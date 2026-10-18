# Why this move, and what it's looking at

Two rows of the panel answer the question you ask most while watching a bot: *why did it do that?*

<p align="center"><img src="../../assets/overlay-panel.svg" alt="The thoughts panel with its 'Why this move' and 'Looking at' rows" width="300"></p>

Read the middle of that panel like this:

```text
Next move                     81%
  throttle, steer right, boost         <- what it's about to do
Why this move
 + lined up for a shot                 <- the named idea pushing hardest toward that move (amber: a strong push)
 + ball on our side                    <- another one pushing toward it
 - low on boost                        <- the one pushing hardest against it
Looking at
 [■■■■■■ car ■■■■■|■■■■ ball ■■■■|opp]
 car 46  ball 38  opp 16               <- what the move rests on
```

**Why this move** names the ideas - your [named features](name-features.md) - that push the bot toward the move it's
about to make, strongest first, and the one pushing hardest against it. Here, "lined up for a shot" is the main
reason it steers right and boosts; "low on boost" argues against boosting.

**Looking at** says what that move rests on: about half on its own car (where it is, how fast, which way it faces,
its boost), about two fifths on the ball, and a little on the opponent. In 2v2 and 3v3 a fourth part, the team-mates,
joins the bar.

Both are worked out for the car the panel shows, live, in the overlay's helper process - never in your bot's.

## Five minutes on the demo bot

The demo bot in `examples/rlbot_demo_bot` has both rows switched on: its pack has what they need and its `bot.py`
passes the `policy=` they use. Add `thoughts.bot.toml` to RLBot, turn rendering on and start a match (see
[the overlay guide](overlay.md#try-it-on-the-demo-bot-first)).

The demo bot is a tiny network that copies a scripted player, so expect its moves to rest mostly on its own car and
the ball, with the opponent close to 0% - the scripted player never looks at the opponent. Some of its named features
are named after what they make it do ("wants to steer right"): those are the ones "why this move" shows most.

## On your own bot

**1. A pack that has what they need.** Packs made by `Study.make_pack` have it already: each named feature's
*decoder direction* (for "why this move") and a few dozen real recorded moments (for "looking at"). For a pack made
before version 0.2, add them:

```python
import cds_interpret as ci

pack = ci.overlay.load_pack("thoughts.npz")
ci.overlay.add_decoders(pack, study.load_features("body3.sae").sae)   # the dictionary the features came from
ci.overlay.add_donors(pack, ci.Moments.load("recordings/my-bot"))     # recorded games of your bot (any recorder)
ci.overlay.save_pack(pack, "thoughts.npz")
```

`add_donors` keeps up to 32 moments per number of cars (kickoffs left out) - only the game state, a few kilobytes,
nothing of your network. For 2v2 and 3v3, add recordings of those modes too (pass a list of Moments): "looking at"
compares a 2v2 moment only with 2v2 moments.

**2. A small function that builds your bot's policy.** The two rows need your network itself, so the overlay's helper
process builds its own copy from a function you name. Put it in a module in your bot's folder:

```python
# my_bot/interp.py
import torch
import cds_interpret as ci
from my_bot.model import MyNet, my_obs, MY_ACTIONS

def make_policy():
    net = MyNet()
    net.load_state_dict(torch.load("model.pt", map_location="cpu", weights_only=True))
    return ci.Policy(net, obs=my_obs, actions=MY_ACTIONS)        # the same Policy you studied the bot with
```

It must be the same network your bot runs (the pack names its layers by their paths in it), with your obs function
for "looking at" ([bring your own bot](bring-your-own-policy.md) explains `ci.Policy`).

**3. Pass it to the overlay:**

```python
with_thoughts(MyBot, pack="thoughts.npz", network="policy", policy="my_bot.interp:make_policy")("me/my-bot").run(
    wants_ball_predictions=True)
```

**4. Check it before a match:**

```bash
cds-interpret check-panel panel.toml thoughts.npz --policy my_bot.interp:make_policy
```

```text
  why this move: works, 0.7 ms a reading on one thread
  looking at: works, 1.8 ms a reading on one thread
panel.toml is fine, and everything it shows is in thoughts.npz.
```

(That's the demo bot. `cds-interpret panel panel.toml` writes a panel file if you don't have one yet.) If something is
missing, it says what and how to add it.

## How to read "Why this move"

Each line is a named feature and its **push**: how much it raises the chance of this exact move. Pushes are in the
network's own units (log-probability): **+0.7 roughly doubles** the move's chance, +0.15 - the smallest the row shows
by default - makes it about a sixth more likely. A push of 0.6 or more is drawn in amber: that feature is driving the
move. The `-` line is the feature pushing hardest *against* the move; it appears only when that push is strong too.

A few things to keep in mind:

- **Only named features take part.** Your network has many more features than you named, and they push too. "no
  named idea" means none of *your named ones* pushed this move - not that nothing did.
- **It's per decision.** Like "Noticing", a feature stays listed a moment (`hold`, 0.6 s) after its push drops, so
  the list doesn't flicker.
- **It's an estimate.** The push is the feature's size times how much the move's chance would change if the layer
  moved along that feature's direction - the first-order estimate called *attribution patching*. On a large
  self-play bot it ranked a decision's features almost exactly like actually removing them one at a time (correlation
  0.96), at a fraction of the cost. Features that only matter together can be under- or over-stated.
- **The word above the car** shows the feature driving the move hardest (`[markers] car_label_from = "why"`), or the
  first one firing when none drives it. Set `car_label_from = "noticing"` for the first one firing, always.

What it computes, for the curious: with z the feature's activation, d its decoder direction, and the gradient of
log p(move) taken at the pack's layer,

```text
push = z * (d . gradient)
```

The gradient comes from running your network once more on the decision's own input, with the layer's values replaced
by a copy that can carry a gradient. [What a decision rests on](what-decisions-rest-on.md#what-each-feature-makes-the-bot-do)
has the exact version of this (removing each feature) for recorded moments.

## How to read "Looking at"

The bar splits the move between the parts of the game: **car** (the bot's own car: position, speed, direction,
boost, its last controls), **ball**, **opp** (the opponents) and, in team modes, **mates** (the team-mates). The
percentages add up to 100.

How it's worked out: each part of the live moment is either kept or swapped for the same part of a real recorded
moment, every combination is run through your bot's own obs function and network (8 combinations in 1v1, 16 in team
modes, all in one batch), and the move's probability over them is split fairly between the parts with Shapley values.
It's done against 3 recorded moments and averaged. A part that pushes *against* the move counts as much as one that
pushes for it: the bar shows how much the move depends on each part, not which way.

Read it as "if this part of the game were different, how much would this move change?" - not as where the bot's
eyes are. It's the live, single-move version of the [entity attribution](what-decisions-rest-on.md#ball-car-or-opponents)
you can run on recorded games, where you can also check how far to trust the split on your bot (its faithfulness
score).

It's the heaviest thing the overlay does, so it's worked out every 4th reading (a few times a second) and the bar
keeps its last reading in between.

## What they cost

Both run in the overlay's helper process, at low priority with one thread, for the panel's car only - your bot never
waits for them, and a busy helper simply skips a reading.

| | what it runs | demo bot (12,000 weights) | a 21-million-weight bot |
|---|---|---|---|
| why this move | your network once more, forward and back, on one input | under 1 ms | about 2 ms |
| looking at | your obs function and network on 24 inputs (1v1) or 48 (team modes), in one batch | about 2 ms | 20-40 ms |

`cds-interpret check-panel ... --policy ...` measures both on your machine.

## Turning them off, or tuning them

They're two rows like any other. To turn one off, leave it out of your panel file's `[[rows]]` - then it isn't even
computed. Without `policy=`, both stay hidden and cost nothing.

```toml
[[rows]]
kind = "why"
label = "Why this move"
max = 2                  # how many features pushing the move at most
min_push = 0.15          # the smallest push worth naming
strong = 0.6             # a push this big is highlighted; one this big against the move gets its "-" line
against = true           # show that "-" line at all
empty = "no named idea"  # what to say when no named feature pushes the move
# features = ["lined up for a shot", "low on boost"]   # only these may appear

[[rows]]
kind = "looking"
label = "Looking at"
every = 4                # worked out every 4th reading: lower is fresher, higher is cheaper
donors = 3               # recorded moments to compare with each time: more is steadier, and costs more
legend = true            # the words and percentages under the bar
words = ["car", "ball", "opp", "mates"]
```

The colours are in `[colors]`: `push`, `push_strong`, `push_against` for "why this move"; `own_car`, `ball`,
`opponents`, `team_mates` for the bar. [Designing your own panel](customize-the-panel.md) has every option.

## When something goes wrong

**The rows don't appear.** The bot's log says why when the overlay starts. The usual reasons: no `policy=` was given;
the pack has no decoder directions or recorded moments (`cds-interpret check-panel panel.toml thoughts.npz --policy
...` says which, and how to add them); the policy's network isn't the one the pack was made from ("has no layer").

**"Looking at" shows in 1v1 but not in 2v2.** The pack has recorded moments of 1v1 only. Add 2v2 ones with
`add_donors(pack, [moments_1v1, moments_2v2])`.

**The log says `'why this move' stopped: ...` or `'looking at' stopped: ...`.** The reading failed once and turned
itself off rather than fail at every decision; the rest of the panel goes on. The message says what failed. Two
things these readings need from your network: its forward pass takes one input tensor (the one the overlay reads -
`input_layer=` if it isn't the network's own input), and the pack's layer runs once per forward pass. Recurrent
networks with a hidden state are not supported yet.

**"no named idea" most of the time.** Your named features rarely drive the moves. That's common - the features that
labels name best are often not the ones decisions rest on. [What a decision rests on](what-decisions-rest-on.md#what-each-feature-makes-the-bot-do)
shows how to find the features that do, by their effect; name a few of those.
