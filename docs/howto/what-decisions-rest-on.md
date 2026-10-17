# See what a decision rests on

Probes tell you what a layer *knows*. This page answers the other question: *why did the bot do that?* First in terms
of the game - the ball, its own car, the opponents - and then in terms of the features of a layer.

The examples use the demo bot with the recording and dictionary from the [naming guide](name-features.md):

```python
import cds_interpret as ci

policy = ci.demo.policy()
study = ci.Study(policy, ci.Moments.load("recordings/demo"))
```

## Ball, car or opponents?

```python
res = study.attribution()
for situation, shares in res["shares"].items():
    print(f"{situation:14s}", "  ".join(f"{g} {v:.2f}" for g, v in shares.items()))
print({method: round(score, 2) for method, score in res["faithfulness"].items()})
```

```text
all moments    own car 0.79  own boost -0.00  ball 0.21  opponents 0.00  team-mates 0.00
kickoff        own car 0.95  own boost -0.00  ball 0.06  opponents -0.01  team-mates 0.00
in_air         own car 0.65  own boost 0.00  ball 0.35  opponents -0.01  team-mates 0.00
{'Shapley': -0.02, 'one swapped at a time': -0.03, 'random': -0.01}
```

Read the first line as: across all moments, about four fifths of what drives the demo bot's choices is its own car,
a fifth is the ball, and the opponents count for nothing. That's honest - the scripted player it copied only ever
looks at itself and the ball. At kickoffs it's almost all the car (the ball is always in the same place then); in the
air the ball counts for more.

How it works, in plain words. For each held-out moment, the toolkit takes each part of the game - the bot's own car,
its boost, the ball, the opponents, the team-mates - and either keeps it or swaps it for the same part from another
recorded moment (a **donor**, from a different game). It runs every combination through your bot's own input function
and network, and splits the change in the bot's choices fairly between the parts using **Shapley values** - a method
from game theory for sharing credit among things that act together. A moment's shares add up to 1; one can come out a
little negative when parts interact.

Why whole parts and not single input numbers? Most of an observation is worked out from several things (the ball
*relative to* the car, the distance *to* the opponent), so one number belongs to two parts. And changing one number
alone can give an input no real game could produce. Swapping a whole part keeps the input a real game state. This
needs your policy's input function.

**Can you trust the ranking?** The `faithfulness` score removes the parts in the method's order of importance and
checks how fast the bot's choices move: 1 is the best possible order, 0 is no better than random.

On the demo bot every method scores about 0 - and that's the score doing its job. The demo bot reads one thing:
where the ball is *relative to* its car. Swap its car alone, or the ball alone, and that relation breaks, so its
choices move a lot; swap both and it's looking at a real moment again, and they move less. When two parts only work
together like this, no ranking of single parts can be right, and the score says so. Read the shares as "car and ball,
together", not as a ranking.

On a large self-play bot (31 million weights), whose parts matter more separately, these whole-part Shapley values
scored 0.46, swapping one part at a time 0.38, and the usual gradient methods (added up per part) 0.26 and 0.19. So
use the parts, and check the score before you read a ranking.

The building blocks are there for your own groupings or for a critic's value: `attribution.FramesCodec(policy)` splits
and rebuilds moments, and `pick_donors`, `coalition_table`, `shapley`, `occlusion`, `kl_table` and `faithfulness` do
the rest. A group can be part of an entity (`Group("cars", role=ROLE_ME, fields=(...,))`), and any function of the
input can be explained - `coalition_table(policy.value_fn, ...)` splits a critic's value the same way.

## What each feature makes the bot do

[Naming a feature](name-features.md) says what it lines up with. This says what it *does*:

```python
from cds_interpret import effects as EF

fs = study.load_features("body3.sae")           # the dictionary you named (or study.features("body.3"))
x = study.activations("body.3")[study.test]
down = policy.net.head                          # the rest of the network after body.3 (the demo bot's "head")
fx = EF.feature_effects(fs.sae, down, x, table=policy.actions)
per = EF.per_feature(fx, fs.sae.m)
names = ["throttle", "steer right", "nose up", "yaw right", "roll right", "jump", "boost", "handbrake"]
for j in sorted(per, key=lambda j: -per[j]["kl"] * per[j]["freq"])[:3]:
    print(f"feature {j}: fires on {per[j]['freq']:.0%}, KL {per[j]['kl']:.2f} when removed: {EF.effect_words(per[j], names)}")
```

```text
feature 150: fires on 38%, KL 0.81 when removed: steer right -0.26, yaw right -0.26, handbrake -0.03
feature 139: fires on 30%, KL 0.80 when removed: steer right +0.14, yaw right +0.14, boost +0.09
feature 470: fires on 30%, KL 0.59 when removed: steer right -0.22, yaw right -0.22
```

At every moment, each active feature is removed on its own and the rest of the network is run from that layer.
`down` is that rest: whatever turns the layer's values into your bot's action scores - for a `torch.nn.Sequential`,
`ci.hooks.tail(seq, i)` gives everything after layer `i`. That gives, for each feature, how it changes each control -
what it "does" - and for each decision, the features it rests on (`EF.explain(fx, i)`).

Put what a feature does next to its suggested name (`names.suggest`). On the large self-play bot, the features the
labels named most clearly ("about to demo", "shot going in") hardly changed any decision; the ones decisions rested on
were about steering - which only their effects reveal.

**Rank features by effect, not by size.** Removing each decision's three biggest features *by effect* moved the bot
twice as much as removing its three most active ones (`EF.joint_ablation` measures any ranking this way). A cheaper
estimate, `EF.attribution_patching` (gradient times contribution), ranked almost exactly like the exact effects
(correlation 0.96), so it's the way to go on a big network.

**Choosing what to test in games.** Pick features to test by what they do on training games, then test them on
held-out ones. On the large bot, the 20 features that pushed boosting most at demo chances, removed together, took a
third of the boosting away at held-out demo chances (0.37 -> 0.25) and almost none at similar fast moments without one
(0.36 -> 0.35). The feature picked by its label alone moved nothing. [Test what your bot relies on](causal-tests.md)
shows how to run such tests over whole games.
