# Check your tools before you trust them

Every technique here can mislead you quietly. A dictionary can rebuild a layer well and still change the bot's
decisions. A feature can be one that only this particular training run happened to find. A probe can read a concept
the bot never uses. A gauge on the panel can say 40% for something that happens 10% of the time. Each of these has a
number that gives it away - this page shows how to get it.

The examples use the demo bot with the recording and dictionary from the [naming guide](name-features.md). They all
start from the same few lines:

```python
import cds_interpret as ci

policy = ci.demo.policy()
study = ci.Study(policy, ci.Moments.load("recordings/demo"))
study.probes()
acts = study.activations("body.3")                      # the layer's values at every moment [N, D]
x_train, x_test = acts[study.train], acts[study.test]   # on the training games / on the held-out games
down = policy.net.head                                  # the rest of the network after the layer
feats = study.load_features("body3.sae")                # the dictionary you named
sae = feats.sae
```

`down` is whatever turns the layer's values into the bot's action scores. For the demo bot that's its `head`; for a
`torch.nn.Sequential`, `ci.hooks.tail(seq, i)` gives everything after layer `i`.

## A dictionary: true to the bot's choices, and found again

```python
from cds_interpret import dict_eval as DE
from cds_interpret import sae as S

print(S.downstream_check(sae, down, x_test))            # the bot run on the dictionary's rebuild of the layer
second = study.features("body.3", seed=1).sae           # a second dictionary of the same layer, another seed
best, r = DE.activation_match(sae, second, x_test, feats.top(50))   # each feature's best partner in it, and how alike
print(f"{(r >= 0.7).mean():.0%} of the top 50 features come back in the second run")
study.load_features("body3.sae")                        # (put the one you named back in the study)
```

Two questions:

- **Does the bot still choose the same?** `downstream_check` puts the rebuilt layer in place and measures how far the
  bot's choices move (KL; lower is better), next to losing the layer altogether. On the demo bot the KL is 97% smaller
  than with the layer gone (`kl_recovered` 0.97), and the chosen action changes on 12% of moments. If that's too
  much, `S.finetune_kl(sae, x_train, down, steps=300)` trains the dictionary a little more on the *choices* rather
  than on the layer's numbers, keeping the same features: on the demo bot it took the KL from 0.051 to 0.036.
- **Would a second run find the same features?** Train a second dictionary with another seed and match the features
  up by when they fire (a correlation of 0.7 or more counts as a match). On the demo bot, all of the top 50 came back.
  Matching by direction alone (`DE.stability`) finds fewer, so measure by firing.

On a large self-play bot's layer (8,192 features), the dictionary kept 86% of the layer yet changed a third of the
chosen actions; `finetune_kl` brought the KL from 0.21 to 0.15 with the same features. Two seeds shared only 34% of
their features, and a feature named "about to demo" had no good partner in the second run. **Name the features that
come back in a second run.**

## A probe: does the bot use what it reads?

A probe shows a concept is *in* a layer. To see whether the bot *uses* it, erase it and watch:

```python
import torch
from cds_interpret import erasure as ER
from cds_interpret import intervene

y = study.labels["goal_side"]
er = ER.fit_leace(x_train, y[study.train])                     # every straight-line trace of the concept, removed
placebo = ER.random_eraser(x_train, er.rank)                    # erase something random of the same size instead
x_in = study.x[torch.from_numpy(study.test)]                    # the network's inputs at the held-out moments
print(intervene.edit_effect(policy, x_in, "body.3", {"goal side erased": ER.erase_edit(er),
                                                     "placebo": ER.erase_edit(placebo)}))
```

```text
{'goal side erased': {'kl': 0.054, 'argmax_changed': 0.145, ...}, 'placebo': {'kl': 0.167, 'argmax_changed': 0.210, ...}}
```

**LEACE** erases every trace of a concept that any straight-line readout could find. If the
bot's choices then hardly move, it either uses the concept in a more tangled way or not at all.

**Always compare with the placebo.** Erasing *anything* readable moves a bot a little; the concept matters only as far
as erasing it moves the bot more than erasing something random of the same size. Here erasing "goal side" changes
15% of the demo bot's choices - and the placebo 21%. Without the placebo, 15% would have looked like a finding.

That's not the whole story, though. The [quickstart](../quickstart.md)'s demo removes just the goal-side probe's own
direction, and that moves the bot about six times as much as removing random directions. Two edits of the same
concept, two answers: LEACE removes the concept along the direction that best hides it, which needn't be the one the
bot reads. So try more than one edit, each with its own control, before you say the bot does or doesn't use
something.

**Look at the behaviour the concept is about, and erase it everywhere.** On a large self-play bot (31 million
weights), erasing "a demolition chance" from every hidden layer moved its choices overall *less* than erasing random
readable features did (KL 0.31 against 0.49-0.68) - and yet it took the bot's boosting at real demolition chances from
0.34 to 0.14 (the placebos: 0.28-0.33). Erasing "they score within 3 s" the same way changed neither. Removing one
probe direction at one layer had shown nothing at all.

The opposite question - a concept a straight-line probe reads poorly at a deep layer: is it gone, or just no longer
in a straight line? - is what `probes.mlp_probe` (a probe with one hidden layer) answers. Compare it on a random
network's layer too, and give it plenty of moments: on small recordings (the demo's 1,200 moments) it does worse than
the plain probe (0.58 against 0.78). On the large self-play bot, where the ball will land read R2 0.43 in a straight
line at the last hidden layer and 0.68 with the MLP probe (0.95 and 0.98 at the first): partly re-encoded, partly
gone - and less than a random network's layer of the same shape keeps (0.91).

## A gauge: does 40% mean 40%?

A probe's score isn't automatically a probability. **Platt scaling** is a small correction that turns it into one.
Learn it on held-out games the probe never saw, and check it on others:

```python
import numpy as np
from cds_interpret import readout as RO

probe = study.probe_results.probe("body.3", "goal_soon")              # a yes / no probe you'd show live
held = study.test
half = np.isin(study.games[held], np.unique(study.games[held])[::2])  # the held-out games, split in two
cal, check = held[half], held[~half]
scores, y = probe.score(acts).numpy(), study.labels["goal_soon"]
a, b = RO.fit_platt(scores[cal], y[cal])                              # a probe's score -> an honest probability
print(RO.calibration(RO.apply_platt((a, b), scores[check]), y[check])["ece"])   # the calibration error
pack = study.make_pack("body.3", probes={"About to score": "goal_soon"})
RO.calibrate_pack_probe(pack, "About to score", a, b)                 # the live panel then shows the corrected number
```

The **calibration error** ("ece") is how far, on average, the shown chance is from how often the thing really
happens: 0.08 here means about 8 percentage points.

On a large self-play bot's panel, a "they score within 3 s" gauge in its top 1% showed 64% for something that happened
92% of the time. After Platt scaling it showed 98%, and the calibration error was three times smaller.

Two more: `RO.smoothing_tradeoff` measures what smoothing a gauge costs (lag, accuracy) against what it buys (less
jitter), and `RO.hysteresis` / `RO.switches_per_minute` do the same for features flicking on and off.

## An explanation: does its ranking find what matters?

`attribution.faithfulness` scores any ranking of the parts of an input by what removing them actually does: 1 is the
best possible order, 0 is chance. See [what a decision rests on](what-decisions-rest-on.md).
