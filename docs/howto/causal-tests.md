# Test what your bot relies on

A probe or a feature tells you a layer *holds* some information. That doesn't mean the bot *uses* it - networks carry
plenty of information they never act on. To find out, change the layer and see whether the bot's choices change.

## Turn one feature off and up

The quickest check, for a feature you've found:

```python
check = study.check_feature("body.3", "threat on goal")   # a name you confirmed, or a feature's number
print(check.summary())
```

```text
feature 139 (threat on goal) of body.3, on the 665 held-out moments it fires on:
  turned off     the chosen action changes on 52% of them (KL 0.803); on average: steer -0.45, yaw -0.45, boost -0.09
  turned up x3   the chosen action changes on 57% of them (KL 3.219); on average: steer +0.60, yaw +0.60, boost -0.28
```

(The examples on this page use the demo bot and the recording, dictionary and names from the
[naming guide](name-features.md): 20 games of 90 seconds. Your feature numbers will differ.)

It switches the feature off (and then makes it three times as strong) inside the network, only on the held-out moments
where the feature fires, and reports:

- how often the bot's chosen action changes;
- the **KL divergence** - how far its whole set of action probabilities moves (0 means not at all);
- how its controls change on average, when it has an action table: "boost -0.09" means it boosts less.

If turning a feature off changes what the bot does, the bot relies on it. If the controls move the way the feature's
name says they should - here the steering changes most, and the demo bot was taught to turn for home when its goal is
threatened - the name is probably right. All checks also appear in the study's report.

## Edits, more generally

An **edit** is any change to what a layer computes, applied every time the network runs. The toolkit has these:

| edit | what it does |
|---|---|
| `study.feature_edit(layer, [j, ...], scale=0.0)` | removes those features (scale 0) or scales them (2 doubles them); the rest of the layer is untouched |
| `study.probe_edit(layer, label, factor=0.0)` | removes the direction a probe reads from (0), or scales it |
| `intervene.clamp_feature(sae, j, value)` | holds a feature at least at a value |
| `intervene.steer(direction, alpha)` | pushes the layer along a direction, everywhere or where a condition holds |
| `intervene.project_out(direction)` | removes a direction |
| `intervene.replace_with(sae)` | replaces the layer with the autoencoder's rebuild of it |
| `intervene.mean_ablate(mean)` | replaces the layer with its average: roughly "losing the layer" |

Measure any of them on the held-out moments:

```python
import torch

effects = study.edits("body.3", {
    "no threat": study.probe_edit("body.3", "threat", 0.0),
    "random direction": ci.intervene.project_out(torch.randn(study.activations("body.3").shape[1])),  # a control
})
for name, e in effects.items():
    print(f"{name}: choice changes on {e['argmax_changed']:.0%} of moments, KL {e['kl']:.3f}")
```

```text
no threat: choice changes on 0% of moments, KL 0.000
random direction: choice changes on 12% of moments, KL 0.034
```

A surprise, and a useful one. The threat probe reads "threat on goal" very well from this layer, yet removing the
direction it reads from changes nothing - less than removing a random direction. The probe found a direction that
*holds* the information but that the rest of the network ignores; the bot acts on its threat feature instead (52%
above). **A probe shows what a layer holds; only an edit shows what the bot uses.**

(A short recording may have too few threats to probe at all: the error says how many it found and how many it needs.
More games fix it, or use a label that's common in your games, like `first_to_ball`.)

**Always include a control** - a random direction, or features that fire about as often but have nothing to do with
it. An edit only means something if it moves the bot much more than its control.

**Look where it moves.** Removing "threat on goal" should change the bot's choices mostly when its goal is threatened:

```python
x = study.x[torch.from_numpy(study.test)]
threatened = study.labels["threat"][study.test] > 0.5
edit = {"no threat": study.probe_edit("body.3", "threat", 0.0)}
print(ci.intervene.edit_effect(policy, x[threatened], "body.3", edit))     # where the goal is threatened
print(ci.intervene.edit_effect(policy, x[~threatened], "body.3", edit))    # everywhere else
```

For an edit that matters, the first number is much bigger than the second. (For the probe edit above, both are close to
zero: it doesn't matter anywhere.)

## The real test: whole games

The strongest evidence is play. Keep the edit on while the bot plays, and compare it with the unedited bot against the
same opponent, from the same kickoffs:

```python
import numpy as np

edit = study.feature_edit("body.3", ["threat on goal"], 0.0)       # names or numbers
with ci.intervene.editing(policy, "body.3", edit):
    edited = ci.record.rocketsim(policy, games=60, opponent="chaser", seed=1)
normal = ci.record.rocketsim(policy, games=60, opponent="chaser", seed=1)


def goals(m):
    g = np.array(m.meta["goals_per_game"])                 # [games, 2]: your bot (blue), the opponent
    return g[:, 0], g[:, 1]


print(ci.intervene.bootstrap_share(*goals(normal)))       # (goal share, low, high): a 95% range over games
print(ci.intervene.bootstrap_share(*goals(edited)))
```

`bootstrap_share` gives your bot's share of the goals with a 95% range. If the two ranges don't overlap, the edit made
a real difference. `rate_ratio_ci(a, na, b, nb)` compares how often something happens (goals conceded per game,
demos per minute) the same way.

A few things that keep this honest:

- **Don't edit both sides.** With `opponent="self"`, both cars are the same network, so the edit would play on both.
  Use a scripted opponent, another bot, or an unedited copy (`ci.Policy(copy.deepcopy(net), ...)`).
- **Same seed, same kickoffs.** Both runs above start from the same kickoffs, so the comparison is fair.
- **Enough games.** Effects on goals are noisy: 50 to 200 games per side is normal. The two runs of 60 games above
  take about 15 minutes on a laptop CPU.
- **Decide before you look.** Pick the features to test before playing the games. `analysis.causal_targets` picks a
  target, a group and matched controls by a fixed rule, so the results can't steer the choice.

To just *watch* an edited bot, `ci.intervene.attach_edits(policy.net, layer, edit)` keeps an edit on until you call
`.detach_hooks()`.
