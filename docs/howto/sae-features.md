# Find the ideas your bot uses

A layer of your bot's network is a list of numbers. Each number on its own rarely means anything you could name - the
ideas the network uses are spread across many of them.

A **sparse autoencoder** is a tool that finds those separate ideas. It learns a dictionary of "features" for one layer,
so that every moment can be rewritten as a mix of just a few of them - say 16 out of 512. Those features often line up
with things you could name: "threat on goal", "far from the ball", "about to score".

## Train one

```python
feats = study.features("body.3")
print(feats.summary(top=5))
```

```text
A sparse autoencoder of body.3 found 512 features (16 active at a time).
It keeps 99.3% of the layer's variance on held-out games; with it in place of the layer, the bot's chosen
action changes on 12% of moments (KL 0.051). For scale: losing the layer altogether (its average in its place)
changes 96% (KL 1.783).

The features most worth a look (AUROC for yes / no facts: 0.5 is chance, 1 perfect; correlation for
numbers: 0 is none, +1 or -1 perfect):
  feature 159   fires on 22.2% of moments   lines up with: in the air (AUROC 0.97), height (correlation +0.23), ...
  feature 87    fires on 29.8% of moments   lines up with: kickoff (AUROC 0.86), about to score (AUROC 0.72), ...
  feature 139   fires on 30.4% of moments   lines up with: threat on goal (AUROC 0.85), not goal side (AUROC 0.85), ...
  feature 219   fires on 19.4% of moments   lines up with: kickoff (AUROC 0.84), distance to the ball (correlation +0.46), ...
  feature 163   fires on 19.7% of moments   lines up with: distance to the ball (correlation +0.66), ...
```

(The demo bot's body.3 is 64 numbers wide. Your feature numbers will be different.)

The first lines tell you whether to trust the dictionary (below). The list shows the features most worth a look: how
often each one fires, and the game facts it lines up with best.

**How many features?** By default the toolkit uses 8 features for every number in the layer (512 for a 64-wide
layer) with 1 in 32 of them active at a time (16 of 512). You can choose:
`study.features(layer, m=1024, k=32, steps=4000)` - `m` is the number of features, `k` how many may be active at once,
`steps` how long it trains. More training steps and more recorded moments both help. On a laptop CPU, a few thousand
steps on a 64-wide layer take about a minute.

**Keep the one you use.** `feats.save("body3.sae")` writes the dictionary to a file, and
`study.load_features("body3.sae")` brings it back - in a new session or on another computer - with the same feature
numbers. Training again gives a different dictionary whose feature 139 is something else, so save the one you name.
(`study.features` is seeded: the same `seed` on the same recording usually gives the same dictionary back.)

## Is the dictionary any good?

Two checks, both on games it never trained on:

- **How much of the layer it keeps.** "Keeps 99.3% of the layer's variance" means the rebuilt layer is very close to
  the real one. A few percent lost is good.
- **Whether the bot still plays the same.** The toolkit puts the rebuilt layer in place of the real one and measures
  how often the bot's chosen action changes, and the **KL divergence** - how far its whole set of action probabilities
  moves (0 means not at all). For scale, it shows the same numbers with the layer replaced by its average, which is
  roughly "losing the layer": 12% against 96% above.

A dictionary can keep 99% of a layer and still change some choices: a few directions matter more to the next layer
than their size suggests. And a bot with many nearly-tied actions flips its top choice easily, so the KL is the
steadier of the two numbers. [Check your tools](check-your-tools.md) has more ways to test a dictionary.

## What is a feature about?

For every feature, the toolkit compares when it fires with every [game fact](../labels.md):

- for a yes / no fact: **AUROC** (how well the feature picks out the "yes" moments; 0.5 is chance, 1 perfect) and
  **lift** (how many times likelier the fact is when the feature fires);
- for a number: **correlation** (from -1 to 1).

```python
feats.guess(139)            # [('threat', 0.71, '+'), ('goal_side', 0.69, '-'), ('concede_soon', 0.65, '+')]
feats.detectors()           # features that pick out a yes / no fact cleanly: [(feature, label, AUROC, lift)]
feats.activation(139)       # how strongly it fires at every moment
feats.sizes(139)            # how big it gets when it fires: {'start': 0.14, 'typical': 0.73, 'highest': 3.81}
```

`guess` gives each match's strength from 0 (none) to 1 (perfect): for a yes / no fact that's how far its AUROC is
from chance (AUROC 0.85 is strength 0.71), for a number the size of its correlation. "-" means the feature fires when
the fact is *not* true.

That's a starting point for a name, not a name. Two features can line up with the same fact for different reasons, and
a feature can line up with nothing you have a label for and still be the most interesting one. The
[naming guide](name-features.md) takes you from these suggestions to names you trust: look at each feature's strongest
moments, write the name down, and check the bot really uses it.

## Without a study

The same tools work on any array of a layer's values:

```python
from cds_interpret import sae as S

sae = S.TopKSAE(d=64, m=512, k=16).init_from(values[:20000])     # values: [N, 64]
S.train_sae(sae, values, rows=train_rows, steps=3000, batch=2048, log=print)
print(S.fvu(sae, values[test_rows]))       # the share of the layer it does NOT keep
codes = sae.encode(values)                 # [N, 512], 16 non-zero per row
S.save(sae, "sae.pt", layer="body.3"); sae = S.load("sae.pt")
```

Here `d` is the layer's width (64 numbers), `m` the number of features (512) and `k` how many are active per moment
(16). `values` is an `[N, 64]` array of the layer's values at N moments, and `train_rows` / `test_rows` say which
moments to train on and which to score on - from different games.

The autoencoder is the "top-k" kind from Gao et al. 2024 ("Scaling and evaluating sparse autoencoders"): exactly k
features are active per moment, and a small extra loss stops features from going unused. Saved dictionaries hold only
plain tensors, so they are safe to share and to load from others.
