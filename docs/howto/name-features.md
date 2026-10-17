# Name your bot's features

A sparse autoencoder hands you your bot's ideas as numbers: feature 139, feature 163. This guide takes you from those
numbers to names you trust - "threat on goal", "far from the ball" - that you can show live on screen.

It's a short loop, and the toolkit has a tool for each step:

1. **Record** games of your bot, with the facts of every moment worked out automatically.
2. **Find** the features of a layer with a sparse autoencoder, and **keep** it.
3. **Get suggestions**: which facts each feature lines up with, and its strongest moments drawn on the field.
4. **Write the names down** in a names file, confirming, renaming or rejecting each suggestion.
5. **Check** that the bot really uses each feature: turn it off and up, and see what changes.
6. **Show** the names live.

The examples use the demo bot, so you can follow along; swap in your own policy at step 1. Your feature numbers will
be completely different from the ones here - that's normal, an autoencoder trained on other games finds its features
in another order - but the story will be the same. Pick your own numbers and names from your own lists.

## 1. Record games

```python
import cds_interpret as ci

policy = ci.demo.policy()                       # or your own: ci.Policy(net, obs=..., actions=...)
moments = ci.record.rocketsim(policy, games=20, seconds=90, opponent="chaser", out="recordings/demo",
                              if_exists="reuse")
```

```text
  game 1/20: blue 3 - orange 0  (450 moments so far, 1 s)
  ...
  game 20/20: blue 2 - orange 0  (9000 moments so far, 28 s)
  saved 9000 moments to recordings/demo
```

`if_exists="reuse"` makes the script safe to run again: the second time, it loads the games already in the folder
instead of playing new ones.

Every moment comes with its **labels**: facts worked out from the game state, like "the ball is going into my goal
within 3 seconds" (`threat`) or "it's a kickoff" (`kickoff`). The suggestions are made from those.
[All the labels](../labels.md).

Recording in the real game works just as well - see [record games](record-moments.md). More games give better
suggestions: 20 is a fine start, and rare situations need more.

## 2. Find the features, and keep them

```python
study = ci.Study(policy, moments)
study.probes()                                  # (the live panel in step 6 uses these)
feats = study.features("body.3")                # a sparse autoencoder of the last hidden layer
print(feats.summary(top=5))
feats.save("body3.sae")                         # keep it: your names will belong to it
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

"16 active at a time" means that at any moment, only 16 of the 512 features are switched on; the rest are zero.
That's what makes each feature a separate idea you can name.

The first lines say whether the dictionary is good enough to name. Here it is: it keeps almost all of the layer, and
the bot mostly plays the same with it in place (KL, a measure of how far the bot's choices move, is small) - nothing
like the mess of losing the layer. If your numbers look poor, train longer, `study.features("body.3", steps=4000)`,
or record more games before naming anything.

**Why save it?** Feature numbers belong to this one dictionary. Train it again on other games and feature 139 is
something else entirely - so names written for this dictionary would land on the wrong features. Saving it keeps it
for good; [coming back later](#coming-back-later) shows how to load it again. (The same seed on the same recording
usually gives the same dictionary back, but a saved file is certain.)

## 3. Get suggestions

```python
for s in study.suggest_names("body.3", top=12):
    print(s)
```

```text
feature   159  in the air                    (AUROC 0.97, 4.3x as likely when it fires; fires on 22.2%)
feature    87  kickoff                       (AUROC 0.86, 3.1x as likely when it fires; fires on 29.8%)
feature   139  threat on goal                (AUROC 0.85, 2.8x as likely when it fires; fires on 30.4%)
feature   163  far from the ball             (correlation +0.66; fires on 19.7%)
feature   189  about to score                (AUROC 0.83, 3.0x as likely when it fires; fires on 28.0%)
feature   128  not goal side                 (AUROC 0.80 for 'not goal side', 1.8x as likely when it fires; fires on 43.3%)
feature    28  shot on target                (AUROC 0.79, 2.8x as likely when it fires; fires on 28.5%)
feature    17  ball landing in our half      (correlation -0.54; fires on 31.9%)
feature   230  high up                       (correlation +0.53; fires on 12.1%)
feature   503  demo chance                   (AUROC 0.72, 2.0x as likely when it fires; fires on 40.2%)
feature   455  expects to score              (correlation +0.40; fires on 3.0%)
feature   411  about to concede              (AUROC 0.68, 4.4x as likely when it fires; fires on 11.1%)
```

Each suggestion is the fact the feature lines up with best:

- **AUROC** - how well the feature picks out the moments where the fact is true. 0.5 is chance, 1.0 is perfect.
- **"x as likely when it fires"** - when feature 139 fires, a threat on goal is 2.8 times as likely as usual.
- **correlation** - for facts that are numbers, from -1 to 1. Feature 163 grows with the distance to the ball, so
  it's suggested as "far from the ball".

The list shows each different suggestion once before any repeats, so twelve features that all go with kickoffs don't
crowd out everything else. When several features do share a suggestion, they're numbered ("kickoff #2") for you to
tell apart. Some suggestions come with a warning, starting with `!`. Read those: they point at the traps at the end of
this page.

**A suggestion is a guess.** It says what a feature goes *with*, not what it *is*. So look at it.

### Look at the strongest moments

```python
study.feature_gallery("body.3", "features.html", n=12)
```

Open `features.html` in your browser. Use the same number as in `suggest_names` (12 here), and the page lists the
same features, in the same order, under the same names. Each card is one feature: its suggested name, how often it
fires, the facts it lines up with, how big it gets when it fires, and its three strongest moments drawn from above.
Your car has a ring and attacks up the page, the ball is the grey dot, and the short lines show where things are
heading. When the recording has every step (RocketSim and RLBot recordings do), the strongest moment is also
animated.

Ask of each card: do these moments look like the name? Feature 139's moments should show the ball heading for your
goal. If they show something else every time, that something else is the better name.

## 4. Write the names down

Start a names file from the suggestions:

```python
ci.names.write_names("names.toml", study.suggest_names("body.3", top=12), layer="body.3")
```

```toml
# What each feature means. Set status = "confirmed" on a name once its moments agree; rename freely;
# "rejected" hides a feature. The live panel shows confirmed names only.
layer = "body.3"
dictionary = "ef1425d35576"   # the numbers below belong to this dictionary only: keep it with feats.save(...)

[features.159]
name = "in the air"
status = "suggested"
# AUROC 0.97, 4.3x as likely when it fires; fires on 22.2% of moments
# when it fires: 0.08 to 2.35, typically 0.47 (the panel counts it from 0.08 unless you set a threshold)

[features.139]
name = "threat on goal"
status = "suggested"
# AUROC 0.85, 2.8x as likely when it fires; fires on 30.4% of moments
# when it fires: 0.14 to 3.81, typically 0.73 (the panel counts it from 0.14 unless you set a threshold)
...
```

The `dictionary` line is the saved dictionary's code: the toolkit refuses to put these names on any other one.

Now go through the file with the gallery open, feature by feature:

- the moments agree with the name: set `status = "confirmed"`;
- they agree with a better name: change `name`, and confirm it;
- they don't make sense to you: set `status = "rejected"`.

```toml
[features.139]
name = "threat on goal"
status = "confirmed"
threshold = 0.7            # optional: on the live panel, only count it as firing above this

[features.163]
name = "far from the ball"
status = "confirmed"
note = "fires at kickoffs too - the ball is far away then"
```

A **threshold** decides when the live panel counts a feature as "on". Without one, the panel uses the feature's
smallest typical firing size (0.14 for feature 139). Set one higher to show only its strong moments - the comment
under each feature gives its sizes, and the toolkit stops you if you pick one it never reaches.

Then give the names to your features, so the rest of the toolkit uses them:

```python
feats.apply_names("names.toml")
```

Found more features later? `ci.names.update_names("names.toml", suggestions, "body.3")` adds the new ones and keeps
everything you wrote. Two features can't share a confirmed name, because the panel tells them apart by name.

## 5. Check the bot uses it

A name says what a feature is about. A check says whether the bot acts on it:

```python
print(study.check_feature("body.3", "threat on goal").summary())      # a confirmed name, or a number
```

```text
feature 139 (threat on goal) of body.3, on the 665 held-out moments it fires on:
  turned off     the chosen action changes on 52% of them (KL 0.803); on average: steer -0.45, yaw -0.45, boost -0.09
  turned up x3   the chosen action changes on 57% of them (KL 3.219); on average: steer +0.60, yaw +0.60, boost -0.28
```

It turns the feature off (and then up, three times as strong) inside the network, on the held-out moments where it
fires. Then it counts how often the bot chooses differently, and how its controls change on average.

Here, switching off "threat on goal" changes half of the bot's choices on those moments: it relies on this feature.
Turning it up pushes the steering the other way - what you'd hope from a real idea rather than an accident. (The demo
bot was taught to turn back and drive home when its goal is threatened, so a feature like this steering it is just
what you'd expect.)

A feature that changes nothing when switched off may still describe something the network sees - it just isn't
steering the bot. For the strongest evidence, play whole games with it switched off:
[test what your bot relies on](causal-tests.md#the-real-test-whole-games).

## 6. Show it live

```python
pack = study.make_pack("body.3", names="names.toml",
                       probes={"First to the ball": "first_to_ball", "Time to the ball": ("ttb_me", {"unit": "s"})})
ci.overlay.save_pack(pack, "thoughts.npz")
```

Only confirmed names make it onto the panel, each with its threshold. Then [wrap your bot](overlay.md#2-wrap-your-bot),
and [style the panel](customize-the-panel.md) if you like.

## Coming back later

Everything above lives in one Python session. In a new one - tomorrow, or after a restart - load the recording and
the dictionary you saved, and carry on:

```python
moments = ci.Moments.load("recordings/demo")
study = ci.Study(policy, moments)
feats = study.load_features("body3.sae")        # the same dictionary: the same feature numbers
feats.apply_names("names.toml")
pack = study.make_pack("body.3", names="names.toml")
```

If you train a new dictionary instead (`study.features("body.3")` on other games, say) and hand it your old names
file, the toolkit stops rather than put the names on the wrong features:

```text
names.toml was written for another dictionary of body.3 (its code: ef1425d35576; this one's: f2fb880699d8). Feature
numbers don't carry over: a dictionary trained again has different features under the same numbers, so these names
would land on the wrong features.
Load the dictionary you named: study.load_features("<file you saved it to>"). ...
```

## Traps to watch for

**The kickoff trap.** A feature can line up with an exciting fact for a boring reason. In one bot, a feature that
lined up well with "demo chance" turned out to fire half the time at kickoffs: two cars racing at each other for the
ball look just like a demo chance. That's why the toolkit's `demo_chance` label leaves kickoffs out. And the
suggestions warn you when many of a feature's firings are kickoffs. The warning looks like this:

```text
      ! 54% of its firings are kickoffs: is it about 'demo chance', or about kickoffs? Check its moments without them
```

When you see it, draw the feature's moments without the kickoffs and name it from what's left:

```python
study.feature_gallery("body.3", "features-no-kickoffs.html", n=12, exclude_kickoffs=True)
```

The same story runs the other way round. In the summary in step 2, feature 219's best match is "kickoff", but it also
goes with the distance to the ball (correlation +0.46). Look at its moments without the kickoffs and you may well find
it's about being far from the ball - which is true at every kickoff.

**A feature only one dictionary has.** Train a second dictionary with another seed and some features come back,
others don't. The ones that come back are the ones worth naming; a feature only one run found may be an accident of
that run. [Check your tools](check-your-tools.md#a-dictionary-true-to-the-bots-choices-and-found-again) shows how to
match two dictionaries up.

**Too few examples.** A feature that lines up perfectly with something that happened ten times proves nothing. The
toolkit only relates features to facts with at least 30 examples (and leaves out facts that only have a value in a
small slice of moments). Rare situations need more games.

**Probes for the live panel.** A probe can score well and still jump around on screen if it leans on numbers that
barely move. The toolkit's probes avoid that by default: they treat every number of a layer alike, instead of rescaling
each one by its own spread. Keep that default for anything you show live. See
[probes you show live](probes.md#probes-you-show-live).

**One layer isn't the whole story.** Different layers hold different things. If a concept you care about has no clean
feature in one layer, try the one before it: `study.features("body.1")`.
