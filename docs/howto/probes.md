# Ask your bot questions

A **probe** is a small readout that asks the network one question - "is the ball about to go in my goal?", "how far
away is the ball?". It takes the numbers of one layer, weighs each one, and adds them up. If something that simple can
answer the question well, on games it never saw, that layer holds the answer.

## Ask every question at once

```python
study = ci.Study(policy, moments)
probes = study.probes()
print(probes.summary(top=8))
```

```text
What each layer knows, scored on games the probes never saw. For a yes / no fact the score is an
AUROC (0.50 is a guess, 1.00 perfect); for a number it's R2 (0.00 is a guess, 1.00 perfect).
"input" and "random" ask the same question of the bot's raw input and of a copy of the network
with random weights. A layer shows something it worked out only where it beats both.
"steady": does the answer hold still when the layer's numbers wobble a little? "yes" is fine to
show live, "so-so" may flicker, "no" will.

  concept                            layer      score  input  random  measure steady
  ----------------------------------------------------------------------------------
  Worked out by the network (at least 0.05 better than both):
  ball_dist (distance to the ball)   body.1      0.90   0.24    0.68  R2      yes
  land_dist (distance to landing)    body.1      0.74   0.21    0.55  R2      yes
  touch_soon (about to touch)        body.3      0.85   0.71    0.70  AUROC   yes
  Readable from the raw input or random weights too:
  ttb_adv (lead to the ball)         body.1      0.27   0.22    0.21  R2      so-so
  outlook                            body.1      0.18   0.13    0.15  R2      yes
  goal_soon (about to score)         body.1      0.90   0.89    0.82  AUROC   yes
  ttb_me (time to the ball)          body.1      0.38   0.24    0.37  R2      so-so
  goal_side (goal side)              body.3      0.97   0.96    0.95  AUROC   yes
  ...
```

(The demo bot on the naming guide's recording: 20 games of 90 seconds.)

How to read it, column by column:

- **concept** - the question (the [label](../labels.md)), with its plain name.
- **layer** - the layer that answers it best.
- **score** - how well, on games kept aside. For yes / no questions it's **AUROC**: the chance a "yes" moment scores
  above a "no" moment (0.5 is a coin flip, 1.0 is perfect). For number questions it's **R²**: how much of the number's
  variation it explains (0 is no better than guessing the average, 1.0 is perfect).
- **input** - the same question asked of your bot's raw input.
- **random** - the same question asked of the same layer in a copy of the network with random weights.
- **steady** - whether the answer would hold still if you showed it live (more below).

The rows come in two groups. First, what the network worked out: questions a layer answers at least 0.05 better than
both baselines, biggest lead first. Here, the demo bot's first hidden layer knows the distance to the ball almost
perfectly (0.90), while its raw input barely does (0.24): the network worked that distance out for itself. Then the
rest. Goal side, for example, scores 0.97 - but the raw input already gets 0.96, so there's little new there.

That's the whole point of the baselines. A high score alone says little. The interesting answers are the ones a layer
gives clearly better than both its raw input and random weights.

## Get at the details

```python
probes.score("body.1", "ball_dist")        # 0.90
probes.best("threat")                      # ("body.1", "threat"): the layer that knows it best (None if not scored)
probes.learned()                           # the labels some layer reads clearly better than both baselines
p = probes.probe("body.1", "ball_dist")    # the readout itself: p.w (one weight per number), p.b, p.direction()
```

A probe's direction (`p.direction()`) is what you remove or push along in a [causal test](causal-tests.md).

## Questions the toolkit skips

A yes / no question needs at least 20 "yes" (and 20 "no") moments in the training games, or there's nothing to learn
from, and at least 10 of each in the held-out games, or there's nothing fair to score it on. A number question needs
its held-out games to vary at least a quarter as much as its training games. `print(probes.summary())` lists every
skipped question and why ("7 yes moments in the held-out games (needs 10)"). Rare events (demo chances, shots on
target) need more and longer games.

(Naming features uses a stricter rule - 30 examples - because a feature is matched against every label at once, and
more matches mean more chances of a lucky one.) Questions that only have an answer at some moments
(when does the ball land, if it's in the air?) use just those moments.

## Probes you show live

A probe can score well and still be useless on screen. The live panel reads it every decision in the real game, where
the network's numbers sit a little differently than in your recordings. A readout that leans on numbers that barely
move turns those small shifts into big jumps.

Four habits keep a live probe honest:

**1. Don't let a quiet number earn a huge weight.** The textbook way to fit a probe first rescales every number to the
same spread. That lets a number that hardly moves earn a weight in the thousands if it happens to help a little. Fine
offline; on screen, a tiny wobble in that number throws the answer far off. So the toolkit's probes treat every number
alike by default. In a test with sixteen ordinary numbers and one very quiet one:

| | weight on the quiet number | score, clean | score, with a little noise |
|---|---|---|---|
| textbook rescaling (`scale="unit"`) | 495 | 0.998 | **-71** |
| the toolkit's default (`scale="global"`) | 0.53 | 0.815 | **0.815** |

The textbook probe looks better on paper and falls apart the moment its numbers wobble. (A yes / no probe goes the same
way: AUROC 0.999 falling to 0.58, against 0.944 kept.)

**2. Look at steadiness, not just the score.** Every probe result carries these numbers:

| number | what it tells you |
|---|---|
| `noise_rel` | how far the answer moves when every number wobbles a little, compared with its normal spread. Under 0.1 is steady |
| `jitter` / `jitter_label` | how far the answer moves from one moment to the next, against how far the real answer moves. Close to each other is honest; far above is noise |
| `wnorm` | the size of the weights. Far bigger than the layer needs means it leans on quiet numbers |
| `miss_med` | (numbers) how far off it typically is |

A landing point that jumps a thousand units between decisions while the real landing point stays put is noise,
whatever its score.

**3. Read places from an early layer.** Later layers keep what the bot needs to act and drop where things are. A place
on the field - where the ball will land, where a car is heading - often reads both better and much more steadily from
the first hidden layers. Check both:

```python
for layer in study.layers:
    s = study.steadiness(layer, ("land_x", "land_y"))      # the landing point, read as one
    print(layer, round(s["miss_med"]), "uu typical miss", round(s["jitter"]), "uu moved per moment")
```

`study.make_pack` does this for you. Its landing ring comes from `study.landing()`, which tries each studied layer
and fits the readout for what you see on screen: the penalty is the one that makes the ring **miss least** - the
median distance from where the ball really lands - on games held out of the fit, not the best score of x and y
apart. It reports the typical miss, the miss nine times out of ten, and how far the ring moves between readings:

```python
fit = study.landing()        # also fit["layer"], fit["median_miss"], fit["p90_miss"], fit["median_jump"], fit["tried"]
```

```text
  landing ring: body.1 misses by 500 uu typically (90% within 1437), moves 217 uu between readings
```

(That's the demo bot. For comparison, the ball is 186 uu across, and the ring is drawn 300 uu wide.)

### A bot that sees the field mirrored

Some bots see the field mirrored left to right at some moments - their observation turns one side into the other, so
one network learns both sides at once. Such a network can't tell the two sides apart, so a readout of where the ball
lands, fitted as usual, is wrong half the time: it can only say "on my near side", not "on the left".

Tell the toolkit when the bot mirrors, with a function of the game state:

```python
def mirrored(frames, me):
    """True where the bot (car slot me[i]) sees moment i mirrored - here: when its car is on the left of its half."""
    rows = np.arange(len(frames))
    side = ci.frames.team_sign(frames.car_team[rows, me])
    return frames.car_pos[rows, me, 0] * side < 0

policy = ci.Policy(net, obs=my_obs, actions=MY_ACTIONS, mirror=mirrored)
```

Then `study.landing()` fits the readout in the bot's own frame (x turned round on the mirrored moments), the pack
remembers it (`frame = "mirrored"`), and the overlay turns each reading back with the same function before drawing -
so the overlay needs the policy too (`with_thoughts(..., policy=...)`, or the [watch script](watch-a-bot.md)). Without
it, the rings stay hidden rather than drawn on the wrong side. The words of a move are turned round as well: such a
bot's "steer right" in its own frame is a "steer left" on the real field.

**4. Smooth what you draw.** Even a good readout wavers a little. The panel smooths its numbers and glides the landing
ring over about a quarter of a second; the [panel file](customize-the-panel.md) lets you change both.

## On your own arrays

The same probes work on any numbers, outside a study:

```python
from cds_interpret import probes as P

train, test = P.group_split(game_ids, test_frac=0.25)              # whole games on each side
res = P.ridge_suite(layer_values, labels, kinds, train, test, groups=game_ids, keep=True,
                    seq=game_ids * 64 + car, order=step)           # (seq and order turn on the jitter numbers)
res["threat"]["auroc"], res["land_t"]["r2"], res["land_t"]["probe"]
```

Here `layer_values` is `[N, D]`, `labels` and `kinds` are dicts by label name (`kinds`: "bin" for yes / no, "reg"
for numbers), and `game_ids`, `car` and `step` say which game, which car and which step each row is. `seq` and `order`
are only needed for the steadiness numbers: `seq` gives each car of each game its own number (any formula that does,
like `game_ids * 64 + car`), and `order` puts its rows in time order.

`ridge_suite` fits every question at once and is quick enough to run on every layer. `probe_layer` fits one probe per
question with logistic regression for yes / no questions, if you want a second opinion.

## Your own questions

Anything you can work out from the game state can be a question: see [adding your own labels](../labels.md#your-own-labels).

## Across training

`ci.tracking.track(...)` asks the same questions of each saved checkpoint, on the same moments: when did your bot learn
about threats? See [follow a bot through training](track-training.md). Remember that the moments come from your final
bot's games; an early checkpoint may never have reached those situations itself.
