# Quickstart

Five minutes from nothing to a look inside a real (tiny) bot - plus the time it takes to download PyTorch the first
time. Then the same on yours.

## 1. Install

You need Python 3.11 or newer. Make a fresh environment so nothing else gets in the way:

```bash
python -m venv .venv
.venv\Scripts\activate             # Windows
source .venv/bin/activate          # macOS / Linux
pip install "cds-interpret[rlgym,rlbot]"
```

This pulls in PyTorch too, which is the biggest download. A CPU build is all the toolkit needs; if you already have
PyTorch in this environment, it is used as it is. On Linux, pip picks PyTorch's GPU build, which is several GB; to get
the small CPU build instead, install it first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Check what's ready:

```bash
cds-interpret doctor
```

```text
cds-interpret 0.1.0 on Python 3.12.10 (Windows)
  torch      2.14.1+cpu
  numpy      1.26.4
  RocketSim  2.2.1
  rlgym      2.0.1
  rlbot      2.0.0-beta.54
  meshes     ...\site-packages\rlgym\rocket_league\sim\collision_meshes
Ready: probes, features, edits, reports (always); recording in RocketSim; RLGym bridges; the RLBot overlay and recorder
```

The extras in the install line bring two things. `rlgym` brings RLGym (the Python toolkit most Rocket League bots are
trained with) and with it RocketSim, a fast Rocket League simulator, and the arena files it needs, so the toolkit can
record games. `rlbot` brings RLBot (the framework that runs bots in the real game), for the live overlay and for
recording real matches.
On Python 3.13, RLGym doesn't install yet: leave it out (`[rlbot]` only). Everything else still works, and the demo
uses random game moments instead of recorded games.

## 2. The tour, on the demo bot

```bash
cds-interpret demo
```

The demo bot is a small network that learned to copy a simple scripted player. When the ball is heading into its own
goal, or it's on the wrong side of the ball, it drives home. Otherwise it drives at a point just behind the ball,
lined up with the other goal. The tour has four steps:

```text
Step 1 of 4 - recording 12 games (60 s each) against a ball chaser in RocketSim
  game 1/12: blue 3 - orange 0  (300 moments so far, 1 s)
  ...
Step 2 of 4 - asking every layer questions about the game (probes)
  ...
  concept                            layer      score  input  random  measure steady
  ----------------------------------------------------------------------------------
  Worked out by the network (at least 0.05 better than both):
  ball_dist (distance to the ball)   body.1      0.89  -0.04    0.50  R2      yes
  land_dist (distance to landing)    body.1      0.64   0.07    0.44  R2      so-so
  supersonic                         body.3      0.74   0.64    0.68  AUROC   so-so
  Readable from the raw input or random weights too:
  goal_side (goal side)              body.3      0.98   0.93    0.94  AUROC   yes
  ...
  ... and 14 labels not scored (too few examples): summary() lists why

Step 3 of 4 - finding the separate ideas in its last hidden layer (a sparse autoencoder)
A sparse autoencoder of body.3 found 512 features (16 active at a time).
It keeps 98.5% of the layer's variance on held-out games; ...
  feature 139   fires on 41.6% of moments   lines up with: not goal side (AUROC 0.88), threat on goal (AUROC 0.87),
          about to concede (AUROC 0.84)
  ...

Step 4 of 4 - switching ideas off to see whether the bot uses them (edits)
  feature 139 (not goal side?) switched off: the bot's choice changes on 25% of the 370 held-out moments it fires on
  'goal_side' probe direction removed at body.3: the bot's choice changes on 15% of moments (KL 0.125)
  a random direction removed at body.3 (the control; 3 tries): the bot's choice changes on 10% of moments (KL 0.022)
  the autoencoder's rebuild in place of body.3: the bot's choice changes on 15% of moments (KL 0.073)

All done. The full picture - every probe, the features with their moments drawn on the field, the edits - is in
cds-demo\report.html
```

Here's what each step did, in plain words:

1. **Recording.** It played 12 short games and saved "moments": what the game looked like, what the bot's network
   saw, and what it chose.
2. **Probes.** A probe is a small readout that asks the network one question, like "how far away is the ball?". The
   toolkit asked dozens of questions at each layer, using some games to learn the readout and other games to score
   it. The table shows the best answers next to two baselines: the same question asked of the bot's raw input, and of
   a copy of the network with random weights. For a yes / no question the score is an **AUROC** (0.5 is a guess, 1.0
   perfect); for a number it's **R2** (0 is a guess, 1 perfect).
3. **Features.** A sparse autoencoder is a tool that finds the separate ideas a network uses. It found 512 of them in
   the bot's last hidden layer - at any moment only 16 are switched on - and listed the ones that line up best with
   things happening in the game.
4. **Edits.** It switched an idea off inside the network and counted how often the bot then chose differently. **KL**
   measures how far the bot's choices moved overall: 0 is not at all.

Open `cds-demo/report.html` in your browser. A few things worth spotting (your numbers will be a little different):

- **Distance to the ball** is read almost perfectly from the first hidden layer (0.89), and not at all from the raw
  input (-0.04). The network worked that distance out for itself.
- **Goal side** (being between the ball and your own goal) is easy to read everywhere - even the raw input gives it
  away (0.93). So the probes alone don't show anything learned there. But removing that direction from the last
  layer moves the bot's choices about six times as much as removing a random direction (KL 0.125 against 0.022):
  the bot *uses* it. That's no surprise - the rule it copied depends on it. **The random direction is the control**:
  an edit means something only when it moves the bot much more than its control does.
- **Feature 139** fires when the bot is on the wrong side of the ball with its goal in danger - exactly when the rule
  it copied says "drive home". Switching it off changes a quarter of its choices at those moments.
- **Some labels aren't scored.** Twelve one-minute games against a ball chaser don't have enough threats or demo
  chances to learn from. In your own scripts, `print(study.probes().summary())` lists every skipped label and why.

## 3. Your own bot

You need three things about your bot:

- **its network** - any `torch.nn.Module`;
- **its input** - a function that builds what your network reads from the game state (or skip this and
  [record the real game](#no-input-function-record-the-real-game) instead);
- **its actions** - the list of controls behind each action number, if your bot picks from a list.

Here's a complete example with a small network of its own. Copy it into a file and run it:

```python
import numpy as np
import torch
import cds_interpret as ci

torch.set_num_threads(2)


class MyNet(torch.nn.Module):                      # stands in for your bot's network
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Linear(18, 64), torch.nn.ReLU(),
                                          torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 90))

    def forward(self, x):
        return self.layers(x)


def my_obs(frames, me):
    """What the network reads, for car slot me[i] at moment i: the ball, my car and the nearest opponent,
    all seen from my team's side."""
    rows = np.arange(len(frames))
    side = ci.frames.team_sign(frames.car_team[rows, me])        # +1 blue, -1 orange
    opp = ci.frames.nearest_car(frames, me)                      # the nearest opponent's car slot
    parts = [
        ci.frames.canon(frames.ball_pos, side) / 4096,           # flipped for orange, so both attack "up"
        ci.frames.canon(frames.ball_vel, side) / 2300,
        ci.frames.canon(frames.car_pos[rows, me], side) / 4096,
        ci.frames.canon(frames.car_vel[rows, me], side) / 2300,
        ci.frames.canon(frames.car_pos[rows, opp], side) / 4096,
        ci.frames.canon(frames.car_vel[rows, opp], side) / 2300,
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)     # [N, 18]


net = MyNet()                                       # load your weights here: net.load_state_dict(torch.load(...))
policy = ci.Policy(net, obs=my_obs, actions=ci.lookup_table(), name="my bot")

moments = ci.record.rocketsim(policy, games=12, seconds=60, opponent="chaser", out="recordings/my-bot",
                              if_exists="reuse")       # run it again: it reuses these games
study = ci.Study(policy, moments)
print(study.probes().summary(top=10))
features = study.features(study.layers[-1])
print(features.summary(top=5))
print("Open", study.report("my-bot.html"), "in your browser.")
```

You should see the games being recorded, then two summaries, then the report's name - a few minutes in all on a
laptop. Run it again and it reuses the recorded games instead of playing new ones.

This example network is untrained, and the table shows it in an instructive way:

```text
  Worked out by the network (at least 0.05 better than both):
  ttb_opp (opponent's time to ball)  layers.3    0.32   0.07    0.14  R2      no
  outlook                            layers.3    0.40   0.29    0.24  R2      no
  ...
  Readable from the raw input or random weights too:
  concede_soon (about to concede)    layers.3    0.99   0.98    0.98  AUROC   no
  ...
```

An untrained network *is* a network with random weights, so whatever it seems to have "worked out" is luck: one random
network happening to read a few things a bit better than another. Notice the small leads, the low scores and the "no"
in the steady column - the signs of noise. A trained bot shows big leads on things that matter, steady answers, and
edits that change its play. (The features of an untrained network line up with the game too, for the same reason: its
layers still mix the game's numbers, at random.)

To use your real bot, swap `MyNet` for your network class and load its weights, and make `my_obs` build exactly what
your bot builds, in the same order. `frames` has every car (`frames.car_pos` is `[moments, cars, 3]`), so team-mates
and opponents are there too; `ci.frames.nearest_car(frames, me, "mate")` finds the nearest team-mate.

**Check your input function once.** If your bot runs in RLBot, record a match with it
([below](#no-input-function-record-the-real-game)): the recorder keeps the inputs your network really saw. Then:

```python
real = ci.Moments.load_all("recordings")
mine = policy.observe(real.frames, real.me).numpy()     # what my_obs builds for the same moments
print(abs(mine - real.obs).max())                        # about 0: my_obs builds what your bot builds
```

The opponent can be `"self"` (your bot against itself), another `ci.Policy`, `"chaser"` or `"idle"`. More games give
steadier numbers: 20 or more is a good start.

### Your bot uses RLGym?

Hand the toolkit your observation builder and action parser directly:

```python
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.action_parsers import LookupTableAction

policy = ci.Policy(net, obs=ci.rlgym.obs_fn(DefaultObs(zero_padding=3)),
                   actions=LookupTableAction.make_lookup_table())
```

### No input function? Record the real game

If your bot runs in RLBot (v5) and you'd rather not write `my_obs`, wrap your bot class where it starts. It plays
exactly as before and saves what its network saw:

```python
from cds_interpret.record import with_recorder

if __name__ == "__main__":
    with_recorder(MyBot, out="recordings", network="policy")("me/my-bot").run(wants_ball_predictions=True)
```

`MyBot` is your bot's class, `"me/my-bot"` its agent id (the one in its bot.toml), and `network` the name of the
attribute your bot keeps its network in. After a few matches:

```python
moments = ci.Moments.load_all("recordings")
study = ci.Study(ci.Policy(my_network, actions=my_actions), moments)     # no obs function needed
```

`my_actions` is your bot's action table: one row of 8 controls per action number, or `ci.lookup_table()`.

## 4. Watch it think, live

Turn what you found into a live panel. First name a few of the ideas: the [naming guide](howto/name-features.md) shows
how, and leaves you a `names.toml` file. Then:

```python
pack = study.make_pack(study.layers[-1], names="names.toml",
                       probes={"First to the ball": "first_to_ball", "Time to the ball": ("ttb_me", {"unit": "s"})})
ci.overlay.save_pack(pack, "thoughts.npz")
```

And in your bot's entry point:

```python
from cds_interpret.overlay import with_thoughts

with_thoughts(MyBot, pack="thoughts.npz", network="policy")("me/my-bot").run(wants_ball_predictions=True)
```

Turn on rendering in RLBot and start a match. [The overlay guide](howto/overlay.md) has the details, and
`examples/rlbot_demo_bot` is a complete bot folder to try first (it's in the repository, not in the pip install:
`git clone https://github.com/adventure-gpt/cds-interpret` gets it). In 2v2 and 3v3 every car gets its own landing
ring and the panel follows the car you watch ([team modes](howto/team-modes.md)). Add `policy=` to see *why* it makes
each move as well ([why this move, and what it's looking at](howto/why-and-looking.md)).

## Where next

- [How it works](concepts.md) - what every number means, and how to read them without fooling yourself.
- [Name your bot's features](howto/name-features.md) - the full loop, from numbers to names you trust.
- [Customize the panel](howto/customize-the-panel.md) - make the overlay your own.
- [Why this move, and what it's looking at](howto/why-and-looking.md) - the panel's two rows about each decision.
- [Coming from rlgym-ppo or GigaLearn](coming-from.md).
