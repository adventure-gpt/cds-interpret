# Bring your own bot

Everything in the toolkit starts with a `ci.Policy`: your bot's network, wrapped with a few facts about it. This page
shows how to wrap any kind of bot.

## The short version

```python
import cds_interpret as ci

policy = ci.Policy(
    net,                        # your torch.nn.Module
    obs=my_obs,                 # how to build its input from the game state (optional - see below)
    actions=ci.lookup_table(),  # the controls behind each action number
    name="my bot",
)
print(policy)
```

```text
Policy('my bot', MyNet, 11,226 parameters, output='logits', 90 actions)
```

(This is the small network from the [quickstart](../quickstart.md#3-your-own-bot).) Check the layers you can study by
running a few inputs through it:

```python
x = policy.observe(ci.demo.synthetic_moments(2, 4).frames, 0)    # 8 random moments, seen from car 0
for layer in policy.layers(x):
    print(layer)
```

```text
(model)                          MyNet              8x90
layers                           Sequential         8x90
layers.0                         Linear             8x64
layers.1                         ReLU               8x64
layers.2                         Linear             8x64
layers.3                         ReLU               8x64
layers.4                         Linear             8x90
```

The `ReLU` rows (`layers.1`, `layers.3`) are the hidden layers; the toolkit studies those by default.

## The input function

`obs(frames, me)` builds your network's input from the game state. It gets:

- `frames`: N moments of game state ([all the fields](../api.md#frames)) - for example `frames.ball_pos` is `[N, 3]`
  and `frames.car_boost` is `[N, cars]`;
- `me`: which car is "your bot" at each moment, as car slot numbers `[N]`.

It returns your bot's input, one row per moment: `[N, D]` (numpy or torch).

You only need it when the toolkit has to build inputs itself - recording in RocketSim, or trying random moments. If
you record your bot in the real game or in your own RLGym environment, the recording keeps the exact inputs your
network saw, and you can leave `obs` out.

Most bots look at the field from their own team's side, so orange sees the field turned around. Two helpers do that:

```python
import numpy as np

def my_obs(frames, me):
    rows = np.arange(len(frames))
    side = ci.frames.team_sign(frames.car_team[rows, me])      # +1 for blue, -1 for orange
    ball = ci.frames.canon(frames.ball_pos, side)              # turned around for orange: both attack +y
    car = ci.frames.canon(frames.car_pos[rows, me], side)
    boost = frames.car_boost[rows, me][:, None] / 100
    return np.concatenate([ball / 4096, car / 4096, boost], axis=1).astype(np.float32)
```

A frame doesn't carry everything the game has. Boost pad timers and jump or flip timers aren't there, so fill those
neutrally (zeros). Recorded frames do carry `frames.prev_controls` `[N, cars, 8]`: every car's controls in the step
before, if your bot reads its previous action.

**Check it once.** Take a few moments your bot really played, build its input both ways - with your bot's own code and
with `my_obs` - and compare. A small mismatch here quietly spoils everything after it.

### Your bot uses an RLGym observation builder?

Then you don't need to write one. Hand over yours:

```python
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.action_parsers import LookupTableAction

policy = ci.Policy(net, obs=ci.rlgym.obs_fn(DefaultObs(zero_padding=3)),
                   actions=LookupTableAction.make_lookup_table())
```

`obs_fn` turns each moment into an RLGym game state and runs your builder on it. A builder that remembers past steps
(stacked frames, previous actions) sees each moment fresh, as if it were the first step. To keep the exact inputs of
real play, [record from your RLGym environment](record-moments.md#your-rlgym-environment) instead. This needs the
`rlgym` extra.

## What your network outputs

Tell the toolkit what comes out of your network with `output=`:

| `output` | your network returns | typical for |
|---|---|---|
| `"logits"` (default) | a score per action | most PPO bots |
| `"probs"` | a probability per action | a network that ends in a softmax |
| `"log_probs"` | a log-probability per action | a network that ends in a log-softmax |
| `"controls"` | 8 controls between -1 and 1 | a bot that outputs controls directly |

## The action table

Most Rocket League bots choose from a fixed list of actions, each a set of 8 controls: throttle, steer, pitch, yaw,
roll, jump, boost, handbrake. `actions=` is that list, one row per action, in the order your network numbers them.

`ci.lookup_table()` builds a 90-action table in the layout many bots use. It's close to RLGym's `LookupTableAction`
table but not identical: RLGym's also holds the handbrake on directional jumps. If your bot was trained with a
different table - even one row different - pass yours (for RLGym's: `LookupTableAction.make_lookup_table()`). The toolkit needs the table to play your bot in RocketSim and to say its actions in words
("throttle, steer left, boost").

### A bot that sees the field mirrored

Some bots turn the field round in their observation at some moments (one side seen as the other), and turn steer,
yaw and roll round again before sending them. Tell the toolkit when, with `mirror=` - a function of the game state
like the input function, returning True for each moment the bot sees mirrored:

```python
policy = ci.Policy(net, obs=my_obs, actions=MY_ACTIONS, mirror=lambda frames, me: my_mirror_flag(frames, me))
```

It then says the bot's moves the way they're really sent, and fits the live landing ring in the bot's own frame and
turns it back for drawing. See [a bot that sees the field mirrored](probes.md#a-bot-that-sees-the-field-mirrored).

## Networks with more than one part

If your network is an actor and a critic together, tell the toolkit which part picks the actions:

```python
policy = ci.Policy(model, forward=lambda x: model.actor(x), value=lambda x: model.critic(x).squeeze(-1))
policy = ci.Policy(agent, forward=agent.get_logits)        # a method of your network works too
```

`forward` is how to run the network on a batch of inputs. `value` is optional; the live overlay can use it for its
outlook bar.

Layer names follow the object you pass in (`"actor.layers.3"`). The live overlay uses the same names, so wrap the same
object your bot runs.

**Per-car networks** (attention, or one encoder per car) work too: `obs` returns whatever shape your network takes
(`[N, cars, features]`, say). The most useful layer to study is one after the cars are pooled together, where a single
vector describes the moment.

**Several inputs** (an observation and a mask, say): pack them into one tensor in `obs` and unpack them in `forward`.

## TorchScript networks and C++ trainers

PyTorch can't attach listeners (hooks) to a TorchScript module, so the toolkit can't read its layers. For networks
built from standard layers - `Sequential`, `Linear`, `LayerNorm`, the usual activations, `Softmax` - make an ordinary
copy with the same weights:

```python
net = ci.eager_copy(torch.jit.load("policy.pt"))
```

Anything with custom code in its `forward`: rebuild the class in Python and load the weights into it with
`load_state_dict`. A network trained in C++ is the same story once its weights are in a file PyTorch can read. See
[coming from GigaLearn](../coming-from.md).

## Handy calls

```python
policy.layers(x)                    # every part of the network, its type and output shape
policy.default_layers(x)            # the hidden layers the toolkit studies by default
policy.activations(x, ["layers.3"]) # {"layers.3": [N, 64]}: what that layer computes for inputs x
policy.act(x)                       # the action number it picks for each input
policy.controls(x)                  # the 8 controls of that action
policy.random_like(seed=0)          # the same network with random weights (a baseline for probes)
```
