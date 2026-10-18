# Thoughts for a bot you can't modify

The overlay usually lives inside your bot: one line in its entry point. But some bots you can't change - a compiled
program, a bot written in another language, a friend's bot you have the weights of. For those, a small RLBot
**script** draws the same panel and landing rings over the bot's cars while the bot itself plays, untouched.

```text
  the bot (its own program) ──── drives its cars, exactly as always
  the script (display only) ──── reads each game packet
                                  -> rebuilds the bot's input with your obs function
                                  -> runs the bot's network (its weights, in memory)
                                  -> draws the thoughts panel and a ring per car
```

It looks just like the overlay: the same panel, the same [team modes](team-modes.md), the same
[panel file](customize-the-panel.md). The difference is where the numbers come from: the script works the bot's
thoughts out *beside* it, from the game, with a copy of its network.

## Five minutes on the demo bot

`examples/watch_a_bot` watches the demo bot this way (the demo bot can be hooked, but here it plays plain, as if it
couldn't be):

1. In RLBot, add `examples/rlbot_demo_bot/bot.toml` as a bot - the plain one, no overlay inside.
2. Add `examples/watch_a_bot/watch.script.toml` as a **script** (RLBot lists scripts apart from bots).
3. Turn on rendering and start the match. The panel reads "DEMO BOT (WATCHED)".

## For your bot

You need three things.

**1. The bot's network with its weights, in PyTorch.** Most bots trained with PyTorch ship their weights; load them
into the same architecture.

**2. How it builds its input from the game** - its observation, rebuilt from the game state. This is the hard part:
it's a port of the bot's own code, written as an `obs(frames, me)` function (see
[bring your own bot](bring-your-own-policy.md)). The script fills `frames` from each packet, with every car's last
controls in `prev_controls`. Check your port before trusting the panel: run it on recorded moments and compare with
what the real bot does - if the port's moves match the real bot's most of the time, the panel shows what the bot
thinks; if not, it shows what your port thinks.

Put both in a function:

```python
# my_port.py, beside the script
import torch
import cds_interpret as ci
from their_bot_arch import TheirNet          # the architecture, rebuilt in PyTorch
from my_obs_port import their_obs            # their observation, ported: obs(frames, me) -> [N, D]

def make_policy():
    net = TheirNet()
    net.load_state_dict(torch.load("their_weights.pt", map_location="cpu", weights_only=True))
    return ci.Policy(net, obs=their_obs, actions=their_action_table())
```

**3. A pack made for that network.** Study it like any bot - its port plays in RocketSim
(`ci.record.rocketsim(policy, ...)`), so you can record it, probe it, find its features and make a pack. To name its
features quickly, let the labels name them: `ci.names.suggest_by_label(features)` gives each label the feature that
detects it best ([naming features](name-features.md)).

Then write the script:

```bash
cds-interpret watch-template my_watch
```

```text
Wrote my_watch\watch.py and my_watch\watch.script.toml. Edit the three lines marked EDIT in watch.py, then add
watch.script.toml to your match as a script (docs/howto/watch-a-bot.md).
```

```python
# my_watch/watch.py - the three lines to edit
PACK = "thoughts.npz"                 # the pack you made
POLICY = "my_port:make_policy"        # the function above
CARS = "TheirBot"                     # a piece of the bot's name in the game
```

`CARS` matches the players whose names contain it, ignoring case: "TheirBot" matches "TheirBot" and "TheirBot (2)".
Give a list of player numbers instead (`CARS = [0, 2]`) or a function of the packet if names don't work.

Add `watch.script.toml` to the match as a script, beside the bot. Check your pack and port before a match:

```bash
cds-interpret check-panel panel.toml thoughts.npz --policy my_port:make_policy
```

### When the obs function isn't enough

The script builds each input from the game state with your policy's obs function. Some bots need more than a packet's
game state gives - boost pads, timers they count between packets, which actions their state allows. Give the script an
`observe` function instead, which gets the raw packet:

```python
class Observer:
    def __init__(self, model):
        self.port = None                                 # kept between packets: timers, a pad map ...
    def __call__(self, packet, cars, field_info):
        if self.port is None:
            self.port = TheirObsPort(field_info)
        x, mirrored = self.port.observe(packet, cars)    # their input for these cars, and how they see the field
        return {"obs": x, "mirrored": mirrored, "mask": self.port.allowed_actions(packet, cars)}

watch_thoughts(POLICY, PACK, cars=CARS, observe=("my_port:Observer", {"model": "their_weights.pt"}))
```

It returns the inputs `[n, ...]`, or a dict with `"obs"` and, when the bot has them, `"mirrored"` (the cars that see
the field turned round: the landing ring and the move's words are turned back), `"mask"` `[n, A]` (the actions each car
may take: the panel's move is chosen among them, as the bot chooses) and `"state"` (more of the game, for a critic or
a codec). Given as `"module:factory"` (with its keyword arguments as a pair), it is built when the script starts.

## What it can and can't show

| row | shows when |
|---|---|
| Next move, Noticing, probes, landing rings | always - they come from the network and the pack |
| Outlook | your policy has a critic (`ci.Policy(..., value=...)`) or the pack has an outlook probe; most shipped bots don't include their critic, so it's usually left out |
| Why this move | the pack has decoder directions (packs from `make_pack` do) |
| Looking at | the pack has recorded moments of that mode (`add_donors`) |

Rows that can't be worked out take no room. Two more limits:

- **Memory.** A bot that remembers between decisions - a recurrent network, previous actions it never shows the game -
  is only seen as far as your obs function can rebuild that memory from the packets.
- **Different brains for different modes.** Some bots play 1v1 with one network and team modes with another. Make a
  pack for the one you have, and say when it applies: `watch_thoughts(..., team_sizes=[2, 3])` draws only in 2v2 and
  3v3.

A bot that sees the field mirrored at some moments (one side as the other) needs its `mirror=` function in the
policy, so the landing ring and the move's words come out right - see
[probes you show live](probes.md#a-bot-that-sees-the-field-mirrored).

Beside your own bot's overlay on the other team, the watch script and the overlay share one panel: it shows the car
the camera is on, whichever of the two bots drives it, at one size, the biggest of the two panels
([both teams](team-modes.md#both-teams-showing-their-thoughts)). Give the script its own `title=` so you can tell
whose thoughts are on screen.

## Safe by design

- **It can't change how the bot plays.** It's a separate program the bot doesn't know about; it only reads the game
  and draws.
- **Nothing is written.** The weights are read into the script's memory and stay there. (With `error_log=` it writes
  its own errors to that file, and nothing else.)
- **It keeps going.** A packet without a ball (the replay after a goal) is skipped, and an error in one draw is
  logged and the next draw goes on. Only a long run of failures in a row switches it off, and then it takes its
  drawing down.

## When something goes wrong

**Nothing is drawn.** Rendering must be on, and `CARS` must match the bot's name in the game (the scoreboard shows
it). The script's log says `cds-interpret watch: on - ...` when it started, or why it didn't.

**The panel is there but the moves look wrong.** Your observation port doesn't match the bot's. Compare them on
recorded moments, field by field.

**The rings sit on the wrong side half the time.** The bot sees the field mirrored at times - give the policy its
`mirror=` function (see above).
