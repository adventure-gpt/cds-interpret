# Watch your bot think in RLBot

[RLBot](https://rlbot.org) is the framework that runs bots in the real Rocket League. Add one line to your RLBot bot
and, while it plays, a panel in the corner of the screen shows what its network is noticing:

<p align="center"><img src="../../assets/overlay-panel.svg" alt="The thoughts panel" width="300"></p>

- **Outlook** - a bar from the middle. Green and to the right when the bot expects to score; red and to the left when
  it's in trouble.
- **Next move** - the action it leans toward, in words, and how sure it is.
- **Why this move** - the ideas (named features) pushing it toward that move, strongest first, and the one pushing
  hardest against it.
- **Looking at** - what the move rests on: its own car, the ball, the opponents, the team-mates, as one coloured bar.
- **Noticing** - the ideas firing right now. Each one stays on the list a moment after it stops, so nothing flickers.
- **Readings** - the probes you chose: a chance with a bar underneath, a time in seconds.
- **On the field** - a bright blue ring with a crosshair where the network thinks the ball will come down, joined to
  the car (one for each of your bot's cars in 2v2 and 3v3), a white cross where RLBot's own prediction says it will,
  and the idea driving the move floating above the car, in its team's colour.

The panel keeps the same size the whole match: every row keeps room for its longest answer, so nothing jumps around
while you read it. "Why this move" and "Looking at" need one more line in your bot's entry point -
[see below](#why-this-move-and-looking-at).

Your bot plays exactly the same with the panel as without it. [Here's why](#why-it-cannot-change-how-your-bot-plays).

## Try it on the demo bot first

`examples/rlbot_demo_bot` is a complete RLBot v5 bot folder with the panel ready to go. It's in the repository, not
in the pip install, so get a copy first:

1. `git clone https://github.com/adventure-gpt/cds-interpret`, and install with the `rlbot` extra (see the
   [quickstart](../quickstart.md)).
2. In RLBot, add `examples/rlbot_demo_bot/thoughts.bot.toml` as a bot. Its pack, `thoughts.npz`, is already in the
   folder.
3. Turn on **rendering** in RLBot's match settings, and start a match.

## On your own bot

### 1. Make a pack

A **pack** is a small file that says what to show: which ideas (features) and which questions (probes) to read, and
from which layer. Make it from a study of your bot:

```python
import cds_interpret as ci

study = ci.Study(policy, moments)
study.probes()
study.features("layers.3")
pack = study.make_pack(
    "layers.3",
    names="names.toml",                                   # the features you named (see the naming guide)
    probes={"First to the ball": "first_to_ball",         # name on the panel: label
            "Time to the ball": ("ttb_me", {"unit": "s"})},
    title="MY BOT'S MIND",
)
print(ci.overlay.save_pack(pack, "thoughts.npz"))      # thoughts.npz - a few kilobytes
```

A few things `make_pack` does for you:

- **Features** come from your names file - only the names you confirmed. Each gets a "firing" threshold worked out
  from your recorded games (you can override it in the names file or the panel file).
- **Probes** are given as `{name on the panel: label}`. Yes / no labels show as a chance with a bar; numbers show as a
  value. Add `{"unit": "s"}` for seconds.
- **The landing ring** is read from whichever layer gives the steadiest landing point - often an early one. Pass
  `land_layer=` to choose yourself.
- **The outlook** comes from a probe of the `outlook` label, when your recordings have it (recordings from RocketSim
  or RLBot do). If your bot has a critic, you can use that instead (below).
- **For "why this move" and "looking at"**, the pack also keeps each feature's decoder direction and 32 real moments
  from your recordings (`decoders=`, `donors=` to change that).

Make the pack from the very same network your bot runs - the pack names its layers by their path in that network.

No names file yet? Pass features directly: `features={"lined up for a shot": 117}`. Or skip them and show only probes.

### 2. Wrap your bot

Where your bot starts - the line that says `MyBot("me/my-bot").run(...)` - wrap the class:

```python
from cds_interpret.overlay import with_thoughts

if __name__ == "__main__":
    with_thoughts(MyBot, pack="thoughts.npz", network="policy")("me/my-bot").run(wants_ball_predictions=True)
```

`MyBot` is your bot's class and `"me/my-bot"` its agent id (the one in its bot.toml); keep whatever your `run(...)`
already passes. `wants_ball_predictions=True` lets the panel draw RLBot's own landing cross.

`network` is the name of the attribute your bot keeps its torch network in (set in `initialize()`). Use a dotted name
for a nested one (`"agent.actor"`), or leave it out and the toolkit picks the largest network it finds on your bot.

Other options:

| option | what it does |
|---|---|
| `panel="compact"` | the built-in compact panel: the same readings in about half the room, on the right below the game's event feed - for watching a match ([customize the panel](customize-the-panel.md#a-smaller-panel-ready-made)) |
| `panel="panel.toml"` | your own panel design - see [customize the panel](customize-the-panel.md) |
| `hooks={"name": function}` | functions your panel file uses for custom rows or markers |
| `policy="my_module:make_policy"` | your bot's `ci.Policy` (the same network, with its obs function), built in the overlay's own process: switches on "why this move" and "looking at" ([how](why-and-looking.md)), and the outlook bar from its critic if it has one |
| `critic="my_module:make_critic"` | use your critic for the outlook bar: `make_critic()` returns a function from inputs to values. It is loaded in the overlay's own process, never in your bot's |
| `input_layer="actor"` | the module whose input is your observation, when your bot doesn't call the network itself on it (read for `critic=` and `policy=`) |
| `title=`, `x=`, `y=`, `draw_every=` | quick overrides of the panel's title, position and drawing rate |
| `worker=False` | do the reading in your bot's process after the controls are sent (for debugging only) |
| `error_log="overlay_errors.log"` | also write the overlay's errors, with where they happened, to this file |

In 2v2 and 3v3 every car of your bot gets its own landing ring, and the panel shows the car you follow with the
spectator camera - a `Hivemind` bot or one copy of your bot per car alike. [Team modes](team-modes.md) explains the
choices (a fixed car, a random one, only while you watch).

When the other team shows its thoughts too - your bot on both sides, or another bot with its own overlay or a
[watch script](watch-a-bot.md) - the overlays share **one panel**, where your panel file puts it. It shows the car the
camera is on, whichever bot drives it; only that bot's overlay draws it. It keeps one size, the biggest of their
panels, so only what's inside changes as the camera moves; each team's title and rings take its colour. They find each
other through small notes in a temp folder; nothing to set up. [Both teams](team-modes.md#both-teams-showing-their-thoughts).

### 3. Play

Add your bot's `bot.toml` to RLBot, turn on **rendering** in the match settings, and start a match. The panel appears
a second or two in, once the overlay's helper process has started; your bot plays from the very first tick.

### Why this move and Looking at

These two rows need your network in the overlay's helper process. Write a small function that builds your bot's
`ci.Policy` - the same network with its weights, its obs function and its action table - and name it:

```python
with_thoughts(MyBot, pack="thoughts.npz", network="policy", policy="my_bot.interp:make_policy")
```

[Why this move, and what it's looking at](why-and-looking.md) shows the function, how to read both rows, and what
they cost (a few milliseconds, in the helper process). Without `policy=` they stay hidden; the rest of the panel is
the same.

## Why it cannot change how your bot plays

1. **It only listens.** The overlay copies your network's layers with read-only hooks while your bot's own forward
   pass runs. Your bot's outputs stay bit-for-bit the same; the tests check it.
2. **It works after your bot.** Everything else happens after your controls have been sent for that tick.
3. **It works somewhere else.** The reading runs in a separate helper process at low priority, with one thread, so it
   never competes with your bot's own threads. If the helper is busy, it simply skips a decision. With `policy=`, the
   helper builds its own copy of your network for "why this move" and "looking at"; your bot's is never touched.
4. **It never makes your bot wait.** Nothing is sent to the helper until it says it's ready, a separate thread does the
   sending, and nothing is drawn on the same tick as a decision.
5. **One bad moment doesn't stop it - or freeze it.** An error in one frame is written to your bot's log (the first
   few, then every 200th, each with where it happened) and the overlay carries on with the next frame. Packets without a
   ball, during the replay after a goal, are skipped. Only a long run of failures in a row - about five seconds -
   switches the overlay off, and then it takes its drawing off the screen, so a panel never sits frozen. A helper
   process that stops is started again. Your bot keeps playing whatever happens. `error_log="overlay_errors.log"`
   also writes the errors to a file.

## When the panel doesn't show

**Nothing at all.** Turn on rendering in RLBot's match settings. Then give it a second or two after kickoff.

**The bot's log says `cds-interpret: thoughts overlay: frame ... failed (...)`.** One frame went wrong and the
overlay carried on; the message says what and where. If it keeps happening, the same message comes every 200th time
(pass `error_log=` to collect them in a file), and after about five seconds of nothing but failures the overlay
switches itself off and clears the screen.

**The bot's log says `cds-interpret: thoughts overlay off (...)`.** The reason is in the brackets. The common ones:

```text
cds-interpret: thoughts overlay off (... has no layer 'layers.3'. Did you mean: 'layers.2'? ...)
```

The pack was made from a different network than the one your bot runs. Make it with `ci.Policy(the_same_module, ...)`.

```text
cds-interpret: thoughts overlay off ([Errno 2] No such file or directory: 'thoughts.npz')
```

The path is relative to where RLBot starts your bot - usually its own folder. Use a full path, or one built from
`Path(__file__).parent`.

```text
cds-interpret: thoughts overlay off (panel.toml [panel]: unknown key 'widht'. Did you mean 'width'? ...)
```

A typo in your panel file. Fix it and restart the match.

**Two panels on top of each other in 2v2 or 3v3.** The copies of your bot don't recognise each other: start them all
from the same `bot.toml`. With bots on both teams showing thoughts, they share one panel by themselves - unless a
panel file says `share = "none"`. See [team modes](team-modes.md#when-something-looks-wrong).

**"Why this move" or "Looking at" doesn't show.** The bot's log says why when the overlay starts; usually `policy=`
is missing, or the pack is from before 0.2. See [the rows' guide](why-and-looking.md#when-something-goes-wrong).

**The landing ring jumps around.** Read it from a steadier layer: `study.steadiness(layer)` for each layer, or let
`make_pack` pick (it does by default). See [probes you show live](probes.md#probes-you-show-live).

More in [when something goes wrong](../troubleshooting.md#the-overlay).
