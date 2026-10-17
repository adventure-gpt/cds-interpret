# Record games

Everything the toolkit finds comes from **moments**: snapshots of your bot's games, each with what the game looked
like, what your bot's network saw, what it chose, and the facts worked out about it. This page shows the four ways to
get them.

| where | you need | best when |
|---|---|---|
| [RocketSim](#in-rocketsim) | an input function for your bot | you want lots of games, fast |
| [The real game](#in-the-real-game-rlbot) | your bot running in RLBot v5 | you want its exact inputs, or have no input function |
| [Your RLGym environment](#your-rlgym-environment) | your RLGym v2 setup | your bot was trained in RLGym |
| [Your own data](#your-own-data) | game states as arrays, RLGym states or RLBot packets | you already have recordings |

Whichever you use, you get a `ci.Moments` back. Save it with `out=` (or `moments.save(folder)`) and load it later:

```python
moments = ci.Moments.load("recordings/my-bot")
moments = ci.Moments.load_all("recordings")       # every recording under a folder, together
print(moments)
```

```text
Moments(9000 moments, 20 games, 36 labels, obs 30-d, tracks)
```

## In RocketSim

RocketSim is a fast Rocket League simulator - it plays a two-minute game in a few seconds.

```python
moments = ci.record.rocketsim(policy, games=20, seconds=120, opponent="self", out="recordings/my-bot")
```

```text
  game 1/20: blue 3 - orange 0  (450 moments so far, 6 s)
  game 2/20: blue 1 - orange 1  (900 moments so far, 11 s)
  ...
  saved 9000 moments to recordings/my-bot
```

The options you'll use most:

| option | what it does |
|---|---|
| `games`, `seconds` | how many games, and how long each is |
| `opponent` | `"self"` (your bot against itself, both sides recorded), another `ci.Policy`, `"chaser"` (drives at the ball), `"idle"`, or your own function |
| `team_size` | 1, 2 or 3 cars a side |
| `every` | keep a moment at every 3rd decision (the default); every decision is still kept for the "what happened next" facts |
| `out` | a folder to save to |
| `if_exists` | when `out` already holds a recording: `"error"` (the default - nothing gets mixed up by accident), `"reuse"` (load it and play nothing - for a script you run again), `"append"` (add these games), `"overwrite"` |
| `deterministic` | `True` (the default): your bot always picks its best action. `False`: it samples, as in training |
| `late` | how often a car's new controls arrive one tick late (default 5%). RocketSim is perfectly repeatable, so without this every game from the same kickoff would be the exact same game |

Each game starts from a random kickoff, and a goal starts a new one.

### The collision meshes

RocketSim needs the arena's shape: Rocket League's collision meshes, a folder holding `soccar/*.cmf`. They come from
the game itself, so this package doesn't include them. The toolkit looks for them, in order:

1. the `meshes=` argument: `ci.record.rocketsim(policy, meshes="path/to/collision_meshes")`;
2. the `CDS_INTERPRET_MESHES` environment variable;
3. a `collision_meshes` folder where you run your script;
4. the copy that comes with RLGym's simulator.

The easiest way is the last one: `pip install "cds-interpret[rlgym]"` (Python 3.11 or 3.12). Otherwise dump them from
your own copy of the game with RLArenaCollisionDumper. `cds-interpret doctor` shows where they were found.

## In the real game (RLBot)

Wrap your bot class where it starts. It plays exactly as before, and writes down what it saw:

```python
from cds_interpret.record import with_recorder
from my_bot import MyBot

if __name__ == "__main__":
    with_recorder(MyBot, out="recordings", network="policy")("me/my-bot").run(wants_ball_predictions=True)
```

Play some matches, then:

```python
moments = ci.Moments.load_all("recordings")
study = ci.Study(ci.Policy(my_network, actions=my_actions), moments)      # no input function needed
```

- `network` is where your bot keeps its torch network: an attribute name (`"policy"`, or `"agent.actor"` for a nested
  one), a function `bot -> network`, or leave it out to use the largest network on the bot.
- `input_layer` / `output_layer`: which part of the network takes your observation and which gives the action scores.
  Both default to the network itself.
- `every`: keep every 3rd decision (a decision is a tick where your network ran).

How it stays out of the way: it copies your network's input and output with read-only listeners while your bot's own
forward pass runs, so its outputs stay exactly the same. Everything else - reading the game state, saving - happens
after your controls are sent. If anything goes wrong in the recorder, it switches itself off and says why in the log;
your bot keeps playing.

Each match goes into its own folder under `out`, saved in pieces while it plays and at the end. Ask for RLBot's ball
prediction (`wants_ball_predictions=True`): the facts about the ball's flight use it. A `Hivemind` bot works too, as
long as its network handles its cars in order.

## Your RLGym environment

If your bot trains in RLGym v2, record it in the very same environment - same observation builder, same action parser -
so the moments hold exactly the inputs it saw in training:

```python
from rlgym.api import RLGym
env = RLGym(...)                      # your training environment (the reward doesn't matter here)
moments = ci.record.rlgym(env, policy, episodes=20, out="recordings/rlgym")
```

Your policy plays every car, and each episode is one game. If your action parser takes something other than an
action number, pass `to_action=` to convert. Record different team sizes into different folders.

## Your own data

Have game states already? Turn them into frames and moments:

```python
fr = ci.frames.from_rlbot_packet(packets)          # RLBot v5 packets
fr = ci.frames.from_rlgym(states)                  # RLGym game states (v1 or v2)
fr = ci.frames.from_arrays(ball_pos=..., ball_vel=..., car_pos=..., car_vel=..., car_fwd=..., car_up=...,
                           car_boost=..., car_on_ground=..., car_team=...)
moments = ci.Moments(fr, me=0, game=game_ids, step=steps).with_labels()
moments.save("recordings/mine")
```

`with_labels()` works out the facts of each moment. The "what happened next" facts (a goal soon, a touch soon) also
need the events of each step (`ev_touch`, `ev_demo_made`, `ev_demoed`, `goal`) and every step of each game; the
recorders above fill those in for you.

## Random moments, for practice

`ci.demo.synthetic_moments()` makes random game states, grouped into pretend games. They're handy for trying the
toolkit and for tests - but they aren't real play, so don't draw conclusions from them.
