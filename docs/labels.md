# Labels: facts about each moment

A **label** is a fact about a moment, worked out from the game state alone: "the ball is going into my goal within
3 seconds", "I'm first to the ball", "the ball lands in 1.2 seconds". They're the questions the toolkit asks your
bot's layers, and the yardstick it holds its features up to.

Labels never look at your bot's own input, so every bot is measured the same way. Each one is worked out from the
point of view of your bot's car, on its own side of the field (as if it always attacked "up").

```python
lab = ci.concept_labels(frames, me)                 # the facts of each moment: {name: [N]}
lab["threat"][:5]                                   # array([0., 0., 1., 1., 0.], dtype=float32)
ci.describe_label("threat")                         # 'the untouched ball goes into our goal within 3 s'
ci.LABEL_KIND["threat"]                             # 'bin': a yes / no fact (numbers are 'reg')
```

Recordings come with all of them already worked out (`moments.labels`), including the "what happened next" ones.

## Every label

Yes / no labels are scored with AUROC, numbers with R² ([what those mean](concepts.md#probes-asking-the-network-a-question)).
A label is NaN (no value) where it doesn't apply - the landing point while the ball stays up, say. Distances are in
**uu** (Unreal units, the game's own: the field is 10,240 uu long and 8,192 wide, a car about 118 long, the ball's
radius about 93), speeds in uu/s (a car's top speed is 2,300).

| label | kind | meaning |
|---|---|---|
| `land_t` | number | seconds until the untouched ball next touches the floor |
| `land_x` | number | x of that landing point (team frame) |
| `land_y` | number | y of that landing point (team frame: + is toward their goal) |
| `land_dist` | number | distance from the car to that landing point (uu) |
| `ttb_me` | number | seconds this car needs to reach the ball |
| `ttb_opp` | number | seconds the nearest opponent needs to reach the ball |
| `ttb_mate` | number | seconds the nearest team-mate needs to reach the ball |
| `ttb_adv` | number | how much sooner this car reaches the ball than the opponents (s) |
| `first_to_ball` | yes / no | no other car reaches the ball sooner |
| `last_back` | yes / no | this car is its team's rearmost |
| `goal_side` | yes / no | this car is between the ball and its own goal |
| `boost` | number | boost / 100 |
| `boost_adv` | number | boost minus the opponents' mean boost, / 100 |
| `threat` | yes / no | the untouched ball goes into our goal within 3 s |
| `threat_t` | number | seconds until it does |
| `on_target` | yes / no | the untouched ball goes into their goal within 3 s |
| `demo_chance` | yes / no | an opponent can be demolished right now |
| `demo_ttc` | number | seconds to contact with that opponent |
| `kickoff` | yes / no | kickoff: the ball still on the centre spot |
| `in_air` | yes / no | wheels off everything |
| `on_wall` | yes / no | driving on a wall |
| `on_ceiling` | yes / no | driving on the ceiling |
| `height` | number | car height / 2044 |
| `speed` | number | car speed / 2300 |
| `supersonic` | yes / no | supersonic |
| `ball_height` | number | ball height / 2044 |
| `ball_dist` | number | car-to-ball distance / 5000 |
| `ball_vy` | number | ball speed toward their goal / 6000 |
| `has_flip` | yes / no | a jump or dodge is still available |
| `shot_soon` | yes / no | within 1 s this car puts the ball on target |
| `touch_soon` | yes / no | within 0.5 s this car touches the ball |
| `demo_soon` | yes / no | within 1 s this car demolishes an opponent |
| `demoed_soon` | yes / no | within 1 s this car is demolished |
| `goal_soon` | yes / no | within 3 s our team scores |
| `concede_soon` | yes / no | within 3 s the other team scores |
| `outlook` | number | goal_soon - concede_soon |

## Details

- **The ball's path.** `land_*`, `threat*`, `on_target` and the times to the ball follow the untouched ball's path:
  RocketSim's prediction when recording in RocketSim, RLBot's when recording the real game, else a simple flight model
  (gravity, drag, bounces off floor, walls and ceiling, the goal mouths open; no ramps or corners). Pass your own with
  `concept_labels(frames, me, path=[N, K, 3], path_t=[K])`.
- **Time to the ball** uses a simple reach model: full throttle, boost while it lasts, a turning cost, more time for a
  ball in the air; capped at the path's horizon (4 s). A demolished car cannot reach it.
- **last_back** is always 1 in 1v1: judge it in team modes.
- **demo_chance** is never set during a kickoff: two cars racing at each other for the ball meet the same geometry.
- **Future labels** are NaN where the recording ends inside their window; a goal ends the window (the kickoff that
  follows resets the game). `outlook` (goal_soon - concede_soon, from -1 to 1) stands in for a critic's value.
- Label values are in seconds, unreal units or fractions as written; `height`, `speed`, `ball_height`, `ball_dist` and
  `ball_vy` are scaled to about 0-1.

## Your own labels

Anything computable from the game state can join the built-in labels - probed, related to features and shown in the
study page like them:

```python
import numpy as np
import cds_interpret as ci

@ci.register_label("ball_in_our_half", kind="bin", doc="the ball is in our half")
def ball_in_our_half(inp):                  # inp: ci.labels.LabelInput
    return inp.ball[:, 1] < 0               # the ball's position in the team frame [N, 3]

@ci.register_label("boost_to_spare", kind="reg", doc="boost minus what a dash to the ball needs")
def boost_to_spare(inp):
    return inp.labels["boost"] - 0.3 * inp.labels["ttb_me"]   # the built-in labels are there too
```

A `LabelInput` holds `frames`, `me`, `sign` (+1 blue, -1 orange), `path` / `path_t` (the ball's path, world frame), and
in the team frame `car_pos`, `car_vel` [N, C, 3], `ball`, `ball_vel` [N, 3], `my_pos`, `my_vel` [N, 3]; `labels` holds
the built-in labels already computed. Registered labels are computed by `concept_labels` from then on (recorders
included); `ci.labels.unregister_label(name)` removes one.
