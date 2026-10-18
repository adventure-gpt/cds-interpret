# When something goes wrong

Something didn't work? You're in the right place. Find the message you saw below; each one says what it means and
what to do.

A good first step for anything: run `cds-interpret doctor`. It lists what's installed and what's ready.

```text
cds-interpret 0.5.0 on Python 3.12.10 (Windows)
  torch      2.14.1+cpu
  ...
Ready: probes, features, edits, reports (always); recording in RocketSim; RLGym bridges; the RLBot overlay and recorder
```

Errors the toolkit raises on purpose explain themselves and suggest a fix. If you ever see a raw error from deep
inside PyTorch or numpy instead, that's a bug in the toolkit - please open an issue.

## Installing

**"No matching distribution found for rlgym-rocket-league", or pip tries to build numpy.**
RLGym needs an older numpy that doesn't install on Python 3.13 or newer. Either use Python 3.11 or 3.12, or leave the
`rlgym` extra out:

```bash
pip install "cds-interpret[rlbot]"
```

Everything works without it except the RLGym bridges. To record in RocketSim you then need its collision meshes
from somewhere else (see below).

**"Could not install packages due to an OSError ... HINT: ... Windows Long Path support".**
PyTorch has some very deep file paths, and Windows refuses paths over 260 characters unless long paths are switched
on. Make your environment in a short folder (like `C:\cds\.venv`) and install again, or switch on long paths in
Windows (search for "Enable Win32 long paths"; needs an administrator).

**"Could not find a version that satisfies the requirement rlbot".**
RLBot v5 needs Python 3.11 or newer. Check with `python --version`.

**PyTorch is huge, or I want a particular build.**
Install PyTorch first, the way [pytorch.org](https://pytorch.org) says for your machine; the toolkit then uses it.
A CPU build is all it needs:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**"... needs the 'rocketsim' package, which isn't installed yet."**

```text
Recording games in RocketSim needs the 'rocketsim' package, which isn't installed yet. Install it with:
    pip install "cds-interpret[sim]"
```

Run the line it gives you. The same goes for `[rlgym]` and `[rlbot]`.

## Recording

**"RocketSim needs the arena's shape - its collision meshes ..."**

```text
RocketSim needs the arena's shape - its collision meshes, a folder holding soccar/*.cmf - and couldn't find them.
It looked in:
  - ...\collision_meshes  (the current folder)
The easiest fix: pip install "cds-interpret[rlgym]" ...
```

RocketSim needs the arena's collision meshes, which come from the game. The `rlgym` extra brings a copy - that's the
easiest fix. Otherwise point to your own: `ci.record.rocketsim(policy, meshes="path/to/collision_meshes")`, or set the
`CDS_INTERPRET_MESHES` environment variable. [More about the meshes](howto/record-moments.md#the-collision-meshes).

**"This policy doesn't know how to build its input from the game state yet."**
Recording in RocketSim means the toolkit has to build your bot's input itself, so your `Policy` needs an `obs`
function. Either write one ([how](howto/bring-your-own-policy.md#the-input-function)), or record somewhere that keeps
your bot's own inputs: the real game, or your RLGym environment.

**"This policy has no action table, so its choices can't be turned into controls."**
Your bot picks actions by number; the toolkit needs to know the controls behind each number. Pass
`actions=ci.lookup_table()` if your bot uses the common 90-action table, or your own table otherwise.

**"recordings/my-bot already holds a recording."**
You ran a recording script a second time, and the toolkit won't write over your games by accident. Pass
`if_exists="reuse"` to the recorder to use the games already there (handy for a script you run again and again),
`"append"` to add new games to them, or `"overwrite"` to replace them - or load them yourself:
`moments = ci.Moments.load("recordings/my-bot")`.

**Every game looks the same.**
RocketSim is very repeatable, so a bot that always picks its best action plays the same game from the same kickoff.
The recorder breaks that up by delaying a car's new controls by one tick now and then (`late=0.05`, the default). If
you turned that off, turn it back on - or sample actions with `deterministic=False`.

**The same recording script gave slightly different games this time.**
That can happen: RocketSim's physics can come out a hair differently from one run of Python to the next, and one
different bounce makes a different game. It doesn't matter for the findings, but it does for feature numbers - so
keep the recording (and the dictionary) you name features from, rather than recording again.

**The RLBot recorder didn't write anything.**
It makes its folder once your bot's network has run. Check the bot's log for a line like
`cds-interpret: recorder off (...)` - the reason is in the brackets. Most often `network=` doesn't name the attribute
your bot keeps its network in (set it in `initialize()`), or the network is a TorchScript module (see below).

## Studying

**"These moments come from 1 game, but the toolkit needs at least 2."**
The toolkit learns on some games and scores on others, so it needs several. Record more: 10 or more gives numbers you
can trust.

**"Your network didn't accept these inputs."**

```text
Your network didn't accept these inputs (30 numbers per moment). PyTorch said: mat1 and mat2 shapes cannot be
multiplied (2x30 and 12x16). Usually the moments were recorded with a different bot, or the obs function builds a
different layout than the network expects.
```

Compare the number in brackets with what your network takes. Then check your `obs` function against your bot's own.

**"There's no layer called 'x' in your ..."**

```text
There's no layer called 'layer.3' in your Sequential.
Its layers are: '0', '1', '2'
(policy.layers(x) lists them with their types and output shapes.)
```

Layer names are paths inside your network, like `"layers.3"` or `"actor.body.1"`. Copy one from the list.

**"... were never called by forward(x)."**
That layer exists, but running your network doesn't go through it. For an actor-critic, point the toolkit at the
actor: `ci.Policy(net, forward=lambda x: net.actor(x))`, and study the actor's layers.

**"This network is a TorchScript module."**
PyTorch can't attach listeners to TorchScript modules. Convert it with `net = ci.eager_copy(net)` (works for networks
built from standard layers), or rebuild your network class in Python and load the weights into it.

**A label is missing from the probes.**
A yes / no label needs at least 20 examples in the training games, and labels that are mostly empty (like the
landing point when the ball stays up) need enough moments where they have a value. Record more or longer games.
Labels about what happened next (`goal_soon` and friends) need real recorded games - random demo moments don't have
them.

**"There's no probe for 'threat' at layer ..."**
Same cause: too few examples of that label. Threats are rare in short games against a weak opponent. Record more
games, or try the bot against itself.

**Every probe scores near 1.**
Look at the "input" column: if the raw input scores as well, the answer is already in your bot's observation. And
check the games differ from each other (see "Every game looks the same").

**The sparse autoencoder changes the bot's choices a lot.**
Train it longer (`steps=3000` or more), give it more features (`m=`) or more active at a time (`k=`), and record more
moments. Judge by the KL as well as the changed-choice share: a bot with many nearly-tied actions flips its choice
easily.

**"There are no features for layer 'x' yet."**
Train them first: `study.features("x")` - or load the ones you saved: `study.load_features("x.sae")`. Naming, checking
and packing features all start from there.

**"names.toml was written for another dictionary of body.3 ..."**
Feature numbers belong to one dictionary: a dictionary trained again (in a new session, or on other games) has
different features under the same numbers, so the toolkit won't put your names on them. Load the dictionary you named
with `study.load_features("body3.sae")` - save it with `feats.save("body3.sae")` next time, right after training it.
If you never saved it, name the new one afresh. [Coming back later](howto/name-features.md#coming-back-later).

**"Feature 22 of body.3 never fires on these moments."**
Usually the same story: a feature number from another dictionary. If it's really this dictionary's, the feature is
one of the few that never switch on - pick another.

**"The names file gives 'x' (feature 180) a threshold of 1.5, but it never got above 0.70 in these games."**
The threshold in your names file is higher than anything that feature ever reaches, so it would never light up on
the panel. The message gives its sizes; the comments in your names file do too (`# when it fires: 0.06 to 0.70,
typically 0.23`). Pick a threshold in that range, or delete the line to use the automatic one.

**"'threat on goal' isn't a name of a feature of body.3."**
`check_feature` and `feature_edit` take a feature's number or a name you confirmed - but only once the names are given
to the features: `feats.apply_names("names.toml")`.

## The overlay

**Nothing is drawn.**
Turn on rendering in RLBot's match settings. Then give it a second or two after the match starts while the overlay's
helper process starts up.

**The bot's log says `cds-interpret: thoughts overlay off (...)`.**
The overlay switched itself off; your bot is still playing. The reason is in the brackets:

- *"There's no layer called ..."* - the pack was made from a different network than the one your bot runs. Make it
  with `ci.Policy(the_same_module, ...)`.
- *"No such file or directory: 'thoughts.npz'"* - the path is relative to where RLBot starts your bot (usually its
  folder). Use a full path, or `Path(__file__).parent / "thoughts.npz"`.
- *"panel.toml [panel]: unknown key 'widht'. Did you mean 'width'?"* - a typo in your panel file. Check the file
  before a match with `cds-interpret check-panel panel.toml thoughts.npz`.
- *"panel.toml is not valid TOML: Cannot declare ('markers',) twice ..."* - a `[table]` may appear only once in a
  file. The example file already has every table; add your lines to the one that's there (remove the `#` in front of
  a commented-out line to use it).

**The panel froze on screen.**
Since 0.2 it shouldn't: an error in one frame is logged and the next frame draws again, packets without a ball (the
replay after a goal) are skipped, and if the overlay ever gives up it takes its drawing down. Look in the bot's log for
`frame ... failed`, or pass `error_log="overlay_errors.log"` to `with_thoughts` (or `watch_thoughts`) to keep the
errors in a file.

**The landing rings sit on the wrong side half the time.**
The bot sees the field mirrored at some moments. Give its policy a `mirror=` function and refit the ring
([a bot that sees the field mirrored](howto/probes.md#a-bot-that-sees-the-field-mirrored)).

**A row or a feature style of my panel file doesn't show.**
Its name doesn't match the pack exactly - names are case-sensitive, and a probe is called by the name you gave it in
`make_pack`. `cds-interpret check-panel panel.toml thoughts.npz` lists every mismatch with what you probably meant;
the overlay also writes them to the bot's log when it starts.

**Two panels on top of each other in 2v2 or 3v3.**
The copies of your bot don't recognise each other as the same bot, so each draws a panel. Start them all from the
same `bot.toml` (the same agent id). See [team modes](howto/team-modes.md#when-something-looks-wrong).

**Both teams show their thoughts and the panels overlap, or both show.**
They should share one panel by themselves. Check that neither panel file says `share = "none"`, and that both bots
run as the same Windows user (they meet in a folder in that user's temp directory; `$CDS_INTERPRET_OVERLAYS` points
both at another one).

**The panel shows a car I'm not watching.**
RLBot tells bots only which player the host's camera followed *most recently*, and says player 0 before anything was
followed. Follow the car you want once; on the free camera the panel stays on the last car you followed.
[What RLBot can and can't tell](howto/team-modes.md#what-rlbot-can-and-cant-tell-the-overlay).

**"Why this move" or "Looking at" doesn't show.**
The bot's log says why when the overlay starts. Usually `policy=` is missing from `with_thoughts`, or the pack is from
before 0.2 and lacks what the rows need. `cds-interpret check-panel panel.toml thoughts.npz --policy
my_bot.interp:make_policy` says which, and how to add it. [The rows' guide](howto/why-and-looking.md#when-something-goes-wrong).

**The landing ring jumps around.**
Read it from a steadier layer: `study.steadiness(layer)` compares layers, and `make_pack` picks the steadiest by
itself. See [probes you show live](howto/probes.md#probes-you-show-live).

**My bot feels slower with the overlay.**
The overlay adds a few read-only listeners to your bot's network and nothing else to its decision. All the rest runs
in a low-priority helper process after your controls are sent, and nothing is drawn on a decision's tick. If your
computer is already fully busy (training in the background, say), compare with the overlay off: that's the honest
test.

## Fair play

RocketSim's author asks that it be used for machine learning and hobby projects that respect Rocket League's terms
of use - never to sell a playable clone, and never to cheat in online matches. The same goes for this toolkit and
anything made with it.
