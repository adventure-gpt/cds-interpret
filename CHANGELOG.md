# Changelog

All notable changes to cds-interpret are written down here. The project follows
[Semantic Versioning](https://semver.org); before 1.0, minor versions may change the API (each change is listed).

## [0.3.0] - unreleased

Hooks for bots whose world is bigger than the generic game state, so the overlay and the watch script serve them
exactly - found moving a large self-play bot's own overlay onto the toolkit.

### Added

- **Factories with arguments**: `policy=`, `critic=` and `observe=` may be `("package.module:factory", {keyword
  arguments})`, so one function builds several checkpoints in the worker or the script.
- **A critic that sees the game** (`with_thoughts(..., critic=...)`): a critic function taking `state` (and `cars`)
  gets the game as one packet's fields - plus whatever `extras(bot, packet)` adds from your bot's own state after each
  decision (`with_thoughts(..., extras=...)`), for a critic with privileged inputs.
- **"Looking at" by the policy's own codec** (`Policy(codec=...)`, `overlay.add_input_donors`): a bot whose input can't
  be rebuilt from game state splits its input by entity itself (decompose / compose / groups); the pack keeps recorded
  inputs per number of cars, and the same exact Shapley split runs on the input's own parts.
- **`watch_thoughts(..., observe=...)`**: the bot's inputs from the raw packet (boost pads, timers kept between
  packets), with optional mirror flags (the ring and the move's words turned back) and action masks.
- **A compact panel** (`panel="compact"`, `cds-interpret panel --compact`): the same readings in about half the
  room, top-right, smaller text, a lighter background - for watching rather than studying. `panel="default"` names
  the default one; a file of the same name still wins.
- **Action masks** in the worker (`compute` reads `mask` [n, A]): the panel's move is chosen among the actions each car
  may take, as the bot chooses.

## [0.2.0] - 2026-10-17

The live overlay in 2v2 and 3v3, two new rows that say why the bot makes each move, a panel that keeps going through
bad frames, thoughts for bots you can't modify, and packs carried over to newer checkpoints.

### Added

- **Team modes** ([guide](docs/howto/team-modes.md)). A landing ring for every car of your bot, each smoothed on its
  own and joined to its own car (`[markers] landing_for`: `"all"`, `"panel"`, `"camera"`, `"none"` or car numbers
  like `[1, 3]`), and one panel that follows a car by rules (`[panel] follow`, default `["camera", "last",
  "random"]`): the car the host's camera follows (RLBot's `last_spectated`), else the last of yours it followed, else
  one picked for the match - or `"first"`, a car number, or `"hide"`. A line under the title says whose thoughts
  these are and why (`[panel] subtitle`, `""` for none). Every car keeps its own smoothing and held features. Works
  for a Hivemind and for one copy of the bot per car alike: the copies find each other by agent id, agree on the
  panel car from the packet, and only that copy draws the panel. `overlay.Follow`, `overlay.landing_cars`.
- **"Why this move"** ([guide](docs/howto/why-and-looking.md)), a new row kind `why`: the named features pushing the
  move the policy leans to hardest, and the one pushing hardest against it - each feature's activation x its decoder
  direction . the gradient of the move's log-probability at the pack's layer (attribution patching), held a moment
  like "Noticing". Options `max`, `min_push`, `strong`, `against`, `empty`, `features`; colours `push`,
  `push_strong`, `push_against`. The word above the car can show the move's driver (`[markers] car_label_from`).
- **"Looking at"**, a new row kind `looking`: exact Shapley shares of the move over the bot's own car, the ball, the
  opponents and the team-mates, each swapped for the same part of real recorded moments through your bot's own obs
  function, every combination in one batch, for the panel's car every few readings. Options `every`, `donors`,
  `legend`, `words`; colours `own_car`, `ball`, `opponents`, `team_mates`.
- **`with_thoughts(..., policy="module:factory")`**: your bot's `ci.Policy`, built in the worker process (never in
  your bot's), for the two new rows; its critic, if any, also feeds the outlook bar. `overlay.Explainer`,
  `overlay.load_policy`.
- **Packs carry what the rows need** (format version 2; version 1 packs still load): `make_pack` keeps the named
  features' decoder directions (`decoders=True`), `add_decoders(pack, sae)` adds them to an older pack (and refuses
  an SAE the features didn't come from), `add_donors(pack, moments, n=32)` keeps real recorded moments per number of
  cars. `Study.make_pack` adds both (`decoders=True`, `donors=32`).
- **`cds-interpret check-panel panel.toml thoughts.npz --policy module:factory`** tries both rows with your policy
  and times them on one thread (`overlay.try_readings`); `check_panel` says what custom `why` / `looking` rows lack.
- **A panel that keeps its size** (`[panel] keep_size = true`): every row keeps room for its longest answer - the
  move its `lines`, "Noticing" and "Why this move" their `max` (plus the "against" line), "Looking at" its bar from
  the first draw - so nothing below ever moves and the panel never resizes. `keep_size = false` lets rows shrink.
- The word above the panel's car in **its team's colour** (`[colors] team_blue`, `team_orange`);
  `car_label_color` still sets one colour for it.
- `RowDrawer.stacked` (a bar in coloured pieces), `RowDrawer.spans` (a line in several colours) and
  `RowDrawer.blank` (room kept empty) for custom rows.
- Pictures of the new panel, of a team-mode panel and of a 3v3 moment from above with every car's ring, drawn by the
  overlay's own code (`scripts/render_panel.py`).
- **Thoughts for a bot you can't modify** ([guide](docs/howto/watch-a-bot.md)): `overlay.watch_thoughts(policy,
  pack, cars=...)`, a display-only RLBot v5 script that rebuilds a bot's input from each packet with your obs
  function, runs its network (weights in memory, nothing written) and draws the same panel and rings over the cars it
  matches (a piece of their name, player indices or a function), with every panel option, `team_sizes=` for a pack
  made for one brain of a bot, and only the rows it can compute. `cds-interpret watch-template` writes a ready
  script and its `script.toml`; `examples/watch_a_bot` watches the demo bot this way.
- **Keep your panel when your bot trains on** ([guide](docs/howto/keep-your-panel.md)): `refit.carry_names` finds
  each named feature again in a new checkpoint's dictionary by what it fires on (correlation over the same moments,
  the runner-up reported, below `min_r` left out), `refit.write_carried_names` writes the new names file, and
  `refit.refit_pack` / `cds-interpret refit-pack` does every step - recording, dictionary, names, probes on their
  labels, the landing ring, calibration, decoder directions and recorded moments - resumably, into one folder.
- **Naming by labels** (`names.suggest_by_label`): for each yes / no label the feature that detects it best, one name
  per feature - the quick way to name a new network.
- **The landing ring, fitted for the screen** (`probes.fit_landing`, `Study.landing`): its ridge penalty chosen on
  validation games by the median miss, with the miss nine times in ten and the move between readings reported; the
  layer with the smallest miss is used by `make_pack`.
- **Bots that see the field mirrored** (`Policy(mirror=...)`, `Policy.mirrored`): the landing readout is fitted in
  the bot's own frame and turned back for drawing (`add_land(frame="mirrored")`; without the policy the rings stay
  hidden rather than drawn on the wrong side), and its moves are said as sent (`policy.unmirror_controls`).
- **`Study.calibrate_pack(pack)`**: every yes / no gauge of a pack calibrated on half the held-out games, checked on
  the other half, kept only where it helped.

### Changed

- **The overlay keeps going through bad frames** instead of switching off at the first error (a panel left frozen on
  screen): an error is logged with its traceback (the first few, then every 200th and any new kind;
  `with_thoughts(..., error_log=...)` copies them to a file) and the next frame draws again; packets without a ball
  (the goal replay) are skipped; only about five seconds of failures in a row, or a worker failing 50 readings in a
  row, switch it off - and then it takes its drawing down. A worker process that stops is started again (three times
  at most).
- **Precision for sparse features**: relations to yes / no labels carry the average precision (`ap`, the moments where
  a feature is off tied at the bottom), and a relation's strength is the larger of |2 AUROC - 1| and the precision
  above the base rate - a sparse feature's AUROC stays near 0.5 even when it detects its label well (on the demo bot
  one read AUROC 0.57 at 24 times its label's base rate). `FeatureSet.detectors` counts a feature whose precision is
  at least max(0.1, 4 x the base rate), or whose AUROC is high; suggestions say the precision.
- `Study.make_pack` fits the landing ring with `Study.landing` (it chose the layer by the probes' steadiness), and its
  probes and meta remember their labels and the dictionary's size, for refits.
- The default panel has two more rows, "Why this move" and "Looking at", between "Next move" and "Noticing". They take
  no room unless the overlay can compute them (a pack that has what they need and `policy=`), so without `policy=` the
  panel shows what it did - but, with `keep_size` on by default, rows keep room for their longest answer and the panel
  is a little taller than in 0.1.
- A Hivemind's panel follows the car on camera instead of always its first car; one-copy-per-car teams draw one
  panel instead of one per copy, on top of each other.
- The word above the car is in the car's team colour (it was the "notice" colour) and, with "why this move" on, names
  the feature driving the move. `car_label_from = "noticing"` gives the 0.1 behaviour.
- The worker's request and result carry the panel car's readings (`compute(..., explainer=)`; results have `notes`);
  its ready message says what it can compute (`WorkerLink.hello`). A critic now gets the network's input in its own
  shape (unchanged for the usual [n, D] input).
- `RowSpec.max`, `empty` and `words` default per kind (`RowSpec("noticing").max` is still 3, `RowSpec("why").max` 2).
- The demo bot's pack also names a few features after what they make it do, and its `bot.py` passes
  `policy="cds_interpret.demo:policy"`, so every row shows on the demo.

## [0.1.0] - 2026-10-17

The first public version: the interpretability toolkit of the Cat Distribution System, on its own, for any PyTorch
Rocket League bot.

### Added

- **Bring your own policy** (`Policy`): any eager PyTorch network, a function from game state to its input, its
  action table and what its output means (logits, probabilities, log-probabilities or continuous controls).
  `eager_copy` turns a TorchScript MLP back into hookable modules; `lookup_table()` builds the common 90-action
  table; `describe_action` says an action in words.
- **Game state** (`Frames`) from RLGym v1 / v2 states, RLBot v5 packets or plain arrays, and back to RLGym v2 states
  (`cds_interpret.rlgym`, with `obs_fn` to run any RLGym ObsBuilder on any moment).
- **Concept labels** for any bot's moments, from the game state alone: the ball's landing, time to the ball, last man
  back, goal side, threats and shots, demolition chances, kickoffs, car state - and future labels from recorded play
  (a shot, a touch, a demolition, a goal for or against soon; an `outlook` for bots without a critic).
  `register_label` adds your own.
- **Recorders**: `record.rocketsim` (your bot in RocketSim against itself, another policy or a scripted player, with
  RocketSim's ball prediction for the labels), `record.with_recorder` (your RLBot v5 bot class records the real game,
  unchanged) and `record.rlgym` (your own RLGym v2 environment). `Moments` save and load as `.npy` column shards.
- **Probes** with game-wise held-out scores, the raw-input and random-network baselines, and steadiness numbers for
  probes shown live (weight norm, noise sensitivity, jitter against the label's own); the penalty acts on one scale
  for the whole layer by default, so a near-silent unit cannot earn a giant weight.
- **Sparse autoencoders** (top-k, with AuxK), their checks (variance explained, the policy run on the reconstruction)
  and feature analysis (label relations, name guesses, strongest distinct moments, firing thresholds,
  `causal_targets`).
- **Edits** for causal tests: replace with an SAE's reconstruction, ablate or scale features, clamp, steer, project
  out or scale a probe's direction, mean-ablate; their effect on the policy (`edit_effect`), `editing(...)` for
  whole games, and `bootstrap_share` / `rate_ratio_ci` for intervals.
- **`Study`**: the whole flow in a few calls, and an HTML **report** (light and dark, no internet needed) whose
  sections are extensible (`report.section`).
- **The live thoughts overlay** for any RLBot v5 bot (`overlay.with_thoughts`): the outlook, the next move and how
  sure it is, the features firing (held 0.6 s), probes as bars, and where the network thinks the ball will land.
  Display only: read-only hooks, a low-priority worker process that is never sent anything before it is ready and
  is written to by a thread of its own, nothing drawn on a decision's tick. Packs are plain `.npz` files.
- **A panel you design yourself**: rows, names, colours, firing thresholds, hold time, position, width, smoothing,
  title and markers in a small TOML file, without touching library code (`cds-interpret panel` writes an example of
  every option, which draws the default panel), plus Python hooks for rows and markers of your own. Mistakes show
  when the file loads, with where they are and what you probably meant; `cds-interpret check-panel` (and the overlay
  at start) lists rows and styles naming probes or features the pack doesn't have.
- **Naming features, end to end** (`names`): suggested names from the labels with warnings (a feature that fires
  mostly at kickoffs, say), each different suggestion before any repeats, a gallery page of each feature's strongest
  moments, a names file you confirm names in (with each feature's sizes, so thresholds can be picked), and
  `Study.check_feature` (turn a feature off and up and see what the bot does differently - by number or by name)
  and `make_pack(names=...)`. Names stay on their features: `Study.features` is seeded, `FeatureSet.save` /
  `Study.load_features` keep a dictionary for good, and a names file records its dictionary's code
  (`sae.fingerprint`) and refuses to go on any other. A threshold the feature never reaches is refused too.
- **Tracking** a concept across training snapshots.
- **Following a bot through its training** (`emergence`): its dispositions on a fixed bank of moments (the action
  classes it puts its probability on in a demo chance, an aerial chance, a 50/50 ...), its habits in recorded games
  with intervals (demolitions, air and wall play, flip resets, hard hits, kickoffs and kickoff goals, the goal share in
  open play, contact), the health of its
  layers (dead / dormant units, effective rank, CKA to another snapshot, weight norms) and the critic's calibration
  against the next goal, and `interval_changes` for what moved beyond its intervals since the last snapshot.
  `docs/howto/track-training.md`.
- **Choosing what to practise** (`curriculum`): a situation map for both sides of recorded games (how often each side
  is in each situation, and what follows), what is over-represented before the goals a bot concedes (`enrichment`),
  a small `kmeans` to group those moments and `eta_squared` to compare descriptions of a state.
- **Friendly by default**: plain-words errors that say what to do (with "did you mean" for layers and labels),
  probes scored only when the held-out games have enough examples (and the skipped ones listed with the reason),
  recordings that are never mixed up by accident (`if_exists`, with `"reuse"` for scripts you run again), a probe
  table that separates what the network worked out from what its input already gave, and a step-by-step demo whose
  edits come with a random-direction control.
- **What a decision rests on** (`attribution`, `Study.attribution`, a report section): the ball, the bot's car and
  boost, the opponents and the team-mates swapped for another moment's through the bot's own obs function, every
  combination run, exact Shapley shares, and a faithfulness score for any ranking (1 = the best possible order).
  `docs/howto/what-decisions-rest-on.md`.
- **What each feature makes the bot do** (`effects`): every active feature removed on its own, its change in each
  control and in the chosen action; the features a decision rests on; rankings scored by removing them together;
  attribution patching as the cheap estimate. `hooks.tail` gives the network after a layer.
- **Checks for the tools themselves** (`docs/howto/check-your-tools.md`): `sae.downstream_check`, `sae.finetune_kl`
  and a KL term in `train_sae` (a dictionary faithful to the decisions), `sae.BatchTopKSAE`, `dict_eval` (seed
  stability by decoder and by firing, single-feature label detection, liveness), `erasure` (LEACE with a same-rank
  placebo: does the policy use what a probe reads?), `probes.mlp_probe`, and `readout` (calibrated gauges -
  `calibrate_pack_probe` - smoothing with its measured lag, flag flicker).
- **A crosscoder over training snapshots** (`crosscoder`): one dictionary over the same layer of several snapshots;
  each feature's decoder norm per snapshot says when it is there.
- **The demo bot**: a tiny network trained to imitate a scripted player, so everything runs without anyone's weights;
  `cds-interpret demo` runs the whole tour and `cds-interpret doctor` says what is ready. A complete RLBot v5 bot
  folder for it in `examples/rlbot_demo_bot`.
