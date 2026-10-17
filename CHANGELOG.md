# Changelog

All notable changes to cds-interpret are written down here. The project follows
[Semantic Versioning](https://semver.org); before 1.0, minor versions may change the API (each change is listed).

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
