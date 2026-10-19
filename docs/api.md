# API reference

Every function and class, grouped by what it's for. Start with `import cds_interpret as ci`: the names you'll use most
are right there (`ci.Policy`, `ci.Moments`, `ci.Study`, `ci.record`, `ci.overlay`). Shapes are written in brackets:
N moments, C cars, D numbers per layer or input, A actions. Every function's own help says more:
`help(ci.Study.probes)`.

- [The policy](#the-policy) · [Frames](#frames) · [Labels](#labels) · [Moments](#moments) · [Recording](#recording)
- [Study](#study) · [Hooks](#hooks) · [Probes](#probes) · [Sparse autoencoders](#sparse-autoencoders) ·
  [Edits](#edits) · [Analysis](#analysis) · [Habits](#habits) · [Tracking](#tracking) ·
  [Following training](#following-training-and-choosing-practice)
- [Naming features](#naming-features) · [The overlay](#the-overlay) · [The panel](#the-panel) ·
  [The study page and figures](#the-study-page-and-figures) · [RLGym bridges](#rlgym-bridges) ·
  [The demo bot](#the-demo-bot) · [Errors](#errors) · [Command line](#command-line)

## The policy

`cds_interpret.policy` - [how-to](howto/bring-your-own-policy.md)

**`Policy(net, obs=None, forward=None, output="logits", actions=None, value=None, name="policy", device="cpu", mirror=None, codec=None)`**
Your bot as the toolkit sees it.

| member | |
|---|---|
| `forward(x)` / `policy(x)` | the action output [N, A] (or [N, 8] controls) for inputs x |
| `log_probs(x)` | [N, A] log-probabilities (discrete policies) |
| `act(x, deterministic=True)` | [N] action indices (argmax or sampled); [N, 8] controls for a continuous policy |
| `controls(x, deterministic=True)` | [N, 8] controls, through the action table |
| `value(x)` | the critic's values [N], or None without one |
| `observe(frames, me)` | the network's input for moments seen by car slots `me` (needs `obs`) |
| `mirrored(frames, me)` | [N] bool: the moments a bot that mirrors the field sees mirrored (`mirror=`; all False without) |
| `codec` | an object splitting the network's input by entity - `decompose(x)`, `compose(parts)`, `groups(parts)` - for "looking at" on a bot whose input can't be rebuilt from game state |
| `layers(example)` | `list[LayerInfo]`: every module, its type and output shape on the example inputs |
| `default_layers(example)` | the hidden layers after their nonlinearity |
| `activations(x, layers, batch=4096)` | `{path: [N, D]}` |
| `random_like(seed=0)` | the same architecture with random weights (the probe baseline) |
| `discrete`, `actions`, `output`, `net`, `name` | |

**`lookup_table() -> [90, 8]`** the common discrete action table (24 ground rows, 66 air rows).
**`unmirror_controls(row)`** controls as a mirroring bot really sends them (steer, yaw, roll turned round).
**`describe_action(row) -> str`** controls in words: "throttle, steer left, boost".
**`eager_copy(scripted) -> nn.Module`** an eager copy of a TorchScript MLP (Sequential, Linear, LayerNorm, the usual
activations, Softmax, Dropout, Identity) with the same weights.

## Frames

`cds_interpret.frames` - the game-state format.

**`Frames(data)`**: `{field: array}` with a leading axis N. Fields: `ball_pos`, `ball_vel`, `ball_angvel` [N, 3];
`car_pos`, `car_vel`, `car_angvel`, `car_fwd`, `car_up` [N, C, 3]; `car_boost` (0-100), `car_on_ground`,
`car_demoed`, `car_has_flip`, `car_supersonic`, `car_team` (0 blue, 1 orange), `car_present` [N, C]. Optional:
`ev_touch`, `ev_demo_made`, `ev_demoed` [N, C] (events of the following step), `goal` [N] (-1 or the scoring team),
`time` [N], `prev_controls` [N, C, 8]. Read fields as attributes (`fr.car_pos`); `len(fr)`, `fr.n_cars`,
`fr.has(name)`, `fr.take(idx)`, `Frames.concat(parts)`.

| function | |
|---|---|
| `from_arrays(**arrays)` | Frames from keyword arrays; fills `ball_angvel`, `car_angvel`, `car_has_flip`, `car_demoed`, `car_present`, `car_supersonic` when absent |
| `from_rlgym(states, max_cars=None)` | from RLGym v1 or v2 game states (duck-typed) |
| `from_rlbot_packet(packets)` | from RLBot v5 game packets (one or many; duck-typed) |
| `team_sign(team)` | +1 blue, -1 orange |
| `canon(v, sign)` | vectors in a team's frame (x and y times the sign) |
| `axes_from_rotator(pitch, yaw, roll)` | forward and up vectors from an Unreal rotator |

## Labels

`cds_interpret.labels` - [every label](labels.md)

| | |
|---|---|
| `concept_labels(frames, me, path=None, path_t=None)` | `{label: [N]}` - the labels of the moment, plus registered ones |
| `future_labels(game_frames, car, step_seconds, shot=None)` | `{label: [T]}` - what happened next, for one recorded game in order |
| `register_label(name, kind, doc="")` | a decorator adding your label: `fn(inp: LabelInput) -> [N]` |
| `unregister_label(name)`, `registered_labels()` | |
| `LABEL_KIND`, `describe_label(name)`, `short_label(name)` | "bin" / "reg"; the meaning; a short display name |
| `ball_path(pos, vel, times=PATH_T)` | a simple untouched-ball flight [N, K, 3] |
| `time_to_ball(frames, path, times)`, `arrival_time(...)`, `landing(...)`, `goal_entry(...)`, `is_kickoff(frames)` | the pieces behind the labels |

## Moments

`cds_interpret.moments`

**`Moments(frames, me, game, step, obs=None, output=None, labels={}, meta={}, tracks=None)`**

| member | |
|---|---|
| `Moments.load(path)` | moments saved by `save()` or a recorder (future labels added from tracks when missing) |
| `Moments.load_all(root)` | every recording under a folder, game ids renumbered |
| `Moments.concat(parts)` | |
| `save(path, meta=None, if_exists="error", renumber=False)` | as `.npy` column shards; for a folder that already holds a recording: `"error"`, `"append"` (`renumber=True` numbers the new games after the old) or `"overwrite"` |
| `with_labels(path=None, path_t=None, overwrite=False)` | computes the labels of the moment |
| `take(idx)`, `len(m)`, `n_games`, `seq` (game * 64 + car: one key per car per game) | |

**`Tracks(frames, game, step, step_seconds)`** every decision step of every game: `games`, `game_frames(g)`,
`window(g, t0, t1)`. **`store.MomentStore(folder)`** the column store under both: `write_shard`, `load(cols, rows)`,
`iter(cols, batch)`, `adopt(other)`.

## Recording

`cds_interpret.record` - [how-to](howto/record-moments.md)

| | |
|---|---|
| `rocketsim(policy, games=10, seconds=120, team_size=1, opponent="self", tick_skip=8, every=3, deterministic=True, seed=0, meshes=None, out=None, record_opponent=None, ball_path="rocketsim", late=0.05, if_exists="error", log=print)` | your policy in RocketSim; `if_exists`: `"error"`, `"reuse"` (load what's in `out` and play nothing), `"append"` or `"overwrite"` |
| `with_recorder(BotClass, out="recordings/rlbot", network=None, input_layer="", output_layer="", every=3, flush_every=7200, tick_skip=None)` | your RLBot v5 bot class, recording the real game |
| `rlgym(env, policy, episodes=10, every=1, max_steps=None, deterministic=True, tick_skip=8, to_action=None, out=None, if_exists="error", log=print)` | your RLGym v2 environment (`if_exists` as above) |
| `find_meshes(path=None)`, `init_rocketsim(meshes=None)` | RocketSim's collision meshes; RocketSim, initialised |
| `scripted.chase`, `scripted.idle`, `scripted.SCRIPTED` | scripted players: `controls(frames, me) -> [N, 8]` |

New sources build on `record._builder.Recording` (`add_step`, `set_events`, `add_moment`, `finish`).

## Study

`cds_interpret.study`

**`Study(policy, moments, layers=None, labels=None, test_frac=0.25, seed=0, log=print, batch=4096)`** builds the
inputs (or uses the recorded ones), computes missing labels, picks the layers and splits the games.

| member | returns |
|---|---|
| `probes(layers=None, labels=None, baselines=True, scale="global")` | `ProbeResults` |
| `features(layer, m=None, k=None, steps=2000, batch=1024, lr=1e-3, seed=0)` | `FeatureSet`: a seeded sparse autoencoder of the layer (`m` features, default 8 per unit; `k` active at a time, default `m / 32`) |
| `load_features(path, layer=None)` | `FeatureSet`: a dictionary kept with `FeatureSet.save`, with the same feature numbers |
| `edits(layer, {name: edit})` | `{name: {"kl", "argmax_changed", "tv"}}` on the held-out moments |
| `probe_edit(layer, label, factor=0.0)`, `feature_edit(layer, features, scale=0.0)` | edits (`features`: numbers or confirmed names) |
| `steadiness(layer, labels=("land_x", "land_y"))` | `probes.readout_stability` of labels read together |
| `make_pack(layer, features=None, probes=None, land_layer=None, title="THOUGHTS", value_scale=1.0, outlook=True, names=None, decoders=True, donors=32, land_frame="team")` | a thoughts pack (`names`: a names file whose confirmed names become its features; `decoders` / `donors`: what the panel's "why this move" and "looking at" need; its probes remember their labels) |
| `landing(layer=None, layers=None, frame="team")` | the landing ring's readout (`probes.fit_landing`) on the layer that misses least on held-out games: `w`, `b`, `layer`, `lambda`, `frame`, `median_miss`, `p90_miss`, `median_jump`, `tried`; kept as `landing_results`. `frame="car"`: relative to the bot's own car, (ahead, left) of it - for a network that sees the field from its car |
| `calibrate_pack(pack, min_gain=0.0)` | each yes / no gauge of a pack Platt-calibrated on half the held-out games, checked on the other half, applied where it helps: {name: {"ece_before", "ece_after", "applied"}} |
| `suggest_names(layer, top=24)` | a suggested name for each top feature (`names.Suggestion`) |
| `feature_gallery(layer, path="features.html", features=None, n=24, exclude_kickoffs=False)` | a page of each feature's strongest moments |
| `check_feature(layer, feature, scales=(0.0, 3.0))` | `FeatureCheck`: the feature (a number or a confirmed name) turned off and up, and what changes |
| `why_not(label)` | why a label has no probe, in words (None if it can have one) |
| `attribution(n=400, donors=4, pool=4000, by_labels=(...), seed=0)` | what decisions rest on, by entity: `{"shares": {situation: {entity: share}}, "faithfulness", ...}` ([how-to](howto/what-decisions-rest-on.md)) |
| `report(path="report.html", title=None)` | the path of the HTML page |
| `activations(layer)`, `x`, `labels`, `kinds`, `games`, `train`, `test`, `layers`, `extra_sections` | |

**`ProbeResults`**: `by_layer[layer][label]` (a result: `r2` or `auroc`, the steadiness numbers, `probe`),
`baselines["input" | "random:<layer>"]`, `labels` (the labels asked about), `layers` (the layers probed), `skipped`
(label -> why it has no probe), `score(layer, label)`, `best(label)`, `probe(layer, label)`, `learned(margin=0.05)`,
`summary(top=None, margin=0.05)` (first what some layer reads `margin` better than both baselines, then the rest).

**`FeatureSet`**: `layer`, `sae`, `fvu`, `effect`, `freq`, `relations`, `activation(j)`, `sizes(j)` (`start`,
`typical`, `highest`: how big it gets when it fires), `guess(j, top=3)`, `name(j)`, `index(number_or_name)`,
`name_feature(j, name)`, `apply_names(names_file)`, `check_names(names)`, `fingerprint` (this exact dictionary's code),
`save(path)`, `top(n=12)`, `detectors(min_auroc=0.8, min_lift=3.0, min_freq=0.002, top=8, min_ap=0.1,
min_ap_lift=4.0)` (precise or clean detectors: `analysis.detects`), `summary(top)`.

**`FeatureCheck`**: `layer`, `feature`, `name`, `n` (the moments it fires on), `results[scale]` (`kl`, `changed`,
`shifts` {control: mean change}, `top_shifts`), `summary()`.

## Hooks

`cds_interpret.hooks`

| | |
|---|---|
| `LayerTap(model, layers, inputs=False, edits=None, keep="last", detach=True, read=True)` | read (and optionally edit) named modules; a context manager, or `attach()` / `detach_hooks()`; `out[name]` |
| `run_layers(model, forward, x, layers, inputs=None, batch=4096)` | `{name: [N, D]}` in batches, on the CPU |
| `list_layers(model, example=None, forward=None)` | `list[LayerInfo]` (path, type, shape, width, leaf, activation) |
| `default_layers(infos)`, `sequential_layers(seq, prefix="", tag="")`, `get_module(model, path)` | |
| `tail(seq, after)` | the rest of a `Sequential` after module `after`: a layer's output -> the network's output |

## Probes

`cds_interpret.probes` - [how-to](howto/probes.md)

| | |
|---|---|
| `group_split(groups, test_frac=0.2, seed=0, strata=None)` | (train rows, test rows) with whole games on one side |
| `ridge_suite(x, labels, kinds, train, test, groups=None, keep=False, scale="global", seq=None, order=None)` | every label's ridge probe; `{label: result}` |
| `probe_layer(x, labels, kinds, train, test, ...)` | one fit a label (logistic for yes / no) |
| `fit_probe(x, y, kind, groups=None, scale="global")` | one `LinearProbe` |
| `readout_stability(w, b, x, rows=None, y=None, seq=None, order=None, noise_sd=0.02)` | wnorm, noise, noise_rel, jitter, jitter_label, jitter_err, miss_med, miss_p90 |
| `fit_landing(x, xy, train, test, groups, seq=None, order=None, mirrored=None, car=None)` | a place readout for the panel, its penalty chosen by the median miss on validation games: `w` [D, 2], `b`, `lambda`, `frame`, `median_miss`, `p90_miss`, `median_jump`, `tried`. `car=(car_xy, forward_xy)`: fitted relative to the car (frame `"car"`) |
| `LinearProbe(w, b, kind, lam, link)` | `score(x)`, `direction()`, `to_dict()`, `LinearProbe.from_dict(d)` |
| `auroc`, `average_precision`, `r2`, `headline(result)` | metrics |

A result holds `r2`, `r` (numbers) or `auroc`, `ap`, `base`, `pos` (yes / no), `n`, `lam`, the steadiness numbers,
and `probe` when `keep=True`.

## Sparse autoencoders

`cds_interpret.sae` - [how-to](howto/sae-features.md)

`TopKSAE(d, m, k, k_aux=None)` with `init_from(sample)`, `encode(x)`, `decode(z)`, `forward(x)`,
`feature_direction(j)`; `train_sae(sae, acts, rows=None, steps=3000, batch=2048, lr=2e-4, seed=0, log=None,
time_limit=None)`; `fvu(sae, x)`; `latent_stats(sae, x)` (freq, mean_when_on, max); `encode_all(sae, x)` (indices and
values [N, k]); `dense_column(idx, val, j)`; `firing_thresholds(sae, x, features, q=0.05)`; `save(sae, path, **meta)`;
`load(path)`; `fingerprint(sae)` (a short code that is the same only for the same dictionary).

Variants and checks: `BatchTopKSAE(d, m, k, k_store=None)` (a batch-wide top-k; one threshold at inference);
`train_sae(..., downstream=f, kl_coef=c)` and `finetune_kl(sae, acts, downstream, steps=1000)` (the policy's KL in the
loss, `f` the network after the layer); `downstream_check(sae, downstream, x)` (KL, argmax changed, KL recovered, FVU,
L0 from stored activations).

`cds_interpret.dict_eval` - [how-to](howto/check-your-tools.md): `stability(a, b, freq_a=None)` and
`decoder_match(a, b)` (the same features by decoder direction), `activation_match(a, b, x, features)` (by firing:
the better test), `label_alignment(idx, val, labels, kinds, m)`, `liveness(idx, val, m)`, `scorecard(...)`.

`cds_interpret.crosscoder`: `Crosscoder(n_models, d, m, k)` - one dictionary over the same layer of several training
snapshots (`init_from(sample [n, M, d])`, `encode`, `decode`, `decoder_norms()` [m, M]); `train_crosscoder(cc,
batches, steps)`, `fvu_by_model(cc, x)`, `presence(norms)` (each feature's profile across snapshots, when it appears,
shared / emerged / faded), `save`, `load`.

## What decisions rest on

`cds_interpret.attribution` - [how-to](howto/what-decisions-rest-on.md)

| | |
|---|---|
| `FramesCodec(policy)` | `decompose(frames, me)` -> parts (ball, cars with roles), `compose(parts)` -> the policy's input, `frames(parts)`, `groups()` |
| `Group(key, role=None, fields=None)` | an entity, or part of one, to swap |
| `pick_donors(signature, n_donors=4, groups=None, seed=0, pool=None)` | [n_donors, N] donor rows with the same roles, from other games |
| `coalition_table(f, parts, donor_rows, groups, compose)` | `f` on every coalition: [n_donors, N, 2^G, K] |
| `shapley(tab)`, `occlusion(tab)`, `kl_table(logp_tab)` | exact Shapley values, one-at-a-time effects, the policy's KL per coalition |
| `faithfulness(metric, orders, average_donors=False)` | a ranking's deletion score: 1 = the best order, 0 = chance (`best_order_auc`, `deletion_curve`) |
| `integrated_gradients`, `grad_x_input`, `group_sum`, `order_from`, `rank_agreement`, `control_logodds` | column-level methods and helpers |

`cds_interpret.effects`: `feature_effects(sae, downstream, x, table=None)` (every active feature removed on its own:
`kl`, `dlogp`, `dctl` [N, k(, C)]), `per_feature(fx, m)`, `effect_words(per[j], control_names)`, `explain(fx, i, k=3)`,
`joint_ablation(sae, downstream, x, fx, orders, ks)`, `attribution_patching(sae, downstream, x, fx)`, `spearman_rows`.

`cds_interpret.erasure`: `fit_leace(x, z)` -> an `Eraser` (callable on [N, D]), `random_eraser(x, rank, mode="feature")`
(the placebo), `erase_edit(er)` (an edit), `edit_size(er, x)`. `probes.mlp_probe(x, y, kind, train, test, groups=None,
hidden=256)` - a non-linear probe with the linear probes' metrics.

`cds_interpret.readout`: `fit_platt` / `apply_platt`, `fit_isotonic` / `apply_isotonic`, `calibration(p, y)` (ECE,
Brier, reliability), `ema_seq`, `smoothing_tradeoff(x, y, seq, order, alphas, dt)`, `hysteresis`, `hold`,
`switches_per_minute`, `flag_quality`, `calibrate_pack_probe(pack, name, a, b)`.

## Edits

`cds_interpret.intervene` - [how-to](howto/causal-tests.md)

Edits: `replace_with(sae)`, `ablate_features(sae, features, scale=0.0)`, `clamp_feature(sae, j, value, mode="max")`,
`steer(direction, alpha, where=None)`, `project_out(direction)`, `scale_direction(direction, factor)`,
`mean_ablate(mean)`. Applying them: `editing(policy_or_net, layer, edit)` (a context manager),
`attach_edits(policy_or_net, layer, edit)`. Measuring: `edit_effect(policy, x, layer, {name: edit})`,
`policy_kl(model, forward, x, layer, edits)`. Intervals: `bootstrap_share(goals_for, goals_against)`,
`rate_ratio_ci(a, na, b, nb)`.

## Analysis

`cds_interpret.analysis`

`label_relations(act, labels, kinds)` (AUROC, base, P(label | on), lift, recall, ap; or r), `sparse_relations(rows,
vals, labels, kinds, n)` (the same, in time proportional to how often a feature fires), `sparse_columns(idx, val, m)`,
`sparse_ap(v_on, pos_on, n_pos, n)` (average precision with the off moments tied at the bottom),
`relation_strength(rel)` (|2 AUROC - 1|, or the precision above the base rate when that is larger: a sparse feature's
AUROC stays low even when it detects well), `detects(rel, min_auroc=0.8, min_lift=3.0, min_ap=0.1, min_ap_lift=4.0)`,
`guess_name(rel, top=3)`, `top_moments(act, k=12, groups=None, steps=None, min_gap=8)`,
`describe(frames, me, t)`, `causal_targets(features, label="demo_chance", ...)`.

## Habits

`cds_interpret.behavior`: `profile_game(game_frames, side_a, step_seconds=8/120)`, `summarize_profiles(profiles, side)`,
`SideCounter(n_games)` (`update(frames, side_a)`, `summary()`, `per_game(side)`).

## Tracking

`cds_interpret.tracking`: `track(snapshots, x, layers, labels, kinds, train, test, groups=None, targets=None, log=None)`
where a snapshot is `(name, step, policy_or_loader[, transform])`; `curves(rows, layer, labels)`.

## Following training and choosing practice

`cds_interpret.emergence` - [how-to](howto/track-training.md): `game_habits`, `summarize_habits`, `action_classes`,
`situations`, `dispositions`, `kl_rows`, `layer_health`, `linear_cka`, `weight_norms`, `first_goal_returns`,
`calibration`, `interval_changes`.

`cds_interpret.curriculum`: `next_goal_within`, `pair_situations`, `situation_outcomes`, `enrichment`, `eta_squared`,
`kmeans`, `car_situation_rows`, `situation_map`.

Each function's docstring gives its arguments and what it returns.

## Naming features

`cds_interpret.names` - [how-to](howto/name-features.md)

| | |
|---|---|
| `suggest(feature_set, features=None, labels=None, top=24)` | `list[Suggestion]`: a name per feature (`feature`, `name`, `label`, `strength`, `detail`, `freq`, `warnings`, `sizes`, `dictionary`); by default each different name once before any repeats |
| `suggest_by_label(feature_set, labels=None, min_ap=0.1, min_ap_lift=4.0, min_auroc=0.85, min_lift=3.0, report=None)` | the other way round: for each yes / no label the feature that detects it best (by precision), one name per feature, kept only when clear; `report` gets every label's numbers |
| `write_names(path, suggestions, layer, overwrite=False)` | start a names file (every name "suggested", the dictionary's code, each feature's sizes as a comment) |
| `update_names(path, suggestions, layer)` | add new features to a names file, keeping your edits |
| `read_names(path)` | `Names`: `layer`, `dictionary`, `features` {index: `FeatureName(name, status, threshold, note)`}, `confirmed()`, `thresholds()` |
| `NUMBER_NAMES` | what a feature that goes with a number label is called, high and low ("far from the ball", "close to the ball") |

## The overlay

`cds_interpret.overlay` - how-tos: [the overlay](howto/overlay.md), [why this move and looking at](howto/why-and-looking.md),
[team modes](howto/team-modes.md)

| | |
|---|---|
| `make_pack(layer, sae=None, features=None, thresholds=None, probes=None, actions=None, output="logits", output_layer="", title="THOUGHTS", value_scale=1.0, outlook_probe=None, meta=None, decoders=True)` | a pack from parts you made yourself (`probes`: {name: (LinearProbe, look)}; `decoders`: keep each named feature's decoder direction). `Study.make_pack` makes one from a study, with `probes` as {name: label} |
| `add_land(pack, module, w [D, 2], b [2], info=None, frame="team")` | the landing marker from its own layer (`frame="mirrored"`: fitted in a mirroring bot's own frame; `frame="car"`: (ahead, left) of the car) |
| `car_frame_offsets(points_xy, car_xy, forward_xy)` / `car_frame_point(offset, car_xy, forward_xy)` | a point as (ahead, left) of a car, and back - the targets of a `frame="car"` readout |
| `heading(yaw)` / `land_point(frame, xy, team, physics=None)` | a car's unit heading on the floor; a landing reading put on the field (world x, y) for its frame |
| `add_decoders(pack, sae)` | the named features' decoder directions, for "why this move" (refuses an SAE the features didn't come from) |
| `add_donors(pack, moments, me=None, n=32, seed=0, skip_kickoffs=True)` | up to `n` real recorded moments per number of cars (a Moments, a list of them, or Frames with `me`), for "looking at" |
| `add_input_donors(pack, {cars: inputs}, n=32, seed=0)` | recorded network inputs per number of cars, for "looking at" by the policy's own codec |
| `save_pack(pack, path)`, `load_pack(path)` | `.npz`, no pickles (format version 2; version 1 packs load) |
| `Thoughts(pack)` | `read(h, output)` (h: the pack layer's values), `land_xy(h_land)` (the values of the layer named in `pack["land"]["module"]`, which may be another layer), `outlook_value(h)`, `names`, `fd` [F, D] decoder directions (or None), `donors` [(Frames, me)], `donor_cars()` |
| `with_thoughts(BotClass, pack, network=None, panel=None, hooks=None, input_layer="", critic=None, worker=True, title=None, draw_every=None, x=None, y=None, policy=None, error_log=None, extras=None)` | your RLBot v5 bot class (a Bot or a Hivemind) with the overlay. `policy` / `critic`: `"module:factory"` or `("module:factory", {kwargs})`, built in the worker (a Policy object with `worker=False`); a critic taking `state` gets the game; `extras(bot, packet) -> {name: array}` adds your bot's own state to it; `error_log`: a file for its errors |
| `watch_thoughts(policy, pack, cars, panel=None, hooks=None, title=None, team_sizes=None, error_log=None, observe=None)` | an RLBot v5 Script class drawing the thoughts of a bot you can't modify: `cars` a piece of its name, player indices or a function of the packet (`match_cars`); `observe(packet, cars, field_info)` -> inputs or {"obs", "mirrored", "mask", "state"} when the obs function isn't enough; `cds-interpret watch-template` writes one ([how-to](howto/watch-a-bot.md)) |
| `Explainer(thoughts, policy, seed=0, freeze=True)` | the two decision readings: `why(x, h, action, acts=None)` -> [{"name", "push"}], `looking(state, me, action, donors=3)` -> {part: share}; `available`, `notes` |
| `load_policy(spec)`, `try_readings(pack, policy, donors=3)` | a Policy from `"module:factory"`; both readings tried on the pack's own moments: {"why_ms", "looking_ms", "problems"} |
| `Follow(rules=("camera", "last", "random"), seed=0)` | team modes: `pick(ours, spectated)` -> (player index or None, rule), `update(ours, spectated)` every packet |
| `landing_cars(landing_for, ours, panel_car, spectated)` | which cars get a landing ring |
| `draw_thoughts(renderer, flat, th, ..., marks=None, subtitle=None, available=None, team=None)`, `smooth_point(prev, new, dt)` | the panel, for your own drawing code |

`cds_interpret.overlay.explain`: `why_pushes(policy, module, x, h, action, acts, directions)` and
`looking_shares(policy, codec, live, me, action, dparts, dsig, donors=3, rng=None)`, the two readings as plain
functions. `cds_interpret.overlay.team`: `check_follow`, `check_landing_for`, `check_subtitle`, `subtitle_text`,
`car_number`, `match_seed`, `RULE_WORDS`.

## The panel

`cds_interpret.overlay.panel` - [how-to](howto/customize-the-panel.md)

| | |
|---|---|
| `PanelConfig(...)`, `PanelConfig.load(path)`, `PanelConfig.from_dict(d)` | the whole panel: `title`, `x`, `y`, `share`, `width`, `text_scale`, `title_scale`, `smoothing`, `hold`, `draw_every`, `value_scale`, `keep_size`, `follow`, `subtitle`, `colors`, `markers`, `features`, `rows`; `rows_of(kind)`, `wants` (what it asks the worker for), `place(team, crowded)` (its left edge), `team_coloured(team, crowded)`. `load` also takes a built-in panel by name: `"default"`, `"compact"` (`preset_text(name)` gives its TOML) |
| `RowSpec(kind, label, source, color, scale, format, text, lines, max, empty, features, words, threshold, height, hook, min_push, strong, against, every, donors, legend)` | one row; kinds: outlook, action, why, looking, noticing, probes, bar, percent, seconds, value, centred, text, spacer, custom |
| `FeatureStyle(label, threshold, show, color)` | a feature's look |
| `Markers(landing, landing_for, landing_color, landing_radius, landing_line, landing_smooth, prediction, prediction_color, car_label, car_label_from, car_label_color, hook)` | the 3D markers; `landing_cars` ("none" when `landing` is off) |
| `Panel(config, hooks=None)` | ready to draw: `draw(renderer, flat, thoughts, car_index, landing_true, landing_think, state, frame, value_scale, team, marks=None, subtitle=None, available=None, crowded=False, box=None)` (thoughts None: only the markers; `crowded`: overlays of both teams run; `box`: the shared panel's (x, y, width, height)); `measure(flat, thoughts, subtitle=None, available=None, team=None, crowded=False)` -> its own (x, y, width, height), nothing drawn |
| `RowDrawer` | what a custom row hook draws with: `line()`, `bar()`, `stacked()`, `spans()`, `blank()`, `gap()`, `color()`, `named()` |
| `check_panel(config, pack)` | what in a config doesn't match a pack (rows, styles and lists naming probes or features it doesn't have; why / looking rows of your own without what they need), in words; `[]` when all is well |
| `WorkerLink(target, *args)`, `compute(thoughts, msg, critic=None, explainer=None)` | the worker process (`hello`: its ready message, with what it can compute) and what it computes |

`cds_interpret.overlay.share` - both teams' overlays in one panel ([how-to](howto/team-modes.md#both-teams-showing-their-thoughts))

| | |
|---|---|
| `Neighbours(folder=None, pid=None, pick=None)` | this overlay's note and the others': `beat(cars, teams, spectated, box=None)` every frame, `alive()`, `owns_panel(cars, spectated)` (does this one draw the shared panel now), `crowded(teams)` (an overlay of another team runs), `shared_box(own)` (the box holding every overlay's panel), `others()`, `close()` |
| `overlay_dir()` | where the notes are: `$CDS_INTERPRET_OVERLAYS`, else `cds-interpret-overlays` in the temp folder |

## Keeping a pack across checkpoints

`cds_interpret.refit` - [how-to](howto/keep-your-panel.md)

| | |
|---|---|
| `carry_names(old_pack, old_values [N, D], new_features, min_r=0.5)` | each named feature found again in a new dictionary by what it fires on: [{"name", "old", "new", "r", "runner_up", "r2", "kept"}] |
| `describe_carry(report)`, `write_carried_names(path, report, new_features)` | the matching in words; a names file for the new dictionary |
| `feature_activations(pack, h)` | the pack's named features' activations [N, F] on a layer's values |
| `refit_pack(old_pack, policy, old_policy, out, moments=None, games=24, seconds=90, team_size=1, opponent="chaser", layer=None, m=None, k=None, steps=2000, min_r=0.5, probes=None, donors=32, calibrate=True)` | every step for a newer checkpoint into `out` (resumable): recording, dictionary, names, pack, calibration; `cds-interpret refit-pack` |

## The study page and figures

`cds_interpret.report`: `build(study, title)`, `section(title, order)` (a decorator), `SECTIONS`.
`cds_interpret.viz.plots`: `heatmap`, `lines`, `bars`, `save`. `cds_interpret.viz.field`: `moment_svg`,
`field_heatmap`, `clip_data`, `clip_canvas`, `CLIP_JS`. `cds_interpret.viz.gallery`: `feature_card`, `gallery_html`.
[How-to](howto/figures.md).

## RLGym bridges

`cds_interpret.rlgym` (needs the `rlgym` extra): `to_rlgym(frames) -> (states, agent_of_slot)`,
`obs_fn(builder, shared_info=None)`, `agent_ids(frames, t)`.

## The demo bot

`cds_interpret.demo`: `policy(weights=None)`, `synthetic_moments(n_games=60, per_game=60, seed=1)`, `run(out, source,
games, seconds)`, `obs(frames, me)`, `teacher(frames, me)`, `DemoNet`, `train(steps, n, seed)`, `random_frames(n, rng)`,
`nearest_actions(controls, table)`.

## Errors

`cds_interpret.errors`: `CDSInterpretError` (the base of all), `MissingDependency` (an optional package; the message
has the install line), `MeshesNotFound`, `ShapeError`, `LayerNotFound`; `require(module, feature)` imports an optional
module or raises `MissingDependency`.

## Command line

`cds-interpret demo [--out cds-demo] [--source auto|rocketsim|synthetic] [--games 12] [--seconds 60] [--threads 2]`,
`cds-interpret doctor`, `cds-interpret panel [panel.toml] [--force]` (writes an example panel config),
`cds-interpret check-panel panel.toml [thoughts.npz] [--policy module:factory]` (checks a panel config, against a
pack, and with `--policy` tries "why this move" and "looking at" and times them),
`cds-interpret watch-template [folder] [--force]` (a display-only script for a bot you can't modify),
`cds-interpret refit-pack old.npz --policy NEW --old-policy OLD --out DIR [--moments DIR] [--games 24] [--seconds 90]
[--team-size 1] [--opponent chaser] [--min-r 0.5]` (a pack carried over to a newer checkpoint),
`cds-interpret --version`; also `python -m cds_interpret`.
