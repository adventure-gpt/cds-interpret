# Contributing to cds-interpret

Thank you for wanting to make it better. New techniques, new labels, new recorders, fixes, docs and examples are all
welcome. This page says how the code is laid out, where new things slot in, and what a change needs before it is
merged.

## Set up

```bash
git clone https://github.com/adventure-gpt/cds-interpret
cd cds-interpret
python -m venv .venv && . .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # a CPU build is all the tests need
pip install -e ".[sim,rlgym,rlbot]" pytest ruff mypy  # rlgym needs Python 3.11-3.12
pytest                                                # about two minutes on a laptop CPU
ruff check src tests scripts examples
```

The tests that need RocketSim's collision meshes or RLBot skip themselves when those are missing; CI runs them on
Python 3.12 with every extra installed.

## How the package is laid out

One module per technique, each usable on its own with plain arrays and tensors; `Study` ties them together.

| module | what it is |
|---|---|
| `frames.py` | `Frames`, the one game-state format, and its adapters (RLGym, RLBot, arrays) |
| `labels.py` | concept labels from game state, future labels, and the label registry |
| `policy.py` | `Policy` (bring your own network), action tables, `eager_copy` |
| `hooks.py` | `LayerTap` and friends: reading and editing layers through forward hooks |
| `moments.py`, `store.py` | recorded moments and the column store under them |
| `record/` | one module per source: `sim.py` (RocketSim), `rlbot.py` (the real game), `rlgym_env.py`, `scripted.py` |
| `probes.py` | linear probes, held-out metrics, steadiness |
| `sae.py` | top-k sparse autoencoders |
| `intervene.py` | edits and their effect on the policy |
| `analysis.py` | features against labels, top moments, causal target selection |
| `names.py` | suggested feature names, the names file |
| `behavior.py` | habit counts from game state alone |
| `tracking.py` | probes across training snapshots |
| `emergence.py` | habits, dispositions and layer health, snapshot after snapshot |
| `curriculum.py` | situations, outcomes and enrichment: what to practise next |
| `study.py` | `Study`, `ProbeResults`, `FeatureSet`: the high-level flow |
| `report.py` | the HTML report and its section registry |
| `viz/` | dependency-free SVG charts, field drawings, the feature gallery |
| `overlay/` | the live thoughts overlay: `pack.py` (what is read), `panel.py` (how it is drawn, and its TOML config), `team.py` (whose thoughts in team modes, whose rings), `explain.py` ("why this move" and "looking at"), `worker.py`, `bot.py`, `watch.py` (a display-only script for bots you can't hook) |
| `refit.py` | carrying a pack to a newer checkpoint: names by what they fire on, the whole refit |
| `demo/` | the demo bot and the tour |
| `_rlbot.py` | what the RLBot recorder and the overlay share |

## Where new things slot in

- **A new technique** (say, attribution or a new kind of probe): a new module next to `probes.py` / `sae.py` that
  works on arrays and tensors and knows nothing of `Study`. Then, if it belongs in the everyday flow, a small
  `Study` method that calls it on `self.activations(layer)`, `self.labels`, `self.train` / `self.test`, and a report
  section:

  ```python
  from cds_interpret.report import section

  @section("Attributions", order=35)          # built-in sections: 0, 10, 20, 30, 40, 45, 47, 48, 50, 90
  def _attributions(study):
      res = getattr(study, "attribution_results", None)
      return "" if res is None else render(res)  # return "" to leave the section out
  ```

- **A new label**: `labels.register_label("name", kind="bin" | "reg", doc="...")` on a function of a `LabelInput`
  (frames, the car of each moment, team-frame positions, the ball's path, the built-in labels). A label that every
  bot needs belongs in `concept_labels` itself, with a line in the module docstring's table, `LABEL_KIND` and
  `LABEL_DOC`, and a test with a hand-made moment.
- **A new recorder** (another simulator, replays): a module in `record/` that fills a
  `record._builder.Recording` - `add_step` for every decision step (all cars, with events), `add_moment` for the
  moments the bot decided at - and returns `recording.finish()`. Labels and future labels then come for free.
- **A new edit**: a function in `intervene.py` returning `edit(x) -> x'`; `edit_effect` and `editing` work with it
  unchanged.
- **The overlay**: what is read lives in the pack (`overlay/pack.py`); how it is drawn in `overlay/panel.py`. A new
  kind of panel row is a branch in `Panel._row`, a name in `ROW_KINDS`, a line in `panel_example.toml` and a row in
  `docs/howto/customize-the-panel.md`. Keep the rules of `overlay/bot.py`: read-only hooks, the work after the
  controls are sent, nothing that can block the bot. A reading that needs the network itself goes in
  `overlay/explain.py`: it runs in the worker on the copy built from `policy=`, for the panel's car only, and turns
  itself off with a note when it fails. Per-frame work goes through `_rlbot.FrameGuard`: a bad frame is logged and
  the next one goes on; only a long run of failures switches the overlay off, and then it clears its drawing. With `keep_size` on, a row keeps the same height whatever it holds
  (`RowDrawer.blank`), and `tests/test_overlay_team.py::test_panel_keeps_its_size` checks it.

## What a change needs

- **Tests** for the new behaviour, on tiny hand-made inputs where possible (`tests/conftest.py` has helpers). A
  read-only hook must leave a network's outputs bit for bit the same - test it when you add one.
- **Type hints** on public functions, and a docstring that says what goes in and what comes out (shapes in
  brackets: `[N, D]`).
- **Friendly errors**: raise `CDSInterpretError` (or `ShapeError`, `LayerNotFound`, `MissingDependency`) with a
  message in plain words that says what went wrong and what to do, like the docs; import optional packages through
  `errors.require`.
- **Friendly docs**: lead with what the reader gets, then a snippet that runs and the output it prints, then the
  explanation. Plain words and short sentences; explain a term the first time it appears.
- **Safe defaults**: CPU, small threads, nothing written outside the paths a user gives, no pickled code in files
  meant to be shared (`torch.load(..., weights_only=True)`, `np.load(..., allow_pickle=False)`).
- **Docs**: the how-to that covers it, `docs/api.md`, and a line in `CHANGELOG.md` under "not yet released".
- `ruff check` clean, `pytest` green.
- Nothing from other people's bots: no other authors' code, weights, ports or replays, and no machine-specific paths
  or personal data in code, tests, docs or commit messages.

## Working on the same branch as others

Several people may be porting improvements at the same time. Pull before you push
(`git pull --rebase`), keep commits small and about one thing, and say in the message what changed and why.

## Releasing (maintainers)

1. Move the "not yet released" section of `CHANGELOG.md` under the new version and date; set
   `src/cds_interpret/_version.py`.
2. `python -m build` and check the wheel installs in a clean environment (`pip install dist/*.whl`;
   `cds-interpret demo`).
3. Tag `vX.Y.Z` and push the tag.
