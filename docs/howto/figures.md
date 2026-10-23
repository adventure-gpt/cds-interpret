# Reports and drawings

Every study can become one web page you can open, keep and share: what your bot knows, the ideas it uses with their
moments drawn on the field, and what happened when you switched them off.

## The study page

```python
study.report("report.html", title="Inside my bot")
```

Open `report.html` in any browser. It's one self-contained HTML page (inline SVG and a few lines of script for the clips; no internet needed), in light and
dark following the reader's system. It shows what the study has computed so far:

| section | shows |
|---|---|
| Overview | the policy, the moments (games, held out, source, opponent), the layers |
| What it knows | heatmaps of every probe's held-out score by layer, beside the raw input and random weights; the labels a layer reads clearly better than both |
| How steady each readout is | each label's best probe: weight norm, noise sensitivity, jitter against the label's own |
| Features | for each SAE: variance unexplained and the policy on the reconstruction; a card per top feature - its guessed (or given) name, how often it fires, what it goes with, its three strongest distinct moments drawn from above and described in words, and an animated clip of the strongest when the recording has tracks |
| Edits | each edit's KL, share of changed actions, total variation |
| What decisions rest on | each entity's Shapley share of the decisions, by situation, and how faithful the ranking is |
| What each attention head looks at | a sentence per head, what each head looks at, every head's map, the shifts by situation and each token's change in the biggest ones, each head's spread ([guide](attention-heads.md)) |
| More | anything you put in `study.extra_sections` (`{title: html}`) |
| Reading this page well | the four habits |

## Your own sections

A new technique can add its own section to every page:

```python
from cds_interpret.report import section

@section("Attributions", order=35)            # built-in orders: 0, 10, 20, 30, 40, 45, 47, 50, 90
def attributions(study):
    res = getattr(study, "attribution_results", None)
    if res is None:
        return ""                             # nothing to show: the section is left out
    return "<p>...</p>"
```

## Figures on their own

`cds_interpret.viz` draws SVG with no plotting library; every figure carries its own light and dark colours and a
tooltip on every mark.

```python
from cds_interpret.viz import field, gallery, plots

svg = plots.heatmap(rows, cols, values, "AUROC by layer", vmin=0.5, vmax=1.0)
svg = plots.lines({"threat": (steps, scores)}, "When did it learn threats?", xlabel="steps", ylabel="AUROC")
svg = plots.bars(labels, values, "Goal share by arm", lo=lows, hi=highs)
svg = plots.heatmap(rows, cols, changes, "Change by situation", vmax=20, diverging=True)   # red below 0, blue above
svg = plots.head_grid(maps, layers, heads, queries, keys, "Each head's map")        # one small map per head
svg = field.moment_svg(moments.frames, t, me, landing=(x, y), title="a moment")   # one moment from above
svg = field.field_heatmap(values, xs, ys, "value by ball position")                 # a value over the field
clip = field.clip_data(frames, t0, t1, me)                                          # for an animated canvas
page = gallery.gallery_html("My features", intro_html, [gallery.feature_card(...), ...])
plots.save(svg, "figure.svg")
```

`analysis.describe(frames, me, t)` puts a moment into words (positions in uu, Unreal units: the field is 10,240 uu
long and 8,192 wide, and the ball's radius is about 93) - "us (0, -4608, 17) 0 uu/s on the ground, 33 boost
[own third, middle]; opp ...; ball ...").

## Habits, from the game state alone

`cds_interpret.behavior` counts what a bot *does* - for any bot you can watch, no network needed: goals, touches,
demolitions made and suffered, demolition chances and how often they are taken, how often it is goal-side when it is
last back (team modes) and when a shot is coming, its boost, supersonic and air time.

```python
import numpy as np
from cds_interpret import behavior

tracks = moments.tracks                                    # every step of every game (RocketSim and RLBot recordings)
side_a = np.array([True, False])                           # which car slots are "A": here slot 0, your bot in 1v1
profiles = [behavior.profile_game(tracks.game_frames(g), side_a, tracks.step_seconds) for g in tracks.games]
print(behavior.summarize_profiles(profiles, "A"))          # per game: goals, touches, demos, air time, ...
```
