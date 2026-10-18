# Design your own thoughts panel

The thoughts panel is the box the overlay draws in the corner of the screen while your bot plays. The default one
looks like this:

<p align="center"><img src="../../assets/overlay-panel.svg" alt="The default thoughts panel" width="300"></p>

You can change almost everything about it - which rows appear, their names and colours, when a feature counts as
"firing", where the panel sits, how wide it is, whose thoughts it shows in 2v2 and 3v3 - with a small text file. No
library code to edit.

## A smaller panel, ready-made

If the default panel takes too much of the screen while you just want to watch, use the compact one: the same
readings in about half the room, on the right below the game's event feed (clear of the spectator card and the boost
meter), with smaller text.

```python
with_thoughts(MyBot, pack="thoughts.npz", panel="compact")
```

To tweak it, write it out and edit it like any panel file:

```bash
cds-interpret panel --compact panel.toml
```

## The two-minute version

Write the example file, which draws exactly the default panel:

```bash
cds-interpret panel panel.toml
```

```text
Wrote panel.toml. Change what you like, then pass it to the overlay: with_thoughts(..., panel="panel.toml")
```

Open `panel.toml`. Every option is there with a comment. Change the title, say:

```toml
[panel]
title = "WHAT MY BOT SEES"
```

Then give the file to the overlay in your bot's entry point:

```python
from cds_interpret.overlay import with_thoughts

if __name__ == "__main__":
    with_thoughts(MyBot, pack="thoughts.npz", network="policy", panel="panel.toml")("me/my-bot").run(
        wants_ball_predictions=True)
```

That's it. Restart the match and the panel has your title.

Anything you leave out of the file keeps its default, so your file can be as short as the one line you changed.
One exception: **rows**. If your file has any `[[rows]]` at all, they replace the default rows, so list every row you
want. The example file already lists them all - add yours among them.

Before a match, check your file against your pack:

```bash
cds-interpret check-panel panel.toml thoughts.npz
```

```text
panel.toml is fine, and everything it shows is in thoughts.npz.
```

A row showing a probe the pack doesn't have, a feature name with a typo, a style for a feature that isn't there - any
of these would otherwise just not show. The check lists them, with what you probably meant. (The overlay runs the same
check when it starts and writes what it finds to the bot's log.)

## What you can change

### The panel itself: `[panel]`

| key | default | what it does |
|---|---|---|
| `title` | the pack's | the line at the top: the title given to `make_pack` ("THOUGHTS" unless you gave one); `""` for none |
| `x`, `y` | `0.012`, `0.30` | the top-left corner, as fractions of the screen (`0.5` is the middle) |
| `width` | `0.17` | how wide, as a fraction of the screen (about 30 characters of text); 0.15-0.25 suits most screens |
| `text_scale` | `1.0` | bigger or smaller text; the panel grows taller, never wider |
| `title_scale` | `1.2` | the title's size, on top of `text_scale` |
| `smoothing` | `0.35` | how far numbers move toward each new reading per draw; `1` means no smoothing |
| `hold` | `0.6` | seconds a feature stays listed after it stops firing, so the list doesn't flicker |
| `draw_every` | `8` | frames between draws; 8 is 15 times a second |
| `value_scale` | from the pack | the outlook value that fills its bar |
| `keep_size` | `true` | every row keeps room for its longest answer, so the panel never changes size ([below](#a-panel-that-keeps-its-size)) |
| `follow` | `["camera", "last", "random"]` | in 2v2 and 3v3, whose thoughts the panel shows ([team modes](team-modes.md#choosing-whose-thoughts-follow)) |
| `subtitle` | `"car {car} of {cars} - {rule}"` | in 2v2 and 3v3, the line under the title; `""` for none ([team modes](team-modes.md#the-line-under-the-title-subtitle)) |

Long text is always cut to fit the width (with a `~` at the end), so a narrow panel never spills over.

### A panel that keeps its size

A move that takes one line now and two lines a moment later, a "Noticing" list that grows from one feature to three:
if the panel followed them, every row below would jump up and down while you read it. So by default each row keeps
room for its longest answer - the move's `lines`, the `max` features of "Noticing" and "Why this move" (plus the line
for the feature pushing against the move), and "Looking at" its bar and legend from the first draw. Rows that are
always the same height (bars, numbers) need nothing. The panel is the same size the whole match.

If you would rather have a compact panel that grows and shrinks with what it says, set:

```toml
[panel]
keep_size = false
```

Lowering `lines` or `max` makes the panel shorter while keeping it steady.

### Colours: `[colors]`

Colours are `[red, green, blue]` or `[red, green, blue, alpha]`, each from 0 to 255, or `"#rrggbb"` /
`"#rrggbbaa"` as in web colours.

| key | used for |
|---|---|
| `background` | the panel's box (default slightly see-through black) |
| `track` | the empty part of every bar |
| `title`, `label`, `value` | the title, the names on the left, the numbers on the right |
| `good`, `bad` | the outlook bar and its word when things look good or bad |
| `notice`, `notice_held` | a feature firing now, and one that stopped a moment ago |
| `bar` | any bar without a colour of its own |
| `divider` | the middle mark of a bar that grows from the middle |
| `push`, `push_strong`, `push_against` | "Why this move": a feature pushing the move, pushing it hard, and the one pushing hardest against it |
| `own_car`, `ball`, `opponents`, `team_mates` | "Looking at": the four parts of its bar and legend |
| `team_blue`, `team_orange` | the word floating above the panel's car, by its team |

The defaults keep every piece of text bright on a dark box; if you change them, keep it readable at a glance.

### The rows: `[[rows]]`

The panel is a list of rows, drawn top to bottom in the order you write them. Each `[[rows]]` table is one row:

```toml
[[rows]]
kind = "outlook"
label = "Outlook"

[[rows]]
kind = "bar"
source = "probe:First to the ball"
label = "First!"
color = [90, 220, 120]
```

The kinds of row:

| kind | shows | its keys |
|---|---|---|
| `outlook` | a bar from the middle with a word: green and right when the bot expects to score, red and left when it is in trouble | `label`, `scale`, `words` (three: below, around, above the middle), `threshold` |
| `action` | the move the policy leans toward, in words, and how sure it is | `label`, `lines` (how many lines the words may take) |
| `why` | the named features pushing that move hardest, and the one pushing hardest against it ([guide](why-and-looking.md)) | `label`, `max`, `min_push`, `strong`, `against`, `empty`, `features` |
| `looking` | what the move rests on - own car, ball, opponents, team-mates - as one coloured bar ([guide](why-and-looking.md)) | `label`, `every`, `donors`, `legend`, `words` (four: car, ball, opponents, team-mates) |
| `noticing` | the named features firing right now | `label`, `max`, `empty` (what to say when nothing fires), `features` (which ones may appear) |
| `probes` | every probe of the pack that doesn't have a row of its own, each in its natural style | - |
| `bar` | a name, a percentage at the right, and a bar under the name | `source`, `label`, `color`, `scale` |
| `percent` | a name and a percentage | `source`, `label`, `scale` |
| `seconds` | a name and a time | `source`, `label` |
| `value` | a name and a number | `source`, `label`, `format` (Python style, e.g. `"{:.0f} uu"`) |
| `centred` | a bar from the middle for any number, like the outlook | `source`, `label`, `scale`, `words`, `threshold`, `color` |
| `text` | a fixed line of text | `text`, `color` |
| `spacer` | a gap | `height` (in lines) |
| `custom` | anything you draw in Python (below) | `hook` |

A row's `source` says what it shows:

- `"probe:<name>"` - a probe of the pack, by the name you gave it in `make_pack` (`"probe:First to the ball"`),
  written exactly the same, capitals and all;
- `"feature:<name>"` - a named feature's activation, by its confirmed name. Use `scale` to say what fills the bar:
  the "highest" size in the feature's comment in your names file is a good start;
- `"value"` - the outlook value (your critic, or the pack's outlook probe).

Bars always sit under their names, so a row never needs more width than its words.

`why` and `looking` appear only when the overlay can work them out: the pack has what they need and the overlay was
given your bot's policy (`with_thoughts(..., policy=...)`; [how](why-and-looking.md)). Without, they take no room at
all - which is why the default rows can include them. Their keys:

| key | row | default | what it does |
|---|---|---|---|
| `max` | why | `2` | how many features pushing the move to list |
| `min_push` | why | `0.15` | the smallest push worth naming (log-probability: 0.7 roughly doubles the move's chance) |
| `strong` | why | `0.6` | a push this big is highlighted; one this big against the move gets its `-` line |
| `against` | why | `true` | show the feature pushing hardest against the move |
| `empty` | why | `"no named idea"` | what to say when no named feature pushes it |
| `features` | why | all | which features may appear |
| `every` | looking | `4` | worked out every 4th reading (the overlay's heaviest reading) |
| `donors` | looking | `3` | recorded moments each reading compares with: more is steadier and slower |
| `legend` | looking | `true` | the words and percentages under the bar |
| `words` | looking | `["car", "ball", "opp", "mates"]` | the legend's words |

To turn one off, leave it out of your `[[rows]]`; then it isn't worked out at all.

### Features: `[features."name"]`

One table per named feature, by the name it has in the pack:

```toml
[features."threat on goal"]
label = "Ball coming in!"    # the name shown on the panel
threshold = 0.7              # counts as firing above this (default: the threshold set when the pack was made)
show = true                  # false hides it everywhere
color = [240, 90, 80]        # its colour in the Noticing list and above the car
```

Raising a feature's `threshold` makes it fire less often - useful when it flashes on for moments that are only a
little like what you named it for. Pick it from the feature's sizes: your names file has a comment under each feature
("when it fires: 0.14 to 3.81, typically 0.73"), and a threshold above the highest means it never shows.

### The markers in the game: `[markers]`

| key | default | what it does |
|---|---|---|
| `landing` | `true` | a ring with a crosshair where the bot's own layer thinks the ball will land (`false`: no rings at all) |
| `landing_for` | `"all"` | whose rings in 2v2 and 3v3: `"all"` your bot's cars, `"panel"`, `"camera"`, `"none"`, or car numbers like `[1, 3]` ([team modes](team-modes.md#choosing-whose-rings-landing_for)) |
| `landing_color` | `[0, 170, 255]` | its colour |
| `landing_radius` | `150.0` | its size in unreal units (the ball's radius is about 93) |
| `landing_line` | `true` | a line from each car to its own ring |
| `landing_smooth` | `0.25` | seconds of smoothing, so the ring glides instead of jumping (`0`: none) |
| `prediction` | `true` | a white cross where RLBot's ball prediction says the ball will land |
| `prediction_color` | white | its colour |
| `car_label` | `true` | a feature's name floating above the panel's car |
| `car_label_from` | `"why"` | `"why"`: the feature driving the move hardest, else the first one firing; `"noticing"`: the first one firing |
| `car_label_color` | the car's team colour | one colour for it, whatever the team (a feature's own `color` is used when its style has one) |
| `hook` | none | your own 3D drawing (below) |

## Your own rows and markers in Python

When the row kinds aren't enough, draw your own. Write a function and name it in the file:

```toml
[[rows]]
kind = "custom"
hook = "boost_row"
```

This row shows a probe called "Boost", so make the pack with one: add `"Boost": "boost"` to the `probes` you give
`make_pack` (`boost` is the label for the car's boost, 0 to 1).

```python
def boost_row(draw, thoughts, state):
    """draw: a RowDrawer; thoughts: what the worker read this decision; state: a dict kept between draws."""
    boost = thoughts["probes"].get("Boost", 0.0)
    draw.line("Boost", f"{boost * 100:.0f}")          # a name on the left, a value on the right
    draw.bar(boost, draw.color(255, 140, 0))          # a bar under it, 0 to 1
    if boost < 0.1:
        draw.line("running on empty!", left_color=draw.named("bad"))

with_thoughts(MyBot, pack="thoughts.npz", panel="panel.toml", hooks={"boost_row": boost_row})(...)
```

`draw` has these tools: `line(left, right=None, left_color=None, right_color=None, indent=0.0)`,
`bar(fraction, color=None, centred=False)`, `stacked([(fraction, color), ...])` (one bar in coloured pieces),
`spans([(text, color), ...])` (one line of text in several colours), `blank(lines=1)` (the room of some lines of
text, left empty - to keep your row the same height), `gap(lines=0.2)`, and `color(r, g, b, a=255)` (or
`named("bad")` for a colour of your config). Text is cut to the panel's width for you.

`thoughts` holds everything the panel knows this decision:

```python
{"value": 0.42,                                          # the outlook value, or None
 "action": {"index": 17, "p": 0.81, "words": "throttle, boost"},
 "features": [{"name": "lined up for a shot", "act": 1.9, "on": True}, ...],
 "probes": {"First to the ball": 0.74, "Time to the ball": 1.3},
 "looks": {"First to the ball": {"unit": "%", "color": None}, ...},
 "land_xy": [x, y],                                      # the landing guess in the team's frame, when the pack has one
 "why": [{"name": "lined up for a shot", "push": 0.92}, ...],   # with policy=, for the panel's car
 "looking": {"own car": 0.46, "ball": 0.38, "opponents": 0.16}}  # with policy=, every few readings
```

A marker hook draws in 3D with RLBot's renderer. It runs inside the same render group as the panel. Name it in the
`[markers]` table - the example file already has one, with a `# hook = ...` line: remove the `#` and put your
function's name (a table may appear only once in a file, so don't add a second `[markers]`):

```toml
[markers]
hook = "my_markers"
```

```python
def my_markers(renderer, flat, thoughts, info):
    """info: {"car_index", "team", "landing_true", "landing_think", "marks"} (marks: a landing guess per car,
    [(car index, (x, y))])."""
    if any(f["on"] for f in thoughts["features"] if f["name"] == "threat on goal"):
        renderer.draw_string_3d("DEFEND", flat.CarAnchor(info["car_index"], flat.Vector3(0, 0, 200)), 2.0,
                                flat.Color(240, 90, 80, 255))
```

A hook may also be given as `"my_package.my_module:my_function"` in the file, without `hooks=`.

## What never changes

Whatever your file says, the overlay keeps its promises: it reads your bot's layers without changing them, does its
work after your controls are sent, never draws on the tick of a decision, and never makes your bot wait for it. If
your panel file has a mistake, the overlay switches itself off and says why in the bot's log - your bot keeps playing.

## When the file has a mistake

The messages say where the mistake is, what's wrong and, when they can, what you probably meant:

```text
panel.toml [panel]: unknown key 'widht'. Did you mean 'width'? Known keys: draw_every, hold, smoothing, ...
```

```text
panel.toml [[rows]] #2 (percnt): unknown row kind 'percnt'. Did you mean 'percent'? The kinds: outlook, action, ...
```

```text
panel.toml [panel]: x = '0.012' should be a number - without the quotes
```

```text
- [features."Airborne"] styles a feature the pack doesn't have, so the style is never used. The pack's features:
  'in the air', 'threat on goal'.
```

In a match, a mistake switches the overlay off and the bot's log says why; your bot keeps playing. To check a file
before a match, run `cds-interpret check-panel panel.toml thoughts.npz` - or in Python:

```python
from cds_interpret.overlay import PanelConfig, check_panel
check_panel(PanelConfig.load("panel.toml"), "thoughts.npz")      # [] when all is well
```
