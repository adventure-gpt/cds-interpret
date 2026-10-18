# Keep your panel when your bot trains on

You named your bot's features, made a pack, and the panel shows "lined up for a shot" and "low on boost". Then the
bot trains for another few billion steps. The new checkpoint is a different network: feature 117 of the old
dictionary means nothing in it, and a pack made for the old network can't be read from the new one.

Most of the ideas are still there, though. One command finds them again and makes a new pack:

```bash
cds-interpret refit-pack thoughts.npz --old-policy my_bot.interp:make_old_policy --policy my_bot.interp:make_policy --out refit
```

```text
1/5 recording 24 games of the new checkpoint in RocketSim (1v1) ...
2/5 a new dictionary of body.3 ...
3/5 carrying the names over by what they fire on ...
8 of 8 names carried over:
  'lined up for a shot': feature 117 -> 402, r 0.91 (runner-up 77 0.38)
  'low on boost': feature 40 -> 13, r 0.96 (runner-up 288 0.22)
  'ball on our side': feature 9 -> 151, r 0.71 (runner-up 30 0.66) - the runner-up is close: check its moments
  ...
4/5 the pack: probes, landing ring, decoder directions, recorded moments ...
  landing ring: body.1 misses by 498 uu typically (90% within 1190), moves 410 uu between readings
5/5 calibrating the yes / no gauges on held-out games ...
  'Threat on goal': calibration error 0.071 -> 0.024
Wrote refit\thoughts.npz - check it with: cds-interpret check-panel panel.toml refit\thoughts.npz
```

(Illustrative output. On a large self-play bot, 8 of 8 names carried over between checkpoints 10 billion steps apart,
with r from 0.65 to 0.96 and two close runners-up.)

Give your bot the new pack, and the panel shows the same ideas, read from the new network.

## How names are carried over

Both networks read the same recorded moments. For each named feature of the old pack, the toolkit works out how
strongly it fires at every moment (from the old network), and finds the feature of the new dictionary that goes up
and down with it most closely - the highest **correlation**, r, across those moments. That feature takes the name.

- **r near 1**: the same idea, found again.
- **r under 0.5** (`--min-r`): left out of the new pack and listed, so you can look for it yourself. The idea may not
  have survived the training, or the new dictionary may have split it in two.
- **a close runner-up**: two new features fire much like the old one. The name goes to the closer; look at both
  features' moments in a [gallery](name-features.md) before you trust it.

The new names go in `refit/names.toml`, confirmed, each with a note saying where it came from. Edit it like any names
file and run the command again: the steps already done are kept.

## What the command does

| step | what it does | kept in `--out` |
|---|---|---|
| moments | records games of the new checkpoint in RocketSim (`--games`, `--seconds`, `--team-size`, `--opponent`), or uses yours (`--moments FOLDER`) | `recording/` |
| dictionary | a new sparse autoencoder of the pack's layer, the old one's size | `features.sae` |
| names | carried over by what they fire on | `names.toml` |
| pack | the probes refitted on their labels, the landing ring (its readout chosen by how far it misses on held-out games), decoder directions and recorded moments for the panel's two decision rows | |
| calibration | each yes / no gauge calibrated on held-out games, kept only where it helped | |
| | the new pack, and every number of every step | `thoughts.npz`, `refit.json` |

It needs two small functions, one per checkpoint, each returning that checkpoint's `ci.Policy`
([bring your own bot](bring-your-own-policy.md)):

```python
# my_bot/interp.py
def make_policy():
    return ci.Policy(load_net("checkpoints/new.pt"), obs=my_obs, actions=MY_ACTIONS)

def make_old_policy():
    return ci.Policy(load_net("checkpoints/old.pt"), obs=my_obs, actions=MY_ACTIONS)
```

The probes are refitted on the labels they were made from: packs made by `Study.make_pack` remember them. A pack from
before version 0.2 doesn't, so its probes are listed as not refitted - pass them in Python, as below.

## In Python

The same, step by step, when you want to look between steps:

```python
import cds_interpret as ci
from cds_interpret import refit

old_pack = ci.overlay.load_pack("thoughts.npz")
study = ci.Study(new_policy, moments)                                # the NEW checkpoint, on recorded moments
features = study.features("body.3")                                 # its own dictionary
old_values = old_policy.activations(study.x, ["body.3"])["body.3"]  # the OLD network, on the same moments
report = refit.carry_names(old_pack, old_values, features, min_r=0.5)
print(refit.describe_carry(report))
refit.write_carried_names("names.toml", report, features)
pack = study.make_pack("body.3", names="names.toml", probes={"Threat on goal": "threat"})
study.calibrate_pack(pack)
ci.overlay.save_pack(pack, "thoughts_new.npz")
```

Or all at once: `refit.refit_pack("thoughts.npz", policy=new_policy, old_policy=old_policy, out="refit",
probes={"Threat on goal": "threat"})`.

## Naming a network with no earlier pack

A new network without a pack to carry names from - a bot you just started studying - can be named straight from the
labels: `ci.names.suggest_by_label(features)` gives each yes / no label the feature that detects it best, one name per
feature, and leaves out labels nothing detects clearly. See [naming features](name-features.md#name-by-the-labels).
