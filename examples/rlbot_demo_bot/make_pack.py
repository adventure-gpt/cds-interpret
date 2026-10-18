"""Make the demo bot's thoughts pack: record it in RocketSim, find what its layers know and its features, and write
thoughts.npz next to this file (a few minutes on a CPU).

    python make_pack.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import cds_interpret as ci
from cds_interpret import effects as EF

HERE = Path(__file__).resolve().parent
CONTROLS = ["throttle", "steer right", "nose up", "yaw right", "roll right", "jump", "boost", "handbrake"]
DOES = ["drive forward", "steer right", "nose up", "turn right", "roll right", "jump", "boost", "handbrake"]


def main() -> None:
    torch.set_num_threads(2)
    policy = ci.demo.policy()
    moments = ci.record.rocketsim(policy, games=24, seconds=90, opponent="chaser")
    study = ci.Study(policy, moments)
    print(study.probes().summary(top=12))
    feats = study.features("body.3", m=256, k=8, steps=2000)
    print(feats.summary(top=8))
    # name the features that detect a yes / no concept cleanly, after it (a real study names features after looking
    # at their strongest moments in the report)
    named = {}
    for j, label, auroc, lift in feats.detectors(top=6):
        name = ci.short_label(label)
        if name not in named:
            named[name] = j
            print(f"  feature {j}: {name} (AUROC {auroc:.2f}, {lift:.1f}x as likely when it fires)")
    # ... and a few after what they make the bot DO (their effect on its controls), since the features labels name best
    # are rarely the ones its moves rest on: these are the ones the panel's "why this move" row will mostly show
    fx = EF.feature_effects(feats.sae, policy.net.head, study.activations("body.3")[study.test], table=policy.actions)
    per = EF.per_feature(fx, feats.sae.m)
    for j in sorted(per, key=lambda j: -per[j]["kl"] * per[j]["freq"]):
        if len(named) >= 6 or j in named.values():
            continue
        c = int(np.argmax(np.abs(per[j]["dctl"])))
        name = ("wants to " if per[j]["dctl"][c] > 0 else "doesn't want to ") + DOES[c]
        if name not in named:
            named[name] = j
            print(f"  feature {j}: {name} ({EF.effect_words(per[j], CONTROLS)})")
    pack = study.make_pack("body.3", features=named,
                           probes={"Threat on goal": "threat", "First to the ball": "first_to_ball",
                                   "Time to the ball": ("ttb_me", {"unit": "s"})},
                           title="DEMO BOT'S MIND", value_scale=1.0)
    path = ci.overlay.save_pack(pack, HERE / "thoughts.npz")
    study.report(HERE / "report.html", title="The demo bot, inside")
    print(f"wrote {path} ({len(named)} named features: {', '.join(named)}) and report.html")


if __name__ == "__main__":
    main()
