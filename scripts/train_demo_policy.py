"""Train the demo bot's network (cds_interpret.demo) and write its weights into the package.

    python scripts/train_demo_policy.py            # about a minute on a CPU

The demo bot copies cds_interpret.demo.teacher, a simple scripted player, by imitation on random game moments.
The weights are small (about 30 KB) and are part of the package so the quickstart runs without anyone's bot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cds_interpret import demo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(demo.__file__).parent / demo.WEIGHTS))
    args = ap.parse_args()
    torch.set_num_threads(2)
    net = demo.train(steps=args.steps, seed=args.seed, log=print)
    torch.save({k: v.clone() for k, v in net.state_dict().items()}, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
