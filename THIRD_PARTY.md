# Third-party software and notices

cds-interpret's own code, documentation and demo weights are under the MIT license (`LICENSE`, copyright
adventure-gpt). No third-party code is carried in this repository. The projects below are dependencies installed by
pip (or by you), each under its own license, and nothing of theirs is redistributed here.

## Dependencies

| project | license | used for | installed by |
|---|---|---|---|
| [PyTorch](https://pytorch.org) | BSD-3-Clause | networks, hooks, sparse autoencoders | always |
| [NumPy](https://numpy.org) | BSD-3-Clause | arrays, labels, probes | always |
| [RocketSim](https://github.com/ZealanL/RocketSim) (the `rocketsim` Python package) by ZealanL | MIT | recording games in a simulator | the `sim` extra |
| [RLGym](https://rlgym.org) (`rlgym-api`, `rlgym-rocket-league`) | Apache-2.0 | the RLGym v2 bridges and the RLGym recorder; its simulator package also ships RocketSim's collision meshes | the `rlgym` extra |
| [RLBot](https://rlbot.org) (`rlbot`, `rlbot_flatbuffers`) | MIT | the live thoughts overlay and recording in the real game | the `rlbot` extra |

Development only: [pytest](https://pytest.org) (MIT), [Ruff](https://docs.astral.sh/ruff/) (MIT),
[mypy](https://mypy-lang.org) (MIT), [Hatchling](https://hatch.pypa.io) (MIT).

## Notes

- **Collision meshes.** RocketSim needs Rocket League's arena collision meshes. They are game assets: this
  repository does not carry them. Use the copy your own install of `rlgym-rocket-league` brings, or dump them from
  your own copy of the game.
- **Fair play.** RocketSim's author asks that it be used for machine learning and hobby projects that respect
  Rocket League's terms of use, never to sell a playable clone, and never to cheat in online matches. The same goes
  for anything built with this toolkit.
- **The action table.** `cds_interpret.lookup_table()` builds a discrete action table in the layout that is common
  across Rocket League bots; it is this project's own code. A bot trained with another table should pass that table.
- **The demo bot.** Its network (`src/cds_interpret/demo/demo_policy.pt`) was trained by this project's
  `scripts/train_demo_policy.py` to imitate this project's scripted player; it contains no other bot's weights.
- Rocket League is a trademark of Psyonix, LLC. This project is not affiliated with or endorsed by Psyonix, Epic
  Games, RLBot or RLGym.
