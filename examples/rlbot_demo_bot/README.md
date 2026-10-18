# The demo bot in RLBot

A complete RLBot v5 bot folder for the cds-interpret demo bot: the plain bot, the same bot with its thoughts drawn
on screen, and a recorder - each one flag away. It is also the template for adding the overlay to your own bot.

1. Get this folder: `git clone https://github.com/adventure-gpt/cds-interpret` (a pip install doesn't include the
   examples). Then, in the Python that `python` runs (3.11 or 3.12; `rlgym` brings RocketSim and its collision meshes
   for step 2): `pip install "cds-interpret[rlgym,rlbot]"`.
2. Optional: `python make_pack.py` - records the demo bot in RocketSim, studies it, and writes a fresh
   `thoughts.npz` and `report.html` here (5-10 minutes on a CPU). The folder already has a `thoughts.npz`.
3. In RLBot v5, add `thoughts.bot.toml` as a bot, turn rendering on, and start a match. Every row of the panel is
   on, "why this move" and "looking at" included: `bot.py` passes `policy="cds_interpret.demo:policy"`, the demo
   bot's network and obs function, to the overlay.
4. Try 2v2 or 3v3: add `thoughts.bot.toml` two or three times to the same team. Each car gets its own landing ring,
   and the panel follows the car you watch with the spectator camera
   ([team modes](../../docs/howto/team-modes.md)).

`bot.toml` is the plain bot. `python bot.py --record` records the game to `recordings/` while it plays; load it with
`cds_interpret.Moments.load_all("recordings")`.
