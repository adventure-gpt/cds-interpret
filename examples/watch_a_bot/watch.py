"""Watch a bot without touching it: the demo bot's thoughts, drawn by a separate RLBot SCRIPT.

The demo bot runs plain (../rlbot_demo_bot/bot.toml - no overlay in it); this script reads the game, rebuilds the demo
bot's input from each packet, runs the same network and draws the panel and the landing rings over its cars. The very
same way works for a bot you can't modify - a compiled bot, a bot in another language - once you have its weights and
a port of how it builds its input (docs/howto/watch-a-bot.md).

RLBot: add ../rlbot_demo_bot/bot.toml as a bot (once, or twice on a team for 2v2), add watch.script.toml (this folder)
as a script, turn rendering on and start the match.
"""

from pathlib import Path

from cds_interpret.overlay import watch_thoughts

HERE = Path(__file__).resolve().parent
PACK = HERE.parent / "rlbot_demo_bot" / "thoughts.npz"   # a pack made for the demo bot's network
POLICY = "cds_interpret.demo:policy"                     # its network, weights and obs function (a ci.Policy)
CARS = "CDS demo bot"                                    # a piece of its name in the game

if __name__ == "__main__":
    watch_thoughts(POLICY, PACK, cars=CARS, title="DEMO BOT (WATCHED)")("cds-interpret/watch-demo-bot").run(
        wants_match_communications=False, wants_ball_predictions=True)
