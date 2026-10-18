"""Team modes and the camera: whose thoughts the panel shows, and which cars get a landing marker.

In 2v2 and 3v3 your bot drives two or three cars. RLBot tells every bot one thing about the host's camera:
``packet.match_info.last_spectated``, the index of the player the host's spectator camera followed most recently.
It keeps that player when the host switches to the free camera or the ball camera, and it is 0 before anything has
been spectated - so "the car on camera now" and "the car on camera last" are the same number, and player 0 at the
start of a match may just mean "nothing yet".

The panel follows a list of rules (``[panel] follow``); the first rule that gives one of your bot's cars wins:

=========  ==========================================================================================
camera     the car the host's camera follows (last_spectated), when it is one of yours
last       the last of your cars the camera followed this match (when it has moved to another player)
random     one of your cars, picked at the start of the match and kept (every copy of the bot picks the same)
first      your first car (the one with the lowest player index)
2          your second car - any car number (1 = the first)
hide       no panel (so ``["camera", "hide"]`` shows thoughts only while the camera is on one of yours)
=========  ==========================================================================================

Landing markers (``[markers] landing_for``): ``"all"`` your cars, ``"panel"`` the panel's car, ``"camera"`` the car
on camera when it is yours, ``"none"``, or a list of car numbers such as ``[1, 3]``.

"Your cars" are a Hivemind's cars, or - for a plain Bot, which RLBot runs once per car - the cars of its team run by
the same bot (the same agent id in the match configuration). Every copy works the list out the same way, so they
agree on the panel's car without talking to each other: that copy draws the panel, each copy draws its own marker.
"""

from __future__ import annotations

import difflib
import random
import string
from typing import Any

from ..errors import CDSInterpretError

FOLLOW_RULES = ("camera", "last", "random", "first", "hide")
LANDING_FOR = ("all", "panel", "camera", "none")
DEFAULT_FOLLOW = ("camera", "last", "random")
# what the subtitle says about the rule that chose the car
RULE_WORDS = {"camera": "on camera", "last": "last on camera", "random": "picked for this match", "first": "first car",
              "car": "fixed"}
SUBTITLE_KEYS = ("car", "cars", "rule", "name")


def check_follow(rules: Any, where: str = "follow") -> list[str | int]:
    """The ``follow`` rules as a list of rule names and car numbers (1 = the first car), or a friendly error."""
    if isinstance(rules, (str, int)) and not isinstance(rules, bool):
        rules = [rules]
    if not isinstance(rules, (list, tuple)) or not rules:
        raise CDSInterpretError(f"{where} should be a list of rules, e.g. [\"camera\", \"last\", \"random\"] (the "
                                f"rules: {', '.join(FOLLOW_RULES)}, or a car number), not {rules!r}")
    out: list[str | int] = []
    for r in rules:
        if isinstance(r, bool):
            raise CDSInterpretError(f"{where}: {r!r} is not a rule (the rules: {', '.join(FOLLOW_RULES)}, or a car "
                                    "number such as 2)")
        if isinstance(r, int) or (isinstance(r, str) and r.strip().isdigit()):
            n = int(r)
            if n < 1:
                raise CDSInterpretError(f"{where}: car numbers start at 1 (your bot's first car), not {n}")
            out.append(n)
            continue
        if not isinstance(r, str):
            raise CDSInterpretError(f"{where}: {r!r} is not a rule (the rules: {', '.join(FOLLOW_RULES)}, or a car "
                                    "number such as 2)")
        word = r.strip().lower()
        if word not in FOLLOW_RULES:
            close = difflib.get_close_matches(word, FOLLOW_RULES, n=1)
            hint = f" Did you mean {close[0]!r}?" if close else ""
            raise CDSInterpretError(f"{where}: {r!r} is not a rule.{hint} The rules: {', '.join(FOLLOW_RULES)}, or "
                                    "a car number (1 = your bot's first car)")
        out.append(word)
    return out


def check_landing_for(v: Any, where: str = "landing_for") -> str | list[int]:
    """``landing_for`` as one of LANDING_FOR or a list of car numbers, or a friendly error."""
    want = (f"{where} should be one of {', '.join(repr(w) for w in LANDING_FOR)} or a list of car numbers such as "
            "[1, 3] (1 = your bot's first car)")
    if isinstance(v, str):
        word = v.strip().lower()
        if word in LANDING_FOR:
            return word
        close = difflib.get_close_matches(word, LANDING_FOR, n=1)
        hint = f" Did you mean {close[0]!r}?" if close else ""
        raise CDSInterpretError(f"{want}, not {v!r}.{hint}")
    if isinstance(v, int) and not isinstance(v, bool):
        v = [v]
    if isinstance(v, (list, tuple)) and v and all(isinstance(n, int) and not isinstance(n, bool) for n in v):
        if any(n < 1 for n in v):
            raise CDSInterpretError(f"{where}: car numbers start at 1 (your bot's first car), not {list(v)}")
        return [int(n) for n in v]
    raise CDSInterpretError(f"{want}, not {v!r}")


def check_subtitle(fmt: str, where: str = "subtitle") -> str:
    """A subtitle template: text with {car}, {cars}, {rule} and {name} in it ("" for no subtitle)."""
    try:
        keys = [k for _, k, _, _ in string.Formatter().parse(fmt) if k is not None]
    except ValueError as e:
        raise CDSInterpretError(f"{where} = {fmt!r}: {e}. Write the fields in curly brackets, e.g. "
                                "\"car {car} of {cars} - {rule}\"") from None
    bad = [k for k in keys if k not in SUBTITLE_KEYS]
    if bad:
        close = difflib.get_close_matches(bad[0], SUBTITLE_KEYS, n=1)
        hint = f" Did you mean {{{close[0]}}}?" if close else ""
        raise CDSInterpretError(f"{where} = {fmt!r}: {{{bad[0]}}} is not something it can say.{hint} It can say "
                                f"{', '.join('{' + k + '}' for k in SUBTITLE_KEYS)}")
    return fmt


def subtitle_text(fmt: str, car: int, ours: list[int], rule: str, name: str = "") -> str:
    """The subtitle for the panel's car: ``fmt`` with {car} (its number among yours, 1 = the first), {cars} (how many
    you have), {rule} (in words: "on camera", "last on camera", ...) and {name} (the player's name in the game)."""
    return fmt.format(car=car_number(ours, car), cars=len(ours), rule=RULE_WORDS.get(rule, rule), name=name)


def car_number(ours: list[int], car: int) -> int:
    """A car's number among the bot's cars (1 = the lowest player index), 0 when it isn't one of them."""
    s = sorted(ours)
    return s.index(car) + 1 if car in s else 0


class Follow:
    """The panel's rules, with what they remember between packets: the last of the bot's cars the camera followed,
    and the car picked at random for the match. ``seed``: the same number in every copy of the bot (e.g. made from
    the players' ids), so every copy picks the same car."""

    def __init__(self, rules: Any = DEFAULT_FOLLOW, seed: int = 0):
        self.rules = check_follow(list(rules))
        self.seed = int(seed)
        self.last: int | None = None
        self.picked: int | None = None

    def update(self, ours: list[int], spectated: int | None) -> None:
        """Call on every packet (not only on draws), so ``last`` never misses where the camera went."""
        if spectated is not None and spectated in ours:
            self.last = int(spectated)

    def pick(self, ours: list[int], spectated: int | None) -> tuple[int | None, str]:
        """(the player index whose thoughts the panel shows, the rule that chose it: "camera", "last", "random",
        "first", "car" or "hide"). (None, "hide") means no panel. Without a rule that gives a car, the first car."""
        cars = sorted(int(c) for c in ours)
        if not cars:
            return None, "hide"
        self.update(cars, spectated)
        for rule in self.rules:
            if rule == "camera" and spectated is not None and spectated in cars:
                return int(spectated), "camera"
            if rule == "last" and self.last in cars:
                return self.last, "last"
            if rule == "random":
                if self.picked not in cars:
                    self.picked = cars[random.Random(self.seed).randrange(len(cars))]
                return self.picked, "random"
            if rule == "first":
                return cars[0], "first"
            if isinstance(rule, int) and rule <= len(cars):
                return cars[rule - 1], "car"
            if rule == "hide":
                return None, "hide"
        return cars[0], "first"


def landing_cars(landing_for: str | list[int], ours: list[int], panel_car: int | None,
                 spectated: int | None) -> list[int]:
    """Which of the bot's cars get a landing marker (player indices, in car order)."""
    cars = sorted(int(c) for c in ours)
    if landing_for == "all":
        return cars
    if landing_for == "none":
        return []
    if landing_for == "panel":
        return [panel_car] if panel_car in cars else []
    if landing_for == "camera":
        return [int(spectated)] if spectated is not None and spectated in cars else []
    if isinstance(landing_for, str):
        raise CDSInterpretError(f"landing_for {landing_for!r}: use one of {', '.join(LANDING_FOR)} or car numbers")
    return [cars[n - 1] for n in landing_for if 1 <= n <= len(cars)]


def match_seed(packet: Any, ours: list[int]) -> int:
    """A number every copy of the bot computes alike for one match, from the player ids RLBot gives each player (so
    "random" picks the same car in every copy, and usually another one next match)."""
    s = 0
    for k, i in enumerate(sorted(ours)):
        pid = getattr(packet.players[i], "player_id", i) if i < len(packet.players) else i
        try:
            s = (s * 1_000_003 + (k + 1) * int(pid)) % (1 << 61)
        except (TypeError, ValueError):
            s = (s * 1_000_003 + i) % (1 << 61)
    return s
