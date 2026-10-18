"""Both teams showing thoughts at once: the same bot on both teams, and two different bots that each have an
overlay (a bot's own with_thoughts beside a watch_thoughts script). Alone, a panel stays where its config puts it;
with the other team's overlay showing too, the two go to their teams' sides of the screen (blue left, orange right),
never overlapping, each title and landing ring in its team's colour - and back when the other one stops."""

import numpy as np
import pytest

from cds_interpret.errors import CDSInterpretError
from cds_interpret.overlay import PanelConfig, add_land, save_pack
from cds_interpret.overlay import share as SH
from cds_interpret.overlay.share import Neighbours
from test_overlay_team import _hivemind_class, team_pack, team_packet, team_policy

BLUE, ORANGE = (80, 160, 255), (255, 150, 40)              # colors.team_blue / team_orange


@pytest.fixture
def notes(tmp_path, monkeypatch):
    """The overlays' notes in a folder of the test's own, read and written on every frame (no 1 s wait)."""
    folder = tmp_path / "notes"
    monkeypatch.setenv("CDS_INTERPRET_OVERLAYS", str(folder))
    monkeypatch.setattr(SH, "BEAT", 0.0)
    monkeypatch.setattr(SH, "READ_EVERY", 0.0)
    return folder


def test_the_notes_say_who_shows_which_team(tmp_path):
    folder = tmp_path / "n"
    a, b = Neighbours(folder, pid=1), Neighbours(folder, pid=2)
    assert not a.crowded(0, now=100.0)                       # nobody else yet
    a.beat(0, True, now=100.0)
    b.beat(1, True, now=100.0)
    assert a.crowded(0, now=101.0) and b.crowded(1, now=101.0)
    assert not a.crowded(1, now=102.0)                       # another overlay of the same team: not the other side
    assert not a.crowded(None, now=103.0)
    assert not Neighbours(folder, pid=3).crowded(0, now=107.0)      # b's note not rewritten for 7 s: not running
    b.beat(None, False, now=140.0)                           # running, but no panel for 40 s: not showing any team
    assert not Neighbours(folder, pid=3).crowded(0, now=140.5)
    b.beat(1, True, now=141.0)
    assert Neighbours(folder, pid=3).crowded(0, now=141.5)
    b.close()
    assert not (folder / "2.json").exists() and not Neighbours(folder, pid=3).crowded(0, now=142.0)
    (folder / "99.json").write_text('{"pid": 99, "team": 1, "shown": 0, "t": 0}', encoding="utf-8")
    (folder / "98.json").write_text("not json", encoding="utf-8")
    assert not Neighbours(folder, pid=3).crowded(0, now=500.0)
    assert not (folder / "99.json").exists()                 # a note left behind long ago is cleaned up
    (tmp_path / "a-file").write_text("x")                     # a folder that can't be made: no neighbours, no error
    bad = Neighbours(tmp_path / "a-file", pid=4)
    bad.beat(0, True)
    assert not bad.ok and not bad.crowded(0)


def test_the_teams_sides_mirror_the_configured_spot():
    c = PanelConfig.load("compact")                          # x 0.836, width 0.15: on the right
    assert c.side == "auto" and c.place(0) == c.place(1) == 0.836          # alone: where it is
    assert c.place(0, crowded=True) == pytest.approx(1 - 0.836 - 0.15) and c.place(1, crowded=True) == 0.836
    assert c.place(None, crowded=True) == 0.836 and not c.sided(None, True)
    d = PanelConfig()                                        # x 0.012: on the left
    assert d.place(0, True) == 0.012 and d.place(1, True) == pytest.approx(1 - 0.012 - 0.17)
    team = PanelConfig.from_dict({"panel": {"side": "team", "x": 0.836, "width": 0.15}})
    assert team.place(0) == pytest.approx(0.014) and team.sided(1)
    fixed = PanelConfig.from_dict({"panel": {"side": "fixed", "x": 0.836, "width": 0.15}})
    assert fixed.place(0, True) == 0.836 and not fixed.sided(0, True)
    with pytest.raises(CDSInterpretError, match="Did you mean 'team'"):
        PanelConfig.from_dict({"panel": {"side": "teams"}})


def _orange_class():
    Base = _hivemind_class()

    class Orange(Base):
        team, indices = 1, [1, 3, 5]
    return Orange


def _pack(tmp_path):
    pk, _ = team_pack(team_policy())
    add_land(pk, "1", np.ones((32, 2), np.float32) * 10, [100.0, 200.0])
    return save_pack(pk, tmp_path / "team.npz")


def _panel(calls):
    """(the panel's left edge, its title colour, the landing rings' colours) of one drawing, or None without a panel."""
    rects = [c[1] for c in calls if c[0] == "draw_rect_2d"]
    if not rects:
        return None
    title = next(c[1] for c in calls if c[0] == "draw_string_2d")
    rings = {(c[1][1].r, c[1][1].g, c[1][1].b) for c in calls if c[0] == "draw_polyline_3d"}
    return rects[0][0], (title[4].r, title[4].g, title[4].b), rings


def _apart(a, b, width):
    return a + width <= b or b + width <= a


def _run(objs, frames, step):
    """Each overlay through the frames; the last drawing with a panel of each."""
    last = {}
    for f in frames:
        pk = team_packet(f / 120, 2)                         # the camera on blue car 2
        for name, o in objs.items():
            o.renderer.calls.clear()
            step(o, pk)
            got = _panel(o.renderer.calls)
            if got is not None:
                last[name] = got
    return last


def test_the_same_bot_on_both_teams(tmp_path, notes):
    pytest.importorskip("rlbot")
    from cds_interpret.overlay import with_thoughts
    path = _pack(tmp_path)
    kw = dict(network="policy", worker=False, draw_every=4, panel="compact")
    blue = with_thoughts(_hivemind_class(), path, **kw)()
    orange = with_thoughts(_orange_class(), path, **kw)()
    for pid, b in ((11, blue), (12, orange)):
        b.initialize()
        b._cds_th_share = Neighbours(pid=pid)                # (two processes in a real match; one here)
    step = type(blue)._packet_processor

    alone = _run({"blue": blue}, range(1, 40), step)        # only blue's overlay: where the config puts it
    x, title, rings = alone["blue"]
    assert x == 0.836 and title != BLUE and rings == {(0, 170, 255)}

    both = _run({"blue": blue, "orange": orange}, range(40, 100), step)
    (bx, bt, br), (ox, ot, orr) = both["blue"], both["orange"]
    assert bx == pytest.approx(0.014) and ox == 0.836 and _apart(bx, ox, 0.15)   # blue left, orange right
    assert bt == BLUE and ot == ORANGE and br == {BLUE} and orr == {ORANGE}
    for a, b in zip(blue.sent, orange.sent):                 # (display only: both still play)
        assert a.keys() == {0, 2, 4} and b.keys() == {1, 3, 5}

    orange.retire()                                          # orange's overlay stops: blue's goes back
    back = _run({"blue": blue}, range(100, 140), step)
    assert back["blue"][0] == 0.836 and back["blue"][1] != BLUE


def test_a_bot_and_a_watch_script_on_the_other_team(tmp_path, notes):
    pytest.importorskip("rlbot")
    from cds_interpret.overlay import with_thoughts
    from test_watch_refit import _watcher
    blue = with_thoughts(_hivemind_class(), _pack(tmp_path), network="policy", worker=False, draw_every=4,
                         panel="compact")()
    blue.initialize()
    blue._cds_th_share = Neighbours(pid=21)
    watch = _watcher(tmp_path, cars=[1, 3, 5], panel="compact")   # another bot's thoughts, over the orange cars
    assert watch._cds_on
    watch._cds_share = Neighbours(pid=22)

    def step(o, pk):
        if o is watch:
            o.handle_packet(pk)
        else:
            o._packet_processor(pk)

    both = _run({"blue": blue, "watch": watch}, range(1, 80), step)
    (bx, bt, br), (wx, wt, wr) = both["blue"], both["watch"]
    assert bx == pytest.approx(0.014) and wx == 0.836 and _apart(bx, wx, 0.15)
    assert bt == BLUE and wt == ORANGE and br == {BLUE} and wr <= {ORANGE}
    watch.retire()
    back = _run({"blue": blue}, range(80, 120), step)
    assert back["blue"][0] == 0.836
