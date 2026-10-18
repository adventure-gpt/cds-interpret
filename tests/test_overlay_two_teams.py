"""Both teams showing thoughts at once - the same bot on both teams, or two different bots that each have an overlay
(a bot's own with_thoughts beside a watch_thoughts script): one panel where the configs put it, showing the car on
camera (else the last of theirs on camera, else one picked at random) - only the overlay with that car draws it, each
draws its own cars' rings - as big as the biggest of their panels whoever draws it; titles and rings in the teams'
colours while both run; alone, nothing changes."""

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
    """The overlays' notes in a folder of the test's own, read and written on every frame."""
    folder = tmp_path / "notes"
    monkeypatch.setenv("CDS_INTERPRET_OVERLAYS", str(folder))
    monkeypatch.setattr(SH, "BEAT", 0.0)
    monkeypatch.setattr(SH, "READ_EVERY", 0.0)
    return folder


def test_who_draws_the_one_panel(tmp_path):
    folder = tmp_path / "n"
    a, b = Neighbours(folder, pid=1, pick=0.2), Neighbours(folder, pid=2, pick=0.9)
    a.beat([0, 2], [0], 5, now=100.0)                         # (5: a car neither has)
    b.beat([1, 3], [1], 5, now=100.0)
    assert b.owns_panel([1, 3], 5, now=100.5) and not a.owns_panel([0, 2], 5, now=100.6)   # never on camera: the pick
    a.beat([0, 2], [0], 2, now=101.0)
    assert a.owns_panel([0, 2], 2, now=101.5) and not b.owns_panel([1, 3], 2, now=101.6)   # a's car on camera
    b.beat([1, 3], [1], 3, now=102.0)
    a.beat([0, 2], [0], 3, now=102.0)
    assert b.owns_panel([1, 3], 3, now=102.5) and not a.owns_panel([0, 2], 3, now=102.6)   # b's car on camera
    a.beat([0, 2], [0], 5, now=103.0)
    b.beat([1, 3], [1], 5, now=103.0)
    assert b.owns_panel([1, 3], 5, now=103.5) and not a.owns_panel([0, 2], 5, now=103.6)   # b's was on camera last
    c = Neighbours(folder, pid=3, pick=0.1)                  # a copy of a's bot (one process a car): the same group
    c.beat([0, 2], [0], 2, now=104.0)
    a.beat([0, 2], [0], 2, now=104.0)
    assert a.owns_panel([0, 2], 2, now=104.5) and c.owns_panel([0, 2], 2, now=104.6)
    assert a.crowded([0], now=104.7) and not c.crowded([0, 1], now=104.8)
    assert not Neighbours(folder, pid=4).owns_panel([], 2, now=105.0)
    b.close()
    assert a.owns_panel([0, 2], 5, now=106.0)                # b gone: a alone
    a.beat([0, 2], [0], 5, box=(0.8, 0.2, 0.15, 0.3), now=107.0)
    d = Neighbours(folder, pid=5)
    assert d.shared_box((0.82, 0.25, 0.16, 0.2), now=107.5) == pytest.approx((0.8, 0.2, 0.18, 0.3))
    (folder / "99.json").write_text('{"pid": 99, "cars": [1], "teams": [1], "t": 0}', encoding="utf-8")
    (folder / "98.json").write_text("not json", encoding="utf-8")
    assert not d.crowded([0], now=500.0) and not (folder / "99.json").exists()   # an old note left behind: cleaned up
    (tmp_path / "a-file").write_text("x")                     # a folder that can't be made: no neighbours, no error
    bad = Neighbours(tmp_path / "a-file", pid=6)
    bad.beat([0], [0], 0)
    assert not bad.ok and not bad.crowded([0]) and bad.owns_panel([0], 0)


def test_the_share_setting():
    c = PanelConfig.load("compact")                          # x 0.836, width 0.15: on the right
    assert c.share == "one" and c.place(0, True) == c.place(1, True) == 0.836     # one panel, where it is
    assert c.team_coloured(0, True) and not c.team_coloured(0, False) and not c.team_coloured(None, True)
    sides = PanelConfig.from_dict({"panel": {"share": "sides", "x": 0.836, "width": 0.15}})
    assert sides.place(0, True) == pytest.approx(0.014) and sides.place(1, True) == 0.836 and sides.place(0) == 0.836
    none = PanelConfig.from_dict({"panel": {"share": "none"}})
    assert not none.team_coloured(0, True)
    with pytest.raises(CDSInterpretError, match="Did you mean 'one'"):
        PanelConfig.from_dict({"panel": {"share": "on"}})


def _orange_class():
    Base = _hivemind_class()

    class Orange(Base):
        team, indices = 1, [1, 3, 5]
    return Orange


def _pack(tmp_path):
    pk, _ = team_pack(team_policy())
    add_land(pk, "1", np.ones((32, 2), np.float32) * 10, [100.0, 200.0])
    return save_pack(pk, tmp_path / "team.npz")


def _drawn(calls):
    """(the panel's background (x, y, w, h) or None, its title colour, the rings' colours) of one drawing."""
    rects = [c[1] for c in calls if c[0] == "draw_rect_2d"]
    title = next((c[1] for c in calls if c[0] == "draw_string_2d"), None)
    rings = {(c[1][1].r, c[1][1].g, c[1][1].b) for c in calls if c[0] == "draw_polyline_3d"}
    return (tuple(rects[0][:4]) if rects else None,
            (title[4].r, title[4].g, title[4].b) if title is not None else None, rings)


def _run(objs, frames, step, spectated):
    """Each overlay through the frames with the camera on `spectated`; each one's last drawing (rings or panel)."""
    last = {}
    for f in frames:
        pk = team_packet(f / 120, spectated)
        for name, o in objs.items():
            o.renderer.calls.clear()
            step(o, pk)
            if any(c[0] in ("draw_rect_2d", "draw_polyline_3d") for c in o.renderer.calls):
                last[name] = _drawn(o.renderer.calls)
    return last


def _hivemind_step(o, pk):
    o._packet_processor(pk)


def test_the_same_bot_on_both_teams_shares_one_panel(tmp_path, notes):
    pytest.importorskip("rlbot")
    from cds_interpret.overlay import with_thoughts
    path = _pack(tmp_path)
    kw = dict(network="policy", worker=False, draw_every=4, panel="compact")
    blue = with_thoughts(_hivemind_class(), path, **kw)()
    orange = with_thoughts(_orange_class(), path, **kw)()
    for pid, b in ((11, blue), (12, orange)):
        b.initialize()
        b._cds_th_share = Neighbours(pid=pid)                # (two processes in a real match; one here)

    alone = _run({"blue": blue}, range(1, 40), _hivemind_step, 2)     # only blue's: as it always was
    box, title, rings = alone["blue"]
    assert box[0] == 0.836 and title != BLUE and rings == {(0, 170, 255)}

    on_blue = _run({"blue": blue, "orange": orange}, range(40, 100), _hivemind_step, 2)   # the camera on blue car 2
    (bb, bt, br), (ob, _, orr) = on_blue["blue"], on_blue["orange"]
    assert bb is not None and bb[0] == 0.836 and bt == BLUE        # blue's panel, in its place, title in blue
    assert ob is None and orr == {ORANGE} and br == {BLUE}          # orange: its rings only, in orange

    on_orange = _run({"blue": blue, "orange": orange}, range(100, 160), _hivemind_step, 3)   # now on orange car 3
    (bb2, _, _), (ob2, ot2, _) = on_orange["blue"], on_orange["orange"]
    assert bb2 is None and ob2 is not None and ob2 == pytest.approx(bb) and ot2 == ORANGE   # the same box, orange's

    elsewhere = _run({"blue": blue, "orange": orange}, range(160, 200), _hivemind_step, 6)  # a car neither has
    assert elsewhere["blue"][0] is None and elsewhere["orange"][0] is not None             # the last on camera
    for a, b in zip(blue.sent, orange.sent):                 # (display only: both still play)
        assert a.keys() == {0, 2, 4} and b.keys() == {1, 3, 5}

    orange.retire()                                          # orange's overlay stops: blue's is alone again
    back = _run({"blue": blue}, range(200, 240), _hivemind_step, 6)
    assert back["blue"][0] is not None and back["blue"][0][0] == 0.836 and back["blue"][1] != BLUE


def test_a_bot_and_a_watch_script_share_the_biggest_panel(tmp_path, notes):
    pytest.importorskip("rlbot")
    from cds_interpret.overlay import with_thoughts
    from test_watch_refit import _watcher
    blue = with_thoughts(_hivemind_class(), _pack(tmp_path), network="policy", worker=False, draw_every=4,
                         panel="compact")()
    blue.initialize()
    blue._cds_th_share = Neighbours(pid=21)
    big = tmp_path / "big.toml"                              # the other bot's own panel: wider, taller, elsewhere
    big.write_text('[panel]\nx = 0.80\ny = 0.20\nwidth = 0.19\n', encoding="utf-8")
    watch = _watcher(tmp_path, cars=[1, 3, 5], panel=str(big))      # another bot's thoughts over the orange cars
    assert watch._cds_on
    watch._cds_share = Neighbours(pid=22)

    def step(o, pk):
        if o is watch:
            o.handle_packet(pk)
        else:
            o._packet_processor(pk)

    warm = _run({"blue": blue, "watch": watch}, range(1, 80), step, 3)        # (both measure their panels)
    on_watch = _run({"blue": blue, "watch": watch}, range(80, 140), step, 3)   # the camera on orange car 3
    on_blue = _run({"blue": blue, "watch": watch}, range(140, 200), step, 2)   # then on blue car 2
    wb, bb = on_watch["watch"][0], on_blue["blue"][0]
    assert warm and wb is not None and bb is not None and on_watch["blue"][0] is None and on_blue["watch"][0] is None
    assert bb == pytest.approx(wb)                            # the same box whoever draws it: the biggest of the two
    assert bb[0] == pytest.approx(0.80) and bb[1] == pytest.approx(0.20 - 0.006) and bb[2] >= 0.19 - 1e-9
    assert on_watch["watch"][1] == ORANGE and on_blue["blue"][1] == BLUE
    watch.retire()
    back = _run({"blue": blue}, range(200, 240), step, 2)
    assert back["blue"][0][0] == 0.836 and back["blue"][0][2] == pytest.approx(0.15)   # alone: its own panel again
