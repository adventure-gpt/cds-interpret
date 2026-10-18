"""The built-in panels: "default" is the example file's panel, "compact" a smaller one in the top-right corner; both
load by name, a file of the same name wins, and `cds-interpret panel --compact` writes the compact one out."""
from __future__ import annotations

import pytest

from cds_interpret.cli import main
from cds_interpret.errors import CDSInterpretError
from cds_interpret.overlay.panel import PRESETS, PanelConfig, preset_text


def test_default_by_name_is_the_default_panel():
    named = PanelConfig.load("default")
    plain = PanelConfig()
    assert (named.x, named.y, named.width, named.text_scale) == (plain.x, plain.y, plain.width, plain.text_scale)
    assert [r.kind for r in named.rows] == [r.kind for r in plain.rows]


def test_compact_is_smaller_and_in_the_top_right():
    c, d = PanelConfig.load("compact"), PanelConfig()
    assert c.x + c.width <= 1.0 and c.x > 0.5 and c.y < d.y
    assert c.text_scale < d.text_scale and c.width < d.width
    kinds = [r.kind for r in c.rows]
    assert kinds == ["outlook", "action", "why", "looking", "noticing", "probes"]
    action = next(r for r in c.rows if r.kind == "action")
    assert action.lines == 1


def test_a_file_of_the_same_name_wins(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "compact").write_text('[panel]\ntitle = "MINE"\n', encoding="utf-8")
    assert PanelConfig.load("compact").title == "MINE"


def test_unknown_name_says_what_there_is():
    with pytest.raises(CDSInterpretError, match="compact"):
        PanelConfig.load("tiny")
    with pytest.raises(CDSInterpretError):
        preset_text("tiny")
    assert set(PRESETS) == {"default", "compact"}


def test_cli_writes_the_compact_panel(tmp_path, capsys):
    out = tmp_path / "panel.toml"
    assert main(["panel", str(out), "--compact"]) == 0
    assert out.read_text(encoding="utf-8") == preset_text("compact")
    assert PanelConfig.load(out).x == PanelConfig.load("compact").x
