"""The customizable panel (TOML config, hooks), the worker's writer thread, and the feature-naming tools."""

import time
import tomllib
from importlib.resources import files

import numpy as np
import pytest

import cds_interpret as ci
from cds_interpret.errors import CDSInterpretError
from cds_interpret.overlay import Panel, PanelConfig, RowSpec, WorkerLink
from cds_interpret.overlay import worker as W

from test_overlay import FakeRenderer, fake_flat

RESULT = {"value": 0.6, "action": {"words": "throttle, steer right, boost", "p": 0.8},
          "features": [{"name": "lined up", "act": 2.0, "on": True}, {"name": "defending", "act": 0.0, "on": False}],
          "probes": {"Threat": 0.1, "Time to ball": 1.2, "Speed": 1500.0},
          "looks": {"Threat": {"unit": "%", "color": [240, 90, 80]}, "Time to ball": {"unit": "s"},
                    "Speed": {"unit": ""}}}


def _texts(r):
    return [c[1][0] for c in r.calls if c[0] == "draw_string_2d"]


# ------------------------------------------------------------------------------------------------- the panel
def test_the_example_file_is_the_default_panel():
    text = files("cds_interpret.overlay").joinpath("panel_example.toml").read_text(encoding="utf-8")
    assert PanelConfig.from_dict(tomllib.loads(text)) == PanelConfig()


def test_default_panel_draws_every_part_within_its_width():
    r = FakeRenderer()
    Panel().draw(r, fake_flat(), RESULT, car_index=0, landing_true=(0, 0, 93), landing_think=(10, 10), state={},
                 frame=5)
    texts = _texts(r)
    assert texts[0] == "THOUGHTS" and "Outlook" in texts and "Next move" in texts and "> lined up" in texts
    assert "Threat" in texts and "1.2 s" in texts and "1500.00" in texts
    assert all(len(t) <= 30 for t in texts)
    names = [c[0] for c in r.calls]
    assert "draw_polyline_3d" in names and "draw_string_3d" in names


def test_a_config_file_changes_rows_names_colours_and_markers(tmp_path):
    (tmp_path / "panel.toml").write_text('''
[panel]
title = "MY BOT"
width = 0.25
hold = 0.0

[markers]
landing = false
car_label = false

[features."lined up"]
label = "Lined up!"
threshold = 5.0

[[rows]]
kind = "bar"
source = "probe:Threat"
label = "Danger"
color = [255, 0, 0]

[[rows]]
kind = "noticing"
empty = "calm"

[[rows]]
kind = "custom"
hook = "mine"
''', encoding="utf-8")
    cfg = PanelConfig.load(tmp_path / "panel.toml")
    seen = []

    def mine(draw, th, state):
        seen.append(th["value"])
        draw.line("custom row", f"{th['value']:.1f}")
        draw.bar(th["value"], draw.color(1, 2, 3))

    r = FakeRenderer()
    Panel(cfg, hooks={"mine": mine}).draw(r, fake_flat(), RESULT, car_index=0, landing_think=(1, 1), state={}, frame=1)
    texts = _texts(r)
    assert texts[0] == "MY BOT" and texts[1] == "Danger" and "Outlook" not in texts and "Threat" not in texts
    assert "calm" in texts                                        # "lined up" is under its new threshold of 5
    assert "custom row" in texts and seen == [0.6]
    names = [c[0] for c in r.calls]
    assert "draw_polyline_3d" not in names and "draw_string_3d" not in names
    rects = [c[1] for c in r.calls if c[0] == "draw_rect_2d"]
    assert any(len(a) > 4 and a[4] == (255, 0, 0, 255) for a in rects)       # the bar's own colour


def test_friendly_config_errors(tmp_path):
    with pytest.raises(CDSInterpretError, match="Did you mean 'width'"):
        PanelConfig.from_dict({"panel": {"widht": 0.2}})
    with pytest.raises(CDSInterpretError, match="Did you mean 'percent'"):
        RowSpec("percnt", source="probe:x")
    with pytest.raises(CDSInterpretError, match="needs a source"):
        RowSpec("bar")
    with pytest.raises(CDSInterpretError, match="colour"):
        PanelConfig.from_dict({"colors": {"bar": [300, 0, 0]}})
    (tmp_path / "bad.toml").write_text("[panel\n", encoding="utf-8")
    with pytest.raises(CDSInterpretError, match="not valid TOML"):
        PanelConfig.load(tmp_path / "bad.toml")
    with pytest.raises(CDSInterpretError, match="cds-interpret panel"):
        PanelConfig.load(tmp_path / "missing.toml")


def test_cli_writes_the_example(tmp_path):
    from cds_interpret.cli import main
    out = tmp_path / "panel.toml"
    assert main(["panel", str(out)]) == 0 and PanelConfig.load(out) == PanelConfig()
    assert main(["panel", str(out)]) == 1                         # never overwrites without --force


def test_writes_never_wait_for_a_slow_worker():
    """A large request to a worker that is slow to read: the writer thread waits, the caller does not."""
    link = WorkerLink(W.echo, 0.0, 1.5)
    try:
        deadline = time.time() + 120
        while not link.ready and time.time() < deadline:
            link.poll()
            time.sleep(0.05)
        big = {"x": np.zeros(500_000, np.float32)}                 # 2 MB, far beyond a pipe's buffer
        t = time.time()
        assert link.send(big)
        assert time.time() - t < 0.2
        got = None
        while got is None and time.time() < deadline:
            got = link.poll()
            time.sleep(0.02)
        assert got is not None and got["x"].shape == (500_000,)
    finally:
        link.close()


# ------------------------------------------------------------------------------------------- naming features
@pytest.fixture(scope="module")
def named_study():
    pol = ci.demo.policy()
    m = ci.demo.synthetic_moments(n_games=20, per_game=50)
    st = ci.Study(pol, m, log=None)
    st.probes(baselines=False)
    st.features("body.3", m=64, k=4, steps=300)
    return st


def test_suggest_write_read_and_pack_from_names(named_study, tmp_path):
    st = named_study
    sugg = st.suggest_names("body.3", top=6)
    assert sugg and all(isinstance(s.name, str) for s in sugg)
    path = ci.names.write_names(tmp_path / "names.toml", sugg, "body.3")
    with pytest.raises(CDSInterpretError, match="already exists"):
        ci.names.write_names(path, sugg, "body.3")
    nm = ci.names.read_names(path)
    assert nm.layer == "body.3" and set(nm.features) == {s.feature for s in sugg} and not nm.confirmed()
    with pytest.raises(CDSInterpretError, match="no confirmed names"):
        st.make_pack("body.3", names=path)
    j = sugg[0].feature
    text = path.read_text(encoding="utf-8").replace(f'[features.{j}]\nname = ', f'[features.{j}]\nthreshold = 0.25\nname = ')
    text = text.replace('status = "suggested"', 'status = "confirmed"', 1)
    path.write_text(text, encoding="utf-8")
    nm = ci.names.read_names(path)
    assert nm.confirmed() == {sugg[0].name: j} and nm.thresholds() == {sugg[0].name: 0.25}
    pack = st.make_pack("body.3", names=path)
    assert [f["name"] for f in pack["features"]] == [sugg[0].name] and pack["features"][0]["threshold"] == 0.25
    st.feature_sets["body.3"].apply_names(path)
    assert st.feature_sets["body.3"].name(j) == sugg[0].name
    path.write_text('layer = "body.3"\n[features.x]\nname = "a"\n', encoding="utf-8")
    with pytest.raises(CDSInterpretError, match="by its number"):
        ci.names.read_names(path)


def test_kickoff_warning_and_gallery(named_study, tmp_path):
    st = named_study
    fs = st.feature_sets["body.3"]
    labels = dict(st.labels)
    j = fs.top(1)[0]
    labels["kickoff"] = (fs.activation(j) > 0).astype(np.float32)   # pretend all its firings are kickoffs
    s = ci.names.suggest(fs, [j], labels)[0]
    assert any("kickoffs" in w for w in s.warnings)
    page = st.feature_gallery("body.3", tmp_path / "f.html", n=4, exclude_kickoffs=True).read_text(encoding="utf-8")
    assert page.count("class='card'") == 4 and "suggested:" in page


def test_check_feature_says_what_changes(named_study, tmp_path):
    st = named_study
    j = st.feature_sets["body.3"].top(1)[0]
    c = st.check_feature("body.3", j)
    assert set(c.results) == {0.0, 3.0} and c.n > 0
    assert all(0 <= r["changed"] <= 1 and r["kl"] >= 0 and "shifts" in r for r in c.results.values())
    assert "turned off" in c.summary()
    with pytest.raises(CDSInterpretError, match=r"study\.features"):
        st.check_feature("body.1", 0)
    assert "Feature checks" in st.report(tmp_path / "r.html").read_text(encoding="utf-8")


def test_the_default_panel_draws_exactly_the_reference_picture(tmp_path):
    """scripts/render_panel.py draws the default panel to SVG; it must match assets/overlay-panel.svg, the panel's
    reference design (bars under labels, the narrow panel)."""
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("render_panel", root / "scripts" / "render_panel.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "panel.svg"
    mod.main(str(out))
    assert out.read_text(encoding="utf-8") == (root / "assets" / "overlay-panel.svg").read_text(encoding="utf-8")
