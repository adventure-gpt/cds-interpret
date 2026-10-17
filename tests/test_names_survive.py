"""Names stay on the features they were given to: seeded dictionaries, saved dictionaries, names files that know
their dictionary, thresholds that can be reached - and the panel checks that catch what would silently not show."""

import numpy as np
import pytest
import torch

import cds_interpret as ci
from cds_interpret.errors import CDSInterpretError
from cds_interpret.overlay import PanelConfig, check_panel

from conftest import meshes_or_skip


@pytest.fixture(scope="module")
def st():
    pol = ci.demo.policy()
    m = ci.demo.synthetic_moments(n_games=20, per_game=50)
    s = ci.Study(pol, m, log=None)
    s.probes(baselines=False)
    s.features("body.3", m=64, k=4, steps=200)
    return s


def test_the_same_seed_gives_the_same_dictionary(st):
    a = st.feature_sets["body.3"].fingerprint
    torch.rand(1000)                                     # whatever ran before must not matter
    b = st.features("body.3", m=64, k=4, steps=200).fingerprint
    assert a == b
    assert st.features("body.3", m=64, k=4, steps=200, seed=1).fingerprint != a
    st.features("body.3", m=64, k=4, steps=200)          # (back to seed 0 for the other tests)


def test_a_saved_dictionary_comes_back_with_its_numbers(st, tmp_path):
    fs = st.feature_sets["body.3"]
    path = fs.save(tmp_path / "body3.sae")
    back = st.load_features(path)
    assert back.fingerprint == fs.fingerprint and back.layer == "body.3"
    j = back.top(1)[0]
    assert np.array_equal(back.activation(j), fs.activation(j))
    with pytest.raises(CDSInterpretError, match="no saved dictionary"):
        st.load_features(tmp_path / "missing.sae")


def test_names_refuse_another_dictionary(st, tmp_path):
    fs = st.feature_sets["body.3"]
    sugg = st.suggest_names("body.3", top=6)
    path = ci.names.write_names(tmp_path / "names.toml", sugg, "body.3")
    text = path.read_text(encoding="utf-8")
    assert f'dictionary = "{fs.fingerprint}"' in text and "when it fires:" in text
    path.write_text(text.replace('status = "suggested"', 'status = "confirmed"', 1), encoding="utf-8")
    assert st.make_pack("body.3", names=path)["features"]            # the dictionary it was written for: fine
    saved = fs.save(tmp_path / "body3.sae")
    st.features("body.3", m=64, k=4, steps=200, seed=7)                # another dictionary of the same layer
    with pytest.raises(CDSInterpretError, match="another dictionary"):
        st.make_pack("body.3", names=path)
    with pytest.raises(CDSInterpretError, match="another dictionary"):
        st.feature_sets["body.3"].apply_names(path)
    with pytest.raises(CDSInterpretError, match="another dictionary"):
        ci.names.update_names(path, st.suggest_names("body.3", top=4), "body.3")
    st.load_features(saved)                                            # the one you named, back again
    assert st.make_pack("body.3", names=path)["features"]


def test_thresholds_must_be_reachable_and_checks_take_names(st, tmp_path):
    fs = st.feature_sets["body.3"]
    j = fs.top(1)[0]
    path = tmp_path / "names.toml"
    path.write_text(f'layer = "body.3"\n[features.{j}]\nname = "busy"\nthreshold = 1000.0\n', encoding="utf-8")
    with pytest.raises(CDSInterpretError, match="never got above"):
        st.make_pack("body.3", names=path)
    fs.apply_names(path)
    assert st.check_feature("body.3", "busy").feature == j
    with pytest.raises(CDSInterpretError, match="Did you mean 'busy'"):
        st.check_feature("body.3", "bussy")
    with pytest.raises(CDSInterpretError, match="no feature 9999"):
        st.check_feature("body.3", 9999)
    z = fs.sizes(j)
    assert 0 < z["start"] <= z["typical"] <= z["highest"]


def test_suggestions_show_each_different_name_before_repeats(st):
    sugg = st.suggest_names("body.3", top=8)
    names = [s.name.split(" #")[0] for s in sugg]
    first_repeat = next((i for i, n in enumerate(names) if n in names[:i]), len(names))
    assert len(set(names[first_repeat:]) - set(names[:first_repeat])) == 0
    gallery_names = ci.names.suggest(st.feature_sets["body.3"], None, st.labels, top=8)
    assert [s.feature for s in gallery_names] == [s.feature for s in sugg]


def test_the_probe_table_says_what_was_worked_out(st):
    text = st.probe_results.summary()
    assert "Worked out by the network" in text or "Nothing beats both baselines" in text
    assert "AUROC (0.50 is a guess" in text


def test_a_second_recording_into_the_same_folder_says_how_to_load_it(tmp_path):
    m = ci.demo.synthetic_moments(n_games=2, per_game=10)
    m.save(tmp_path / "rec")
    with pytest.raises(CDSInterpretError, match=r"ci\.Moments\.load"):
        m.save(tmp_path / "rec")


def test_a_script_run_again_can_reuse_its_recording(tmp_path):
    meshes_or_skip()
    pol = ci.demo.policy()
    m = ci.record.rocketsim(pol, games=1, seconds=4, out=tmp_path / "rec", log=None)
    with pytest.raises(CDSInterpretError, match='if_exists="reuse"'):
        ci.record.rocketsim(pol, games=1, seconds=4, out=tmp_path / "rec", log=None)
    lines = []
    again = ci.record.rocketsim(pol, games=1, seconds=4, out=tmp_path / "rec", if_exists="reuse", log=lines.append)
    assert len(again) == len(m) and "using it" in lines[0]
    with pytest.raises(CDSInterpretError, match="must be one of"):
        m.save(tmp_path / "rec", if_exists="reuse")                   # (only a recorder can reuse)


# ------------------------------------------------------------------------------------------------- the panel
def test_panel_values_are_checked_when_the_file_loads(tmp_path):
    assert PanelConfig.from_dict({"colors": {"bar": "#ff8000"}}).colors["bar"] == (255, 128, 0)
    with pytest.raises(CDSInterpretError, match="#rrggbb"):
        PanelConfig.from_dict({"colors": {"bar": "orange"}})
    with pytest.raises(CDSInterpretError, match="should be a number - without the quotes"):
        PanelConfig.from_dict({"panel": {"x": "0.012"}})
    with pytest.raises(CDSInterpretError, match=r"\[\[rows\]\] #2.*Did you mean 'centred'"):
        PanelConfig.from_dict({"rows": [{"kind": "outlook"}, {"kind": "centered", "source": "value"}]})
    with pytest.raises(CDSInterpretError, match="is a \\[panel\\] setting"):
        PanelConfig.from_dict({"rows": [{"kind": "outlook", "title": "MINE"}]})
    (tmp_path / "twice.toml").write_text("[markers]\nlanding = true\n[markers]\ncar_label = false\n", encoding="utf-8")
    with pytest.raises(CDSInterpretError, match="only once in a file"):
        PanelConfig.load(tmp_path / "twice.toml")


def test_check_panel_finds_what_would_not_show(tmp_path):
    pack = {"features": [{"name": "airborne"}], "probes": [{"name": "First to the ball"}], "outlook": None}
    conf = PanelConfig.from_dict({
        "features": {"Airborne": {"label": "Up!"}},
        "rows": [{"kind": "bar", "source": "probe:first to the ball"},
                 {"kind": "noticing", "features": ["airbourne"]},
                 {"kind": "outlook"}]})
    problems = check_panel(conf, pack)
    assert len(problems) == 4
    assert "Did you mean 'First to the ball'" in problems[0] and "Did you mean 'airborne'" in problems[1]
    assert "no outlook" in problems[2] and "Did you mean 'airborne'" in problems[3]
    assert check_panel(PanelConfig.from_dict({"rows": [{"kind": "probes"}]}), pack) == []


def test_cli_checks_a_panel(tmp_path, capsys):
    from cds_interpret.cli import main
    out = tmp_path / "panel.toml"
    main(["panel", str(out)])
    assert main(["check-panel", str(out)]) == 0
    out.write_text("[panel]\nwidth = 5\n", encoding="utf-8")
    assert main(["check-panel", str(out)]) == 1
    assert "share of the screen" in capsys.readouterr().out
