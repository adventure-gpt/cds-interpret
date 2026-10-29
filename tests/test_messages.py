"""A team-mate message channel: usage counts bits right; the meaning finds the label a symbol tracks (and not noise,
against time-shifted copies) and names it; the listening test scores a deaf receiver exactly 0 and a listening one
clearly, with no rounding of its own; the Channel's reading and writing; the Study method, the report section, the
saved numbers and the friendly errors."""

import json

import numpy as np
import pytest
import torch

import cds_interpret as ci
from cds_interpret import messages as M
from cds_interpret.errors import CDSInterpretError, ShapeError


# ------------------------------------------------------------------------------------------------------- channel
def test_the_channel_reads_and_writes_each_slot():
    ch = M.Channel(symbols=3, columns=[range(2, 5), slice(5, 8)])
    x = torch.zeros(4, 8)
    x[0, 2 + 1] = 1.0                    # slot 0 says 1, slot 1 silent
    x[1, 5 + 2] = 1.0                    # slot 1 says 2
    x[2, 2] = x[2, 5] = 1.0              # both say 0
    assert ch.incoming(x).tolist() == [[1, -1], [-1, 2], [0, 0], [-1, -1]]
    y = ch.put(x, 1, 1)
    assert ch.incoming(y)[:, 1].tolist() == [1, 1, 1, 1] and torch.equal(y[:, :5], x[:, :5])
    assert ch.incoming(ch.put(x, 0, None))[:, 0].tolist() == [-1, -1, -1, -1]
    live = M.Channel(3, [range(2, 5)], live=lambda x: x[:, 0] > 0.5)
    assert live.incoming(x)[:, 0].tolist() == [-1, -1, -1, -1]
    with pytest.raises(ShapeError, match="one column per symbol"):
        M.Channel(symbols=4, columns=[range(0, 3)])
    with pytest.raises(CDSInterpretError, match="no sender slots"):
        M.Channel(symbols=4, columns=[])


# -------------------------------------------------------------------------------------------------------- usage
def test_usage_counts_bits():
    k, n = 8, 2000
    u = M.usage(np.full((n, k), 1 / k), np.zeros(n, np.int64))
    assert u["entropy_per_decision"] == pytest.approx(3.0) and u["info_sampled"] == pytest.approx(0.0, abs=1e-12)
    s = np.random.default_rng(0).integers(0, k, n)
    u = M.usage(np.eye(k)[s], s)
    assert u["entropy_per_decision"] == pytest.approx(0.0) and u["info_sampled"] == pytest.approx(u["entropy_sent"])


def test_mutual_information_by_hand():
    a = np.array([0, 0, 1, 1])
    assert M.mutual_info(a, a) == pytest.approx(1.0)
    assert M.mutual_info(a, np.array([0, 1, 0, 1])) == pytest.approx(0.0)
    assert M.mutual_info_soft(np.eye(2)[a], a) == pytest.approx(1.0)
    assert M.mutual_info_soft(np.full((4, 2), 0.5), a) == pytest.approx(0.0)
    codes, names = M.discretize(np.array([0.1, 0.2, 0.3, 0.4, np.nan]), "reg", bins=2)
    assert names == ["low", "high", "n/a"] and codes.tolist() == [0, 0, 1, 1, 2]


# ------------------------------------------------------------------------------------------------------ meaning
def test_meaning_finds_the_tracked_label_and_not_noise():
    rng = np.random.default_rng(1)
    games, per = 30, 200
    n = games * per
    game, step = np.repeat(np.arange(games), per), np.tile(np.arange(per), games)
    lab = np.repeat(rng.random(n // 20) < 0.3, 20).astype(np.float32)        # a label that lasts a while
    noise = np.repeat(rng.random(n // 20) < 0.5, 20).astype(np.float32)
    sent = np.where(lab > 0.5, 2, rng.integers(0, 2, n))
    out = M.meaning(sent, {"closest": lab, "noise": noise}, probs=np.full((n, 4), 0.25), games=game, order=step,
                    shuffles=10)
    good, bad = out["labels"]["closest"], out["labels"]["noise"]
    assert good["beats"] and good["excess"] > 0.3 and 0 < good["share"] <= 1
    assert not (bad["beats"] and bad["excess"] > M.MIN_BITS)
    assert "closest car" in out["symbols"]["sent"][2]["phrase"]
    assert out["symbols"]["sent"][3]["phrase"] == "never sent"


# ---------------------------------------------------------------------------------------------------- listening
def test_a_deaf_receiver_scores_exactly_zero_and_a_listening_one_does_not():
    deaf, ears = ci.demo.message_team(listens=False, n_games=6), ci.demo.message_team(n_games=6)
    for d, listens in ((deaf, False), (ears, True)):
        x = torch.as_tensor(d.moments.obs)
        rows = M.listen(d.policy.log_probs, x, d.channel, batch=128)
        assert len(rows["real"]) == len(x) and np.all(rows["noise"] == 0.0)
        lis = M.listening(rows, 4, [0.25] * 4)
        if listens:
            assert lis["changed_swap"] > 0.3 and lis["kl_swap"] > 0.5
            assert np.isfinite(np.asarray(lis["kl_matrix"])).all()
        else:
            assert lis["kl_swap"] == 0.0 and lis["changed_silence"] == 0.0 and lis["changed_any"] == 0.0
    refs = {"own boost zeroed": lambda x, s: torch.cat([x[:, :4], torch.zeros_like(x[:, 4:5]), x[:, 5:]], 1)}
    rows = M.listen(deaf.policy.log_probs, torch.as_tensor(deaf.moments.obs), deaf.channel, references=refs)
    assert rows["kl_own boost zeroed"].mean() > 0.0


# ------------------------------------------------------------------------------------------------ study, report
def test_the_study_answers_all_three_and_the_report_shows_them(tmp_path):
    d = ci.demo.message_team(n_games=12)
    study = ci.Study(d.policy, d.moments, layers=[], log=None)
    rep = study.messages(logits=d.logits, channel=d.channel, shuffles=5)
    assert rep.verdicts["meaning"].startswith("Yes") and rep.verdicts["listening"].startswith("Yes")
    assert "closest" in rep.verdicts["meaning"]
    ph = rep.phrases()
    assert "closest car" in ph[0] and "last car back" in ph[2]
    assert rep.usage["symbols"] == 4 and abs(sum(rep.usage["share_sent"]) - 1) < 1e-9
    page = study.report(tmp_path / "r.html").read_text(encoding="utf-8")
    assert "Message channel" in page and "team-mates listen" in page
    saved = json.loads(rep.save(tmp_path / "m.json").read_text(encoding="utf-8"))
    assert saved["verdicts"]["listening"] == rep.verdicts["listening"]
    assert "symbols as sent" in rep.summary() and set(rep.figures()) == {"usage", "meaning", "listening"}
    rep2 = study.messages(logits=lambda x: torch.as_tensor(d.logits[:len(x)]) * 0.0, shuffles=2)   # a callable
    assert rep2.listening is None and rep2.usage["entropy_per_decision"] == pytest.approx(2.0)
    assert "Not tested" in rep2.verdicts["listening"]


def test_study_messages_mistakes_are_explained():
    d = ci.demo.message_team(n_games=4)
    study = ci.Study(d.policy, d.moments, layers=[], log=None)
    with pytest.raises(ShapeError, match="moments, symbols"):
        study.messages(logits=d.logits[:10])
    with pytest.raises(CDSInterpretError, match="no label"):
        study.messages(logits=d.logits, labels=["closest", "clostest"])
    with pytest.raises(ShapeError, match="symbols"):
        M.analyze(d.logits, x=torch.as_tensor(d.moments.obs), forward=d.policy.log_probs,
                  channel=M.Channel(5, [range(7, 12)]))
