"""emergence (tracking a policy across training snapshots) and curriculum (situation maps, failure moments) on
hand-made games and tiny arrays."""
import numpy as np
import torch

from cds_interpret import curriculum as K
from cds_interpret import emergence as E
from cds_interpret.errors import ShapeError
from cds_interpret.frames import Frames


def game(T=40, cars=2):
    """A still 1v1 game: car 0 blue at y -2000, car 1 orange at y +2000, ball resting at centre; nothing happens."""
    d = {"ball_pos": np.tile([0.0, 0.0, 92.75], (T, 1)), "ball_vel": np.zeros((T, 3)), "ball_angvel": np.zeros((T, 3)),
         "car_pos": np.tile(np.array([[0.0, -2000.0, 17.0], [0.0, 2000.0, 17.0]])[:cars], (T, 1, 1)),
         "car_vel": np.zeros((T, cars, 3)), "car_angvel": np.zeros((T, cars, 3)),
         "car_fwd": np.tile([1.0, 0.0, 0.0], (T, cars, 1)), "car_up": np.tile([0.0, 0.0, 1.0], (T, cars, 1)),
         "car_boost": np.full((T, cars), 50.0), "car_on_ground": np.ones((T, cars)), "car_demoed": np.zeros((T, cars)),
         "car_has_flip": np.ones((T, cars)), "car_supersonic": np.zeros((T, cars)),
         "car_team": np.tile([0.0, 1.0][:cars], (T, 1)), "car_present": np.ones((T, cars)),
         "ev_touch": np.zeros((T, cars)), "ev_demo_made": np.zeros((T, cars)), "ev_demoed": np.zeros((T, cars)),
         "goal": np.full(T, -1, np.int8)}
    return d


def test_habits_counts_events_once():
    d = game()
    # car 0 airborne from step 5 to 15 with its flip spent, touches the ball at 8 and gets the flip back at 9: a reset
    d["car_on_ground"][5:16, 0] = 0
    d["car_pos"][5:16, 0, 2] = 600.0
    d["car_has_flip"][5:9, 0] = 0
    d["ev_touch"][8, 0] = 1
    d["ev_touch"][12:15, 0] = 1                           # a second contact, three steps long: one touch
    d["ball_pos"][5:16, 2] = 700.0
    d["ball_vel"][13] = [0.0, 3500.0, 0.0]                # the second touch leaves the ball fast (after its first step)
    d["goal"][20] = 0                                     # blue scores within 3 s of the air touch
    h = E.game_habits(Frames(d), np.array([True, False]), labels=False)
    a, b = h["A"], h["B"]
    assert a["touches"] == 2 and a["flip_resets"] == 1 and a["air_touches"] == 2 and a["air_dribble_touches"] == 1
    assert a["hard_hits"] == 1 and a["goals"] == 1 and b["goals_against"] == 1 and a["aerial_goals"] == 1
    assert a["air_touches_off"] == 0 and a["air_touches_def"] == 2    # ball at y 0: not in the opponent's half
    assert b["touches"] == 0 and b["flip_resets"] == 0


def test_kickoff_and_summary():
    games = []
    for first in (0, 1, 0, 0):
        d = game()
        d["ev_touch"][3, first] = 1
        d["ball_pos"][3:, 1] = 1000.0 if first == 0 else -1000.0        # the ball goes the toucher's way
        games.append(E.game_habits(Frames(d), np.array([True, False]), labels=False, step_seconds=0.1))
    s = E.summarize_habits(games, "A", step_seconds=0.1, n_boot=200)
    assert abs(s["kickoff_first_touch"]["v"] - 0.75) < 1e-9 and abs(s["kickoff_their_half"]["v"] - 0.75) < 1e-9
    assert s["kickoff_first_touch"]["lo"] <= 0.75 <= s["kickoff_first_touch"]["hi"]
    assert s["games"]["v"] == 4


def test_situations_and_dispositions():
    n = 6
    lab = {k: np.zeros(n) for k in ("in_air", "on_wall", "on_ceiling", "kickoff", "demo_chance", "speed", "boost",
                                     "supersonic", "ball_height", "ball_dist", "height", "ttb_adv", "land_y", "threat",
                                     "has_flip", "goal_side", "touch_soon")}
    lab["ball_height"][0], lab["ball_dist"][0], lab["ttb_adv"][0], lab["land_y"][0] = 800 / 2044, 1000 / 5000, 0.5, 2000
    lab["demo_chance"][1] = 1
    lab["kickoff"][2] = 1
    m = E.situations({"lab_" + k: v for k, v in lab.items()})
    assert m["off_aerial"].tolist() == [True] + [False] * 5
    assert m["demo_chance"][1] and m["kickoff"][2] and not m["off_aerial"][2]
    table = np.zeros((3, 8))
    table[0, 6] = 1                                       # boost
    table[1, 5] = 1                                       # plain jump
    table[2, [2, 5]] = 1                                  # a dodge
    probs = np.tile([0.5, 0.3, 0.2], (n, 1))
    d = E.dispositions(probs, table, {"all": np.ones(n, bool)})["all"]
    assert abs(d["p_boost"] - 0.5) < 1e-9 and abs(d["p_jump"] - 0.5) < 1e-9 and abs(d["p_dodge"] - 0.2) < 1e-9
    assert np.allclose(E.kl_rows(probs, probs), 0)


def test_health_cka_norms():
    torch.manual_seed(0)
    x = torch.relu(torch.randn(500, 20))
    x[:, 3] = 0                                           # a dead unit
    h = E.layer_health(x, rank_rows=500)
    assert abs(h["dead"] - 1 / 20) < 1e-9 and h["dormant"] >= h["dead"] and 1 < h["erank"] <= 20
    assert abs(E.linear_cka(x, x) - 1) < 1e-9
    assert abs(E.linear_cka(x, 3 * x @ torch.linalg.qr(torch.randn(20, 20))[0]) - 1) < 1e-6    # rotation, scale
    assert E.linear_cka(x, torch.relu(torch.randn(500, 20))) < 0.2
    net = torch.nn.Sequential()
    net.add_module("actor", torch.nn.Module())
    net.actor.trunk = torch.nn.Sequential(torch.nn.Linear(3, 4))
    w = E.weight_norms(net, prefixes=("actor.trunk",))
    assert abs(w["actor.trunk:total"] - float(net.actor.trunk[0].weight.detach().norm())) < 1e-6
    assert abs(E.weight_norms(net)["total"] - w["actor.trunk:total"]) < 1e-6        # by default: every Linear layer


def test_returns_and_calibration():
    goal = np.full(30, -1)
    goal[10], goal[25] = 0, 1
    r, ok = E.first_goal_returns(goal, 0, np.array([0, 10, 11, 29]), 0.9)
    assert ok.tolist() == [True, True, True, False]
    assert np.allclose(r[:3], [0.9 ** 10, 1.0, -(0.9 ** 14)])
    f, a = K.next_goal_within(goal, 0, np.array([0, 5, 12, 20]), 6)
    assert f.tolist() == [False, True, False, False] and a.tolist() == [False, False, False, True]
    v = np.linspace(-1, 1, 200)
    c = E.calibration(3 * v + 0.5, v, {"pos": v > 0})
    assert abs(c["slope"] - 3) < 1e-6 and abs(c["r"] - 1) < 1e-9 and abs(c["groups"]["pos"]["residual"]) < 1e-6


def test_pair_situations_outcomes_enrichment():
    d = game(T=4)
    d["car_pos"][:, 1] = [100.0, 0.0, 17.0]               # the opponent is on the ball
    d["ball_vel"][:, 1] = -800.0                          # moving toward blue's goal
    fr = Frames(d)
    m = K.pair_situations(fr, np.zeros(4, int))
    assert m["facing_carry"].all() and not m["we_carry"].any()
    out = K.situation_outcomes({"x": np.array([1, 1, 0, 0], bool)}, np.array([1, 0, 0, 0], bool), np.array([0, 1, 1, 0], bool),
                               np.array([0, 0, 1, 1]), n_boot=50)
    assert out["x"]["n"] == 2 and out["x"]["net"] == 0.0
    t = np.r_[np.ones(10), np.zeros(30)].astype(bool)
    col = np.r_[np.ones(6), np.zeros(4), np.ones(3), np.zeros(27)]
    e = K.enrichment(t, {"b": col})
    assert abs(e["b"]["lift"] - 0.6 / 0.1) < 1e-9 and e["b"]["z"] > 0
    y = np.r_[np.zeros(50), np.ones(50)]
    assert abs(K.eta_squared(y, np.r_[np.zeros(50), np.ones(50)]) - 1) < 1e-9
    lab, _ = K.kmeans(np.r_[np.random.default_rng(0).normal(0, 0.1, (30, 2)), np.random.default_rng(1).normal(5, 0.1, (30, 2))], 2)
    assert len(set(lab[:30])) == 1 and len(set(lab[30:])) == 1 and lab[0] != lab[30]


def test_friendly_shape_errors():
    import pytest
    with pytest.raises(ShapeError):
        E.dispositions(np.ones((3, 4)) / 4, np.zeros((5, 8)), {"all": np.ones(3, bool)})
    with pytest.raises(ShapeError):
        E.action_classes(np.zeros((5, 7)))
    with pytest.raises(ShapeError):
        E.linear_cka(torch.zeros(4, 3), torch.zeros(5, 3))
    with pytest.raises(ShapeError):
        E.layer_health(torch.zeros(4))


def test_interval_changes():
    old = {"a": {"v": 1.0, "lo": 0.8, "hi": 1.2}, "b": {"v": 2.0, "lo": 1.5, "hi": 2.5}, "c": {"v": 0.5, "lo": 0.4, "hi": 0.6},
           "games": {"v": 48, "lo": 48, "hi": 48}}
    new = {"a": {"v": 1.5, "lo": 1.3, "hi": 1.7}, "b": {"v": 2.2, "lo": 1.9, "hi": 2.6}, "c": {"v": 0.1, "lo": 0.05, "hi": 0.2},
           "games": {"v": 96, "lo": 96, "hi": 96}, "d": {"v": 1.0, "lo": 0.0, "hi": 2.0}}
    ch = E.interval_changes(old, new)
    assert [c["name"] for c in ch] == ["c", "a"]          # b overlaps; games skipped; d has no old reading
    assert ch[0]["dir"] == "down" and ch[1]["dir"] == "up"
    nets = E.interval_changes({"x": {"net": -0.4, "net_lo": -0.5, "net_hi": -0.3}},
                              {"x": {"net": 0.1, "net_lo": 0.0, "net_hi": 0.2}}, value="net", lo="net_lo", hi="net_hi")
    assert nets and nets[0]["name"] == "x" and nets[0]["dir"] == "up"


def test_kickoff_goals_and_open_play():
    d = game(T=200)
    d["ev_touch"][3, 0] = 1                               # kickoff 1: blue touches first, scores 1.7 s later
    d["goal"][20] = 0
    d["ev_touch"][25, 1] = 1                              # kickoff 2: orange touches first; blue scores 12.5 s later
    d["goal"][150] = 0
    d["ev_touch"][155, 1] = 1                             # kickoff 3: orange scores 2 s after its touch
    d["goal"][175] = 1
    h = E.game_habits(Frames(d), np.array([True, False]), labels=False, step_seconds=0.1)
    assert h["A"]["kickoff_goals"] == 1 and h["A"]["open_goals"] == 1 and h["A"]["kickoff_goals_against"] == 1
    assert h["B"]["kickoff_goals"] == 1 and h["B"]["open_goals"] == 0
    s = E.summarize_habits([h], "A", step_seconds=0.1, n_boot=50)
    assert abs(s["goal_share"]["v"] - 2 / 3) < 1e-9 and abs(s["open_goal_share"]["v"] - 1.0) < 1e-9
