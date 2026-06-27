"""Offline tests for the AFML ch.4 sample weights (golden vectors computed by hand)."""
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_sample_weights as SW
import egp_triple_barrier as TB
import egp_meta_model as MM


def test_concurrency_golden():
    # events (0,3) and (2,5) over 6 bars -> overlap at t=2,3
    c = SW.concurrency([(0, 3), (2, 5)], n_bars=6)
    assert c == [1, 1, 2, 2, 1, 1]


def test_average_uniqueness_isolated_event():
    # a single event -> c=1 everywhere -> mean uniqueness = 1.0
    au = SW.average_uniqueness([(0, 4)], n_bars=5)
    assert abs(au[0] - 1.0) < 1e-12


def test_average_uniqueness_full_overlap():
    # two identical events -> c=2 everywhere -> uniqueness 0.5 each
    au = SW.average_uniqueness([(0, 3), (0, 3)], n_bars=4)
    assert all(abs(u - 0.5) < 1e-12 for u in au)


def test_average_uniqueness_partial_overlap_golden():
    # (0,3),(2,5) over 6 bars, c=[1,1,2,2,1,1]
    # u_bar_0 = mean(1,1,0.5,0.5)=0.75 ; u_bar_1 = mean(0.5,0.5,1,1)=0.75
    au = SW.average_uniqueness([(0, 3), (2, 5)], n_bars=6)
    assert abs(au[0] - 0.75) < 1e-12
    assert abs(au[1] - 0.75) < 1e-12


def test_return_attribution_single_event():
    # one event (0,2), c=1, log_returns=[0,0.1,0.2] -> w=|0+0.1+0.2|=0.3 (without normalization)
    w = SW.return_attribution_weights([(0, 2)], [0.0, 0.1, 0.2], n_bars=3, normalize=False)
    assert abs(w[0] - 0.3) < 1e-12


def test_return_attribution_normalized_sums_to_I():
    events = [(0, 2), (1, 4), (3, 5)]
    rets = [0.0, 0.01, -0.02, 0.03, -0.01, 0.02]
    w = SW.return_attribution_weights(events, rets, n_bars=6, normalize=True)
    assert abs(sum(w) - len(events)) < 1e-9      # Sum(w) = I


def test_time_decay_golden_clflastw_half():
    # tW=[0.5,0.5,1.0] -> cumsum=[0.5,1.0,2.0] ; clfLastW=0.5
    # slope=(1-0.5)/2=0.25 ; const=1-0.25*2=0.5 ; decay=[0.625,0.75,1.0]
    d = SW.time_decay([0.5, 0.5, 1.0], clf_last_w=0.5)
    assert abs(d[0] - 0.625) < 1e-12
    assert abs(d[1] - 0.75) < 1e-12
    assert abs(d[2] - 1.0) < 1e-12               # the most recent = 1


def test_time_decay_golden_clflastw_zero():
    # clfLastW=0 -> slope=0.5, const=0 -> decay=[0.25,0.5,1.0]
    d = SW.time_decay([0.5, 0.5, 1.0], clf_last_w=0.0)
    assert abs(d[0] - 0.25) < 1e-12 and abs(d[2] - 1.0) < 1e-12


def test_time_decay_negative_zeros_oldest():
    # clfLastW=-0.5 -> slope=1.0, const=-1.0 -> [-0.5,0,1.0] clip -> [0,0,1.0]
    d = SW.time_decay([0.5, 0.5, 1.0], clf_last_w=-0.5)
    assert d[0] == 0.0 and abs(d[2] - 1.0) < 1e-12


def test_time_decay_unity_no_decay():
    d = SW.time_decay([0.3, 0.4, 0.3], clf_last_w=1.0)
    assert all(abs(x - 1.0) < 1e-12 for x in d)  # clfLastW=1 -> no decay


def test_combined_weights_normalized():
    events = [(0, 3), (2, 5), (4, 7)]
    rets = [0.0, 0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.0]
    w = SW.combined_weights(events, rets, n_bars=8, clf_last_w=0.5)
    assert abs(sum(w) - len(events)) < 1e-9
    assert all(x >= 0 for x in w)


def test_uniqueness_report_effective_size():
    # strong redundancy (all overlapping) -> effective size << n
    events = [(0, 9)] * 5
    rep = SW.uniqueness_report(events, n_bars=10)
    assert rep["n_events"] == 5
    assert abs(rep["mean_avg_uniqueness"] - 0.2) < 1e-9   # c=5 partout -> 1/5
    assert abs(rep["effective_sample_size"] - 1.0) < 1e-9  # 5 obs redondantes ~ 1 independante


def test_integration_weights_feed_meta_model():
    # P3 -> barriers -> AFML ch.4 weights -> weighted secondary model
    prices = [100.0]
    rng = random.Random(0)
    for _ in range(300):
        prices.append(prices[-1] * (1 + rng.gauss(0.0003, 0.01)))
    log_rets = [0.0] + [math.log(prices[i] / prices[i - 1]) for i in range(1, len(prices))]
    sig_idx = list(range(0, 270, 3))             # signaux rapproches -> chevauchements
    sides = [1 if rng.random() > 0.4 else -1 for _ in sig_idx]
    res = TB.triple_barrier_from_signals(prices, sig_idx, sides, span=15,
                                         pt_mult=1.5, sl_mult=1.5, max_horizon=12, cost=0.0005)
    w = SW.weights_from_barriers(res, log_rets, n_bars=len(prices), clf_last_w=0.7)
    assert len(w) == len(res) and abs(sum(w) - len(res)) < 1e-6
    X = [[r["sigma"], float(r["side"]), r["ret"]] for r in res]
    y = [r["meta_label"] for r in res]
    out = MM.evaluate_meta_model(X, y, test_frac=0.3, purge=2, lr=0.3, epochs=200,
                                 sample_weights=w)
    for k in ("precision", "recall", "f1", "accuracy"):
        assert 0.0 <= out["metrics"][k] <= 1.0
