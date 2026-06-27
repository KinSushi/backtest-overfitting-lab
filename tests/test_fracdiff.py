"""Offline tests for fractional differentiation FFD (AFML ch.5), golden vectors."""
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_fracdiff as FD


def test_weights_d0_identity():
    # d=0 -> [1,0,0,...] (no differentiation)
    w = FD.frac_weights(0.0, 5)
    assert abs(w[0] - 1.0) < 1e-12
    assert all(abs(x) < 1e-12 for x in w[1:])


def test_weights_d1_first_difference():
    # d=1 -> [1,-1,0,0,...] (difference premiere)
    w = FD.frac_weights(1.0, 5)
    assert abs(w[0] - 1.0) < 1e-12 and abs(w[1] + 1.0) < 1e-12
    assert all(abs(x) < 1e-12 for x in w[2:])


def test_weights_d_half_golden():
    # d=0.5 -> [1, -0.5, -0.125, -0.0625] (computed by hand via the recursion)
    w = FD.frac_weights(0.5, 4)
    assert abs(w[0] - 1.0) < 1e-12
    assert abs(w[1] - (-0.5)) < 1e-12
    assert abs(w[2] - (-0.125)) < 1e-12
    assert abs(w[3] - (-0.0625)) < 1e-12


def test_ffd_weights_truncation():
    # d=0.5, thresh=0.1 -> on garde |w|>=0.1 : [1, -0.5, -0.125] ; |w_3|=0.0625<0.1 -> stop
    w = FD.frac_weights_ffd(0.5, thresh=0.1)
    assert len(w) == 3
    assert abs(w[-1] - (-0.125)) < 1e-12
    assert FD.ffd_width(0.5, 0.1) == 3


def test_ffd_d1_is_first_difference():
    # d=1 with fine threshold -> first difference: X~_t = X_t - X_{t-1}
    series = [1.0, 2.0, 4.0, 7.0, 11.0]
    res = FD.frac_diff_ffd(series, d=1.0, thresh=1e-6)
    assert res["width"] == 2 and res["offset"] == 1
    # diffs attendues : 1,2,3,4
    assert all(abs(a - b) < 1e-9 for a, b in zip(res["values"], [1.0, 2.0, 3.0, 4.0]))


def test_ffd_d0_returns_series():
    # d=0 -> weights [1] (the only >= threshold) -> series unchanged
    series = [3.0, 1.0, 4.0, 1.0, 5.0]
    res = FD.frac_diff_ffd(series, d=0.0, thresh=1e-6)
    assert res["width"] == 1 and res["offset"] == 0
    assert all(abs(a - b) < 1e-12 for a, b in zip(res["values"], series))


def test_memory_retention_d0_is_one():
    series = [math.sin(i / 5.0) + i * 0.1 for i in range(100)]
    assert abs(FD.memory_retention(series, d=0.0, thresh=1e-6) - 1.0) < 1e-9


def test_memory_decreases_with_d():
    # the more d increases, the more the correlation with the original (memory) drops
    rng = random.Random(0)
    series = [0.0]
    for _ in range(500):
        series.append(series[-1] + rng.gauss(0, 1))  # random walk (non-stationary)
    corr_low = FD.memory_retention(series, d=0.2, thresh=1e-4)
    corr_high = FD.memory_retention(series, d=0.9, thresh=1e-4)
    assert corr_low > corr_high                         # small d keeps more memory


def test_ffd_width_grows_as_d_decreases():
    # the smaller d is, the wider the FFD window (slower decay of the weights)
    w_small_d = FD.ffd_width(0.2, thresh=1e-4)
    w_large_d = FD.ffd_width(0.8, thresh=1e-4)
    assert w_small_d > w_large_d


def test_fracdiff_feature_alignment():
    prices = [100.0 + i for i in range(50)]
    res = FD.frac_diff_ffd(prices, d=0.5, thresh=1e-3)
    feat = FD.fracdiff_feature(prices, event_indices=[0, 10, 25, 49], d=0.5, thresh=1e-3)
    assert len(feat) == 4
    # event 0 is before the 1st FFD value (offset>0) -> 0.0
    assert feat[0] == 0.0
    # event 49 corresponds to the last FFD value
    assert abs(feat[-1] - res["values"][49 - res["offset"]]) < 1e-9
