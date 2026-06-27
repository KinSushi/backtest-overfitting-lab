"""Offline tests for egp_regime (regime/session diagnostic), deterministic."""
import math
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_regime as R


def test_classify_session_boundaries():
    assert R.classify_session(3) == "ASIA"
    assert R.classify_session(8) == "LONDON"
    assert R.classify_session(12) == "LONDON"
    assert R.classify_session(13) == "OVERLAP"
    assert R.classify_session(16) == "OVERLAP"
    assert R.classify_session(17) == "NEW_YORK"
    assert R.classify_session(21) == "NEW_YORK"
    assert R.classify_session(22) == "SYDNEY"
    # datetime UTC
    assert R.classify_session(datetime(2024, 1, 8, 14, 30)) == "OVERLAP"


def test_segment_stats_known_values():
    pnls = [10.0, -5.0, 20.0, -5.0]      # net=20, 2 gains/2 pertes
    st = R.segment_stats(pnls)
    assert st["count"] == 4
    assert abs(st["net"] - 20.0) < 1e-9
    assert abs(st["win_rate"] - 0.5) < 1e-9
    # profit factor = (10+20)/(5+5) = 3.0
    assert abs(st["profit_factor"] - 3.0) < 1e-9


def test_segment_stats_no_losses_infinite_pf():
    st = R.segment_stats([1.0, 2.0, 3.0])
    assert st["profit_factor"] == float("inf")
    assert st["win_rate"] == 1.0


def test_performance_by_session_grouping():
    trades = [
        {"time": datetime(2024, 1, 8, 3, 0), "pnl": 5.0},    # ASIA
        {"time": datetime(2024, 1, 8, 3, 30), "pnl": -2.0},  # ASIA
        {"time": datetime(2024, 1, 8, 14, 0), "pnl": 30.0},  # OVERLAP
        {"time": datetime(2024, 1, 8, 19, 0), "pnl": -1.0},  # NEW_YORK
    ]
    by = R.performance_by_session(trades)
    assert by["ASIA"]["count"] == 2 and abs(by["ASIA"]["net"] - 3.0) < 1e-9
    assert by["OVERLAP"]["count"] == 1 and by["OVERLAP"]["net"] == 30.0
    assert "NEW_YORK" in by and by["NEW_YORK"]["net"] == -1.0


def test_volatility_regimes_quantile_assignment():
    vol = [0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0]
    out = R.volatility_regimes(vol, quantiles=(1/3, 2/3))
    labels = out["labels"]
    assert labels[0] == "LOW" and labels[-1] == "HIGH"
    # the highest vol is HIGH, the lowest LOW
    assert labels[vol.index(2.0)] == "HIGH"
    assert labels[vol.index(0.1)] == "LOW"


def test_performance_by_vol_regime_edge_in_high():
    # edge concentrated in the HIGH regime; 3 vol levels to populate LOW/MID/HIGH
    vol = [0.1]*7 + [0.5]*7 + [2.0]*6
    pnls = [0.0]*7 + [0.0]*7 + [5.0]*6
    by = R.performance_by_vol_regime(pnls, vol, quantiles=(1/3, 2/3))
    assert by["HIGH"]["net"] > 0
    assert by["LOW"]["net"] == 0.0


def test_edge_concentration_flags_fragility():
    # all the profit in a single segment -> fragile
    by = {"OVERLAP": {"net": 100.0}, "ASIA": {"net": -5.0}, "NEW_YORK": {"net": -3.0}}
    ec = R.edge_concentration(by)
    assert ec["fragile"] is True
    assert ec["best_segment"] == "OVERLAP"
    assert ec["n_profitable"] == 1

    # profit reparti -> non fragile
    by2 = {"OVERLAP": {"net": 30.0}, "LONDON": {"net": 25.0}, "NEW_YORK": {"net": 20.0}}
    ec2 = R.edge_concentration(by2)
    assert ec2["fragile"] is False
    assert ec2["n_profitable"] == 3
    assert ec2["best_share"] < 0.6


def test_regime_change_points_detects_shift():
    # series with mean 0 then +1: a break around index 30
    series = [0.0]*30 + [1.0]*30
    pts = R.regime_change_points(series, mu0=0.0, k=0.3, h=4.0, min_gap=5)
    assert len(pts) >= 1
    assert 28 <= pts[0] <= 45  # alarm shortly after the break (CUSUM delay)


def test_rolling_volatility_responds_to_regime():
    # increasing amplitude (alternating) -> the rolling vol must rise
    rets = [0.001 * (1 if i % 2 == 0 else -1) for i in range(30)] + \
           [0.05 * (1 if i % 2 == 0 else -1) for i in range(30)]
    vol = R.rolling_volatility(rets, window=10)
    assert vol[5] < vol[-1]         # the rolling vol rises in the second regime
    assert all(v >= 0 for v in vol)
