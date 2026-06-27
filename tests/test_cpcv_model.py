"""Offline tests for the model-level CPCV backtest (multiple paths -> deflated gate)."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_cpcv_model as CM
import egp_cpcv as CP


def _data(n=240, seed=0):
    rng = random.Random(seed)
    X = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(n)]
    y = [1 if X[i][0] > 0 else 0 for i in range(n)]
    bets = [{"prob": 0.0, "side_return": 0.01 if X[i][0] > 0 else -0.01} for i in range(n)]
    events = [(i, i + 2) for i in range(n)]
    return X, y, bets, events


def test_paths_count_and_coverage():
    X, y, bets, events = _data(n=240)
    paths = CM.cpcv_path_returns(X, y, bets, events, n_groups=4, k_test=2, seed=0)
    assert len(paths) == CP.n_paths(4, 2)            # 3 paths
    assert all(len(p) == len(X) for p in paths)      # each path covers ALL observations


def test_profitable_model_positive_paths():
    # learnable model + winning bets in the right direction -> mean path return > 0
    X, y, bets, events = _data(n=300, seed=1)
    out = CM.cpcv_model_gate(X, y, bets, events, n_groups=5, k_test=2, n_trials=20, seed=0)
    assert out["mean_path_return"] > 0
    assert out["gate"]["frac_positive_paths"] >= 0.5


def test_gate_structure_present():
    X, y, bets, events = _data(n=240, seed=2)
    out = CM.cpcv_model_gate(X, y, bets, events, n_groups=4, k_test=2, n_trials=10, seed=0)
    for k in ("n_paths", "per_path_sharpe", "median_oos_sharpe", "decision"):
        assert k in out["gate"]
    assert out["gate"]["n_paths"] == 3


def test_deterministic_with_seed():
    X, y, bets, events = _data(n=240, seed=3)
    a = CM.cpcv_path_returns(X, y, bets, events, n_groups=4, k_test=2, seed=7)
    b = CM.cpcv_path_returns(X, y, bets, events, n_groups=4, k_test=2, seed=7)
    assert a == b


def test_purge_embargo_run():
    X, y, bets, events = _data(n=300, seed=4)
    out = CM.cpcv_model_gate(X, y, bets, events, n_groups=5, k_test=2, purge=2, embargo=1,
                             n_trials=15, seed=0)
    assert out["n_paths"] == CP.n_paths(5, 2)
