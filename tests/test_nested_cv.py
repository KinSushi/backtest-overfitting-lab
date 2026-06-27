"""Offline tests for purged nested CV (orchestration of verified modules, AFML ch.7)."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_nested_cv as NCV


def _learnable(n=200, seed=0):
    rng = random.Random(seed)
    X = [[rng.random(), rng.random(), rng.random()] for _ in range(n)]
    y = [1 if x[0] + x[1] > 1.0 else 0 for x in X]
    events = [(i, i + 2) for i in range(n)]
    return X, y, events


def test_grid_search_returns_best_and_table():
    X, y, ev = _learnable()
    grid = {"lr": [0.1, 0.5], "l2": [0.0, 0.01], "epochs": [200]}
    out = NCV.grid_search_cv(X, y, ev, grid, n_splits=4, n_bars=len(X) + 3)
    assert len(out["table"]) == 4                       # 2*2*1 combinaisons
    assert out["best_params"] in [c for c, _, _ in out["table"]]
    assert 0.0 <= out["best_score"] <= 1.0


def test_nested_cv_runs_and_unbiased_estimate():
    X, y, ev = _learnable(n=240, seed=1)
    grid = {"lr": [0.1, 0.5], "l2": [0.0, 0.01], "epochs": [200]}
    out = NCV.nested_cv(X, y, ev, grid, outer_splits=4, inner_splits=3, n_bars=len(X) + 3)
    assert out["n_outer_folds"] == 4
    assert len(out["chosen_params_per_fold"]) == 4
    assert out["mean_outer_score"] > 0.8               # probleme apprenable


def test_nested_cv_deterministic():
    X, y, ev = _learnable(n=200, seed=2)
    grid = {"lr": [0.3], "l2": [0.0], "epochs": [150]}
    a = NCV.nested_cv(X, y, ev, grid, outer_splits=3, inner_splits=2, n_bars=len(X) + 3)
    b = NCV.nested_cv(X, y, ev, grid, outer_splits=3, inner_splits=2, n_bars=len(X) + 3)
    assert a["outer_scores"] == b["outer_scores"]  # deterministic


def test_param_combos_cartesian():
    combos = NCV._param_combos({"a": [1, 2], "b": [3, 4, 5]})
    assert len(combos) == 6                             # 2*3
    assert {"a": 1, "b": 3} in combos and {"a": 2, "b": 5} in combos


def test_grid_search_picks_better_config():
    # a degenerate config (epochs=1) must get a score <= a trained config
    X, y, ev = _learnable(n=200, seed=3)
    out = NCV.grid_search_cv(X, y, ev, {"lr": [0.5], "l2": [0.0], "epochs": [1, 300]},
                             n_splits=4, n_bars=len(X) + 3)
    scores = {c["epochs"]: s for c, s, _ in out["table"]}
    assert scores[300] >= scores[1]
