"""Offline tests for the purged K-fold (ch.7) + MDA/SFI importance (ch.8), deterministic."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_cv_importance as CI


def test_purged_kfold_partitions_test():
    events = [(i, i) for i in range(20)]            # non chevauchants
    folds = CI.purged_kfold(events, n_splits=4, embargo_pct=0.0)
    assert len(folds) == 4
    all_test = sorted(p for f in folds for p in f["test_idx"])
    assert all_test == list(range(20))  # the tests partition the observations
    for f in folds:
        assert set(f["train_idx"]).isdisjoint(set(f["test_idx"]))


def test_purge_removes_overlapping_train():
    # observation 0 has a long label (0..10) overlapping the test block -> purged
    events = [(0, 10)] + [(i, i) for i in range(1, 12)]
    folds = CI.purged_kfold(events, n_splits=3, embargo_pct=0.0)
    # find a fold whose test window covers [0,10] and check that obs 0 (if outside test)
    # is not in the train
    for f in folds:
        if 0 not in f["test_idx"]:
            tmin = min(events[p][0] for p in f["test_idx"])
            tmax = max(events[p][1] for p in f["test_idx"])
            overlaps = not (10 < tmin or 0 > tmax)
            if overlaps:
                assert 0 not in f["train_idx"]  # purged because overlapping


def test_embargo_removes_right_neighbors():
    events = [(i, i) for i in range(30)]
    f0 = CI.purged_kfold(events, n_splits=3, embargo_pct=0.0)
    f1 = CI.purged_kfold(events, n_splits=3, embargo_pct=0.2)   # embargo = 6 barres
    # with embargo, the 1st fold's train has FEWER observations (right buffer removed)
    assert len(f1[0]["train_idx"]) < len(f0[0]["train_idx"])


def test_mda_detects_predictive_vs_noise():
    # y depends on x0 only ; x1 = noise -> MDA(x0) >> MDA(x1)
    rng = random.Random(0)
    X, y = [], []
    for _ in range(400):
        x0 = rng.uniform(-2, 2)
        x1 = rng.gauss(0, 1)
        X.append([x0, x1])
        y.append(1 if x0 > 0 else 0)
    events = [(i, i) for i in range(len(X))]
    imp = CI.mda_importance(X, y, events, n_splits=4, seed=1)
    assert imp[0]["mean"] > imp[1]["mean"]
    assert imp[0]["mean"] > 0.1                     # permuter x0 detruit l'accuracy
    assert abs(imp[1]["mean"]) < 0.1  # x1 has no effect


def test_sfi_ranks_predictive_feature_first():
    rng = random.Random(1)
    X, y = [], []
    for _ in range(400):
        x0 = rng.uniform(-2, 2)
        x1 = rng.gauss(0, 1)
        X.append([x0, x1])
        y.append(1 if x0 > 0 else 0)
    events = [(i, i) for i in range(len(X))]
    sfi = CI.sfi_importance(X, y, events, n_splits=4)
    assert sfi[0] > sfi[1]  # x0 alone predicts better than x1 alone
    assert sfi[0] > 0.8                              # x0 seul ~ parfait


def test_cv_score_in_range():
    rng = random.Random(2)
    X = [[rng.gauss(0, 1)] for _ in range(200)]
    y = [1 if x[0] > 0 else 0 for x in X]
    events = [(i, i) for i in range(len(X))]
    res = CI.cv_score(X, y, events, n_splits=4)
    assert 0.0 <= res["mean"] <= 1.0 and res["mean"] > 0.8   # feature parfaite


def test_rank_features_orders_by_mean():
    imp = {0: {"mean": 0.05, "std": 0.0}, 1: {"mean": 0.30, "std": 0.0},
           2: {"mean": 0.10, "std": 0.0}}
    ranked = CI.rank_features(imp)
    assert [j for j, _ in ranked] == [1, 2, 0]
