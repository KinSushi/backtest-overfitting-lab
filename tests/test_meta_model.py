"""Offline tests for the secondary meta-labeling model (deterministic/golden)."""
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_meta_model as MM
import egp_triple_barrier as TB


def test_sigmoid_known_values():
    assert abs(MM.sigmoid(0.0) - 0.5) < 1e-12
    assert MM.sigmoid(100) > 0.99 and MM.sigmoid(-100) < 0.01
    assert 0.0 < MM.sigmoid(50) <= 1.0 and 0.0 <= MM.sigmoid(-50) < 1.0  # stabilite numerique


def test_prf1_golden_vector_sklearn():
    # Vecteur golden (LabEx/sklearn) : TP=4, FP=1, FN=1 -> P=R=F1=0.8
    y_true = [0, 1, 1, 0, 1, 0, 0, 1, 0, 1]
    y_pred = [0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
    m = MM.precision_recall_f1(y_true, y_pred)
    assert m["tp"] == 4 and m["fp"] == 1 and m["fn"] == 1 and m["tn"] == 4
    assert abs(m["precision"] - 0.8) < 1e-9
    assert abs(m["recall"] - 0.8) < 1e-9
    assert abs(m["f1"] - 0.8) < 1e-9
    assert abs(m["accuracy"] - 0.8) < 1e-9


def test_prf1_zero_division_safe():
    m = MM.precision_recall_f1([0, 0, 0], [0, 0, 0])   # no positive -> 0, no error
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0


def test_logreg_learns_separable_problem():
    # frontiere lineaire nette : y=1 si x0+x1>0
    rng = random.Random(0)
    X, y = [], []
    for _ in range(400):
        a, b = rng.uniform(-2, 2), rng.uniform(-2, 2)
        X.append([a, b])
        y.append(1 if (a + b) > 0 else 0)
    res = MM.evaluate_meta_model(X, y, test_frac=0.3, lr=0.5, epochs=400)
    assert res["metrics"]["f1"] > 0.85, res["metrics"]
    assert res["metrics"]["accuracy"] > 0.85


def test_logreg_noise_no_spurious_skill():
    # y independent of X -> test F1 close to chance (low), no illusion of edge
    rng = random.Random(1)
    X = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(400)]
    y = [rng.randint(0, 1) for _ in range(400)]
    res = MM.evaluate_meta_model(X, y, test_frac=0.3, lr=0.3, epochs=300)
    assert res["metrics"]["accuracy"] < 0.65   # no real skill


def test_purged_split_disjoint_and_gap():
    tr, te = MM.train_test_split_purged(100, test_frac=0.3, purge=5)
    assert set(tr).isdisjoint(set(te))
    assert max(tr) < min(te)
    # purge: gap of at least 5 between end of train and start of test
    assert min(te) - max(tr) >= 5


def test_l2_shrinks_weights():
    rng = random.Random(2)
    X = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(200)]
    y = [1 if (x[0] + x[1]) > 0 else 0 for x in X]
    m0 = MM.LogisticRegression(lr=0.5, epochs=300, l2=0.0).fit(X, y)
    m1 = MM.LogisticRegression(lr=0.5, epochs=300, l2=2.0).fit(X, y)
    norm0 = sum(w * w for w in m0.w)
    norm1 = sum(w * w for w in m1.w)
    assert norm1 < norm0    # L2 reduces the norm of the weights


def test_sample_weights_influence_fit():
    # two contradictory groups; heavily weighting the positive group must shift the bias
    X = [[0.0]] * 100
    y = [0] * 50 + [1] * 50
    base = MM.LogisticRegression(lr=0.3, epochs=300).fit(X, y)
    sw = [1.0] * 50 + [9.0] * 50          # 9x weight on the positives
    weighted = MM.LogisticRegression(lr=0.3, epochs=300).fit(X, y, sw)
    assert MM.sigmoid(weighted.b) > MM.sigmoid(base.b)   # proba positive accrue


def test_integration_with_triple_barrier_meta_labels():
    # P3 -> meta-labels; we learn to predict them from features (sigma, side, ret)
    prices = [100.0]
    rng = random.Random(3)
    for _ in range(400):
        prices.append(prices[-1] * (1 + rng.gauss(0.0003, 0.01)))
    sig_idx = list(range(0, 360, 4))
    sides = [1 if rng.random() > 0.4 else -1 for _ in sig_idx]
    res = TB.triple_barrier_from_signals(prices, sig_idx, sides, span=15,
                                         pt_mult=1.5, sl_mult=1.5, max_horizon=10, cost=0.0005)
    X = [[r["sigma"], float(r["side"]), r["ret"]] for r in res]
    y = [r["meta_label"] for r in res]
    out = MM.evaluate_meta_model(X, y, test_frac=0.3, purge=2, lr=0.3, epochs=300)
    # the mechanics run and return valid metrics in [0,1]
    for k in ("precision", "recall", "f1", "accuracy"):
        assert 0.0 <= out["metrics"][k] <= 1.0
    assert out["n_train"] > 0 and out["n_test"] > 0
