"""Offline tests for CART tree + random forest (AFML ch.6/8) : golden + verifiable properties."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_tree_forest as TF


def test_gini_golden():
    assert abs(TF.gini_from_counts([2, 2]) - 0.5) < 1e-12          # 1-(0.25+0.25)
    assert TF.gini_from_counts([4, 0]) == 0.0                       # pur
    assert abs(TF.gini_from_counts([1, 1, 1, 1]) - 0.75) < 1e-12    # 4 classes equilibrees
    assert abs(TF.gini([0, 0, 1, 1]) - 0.5) < 1e-12


def test_tree_learns_separable():
    # clean rule : x0 <= 0.5 -> class 0, else 1
    X = [[v / 100.0, random.Random(v).random()] for v in range(100)]
    y = [0 if x[0] <= 0.5 else 1 for x in X]
    t = TF.DecisionTreeClassifier(max_depth=3, min_samples_leaf=1).fit(X, y)
    preds = t.predict(X)
    acc = sum(1 for p, yy in zip(preds, y) if p == yy) / len(y)
    assert acc == 1.0                                              # separable -> 100%


def test_tree_proba_sums_to_one():
    X = [[v / 50.0] for v in range(50)]
    y = [0 if v < 25 else 1 for v in range(50)]
    t = TF.DecisionTreeClassifier(max_depth=2).fit(X, y)
    for p in t.predict_proba(X):
        assert abs(sum(p) - 1.0) < 1e-12


def test_tree_max_depth_respected():
    rng = random.Random(0)
    X = [[rng.random(), rng.random(), rng.random()] for _ in range(200)]
    y = [rng.randint(0, 1) for _ in range(200)]
    t = TF.DecisionTreeClassifier(max_depth=2).fit(X, y)

    def depth(node):
        if node["leaf"]:
            return 0
        return 1 + max(depth(node["left"]), depth(node["right"]))

    assert depth(t.tree_) <= 2


def test_mdi_identifies_informative_feature():
    # feature 0 determines the label ; features 1,2 = noise -> MDI(0) dominates
    rng = random.Random(1)
    X = [[rng.random(), rng.random(), rng.random()] for _ in range(300)]
    y = [1 if x[0] > 0.5 else 0 for x in X]
    t = TF.DecisionTreeClassifier(max_depth=4).fit(X, y)
    imp = t.mdi()
    assert imp[0] > imp[1] and imp[0] > imp[2]                     # variable utile domine


def test_forest_predict_proba_averaged():
    rng = random.Random(2)
    X = [[rng.random(), rng.random()] for _ in range(200)]
    y = [1 if x[0] + x[1] > 1.0 else 0 for x in X]
    rf = TF.RandomForestClassifier(n_estimators=10, max_depth=4, bootstrap="uniform", seed=0).fit(X, y)
    for p in rf.predict_proba(X):
        assert abs(sum(p) - 1.0) < 1e-12 and 0.0 <= p[1] <= 1.0


def test_forest_oob_close_to_test_accuracy():
    # Breiman : l'erreur OOB approxime l'erreur de test
    rng = random.Random(3)
    X = [[rng.random(), rng.random()] for _ in range(400)]
    y = [1 if x[0] + x[1] > 1.0 else 0 for x in X]
    Xtr, ytr = X[:300], y[:300]
    Xte, yte = X[300:], y[300:]
    rf = TF.RandomForestClassifier(n_estimators=25, max_depth=5, bootstrap="uniform", seed=0).fit(Xtr, ytr)
    oob = rf.oob_score(Xtr, ytr)["oob_accuracy"]
    preds = rf.predict(Xte)
    test_acc = sum(1 for p, yy in zip(preds, yte) if p == yy) / len(yte)
    assert oob > 0.8  # learns the rule
    assert abs(oob - test_acc) < 0.12                              # OOB ~ test


def test_forest_sequential_bootstrap_runs():
    rng = random.Random(4)
    n = 120
    X = [[rng.random(), rng.random()] for _ in range(n)]
    y = [1 if x[0] > 0.5 else 0 for x in X]
    events = [(i, i + 3) for i in range(n)]
    rf = TF.RandomForestClassifier(n_estimators=8, max_depth=4, bootstrap="sequential", seed=0)
    rf.fit(X, y, events=events, n_bars=n + 5)
    acc = sum(1 for p, yy in zip(rf.predict(X), y) if p == yy) / n
    assert acc > 0.85  # learns via sequential draw


def test_forest_feature_importances_normalized():
    rng = random.Random(5)
    X = [[rng.random(), rng.random(), rng.random()] for _ in range(300)]
    y = [1 if x[1] > 0.5 else 0 for x in X]                        # feature 1 informative
    rf = TF.RandomForestClassifier(n_estimators=15, max_depth=4, bootstrap="uniform", seed=0).fit(X, y)
    fi = rf.feature_importances()
    assert abs(sum(fi) - 1.0) < 1e-9                               # normalise a 1
    assert fi[1] == max(fi)                                        # variable utile domine


def test_sample_weights_influence_tree():
    # zero weights on one half -> the tree learns only the other half
    X = [[0.0], [0.0], [1.0], [1.0]]
    y = [0, 1, 0, 1]
    w = [1.0, 0.0, 0.0, 1.0]                                       # keeps only (0->0) and (1->1)
    t = TF.DecisionTreeClassifier(max_depth=2).fit(X, y, sample_weight=w)
    assert t.predict([[0.0]])[0] == 0 and t.predict([[1.0]])[0] == 1
