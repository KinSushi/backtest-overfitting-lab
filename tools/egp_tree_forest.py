"""
egp_tree_forest.py - CART tree + Random forest, pure-Python (AFML ch.6 bagging, ch.8 MDI).

Why a tree rather than bagging logistic regression: the sources (Breiman 1996;
Penn State) show that bagging reduces the variance of UNSTABLE learners (trees) and little
that of stable learners (regression). The random forest = tree bagging + sub-sampling
of features, all on SEQUENTIAL draws (egp_seq_bootstrap) to decorrelate further. Bonus:
the tree provides the MDI (Mean Decrease Impurity) set aside in ch.8 (which required trees).

Formulas VERIFIED against concordant sources:
  - Gini : I = 1 - sum_j p_j^2  (numberanalytics ; scientistcafe ; quantinsti ; Hanane D).
  - CART split criterion (minimize the weighted impurity of the children):
        G = p_L*(1 - sum p_Lj^2) + p_R*(1 - sum p_Rj^2)   (arxiv CART-ELC 2505.05402).
  - Weighted impurity decrease (scikit-learn, verified):
        N_t/N * (I_parent - N_tR/N_t * I_right - N_tL/N_t * I_left).
  - MDI / Mean Decrease Gini (Breiman 2002 ; arxiv 2507.07477, 1911.11901) :
        importance(node) = w_node*G_node - w_L*G_L - w_R*G_R, w = share of samples reaching
        the node; MDG_j = sum over the nodes where j splits (then mean over the trees, normalized).
  - Bagging: aggregation = MEAN of probabilities (classif); OOB = predict each obs with only the
        trees that did NOT draw it (Breiman, "Out-of-bag estimation").

Reference: M. Lopez de Prado, *Advances in Financial Machine Learning* (2018), ch.6 & 8;
Breiman (1996, 2001, 2002). Pure-Python, supports sample weights (bridge to egp_sample_weights).
"""
from __future__ import annotations

import math
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)


# --------------------------------------------------------------------------- gini
def gini_from_counts(counts: Sequence[float]) -> float:
    """Gini impurity from per-class weights/counts: 1 - sum (w_c/W)^2."""
    W = sum(counts)
    if W <= 0:
        return 0.0
    return 1.0 - sum((c / W) ** 2 for c in counts)


def gini(labels: Sequence, weights: Optional[Sequence[float]] = None,
         classes: Optional[Sequence] = None) -> float:
    """Gini impurity of a set of labels (with optional weights)."""
    if classes is None:
        classes = sorted(set(labels))
    idx = {c: i for i, c in enumerate(classes)}
    counts = [0.0] * len(classes)
    if weights is None:
        weights = [1.0] * len(labels)
    for lab, w in zip(labels, weights):
        counts[idx[lab]] += w
    return gini_from_counts(counts)


# --------------------------------------------------------------------------- CART tree
class DecisionTreeClassifier:
    """CART classification tree (Gini impurity, greedy binary splits)."""

    def __init__(self, max_depth: int = 6, min_samples_leaf: int = 1,
                 max_features: Optional[int] = None, seed: int = 0):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.rng = random.Random(seed)
        self.tree_: Optional[dict] = None
        self.classes_: List = []
        self.n_features_: int = 0

    def fit(self, X: Sequence[Sequence[float]], y: Sequence,
            sample_weight: Optional[Sequence[float]] = None) -> "DecisionTreeClassifier":
        n = len(X)
        self.n_features_ = len(X[0]) if n else 0
        self.classes_ = sorted(set(y))
        w = list(sample_weight) if sample_weight is not None else [1.0] * n
        idx = list(range(n))
        self._root_weight = sum(w)
        self.tree_ = self._build(X, y, w, idx, depth=0)
        return self

    def _class_counts(self, y, w, idx) -> List[float]:
        cidx = {c: i for i, c in enumerate(self.classes_)}
        counts = [0.0] * len(self.classes_)
        for i in idx:
            counts[cidx[y[i]]] += w[i]
        return counts

    def _build(self, X, y, w, idx, depth) -> dict:
        counts = self._class_counts(y, w, idx)
        W = sum(counts)
        node_gini = gini_from_counts(counts)
        proba = [c / W if W > 0 else 0.0 for c in counts]
        node = {"leaf": True, "proba": proba, "w": W, "gini": node_gini,
                "n": len(idx), "feature": None, "threshold": None}
        # criteres d'arret
        if depth >= self.max_depth or node_gini == 0.0 or len(idx) < 2 * self.min_samples_leaf:
            return node
        best = self._best_split(X, y, w, idx, W, node_gini)
        if best is None:
            return node
        feat, thr, left_idx, right_idx = best
        node.update({"leaf": False, "feature": feat, "threshold": thr,
                     "left": self._build(X, y, w, left_idx, depth + 1),
                     "right": self._build(X, y, w, right_idx, depth + 1)})
        return node

    def _best_split(self, X, y, w, idx, W, parent_gini):
        feats = list(range(self.n_features_))
        if self.max_features is not None and self.max_features < self.n_features_:
            feats = self.rng.sample(feats, self.max_features)
        best_gain = 0.0
        best = None
        for f in feats:
            vals = sorted(set(X[i][f] for i in idx))
            for a, b in zip(vals, vals[1:]):
                thr = 0.5 * (a + b)
                left_idx = [i for i in idx if X[i][f] <= thr]
                right_idx = [i for i in idx if X[i][f] > thr]
                if len(left_idx) < self.min_samples_leaf or len(right_idx) < self.min_samples_leaf:
                    continue
                gL = gini_from_counts(self._class_counts(y, w, left_idx))
                gR = gini_from_counts(self._class_counts(y, w, right_idx))
                WL = sum(w[i] for i in left_idx)
                WR = sum(w[i] for i in right_idx)
                child = (WL * gL + WR * gR) / W                 # weighted impurity of the children
                gain = parent_gini - child                      # impurity decrease
                if gain > best_gain + 1e-15:
                    best_gain = gain
                    best = (f, thr, left_idx, right_idx)
        return best

    def _leaf_proba(self, x, node):
        while not node["leaf"]:
            node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        return node["proba"]

    def predict_proba(self, X) -> List[List[float]]:
        return [self._leaf_proba(x, self.tree_) for x in X]

    def predict(self, X, threshold: float = 0.5) -> List:
        out = []
        for p in self.predict_proba(X):
            if len(self.classes_) == 2:
                out.append(self.classes_[1] if p[1] >= threshold else self.classes_[0])
            else:
                out.append(self.classes_[max(range(len(p)), key=lambda i: p[i])])
        return out

    def mdi(self) -> List[float]:
        """Raw MDI per variable (sum of weighted Gini decreases at the splitting nodes
        on the variable). w = share of samples reaching the node (node weight / root weight)."""
        imp = [0.0] * self.n_features_
        root_w = self._root_weight if self._root_weight > 0 else 1.0

        def rec(node):
            if node["leaf"]:
                return
            wn = node["w"] / root_w
            wl = node["left"]["w"] / root_w
            wr = node["right"]["w"] / root_w
            imp[node["feature"]] += wn * node["gini"] - wl * node["left"]["gini"] - wr * node["right"]["gini"]
            rec(node["left"])
            rec(node["right"])

        rec(self.tree_)
        return imp


# --------------------------------------------------------------------------- forest
class RandomForestClassifier:
    """Random forest: bagging of CART trees (sequential or uniform draw) + OOB + mean MDI."""

    def __init__(self, n_estimators: int = 25, max_depth: int = 6, min_samples_leaf: int = 1,
                 max_features: Optional[int] = None, bootstrap: str = "uniform", seed: int = 0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.seed = seed
        self.trees_: List[DecisionTreeClassifier] = []
        self.in_bag_: List[set] = []
        self.classes_: List = []
        self.n_features_: int = 0

    def fit(self, X, y, events: Optional[Sequence[Tuple[int, int]]] = None,
            n_bars: Optional[int] = None, sample_weight: Optional[Sequence[float]] = None
            ) -> "RandomForestClassifier":
        n = len(X)
        self.n_features_ = len(X[0]) if n else 0
        self.classes_ = sorted(set(y))
        if self.bootstrap == "sequential" and (events is None or n_bars is None):
            raise ValueError("bootstrap='sequential' requiert events + n_bars")
        for b in range(self.n_estimators):
            if self.bootstrap == "sequential":
                import egp_seq_bootstrap as SB
                draw = SB.seq_bootstrap(events, n_bars, s_length=n, seed=self.seed + b)
            elif self.bootstrap == "uniform":
                rng = random.Random(self.seed + b)
                draw = [rng.randrange(n) for _ in range(n)]
            else:  # 'none': no draw (for tests)
                draw = list(range(n))
            Xb = [X[i] for i in draw]
            yb = [y[i] for i in draw]
            wb = [sample_weight[i] for i in draw] if sample_weight is not None else None
            tree = DecisionTreeClassifier(self.max_depth, self.min_samples_leaf,
                                          self.max_features, seed=self.seed + b)
            tree.fit(Xb, yb, wb)
            self.trees_.append(tree)
            self.in_bag_.append(set(draw))
        return self

    def predict_proba(self, X) -> List[List[float]]:
        acc = [[0.0] * len(self.classes_) for _ in X]
        for tree in self.trees_:
            for r, p in enumerate(tree.predict_proba(X)):
                for c in range(len(self.classes_)):
                    acc[r][c] += p[c]
        B = len(self.trees_)
        return [[v / B for v in row] for row in acc]

    def predict(self, X, threshold: float = 0.5) -> List:
        out = []
        for p in self.predict_proba(X):
            if len(self.classes_) == 2:
                out.append(self.classes_[1] if p[1] >= threshold else self.classes_[0])
            else:
                out.append(self.classes_[max(range(len(p)), key=lambda i: p[i])])
        return out

    def oob_score(self, X, y, threshold: float = 0.5) -> Dict:
        """OOB error (Breiman): each obs predicted only by the trees that did not draw it."""
        n = len(X)
        correct = 0
        evaluated = 0
        for i in range(n):
            acc = [0.0] * len(self.classes_)
            cnt = 0
            for b, tree in enumerate(self.trees_):
                if i not in self.in_bag_[b]:
                    p = tree.predict_proba([X[i]])[0]
                    for c in range(len(self.classes_)):
                        acc[c] += p[c]
                    cnt += 1
            if cnt == 0:
                continue
            evaluated += 1
            if len(self.classes_) == 2:
                pred = self.classes_[1] if (acc[1] / cnt) >= threshold else self.classes_[0]
            else:
                pred = self.classes_[max(range(len(acc)), key=lambda c: acc[c])]
            if pred == y[i]:
                correct += 1
        return {"oob_accuracy": correct / evaluated if evaluated else float("nan"),
                "n_evaluated": evaluated}

    def feature_importances(self, normalize: bool = True) -> List[float]:
        """Mean MDI over the trees (normalized to sum 1 by default, like scikit-learn)."""
        agg = [0.0] * self.n_features_
        for tree in self.trees_:
            for f, v in enumerate(tree.mdi()):
                agg[f] += v
        agg = [v / len(self.trees_) for v in agg]
        if normalize:
            s = sum(agg)
            if s > 0:
                agg = [v / s for v in agg]
        return agg
