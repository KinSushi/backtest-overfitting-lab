"""
egp_meta_model.py - Secondary META-LABELING model (Lopez de Prado, AFML ch.3).

The PRIMARY model decides the SIDE (long/short); the triple-barrier (egp_triple_barrier) produces
binary META-LABELS {0,1} = "was the side taken profitable NET OF COSTS?". This module
trains a SECONDARY classifier to predict these meta-labels from context features,
to FILTER the signals: trade only those whose meta probability is sufficient. This improves
precision (fewer false positives) without touching the primary model (AFML ch.3).

Implementation: PURE-PYTHON logistic regression (gradient descent, L2, sample weights),
zero numpy/scipy. Formulas VERIFIED against sources :
  - sigmoid s(z)=1/(1+e^-z); BCE loss = -(1/n) Sum[y*log p + (1-y)*log(1-p)];
    gradient grad_w = (1/n) X^T (p - y), grad_b = (1/n) Sum(p - y) ; maj w <- w - lr*grad
    (aplab.academy, codefrydev, peterroelants, datagran - concordant).
  - precision = TP/(TP+FP); recall = TP/(TP+FN); F1 = 2*TP/(2*TP+FP+FN) (= harmonic mean)
    (scikit-learn : f1_score, precision_recall_fscore_support). F1=0 si TP+FP+FN=0.

Time-ordered TRAIN/TEST split with PURGE (gap) to avoid leakage (AFML ch.7).
CAVEAT: the real FEATURES must come from the market/strategy context; this module provides
the training/evaluation mechanics, tested on synthetic data.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple


def sigmoid(z: float) -> float:
    """Sigmoide numeriquement stable."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def standardize_fit(X: Sequence[Sequence[float]]) -> Tuple[List[float], List[float]]:
    """Computes mean/std per column (std=1 if zero)."""
    n = len(X)
    d = len(X[0]) if n else 0
    means = [sum(row[j] for row in X) / n for j in range(d)]
    stds = []
    for j in range(d):
        var = sum((row[j] - means[j]) ** 2 for row in X) / n
        s = math.sqrt(var)
        stds.append(s if s > 1e-12 else 1.0)
    return means, stds


def standardize_apply(X: Sequence[Sequence[float]], means: List[float],
                      stds: List[float]) -> List[List[float]]:
    return [[(row[j] - means[j]) / stds[j] for j in range(len(means))] for row in X]


class LogisticRegression:
    """Binary logistic regression, batch gradient descent, L2, sample weights."""

    def __init__(self, lr: float = 0.1, epochs: int = 500, l2: float = 0.0):
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.w: List[float] = []
        self.b: float = 0.0

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int],
            sample_weights: Optional[Sequence[float]] = None) -> "LogisticRegression":
        n = len(y)
        d = len(X[0]) if n else 0
        self.w = [0.0] * d
        self.b = 0.0
        sw = list(sample_weights) if sample_weights is not None else [1.0] * n
        W = sum(sw) if sum(sw) > 0 else 1.0
        for _ in range(self.epochs):
            gw = [0.0] * d
            gb = 0.0
            for i in range(n):
                z = self.b + sum(self.w[j] * X[i][j] for j in range(d))
                p = sigmoid(z)
                diff = (p - y[i]) * sw[i]
                for j in range(d):
                    gw[j] += diff * X[i][j]
                gb += diff
            for j in range(d):
                gw[j] = gw[j] / W + self.l2 * self.w[j]  # L2 (not on the bias)
                self.w[j] -= self.lr * gw[j]
            self.b -= self.lr * (gb / W)
        return self

    def predict_proba(self, X: Sequence[Sequence[float]]) -> List[float]:
        return [sigmoid(self.b + sum(self.w[j] * row[j] for j in range(len(self.w)))) for row in X]

    def predict(self, X: Sequence[Sequence[float]], threshold: float = 0.5) -> List[int]:
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]


def confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, int]:
    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yp == 1 and yt == 1:
            tp += 1
        elif yp == 1 and yt == 0:
            fp += 1
        elif yp == 0 and yt == 0:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def precision_recall_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    """Precision=TP/(TP+FP), recall=TP/(TP+FN), F1=2TP/(2TP+FP+FN). 0 if denominator is zero."""
    c = confusion(y_true, y_pred)
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom = 2 * tp + fp + fn
    f1 = (2 * tp / denom) if denom > 0 else 0.0
    acc_den = tp + fp + fn + c["tn"]
    accuracy = (tp + c["tn"]) / acc_den if acc_den > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy, **c}


def train_test_split_purged(n: int, test_frac: float = 0.3,
                            purge: int = 0) -> Tuple[List[int], List[int]]:
    """Time-ORDERED split with purge: train=[0, cut-purge), test=[cut, n)."""
    cut = int(n * (1.0 - test_frac))
    test_idx = list(range(cut, n))
    train_idx = list(range(0, max(0, cut - purge)))
    return train_idx, test_idx


def evaluate_meta_model(X: Sequence[Sequence[float]], y: Sequence[int], test_frac: float = 0.3,
                        purge: int = 0, lr: float = 0.1, epochs: int = 500, l2: float = 0.0,
                        threshold: float = 0.5, standardize: bool = True,
                        sample_weights: Optional[Sequence[float]] = None) -> Dict:
    """Trains on the train (purge), evaluates on the test. Returns metrics + model + scaler."""
    n = len(y)
    tr, te = train_test_split_purged(n, test_frac, purge)
    Xtr = [list(X[i]) for i in tr]
    ytr = [y[i] for i in tr]
    Xte = [list(X[i]) for i in te]
    yte = [y[i] for i in te]
    means = stds = None
    if standardize and Xtr:
        means, stds = standardize_fit(Xtr)
        Xtr = standardize_apply(Xtr, means, stds)
        Xte = standardize_apply(Xte, means, stds)
    swtr = [sample_weights[i] for i in tr] if sample_weights is not None else None
    model = LogisticRegression(lr=lr, epochs=epochs, l2=l2).fit(Xtr, ytr, swtr)
    yhat = model.predict(Xte, threshold)
    metrics = precision_recall_f1(yte, yhat)
    return {"metrics": metrics, "n_train": len(tr), "n_test": len(te),
            "model": model, "scaler": (means, stds),
            "base_rate_test": (sum(yte) / len(yte)) if yte else 0.0}
