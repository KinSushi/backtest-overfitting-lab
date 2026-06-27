"""
egp_cv_importance.py - Purged K-Fold CV (AFML ch.7) + MDA/SFI feature importance (ch.8).

PROBLEM (ch.7): standard K-Fold CV LEAKS in finance because labels overlap in time:
a training observation whose label covers the test period shares
information with the test -> overestimation. Solution: PURGED K-FOLD = remove from train any
observation whose label [t0,t1] OVERLAPS the test time window (purge), plus an
EMBARGO (buffer) after the test (leakage via delayed reaction).

IMPORTANCE (ch.8):
  - MDA (Mean Decrease Accuracy): model-agnostic, OUT-OF-SAMPLE. Train, measure
    the test score, then for EACH feature PERMUTE its column in the test and re-measure:
    the score DROP = importance. (AFML snippet 8.3.)
  - SFI (Single Feature Importance): out-of-sample importance of a SINGLE feature alone (CV with one
    column only). Ignores joint effects. (AFML snippet 8.4.)
  - MDI (impurity): TREE-SPECIFIC, in-sample -> NOT applicable to the secondary model's
    logistic regression -> out of scope here (would require a forest).

Definitions VERIFIED against sources: Wikipedia "Purged cross-validation"; quantinsti; mlfinlab;
reasonabledeviations (AFML notes); AFML ch.7-8 summary (PhilPapers). Convention: an event =
(t0, t1) INCLUSIVE indices (t0 = signal, t1 = barrier touched), observations ordered by t0.
Pure-Python. Source: M. Lopez de Prado, *Advances in Financial Machine Learning* (2018), ch.7-8.
"""
from __future__ import annotations

import os
import random
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
import egp_meta_model as MM

Event = Tuple[int, int]
FitFn = Callable[[List[List[float]], List[int]], Callable[[List[List[float]]], List[int]]]
ScoreFn = Callable[[Callable, List[List[float]], List[int]], float]


def purged_kfold(events: Sequence[Event], n_splits: int = 5, embargo_pct: float = 0.0,
                 n_bars: Optional[int] = None) -> List[Dict[str, List[int]]]:
    """Purged K-Fold (AFML 7.4). Observations (order = position in 'events') are split
    into n_splits contiguous test blocks. For each block: PURGE from train any observation whose
    label overlaps the test time window, and apply an EMBARGO (in bars) on the
    right. Returns [{train_idx, test_idx}, ...]."""
    N = len(events)
    if n_bars is None:
        n_bars = max((t1 for (_, t1) in events), default=0) + 1
    embargo_bars = int(n_bars * embargo_pct)
    folds = []
    # test-block bounds (observation positions)
    bounds = [(i * N // n_splits, (i + 1) * N // n_splits) for i in range(n_splits)]
    for (a, b) in bounds:
        if a >= b:
            continue
        test_idx = list(range(a, b))
        test_t0_min = min(events[p][0] for p in test_idx)
        test_t1_max = max(events[p][1] for p in test_idx)
        train_idx = []
        for p in range(N):
            if a <= p < b:
                continue
            t0_p, t1_p = events[p]
            # keep if the label ends BEFORE the test start, OR starts AFTER the end+embargo
            if t1_p < test_t0_min or t0_p > test_t1_max + embargo_bars:
                train_idx.append(p)
        folds.append({"train_idx": train_idx, "test_idx": test_idx})
    return folds


# --------------------------------------------------------------------------- #
# Default predictor (secondary model's logistic regression)           #
# --------------------------------------------------------------------------- #
def default_logreg_fit(lr: float = 0.3, epochs: int = 300, l2: float = 0.0,
                       standardize: bool = True) -> FitFn:
    """Builds a fit_fn: (X_train, y_train) -> predictor(X)->labels, with standardization
    fitted on train and applied at prediction time."""
    def fit(Xtr: List[List[float]], ytr: List[int]):
        means = stds = None
        Xs = [list(r) for r in Xtr]
        if standardize and Xs:
            means, stds = MM.standardize_fit(Xs)
            Xs = MM.standardize_apply(Xs, means, stds)
        model = MM.LogisticRegression(lr=lr, epochs=epochs, l2=l2).fit(Xs, ytr)

        def predict(X):
            Xa = [list(r) for r in X]
            if means is not None:
                Xa = MM.standardize_apply(Xa, means, stds)
            return model.predict(Xa)
        return predict
    return fit


def accuracy_score(predict, X: List[List[float]], y: List[int]) -> float:
    yhat = predict(X)
    n = len(y)
    return sum(1 for a, b in zip(yhat, y) if a == b) / n if n else 0.0


def cv_score(X: Sequence[Sequence[float]], y: Sequence[int], events: Sequence[Event],
             n_splits: int = 5, embargo_pct: float = 0.0,
             fit_fn: Optional[FitFn] = None, score_fn: ScoreFn = accuracy_score,
             n_bars: Optional[int] = None) -> Dict:
    """Purged CV score per fold. Returns {fold_scores, mean, std}."""
    fit_fn = fit_fn or default_logreg_fit()
    folds = purged_kfold(events, n_splits, embargo_pct, n_bars)
    scores = []
    for f in folds:
        tr, te = f["train_idx"], f["test_idx"]
        if not tr or not te:
            continue
        predict = fit_fn([list(X[i]) for i in tr], [y[i] for i in tr])
        scores.append(score_fn(predict, [list(X[i]) for i in te], [y[i] for i in te]))
    n = len(scores)
    mean = sum(scores) / n if n else 0.0
    std = (sum((s - mean) ** 2 for s in scores) / n) ** 0.5 if n else 0.0
    return {"fold_scores": scores, "mean": mean, "std": std}


# --------------------------------------------------------------------------- #
# MDA - Mean Decrease Accuracy (permutation, AFML 8.3)                          #
# --------------------------------------------------------------------------- #
def mda_importance(X: Sequence[Sequence[float]], y: Sequence[int], events: Sequence[Event],
                   n_splits: int = 5, embargo_pct: float = 0.0,
                   fit_fn: Optional[FitFn] = None, score_fn: ScoreFn = accuracy_score,
                   seed: int = 0, n_bars: Optional[int] = None) -> Dict[int, Dict[str, float]]:
    """Permutation importance: score drop when each feature is shuffled in the test.
    Returns {j: {mean, std}}; high mean = important feature."""
    fit_fn = fit_fn or default_logreg_fit()
    folds = purged_kfold(events, n_splits, embargo_pct, n_bars)
    d = len(X[0]) if X else 0
    per_feat: Dict[int, List[float]] = {j: [] for j in range(d)}
    for fi, f in enumerate(folds):
        tr, te = f["train_idx"], f["test_idx"]
        if not tr or not te:
            continue
        predict = fit_fn([list(X[i]) for i in tr], [y[i] for i in tr])
        Xte = [list(X[i]) for i in te]
        yte = [y[i] for i in te]
        s0 = score_fn(predict, Xte, yte)
        for j in range(d):
            rng = random.Random(1000 * seed + 31 * fi + j)
            col = [row[j] for row in Xte]
            perm = col[:]
            rng.shuffle(perm)
            Xperm = [row[:] for row in Xte]
            for i in range(len(Xperm)):
                Xperm[i][j] = perm[i]
            sj = score_fn(predict, Xperm, yte)
            per_feat[j].append(s0 - sj)        # score drop = importance
    out = {}
    for j, vals in per_feat.items():
        n = len(vals)
        m = sum(vals) / n if n else 0.0
        sd = (sum((v - m) ** 2 for v in vals) / n) ** 0.5 if n else 0.0
        out[j] = {"mean": m, "std": sd}
    return out


# --------------------------------------------------------------------------- #
# SFI - Single Feature Importance (AFML 8.4)                                    #
# --------------------------------------------------------------------------- #
def sfi_importance(X: Sequence[Sequence[float]], y: Sequence[int], events: Sequence[Event],
                   n_splits: int = 5, embargo_pct: float = 0.0,
                   fit_fn: Optional[FitFn] = None, score_fn: ScoreFn = accuracy_score,
                   n_bars: Optional[int] = None) -> Dict[int, float]:
    """Importance of a SINGLE feature: CV score using only column j.
    Returns {j: mean_cv_score}; high = feature individually predictive."""
    fit_fn = fit_fn or default_logreg_fit()
    d = len(X[0]) if X else 0
    out = {}
    for j in range(d):
        Xj = [[row[j]] for row in X]
        out[j] = cv_score(Xj, y, events, n_splits, embargo_pct, fit_fn, score_fn, n_bars)["mean"]
    return out


def rank_features(importance: Dict[int, Dict[str, float]]) -> List[Tuple[int, float]]:
    """Ranks features by descending MDA importance (mean)."""
    items = [(j, v["mean"] if isinstance(v, dict) else v) for j, v in importance.items()]
    return sorted(items, key=lambda kv: kv[1], reverse=True)
