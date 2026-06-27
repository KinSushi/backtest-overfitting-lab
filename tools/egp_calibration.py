"""
egp_calibration.py - Probability calibration (Platt scaling, isotonic/PAVA, Brier score).

WHY: bet sizing (egp_bet_sizing, AFML ch.10) ASSUMES that p is well calibrated (z and the size
depend on it directly). But a classifier's probabilities (logistic regression, forest) are often
poorly calibrated. So we learn a recalibration function on a HOLD-OUT FOLD (purged) then apply it
before sizing. Otherwise: systematic over/under-sizing of positions.

Formulas VERIFIED against concordant sources:
  - Platt scaling (arxiv 2601.19944; arxiv 1710.08901 "PD calibration"):
        p = 1 / (1 + exp(A*s + B)),  A,B fitted by MLE (min NLL via gradient descent).
        Gradient: dNLL/dA = sum (y_i - p_i)*s_i;  dNLL/dB = sum (y_i - p_i)  [p = sigma(-(A s+B))].
  - Isotonic regression (KDnuggets; MATLAB; arxiv 1511.05191): monotone non-decreasing mapping
        fitted by PAVA (Pool Adjacent Violators Algorithm), order-preserving least squares.
  - Brier score (Train in Data; arxiv): (1/N) * sum (f_t - o_t)^2, o_t in {0,1}; lower = better.

Calibrate on a set DISJOINT from training (otherwise optimism). Pure-Python.
Reference: Platt (2000); Zadrozny & Elkan (2002); Niculescu-Mizil & Caruana (2005).
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

_EPS = 1e-12


def brier_score(probs: Sequence[float], y: Sequence[float]) -> float:
    """Brier score = mean of squares (p - y)^2, y in {0,1}. Lower = better calibrated."""
    n = len(probs)
    return sum((probs[i] - y[i]) ** 2 for i in range(n)) / n if n else float("nan")


def _clamp(p: float) -> float:
    return min(1.0 - _EPS, max(_EPS, p))


class PlattScaling:
    """Sigmoid recalibration: p = 1/(1+exp(A*s+B)), A,B by MLE (gradient descent)."""

    def __init__(self, lr: float = 0.05, epochs: int = 2000):
        self.lr = lr
        self.epochs = epochs
        self.A = 0.0
        self.B = 0.0

    def _p(self, s: float) -> float:
        z = self.A * s + self.B
        # p = 1/(1+exp(z)), rewritten to avoid exp overflow
        if z >= 0:
            ez = math.exp(-z)
            return ez / (1.0 + ez)    # = 1/(1+exp(z)), stable for z>=0
        return 1.0 / (1.0 + math.exp(z))   # stable for z<0 (exp(z)<1)

    def fit(self, scores: Sequence[float], y: Sequence[float]) -> "PlattScaling":
        n = len(scores)
        self.A, self.B = 0.0, 0.0
        for _ in range(self.epochs):
            gA = 0.0
            gB = 0.0
            for i in range(n):
                p = self._p(scores[i])
                diff = y[i] - p                       # (y - p)
                gA += diff * scores[i]
                gB += diff
            self.A -= self.lr * gA / n
            self.B -= self.lr * gB / n
        return self

    def predict(self, scores: Sequence[float]) -> List[float]:
        return [_clamp(self._p(s)) for s in scores]


def _pava(values: Sequence[float], weights: Sequence[float]) -> List[float]:
    """Pool Adjacent Violators Algorithm: monotone non-decreasing fit (weighted least squares).
    `values` must be sorted by ascending score. Returns the fitted value per point (block-wise)."""
    blocks: List[List[float]] = []          # [value, weight, count]
    for v, w in zip(values, weights):
        blocks.append([v, w, 1])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            nw = w1 + w2
            blocks.append([(v1 * w1 + v2 * w2) / nw, nw, c1 + c2])
    out: List[float] = []
    for v, _, c in blocks:
        out.extend([v] * c)
    return out


class IsotonicRegression:
    """Monotone non-parametric recalibration (PAVA). Linear interpolation between points to predict."""

    def __init__(self):
        self.x_: List[float] = []
        self.y_: List[float] = []

    def fit(self, scores: Sequence[float], y: Sequence[float],
            weights: Optional[Sequence[float]] = None) -> "IsotonicRegression":
        order = sorted(range(len(scores)), key=lambda i: scores[i])
        xs = [float(scores[i]) for i in order]
        ys = [float(y[i]) for i in order]
        ws = [float(weights[i]) for i in order] if weights is not None else [1.0] * len(xs)
        fitted = _pava(ys, ws)
        # compact into (x, value) points by averaging consecutive x with equal value
        self.x_, self.y_ = xs, fitted
        return self

    def predict(self, scores: Sequence[float]) -> List[float]:
        out = []
        xs, ys = self.x_, self.y_
        for s in scores:
            if s <= xs[0]:
                out.append(_clamp(ys[0]))
            elif s >= xs[-1]:
                out.append(_clamp(ys[-1]))
            else:
                # linear interpolation between the surrounding points
                lo = 0
                hi = len(xs) - 1
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if xs[mid] <= s:
                        lo = mid
                    else:
                        hi = mid
                x0, x1, y0, y1 = xs[lo], xs[hi], ys[lo], ys[hi]
                t = 0.0 if x1 == x0 else (s - x0) / (x1 - x0)
                out.append(_clamp(y0 + t * (y1 - y0)))
        return out


def calibrate(scores_fit: Sequence[float], y_fit: Sequence[float], method: str = "platt",
              **kwargs):
    """Fits a calibrator ('platt' or 'isotonic') on the calibration fold."""
    if method == "platt":
        return PlattScaling(**kwargs).fit(scores_fit, y_fit)
    if method == "isotonic":
        return IsotonicRegression().fit(scores_fit, y_fit)
    raise ValueError("method must be 'platt' or 'isotonic'")


def calibration_gain(scores: Sequence[float], y: Sequence[float], calibrator) -> dict:
    """Compares the raw Brier (scores as-is) to the recalibrated Brier on an evaluation set."""
    raw = brier_score([_clamp(s) for s in scores], y)
    cal = brier_score(calibrator.predict(scores), y)
    return {"brier_raw": raw, "brier_calibrated": cal, "improved": cal < raw}


class EnsembleCalibrator:
    """Ensemble of calibrators (one per purged fold): prediction = mean. Uses all the
    data without a single leaking hold-out."""

    def __init__(self, calibrators):
        self.calibrators = calibrators

    def predict(self, scores):
        if not self.calibrators:
            return [_clamp(s) for s in scores]
        preds = [c.predict(scores) for c in self.calibrators]
        return [sum(p[i] for p in preds) / len(preds) for i in range(len(scores))]


def calibrate_cv(scores, y, events, method: str = "isotonic", n_splits: int = 5,
                 embargo_pct: float = 0.0, n_bars=None, **kwargs):
    """Calibration in PURGED CV: fits one calibrator per fold (purged train), aggregates into an
    ensemble, and estimates the OUT-OF-FOLD Brier (honest, no leakage). Returns
    {calibrator, oof_brier, n_folds}."""
    import egp_cv_importance as CV
    if n_bars is None:
        n_bars = max(t1 for _, t1 in events) + 1
    folds = CV.purged_kfold(events, n_splits=n_splits, embargo_pct=embargo_pct, n_bars=n_bars)
    cals = []
    oof_pred = [None] * len(scores)
    for fold in folds:
        tr, te = fold["train_idx"], fold["test_idx"]
        if not tr or not te:
            continue
        cal = calibrate([scores[i] for i in tr], [y[i] for i in tr], method=method, **kwargs)
        cals.append(cal)
        for i, p in zip(te, cal.predict([scores[i] for i in te])):
            oof_pred[i] = p
    cov = [(oof_pred[i], y[i]) for i in range(len(scores)) if oof_pred[i] is not None]
    oof_brier = brier_score([p for p, _ in cov], [yy for _, yy in cov]) if cov else float("nan")
    return {"calibrator": EnsembleCalibrator(cals), "oof_brier": oof_brier, "n_folds": len(cals)}
