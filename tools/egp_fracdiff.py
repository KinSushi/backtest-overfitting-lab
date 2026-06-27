"""
egp_fracdiff.py - Fixed-width fractional differentiation (FFD), AFML ch.5 (Lopez de Prado).

Problem: prices are NON-STATIONARY (ML models assume stationarity) but
returns (first-order difference) erase almost all the MEMORY. Differentiating to a
REAL order d in (0,1) makes the series stationary while KEEPING as much memory as possible (Hosking 1981;
Lopez de Prado 2018 ch.5). Useful to build stationary-but-memory-preserving FEATURES for the
secondary model (egp_meta_model).

Formulas VERIFIED against concordant sources (RobotFX/MQL5, Deep Learning Wizard, mlfinlab):
  - Weights (AFML eq. 5.4): w_0 = 1; w_k = -w_{k-1} * (d - k + 1) / k , k = 1, 2, ...
  - FFD (fixed window): truncate the sequence at the FIRST k where |w_k| < tau (threshold), then apply
    the SAME truncated vector to all observations -> X~_t = Sum_{k=0}^{l*} w_k * X_{t-k}.
    (w_0 multiplies the most recent X_t; w_{l*} the oldest.) Method RECOMMENDED by LdP because
    no drift (unlike the expanding window).
Known numerical check: d=1 -> w=[1,-1,0,...] (first difference); d=0 -> [1,0,...].

RESERVATION: the choice of the MINIMAL d* ensuring stationarity is made via an ADF test (Augmented
Dickey-Fuller). The ADF is NOT implemented here (no statsmodels; MacKinnon critical-value table
to be verified separately) -> out of scope. This module provides the verified FFD transform
and a memory-retention diagnostic; the final d* must be confirmed by an external ADF.
Pure-Python. Source: M. Lopez de Prado, *Advances in Financial Machine Learning* (2018), ch.5.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple


def frac_weights(d: float, size: int) -> List[float]:
    """Fractional-differentiation weights (AFML eq. 5.4): w_0=1, w_k=-w_{k-1}(d-k+1)/k."""
    if size <= 0:
        return []
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] * (d - k + 1) / k)
    return w


def frac_weights_ffd(d: float, thresh: float = 1e-5, max_size: int = 100000) -> List[float]:
    """FFD weights: stop at the FIRST k where |w_k| < thresh (fixed-width window).
    Returns [w_0, w_1, ..., w_l*] (most recent to oldest)."""
    w = [1.0]
    k = 1
    while k < max_size:
        wk = -w[-1] * (d - k + 1) / k
        if abs(wk) < thresh:
            break
        w.append(wk)
        k += 1
    return w


def ffd_width(d: float, thresh: float = 1e-5) -> int:
    """FFD window width (number of weights kept) for (d, thresh)."""
    return len(frac_weights_ffd(d, thresh))


def frac_diff_ffd(series: Sequence[float], d: float, thresh: float = 1e-5) -> Dict:
    """Fixed-width fractional differentiation. X~_t = Sum_k w_k * X_{t-k}.
    Returns {values, offset, weights, width}; values starts at index offset=width-1
    (the first bars are dropped, incomplete window)."""
    w = frac_weights_ffd(d, thresh)
    width = len(w)
    n = len(series)
    out: List[float] = []
    for t in range(width - 1, n):
        s = 0.0
        for k in range(width):
            s += w[k] * series[t - k]      # w[0]*X_t (recent) ... w[width-1]*X_{t-width+1} (oldest)
        out.append(s)
    return {"values": out, "offset": width - 1, "weights": w, "width": width}


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n < 2:
        return float("nan")
    a = a[:n]
    b = b[:n]
    ma = sum(a) / n
    mb = sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = sum((a[i] - ma) ** 2 for i in range(n))
    vb = sum((b[i] - mb) ** 2 for i in range(n))
    if va <= 0 or vb <= 0:
        return float("nan")
    return cov / math.sqrt(va * vb)


def memory_retention(series: Sequence[float], d: float, thresh: float = 1e-5) -> float:
    """Pearson correlation between the FFD series and the original (aligned) series. Proxy for the
    MEMORY kept: ~1 = lots of memory (low d); low = little (d close to 1).
    AFML picks the minimal d such that the series is stationary (ADF) AND corr stays high."""
    res = frac_diff_ffd(series, d, thresh)
    aligned = list(series[res["offset"]:])
    return _pearson(res["values"], aligned)


def memory_vs_d(series: Sequence[float], d_grid: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4,
                0.5, 0.6, 0.7, 0.8, 0.9, 1.0), thresh: float = 1e-5) -> List[Tuple[float, float, int]]:
    """For each d: (d, corr with the original, window width). Helps choose the
    stationarity/memory trade-off (the final d* still to be confirmed by an external ADF)."""
    out = []
    for d in d_grid:
        out.append((d, memory_retention(series, d, thresh), ffd_width(d, thresh)))
    return out


def fracdiff_feature(prices: Sequence[float], event_indices: Sequence[int], d: float = 0.4,
                     thresh: float = 1e-5) -> List[float]:
    """Builds an FFD feature aligned on event indices (signals), for the secondary
    model. Events before the first available FFD value receive 0.0 (incomplete window)."""
    res = frac_diff_ffd(prices, d, thresh)
    offset = res["offset"]
    vals = res["values"]
    feat = []
    for idx in event_indices:
        j = idx - offset
        feat.append(vals[j] if 0 <= j < len(vals) else 0.0)
    return feat
