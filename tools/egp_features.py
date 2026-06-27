"""
egp_features.py - Market feature stack aligned on events (pure-Python).

Replaces the runner's 3 demo features ([fracdiff, side, sigma]) with a statistical set
rich and reproducible. All features are causal (at time t, use only P[<=t]: no
look-ahead). Designed for XAUUSD M1 but generic.

Formulas:
      - Log return : r_t = ln(P_t / P_{t-1})                               (by definition).
  - Momentum window w : P_t / P_{t-w} - 1                                 (definitional).
  - Rolling volatility: std of log returns over w                       (definitional).
  - Rolling mean of returns over w                                       (definitional).
  - Skewness (Fisher-Pearson) : g1 = m3 / m2^{3/2}, m_k = (1/n) sum (x-xbar)^k.
      - Excess kurtosis : g2 = m4 / m2^2 - 3.
      - Lag-1 autocorrelation over w (Box-Jenkins) :
        r1 = sum_{t} (x_t - xbar)(x_{t-1} - xbar) / sum (x_t - xbar)^2.
  - Z-score of price over w : (P_t - mean_w) / std_w.
  - Fracdiff at several d : egp_fracdiff.fracdiff_feature (VERIFIED weights, AFML ch.5).
    - EWMA volatility : egp_triple_barrier.ewma_volatility (VERIFIED).

Skewness/kurtosis = moments echantillonnaux standards (Fisher-Pearson) ; ACF = definition Box-Jenkins ;
the rest is definitional. Only the fracdiff brick carries a non-trivial formula, already verified.
Standard sources: Box & Jenkins (ACF); Fisher-Pearson (moments). CAVEAT: these are not
microstructure features (volume/spread/imbalance), unavailable here; to add with real data.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)

import egp_fracdiff as FD
import egp_triple_barrier as TB


def _log_returns(prices: Sequence[float]) -> List[float]:
    r = [0.0]
    for t in range(1, len(prices)):
        p0, p1 = prices[t - 1], prices[t]
        r.append(math.log(p1 / p0) if (p0 > 0 and p1 > 0) else 0.0)
    return r


def _mean(v: Sequence[float]) -> float:
    return sum(v) / len(v) if v else 0.0


def _std(v: Sequence[float]) -> float:
    if len(v) < 2:
        return 0.0
    mu = _mean(v)
    return math.sqrt(sum((x - mu) ** 2 for x in v) / len(v))


def _skew(v: Sequence[float]) -> float:
    n = len(v)
    if n < 3:
        return 0.0
    mu = _mean(v)
    m2 = sum((x - mu) ** 2 for x in v) / n
    m3 = sum((x - mu) ** 3 for x in v) / n
    return m3 / (m2 ** 1.5) if m2 > 0 else 0.0


def _kurt(v: Sequence[float]) -> float:
    n = len(v)
    if n < 4:
        return 0.0
    mu = _mean(v)
    m2 = sum((x - mu) ** 2 for x in v) / n
    m4 = sum((x - mu) ** 4 for x in v) / n
    return (m4 / (m2 ** 2) - 3.0) if m2 > 0 else 0.0


def _acf1(v: Sequence[float]) -> float:
    n = len(v)
    if n < 3:
        return 0.0
    mu = _mean(v)
    denom = sum((x - mu) ** 2 for x in v)
    if denom <= 0:
        return 0.0
    num = sum((v[t] - mu) * (v[t - 1] - mu) for t in range(1, n))
    return num / denom


def default_feature_config() -> Dict:
    return {"windows": [5, 20, 60], "fracdiff_d": [0.3, 0.5], "fracdiff_thresh": 1e-4,
            "ewma_span": 20, "include_skew_kurt": True, "include_acf": True}


def build_features(prices: Sequence[float], event_indices: Sequence[int],
                   config: Optional[Dict] = None) -> Tuple[List[List[float]], List[str]]:
    """Builds the feature matrix (len(event_indices) x n_features) + names.
    Causal: the feature at time t uses only prices up to t (indices <= t)."""
    cfg = config or default_feature_config()
    windows = cfg["windows"]
    r = _log_returns(prices)
    ewma = TB.ewma_volatility(r, span=cfg["ewma_span"])
    # fracdiff per d (aligned on event indices)
    ffd_cols = {d: FD.fracdiff_feature(prices, event_indices, d=d, thresh=cfg["fracdiff_thresh"])
                for d in cfg["fracdiff_d"]}

    names: List[str] = ["ret_1"]
    for w in windows:
        names += [f"mom_{w}", f"vol_{w}", f"meanret_{w}", f"zscore_{w}"]
        if cfg["include_skew_kurt"]:
            names += [f"skew_{w}", f"kurt_{w}"]
        if cfg["include_acf"]:
            names += [f"acf1_{w}"]
    names += ["ewma_vol"]
    names += [f"fracdiff_{d}" for d in cfg["fracdiff_d"]]

    X: List[List[float]] = []
    for ei, t in enumerate(event_indices):
        row: List[float] = [r[t] if t >= 1 else 0.0]
        for w in windows:
            if t >= w:
                win_r = r[t - w + 1:t + 1]
                win_p = prices[t - w + 1:t + 1]
                mom = (prices[t] / prices[t - w] - 1.0) if prices[t - w] > 0 else 0.0
                vol = _std(win_r)
                meanret = _mean(win_r)
                mp, sp = _mean(win_p), _std(win_p)
                z = (prices[t] - mp) / sp if sp > 0 else 0.0
                row += [mom, vol, meanret, z]
                if cfg["include_skew_kurt"]:
                    row += [_skew(win_r), _kurt(win_r)]
                if cfg["include_acf"]:
                    row += [_acf1(win_r)]
            else:
                k = 4 + (2 if cfg["include_skew_kurt"] else 0) + (1 if cfg["include_acf"] else 0)
                row += [0.0] * k
        row.append(ewma[t] if t < len(ewma) else 0.0)
        row += [ffd_cols[d][ei] for d in cfg["fracdiff_d"]]
        X.append(row)
    return X, names
