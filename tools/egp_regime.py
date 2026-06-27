"""
egp_regime.py - Diagnostic: WHERE does the edge live? (session, hour, day, volatility regime)

Goal: decompose a strategy's performance by temporal/regime SEGMENT to answer
the question "is the edge robust or concentrated in a single context?". An edge present in only one
session (or a single vol regime) is FRAGILE: it vanishes as soon as the context changes.

Inputs: `trades` = list of dicts {'time': datetime (UTC), 'pnl': float}. For the vol
volatility: a series of per-bar returns + its rolling volatility.

FX sessions (STANDARD/winter time, UTC; to CALIBRATE on the broker server time + DST):
  ASIA 00:00-08:00 | LONDON 08:00-13:00 | OVERLAP(Londres+NY) 13:00-17:00 | NEW_YORK 17:00-22:00
  | SYDNEY 22:00-24:00. The London/NY overlap (13:00-17:00 UTC) is the most liquid.
      Sources : dukascopy, tmgm, startrader, myfxbook (concordant ; Tokyo without DST 00:00-09:00 UTC).

Regime-change detection: reuses egp_structural.page_cusum (Page CUSUM).
Pure-Python. No external dependency.
"""
from __future__ import annotations

import math
import os
import sys
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
import egp_accept_gate as G
import egp_structural as S

# Day partition (UTC, standard time): (label, start_h, end_h) half-open [start, end).
SESSIONS_UTC_STD: List[Tuple[str, int, int]] = [
    ("ASIA", 0, 8),
    ("LONDON", 8, 13),
    ("OVERLAP", 13, 17),     # London + New York: the most liquid
    ("NEW_YORK", 17, 22),
    ("SYDNEY", 22, 24),
]


def _hour_of(t) -> int:
    if isinstance(t, datetime):
        return t.hour
    return int(t) % 24


def classify_session(t, sessions: List[Tuple[str, int, int]] = SESSIONS_UTC_STD) -> str:
    """Returns the session label for a UTC datetime (or an hour 0-23)."""
    h = _hour_of(t)
    for label, a, b in sessions:
        if a <= h < b:
            return label
    return "OFF"


# --------------------------------------------------------------------------- #
# Per-segment statistics                                                     #
# --------------------------------------------------------------------------- #
def segment_stats(pnls: Sequence[float]) -> Dict[str, float]:
    """Aggregates of a segment: count, net, mean, win_rate, profit_factor, sharpe."""
    n = len(pnls)
    if n == 0:
        return {"count": 0, "net": 0.0, "mean": 0.0, "win_rate": 0.0,
                "profit_factor": float("nan"), "sharpe": 0.0}
    net = sum(pnls)
    wins = [p for p in pnls if p > 0]
    gross_win = sum(wins)
    gross_loss = -sum(p for p in pnls if p < 0)
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    return {"count": n, "net": net, "mean": net / n,
            "win_rate": len(wins) / n, "profit_factor": pf,
            "sharpe": G.sharpe_ratio(list(pnls))}


def performance_by(trades: List[Dict], key_fn: Callable[[Dict], object]) -> Dict[object, Dict]:
    """Groups trades by key_fn(trade) then computes segment_stats per group."""
    buckets: Dict[object, List[float]] = {}
    for tr in trades:
        buckets.setdefault(key_fn(tr), []).append(tr["pnl"])
    return {k: segment_stats(v) for k, v in buckets.items()}


def performance_by_session(trades: List[Dict],
                           sessions: List[Tuple[str, int, int]] = SESSIONS_UTC_STD) -> Dict[str, Dict]:
    return performance_by(trades, lambda tr: classify_session(tr["time"], sessions))


def performance_by_hour(trades: List[Dict]) -> Dict[int, Dict]:
    return performance_by(trades, lambda tr: _hour_of(tr["time"]))


def performance_by_weekday(trades: List[Dict]) -> Dict[int, Dict]:
    # 0=Monday .. 6=Sunday
    return performance_by(trades, lambda tr: tr["time"].weekday())


# --------------------------------------------------------------------------- #
# Regimes de volatilite                                                        #
# --------------------------------------------------------------------------- #
def rolling_volatility(returns: Sequence[float], window: int = 20) -> List[float]:
    """Rolling std (realized volatility per bar). The first bars
    (before 'window' points) use the available window."""
    out = []
    for i in range(len(returns)):
        lo = max(0, i - window + 1)
        w = returns[lo:i + 1]
        m = sum(w) / len(w)
        var = sum((x - m) ** 2 for x in w) / len(w)
        out.append(math.sqrt(var))
    return out


def _quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def volatility_regimes(vol_series: Sequence[float],
                       quantiles: Tuple[float, float] = (1 / 3, 2 / 3)) -> Dict:
    """Classifies each observation as LOW/MID/HIGH by the quantiles of the vol distribution.
    Returns {labels: [...], thresholds: (q_lo, q_hi)}."""
    sv = sorted(vol_series)
    q_lo = _quantile(sv, quantiles[0])
    q_hi = _quantile(sv, quantiles[1])
    labels = []
    for v in vol_series:
        if v <= q_lo:
            labels.append("LOW")
        elif v <= q_hi:
            labels.append("MID")
        else:
            labels.append("HIGH")
    return {"labels": labels, "thresholds": (q_lo, q_hi)}


def performance_by_vol_regime(pnls: Sequence[float], vol_series: Sequence[float],
                              quantiles: Tuple[float, float] = (1 / 3, 2 / 3)) -> Dict[str, Dict]:
    """Performance by volatility regime (pnls and vol_series aligned index by index)."""
    reg = volatility_regimes(vol_series, quantiles)["labels"]
    buckets: Dict[str, List[float]] = {}
    for p, r in zip(pnls, reg):
        buckets.setdefault(r, []).append(p)
    return {k: segment_stats(v) for k, v in buckets.items()}


# --------------------------------------------------------------------------- #
# Points de changement de regime (CUSUM iteratif)                              #
# --------------------------------------------------------------------------- #
def regime_change_points(series: Sequence[float], mu0: float = 0.0, k: float = 0.5,
                         h: float = 5.0, min_gap: int = 5, max_points: int = 50) -> List[int]:
    """Detects mean break points by restarting page_cusum after each alarm.
    Returns the indices (in 'series') of the successive alarms."""
    points: List[int] = []
    start = 0
    n = len(series)
    while start < n and len(points) < max_points:
        out = S.page_cusum(series[start:], mu0=mu0, k=k, h=h)
        a = out.get("alarm")
        if a is None:
            break
        idx = start + a
        points.append(idx)
        start = idx + max(1, min_gap)
    return points


# --------------------------------------------------------------------------- #
# Edge concentration (fragility risk)                                #
# --------------------------------------------------------------------------- #
def edge_concentration(by_segment: Dict[object, Dict],
                       fragile_share: float = 0.6) -> Dict:
    """Measures the concentration of profit across profitable segments.
    - best_share: share of total net profit coming from the BEST segment;
    - hhi: Herfindahl index on the profit shares (1 = all in one segment);
    - fragile: True if <=1 profitable segment OR best_share > threshold.
    An edge concentrated in a single context is deemed FRAGILE."""
    pos = {k: v["net"] for k, v in by_segment.items() if v["net"] > 0}
    total = sum(pos.values())
    if total <= 0:
        return {"n_profitable": 0, "n_total": len(by_segment), "best_segment": None,
                "best_share": 0.0, "hhi": 0.0, "fragile": True,
                "reason": "no profitable segment"}
    shares = {k: v / total for k, v in pos.items()}
    best = max(shares, key=shares.get)
    best_share = shares[best]
    hhi = sum(s * s for s in shares.values())
    fragile = (len(pos) <= 1) or (best_share > fragile_share)
    reason = []
    if len(pos) <= 1:
        reason.append("only one profitable segment")
    if best_share > fragile_share:
        reason.append(f"{best} concentrates {best_share:.0%} of profit (> {fragile_share:.0%})")
    return {"n_profitable": len(pos), "n_total": len(by_segment),
            "best_segment": best, "best_share": best_share, "hhi": hhi,
            "fragile": fragile, "reason": "; ".join(reason) if reason else "edge spread out"}
