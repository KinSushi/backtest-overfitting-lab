"""
egp_triple_barrier.py - Triple-barrier labeling (Lopez de Prado, AFML ch.3), vol-aware,
with COST-NET META-LABELS.

Principle (AFML ch.3, getDailyVol/getEvents/getBins): for each event (signal) at t0 at
price p0, three barriers are set:
  - upper barrier (take-profit)        : position return >= pt_mult * sigma_t0;
  - lower barrier (stop)               : position return <= -sl_mult * sigma_t0;
  - vertical barrier (time)            : t0 + max_horizon bars.
The label = FIRST barrier touched: +1 (profit), -1 (stop), 0 (time). sigma_t0 = local
vol (EWMA) -> WIDE barriers in volatile regimes, NARROW in calm regimes.

POSITION-CENTERED convention (side): for a long (side=+1) profit = price going up;
for a short (side=-1) profit = price going down. Position return = side * (p_t/p0 - 1).

META-LABEL (binary, for AFML meta-labeling): "was the side taken profitable NET OF
COSTS?" = 1 if position_return - cost > 0, else 0. A trade that touches the upper barrier
but whose gain does NOT cover the costs gets meta=0: that is the key contribution of net-of-cost.

Reuses egp_costs to convert a cost (account currency) into return units.
Pure-Python. Source: M. Lopez de Prado, *Advances in Financial Machine Learning* (2018), ch.3.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
import egp_costs as C


def returns_from_prices(prices: Sequence[float]) -> List[float]:
    """Simple bar-to-bar returns r_t = p_t/p_{t-1} - 1 (r_0 = 0)."""
    out = [0.0]
    for i in range(1, len(prices)):
        out.append(prices[i] / prices[i - 1] - 1.0 if prices[i - 1] != 0 else 0.0)
    return out


def ewma_volatility(returns: Sequence[float], span: int = 20) -> List[float]:
    """EWMA volatility (standard deviation), getDailyVol style: recursive variance
    var_t = (1-alpha)*var_{t-1} + alpha*r_t^2, alpha = 2/(span+1). Returns sqrt(var_t)."""
    if not returns:
        return []
    alpha = 2.0 / (span + 1.0)
    var = returns[0] ** 2
    out = [math.sqrt(var)]
    for r in returns[1:]:
        var = (1.0 - alpha) * var + alpha * (r * r)
        out.append(math.sqrt(var))
    return out


def first_touch(prices: Sequence[float], t0: int, sigma: float, pt_mult: float,
                sl_mult: float, max_horizon: int, side: int = 1) -> Dict:
    """First barrier touched from t0 (exclusive) to t0+max_horizon.
    Returns {touch_idx, outcome ('pt'/'sl'/'time'), ret (raw return p_t/p0-1),
    side_return (position return = side*ret), label (+1/-1/0)}."""
    n = len(prices)
    p0 = prices[t0]
    up = pt_mult * sigma            # profit threshold (position return)
    dn = sl_mult * sigma            # stop threshold
    last = min(t0 + max_horizon, n - 1)
    for t in range(t0 + 1, last + 1):
        r = prices[t] / p0 - 1.0
        sr = side * r
        if up > 0 and sr >= up:
            return {"touch_idx": t, "outcome": "pt", "ret": r, "side_return": sr, "label": 1}
        if dn > 0 and sr <= -dn:
            return {"touch_idx": t, "outcome": "sl", "ret": r, "side_return": sr, "label": -1}
    # vertical barrier: no horizontal one touched
    r = prices[last] / p0 - 1.0
    return {"touch_idx": last, "outcome": "time", "ret": r, "side_return": side * r, "label": 0}


def cost_in_return_units(cost_model: "C.CostModel", lots: float, price: float) -> float:
    """Converts the round-trip cost (account currency) into return units:
    cost / notional, notional = price * contract_size * lots."""
    notional = price * cost_model.contract_size * lots
    if notional <= 0:
        return 0.0
    return cost_model.linear_cost(lots) / notional


def meta_label(side_return: float, cost: float) -> int:
    """Binary COST-NET meta-label: 1 if side_return - cost > 0, else 0."""
    return 1 if (side_return - cost) > 0 else 0


def apply_triple_barrier(prices: Sequence[float], events: List[Tuple[int, int]],
                         pt_mult: float, sl_mult: float, vol: Sequence[float],
                         max_horizon: int) -> List[Dict]:
    """Applies the triple-barrier to a list of events (t0, side).
    vol = sigma series aligned with prices (sigma_t0 = vol[t0])."""
    out = []
    for (t0, side) in events:
        sigma = vol[t0] if t0 < len(vol) else 0.0
        ft = first_touch(prices, t0, sigma, pt_mult, sl_mult, max_horizon, side)
        ft["t0"], ft["side"], ft["sigma"] = t0, side, sigma
        out.append(ft)
    return out


def add_meta_labels(barrier_results: List[Dict], cost: float = 0.0) -> List[Dict]:
    """Adds the cost-net meta-label to each result (cost in return units)."""
    for r in barrier_results:
        r["meta_label"] = meta_label(r["side_return"], cost)
        r["net_return"] = r["side_return"] - cost
    return barrier_results


def triple_barrier_from_signals(prices: Sequence[float], signal_idx: Sequence[int],
                                signal_sides: Sequence[int], span: int = 20,
                                pt_mult: float = 2.0, sl_mult: float = 2.0,
                                max_horizon: int = 20, cost: float = 0.0) -> List[Dict]:
    """End to end: prices -> returns -> EWMA vol -> barriers -> meta-labels.
    cost: round-trip cost in RETURN UNITS (cf. cost_in_return_units)."""
    rets = returns_from_prices(prices)
    vol = ewma_volatility(rets, span)
    events = list(zip(signal_idx, signal_sides))
    res = apply_triple_barrier(prices, events, pt_mult, sl_mult, vol, max_horizon)
    return add_meta_labels(res, cost)


def label_summary(barrier_results: List[Dict]) -> Dict:
    """Counts primary and meta labels + rates."""
    n = len(barrier_results)
    if n == 0:
        return {"n": 0}
    prim = {1: 0, -1: 0, 0: 0}
    meta_pos = 0
    for r in barrier_results:
        prim[r["label"]] += 1
        meta_pos += r.get("meta_label", 0)
    return {"n": n, "pt": prim[1], "sl": prim[-1], "time": prim[0],
            "meta_positive": meta_pos, "meta_rate": meta_pos / n}
