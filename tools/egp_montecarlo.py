"""
egp_montecarlo.py - Monte-Carlo over the trade SEQUENCE: drawdown and risk of ruin.

A backtest gives only ONE path. By resampling the per-trade P&L we obtain a
DISTRIBUTION: expected maximum drawdown, loss probability, and risk of ruin
(probability that capital hits a ruin threshold before the end). Two modes:
  - bootstrap (draw WITH replacement): varies the trade sample -> uncertainty
    estimation on the terminal equity AND the drawdown;
  - shuffle (permutation WITHOUT replacement): keeps the SAME set of trades, varies only
    the ORDER -> isolates the SEQUENCING risk (the terminal equity is invariant, only the
    path/drawdown changes).

Sources :
  - Efron & Tibshirani (1993), "An Introduction to the Bootstrap", Chapman & Hall.
  - Gambler's ruin, Feller, "An Introduction to Probability Theory and Its
    Applications": for a +1/-1 game with probability p>q and capital of i units, the
    infinite-horizon ruin probability equals (q/p)^i. Used as GOLDEN VECTOR for the simulator.
Pur-Python.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence


def equity_curve(pnls: Sequence[float], start_equity: float = 0.0) -> List[float]:
    """Equity curve = initial capital + cumulative sum of P&L (includes the initial point)."""
    eq = [start_equity]
    s = start_equity
    for x in pnls:
        s += x
        eq.append(s)
    return eq


def max_drawdown(equity: Sequence[float]):
    """Returns (max_drawdown_absolute, max_drawdown_relative). DD = previous peak - trough."""
    if not equity:
        return 0.0, 0.0
    peak = equity[0]
    mdd = 0.0
    mdd_pct = 0.0
    for x in equity:
        if x > peak:
            peak = x
        dd = peak - x
        if dd > mdd:
            mdd = dd
        if peak > 0:
            ddp = dd / peak
            if ddp > mdd_pct:
                mdd_pct = ddp
    return mdd, mdd_pct


def _percentiles(vals, ps=(50, 95, 99)):
    s = sorted(vals)
    n = len(s)
    out = {}
    for p in ps:
        if n == 1:
            out[p] = s[0]
        else:
            idx = min(n - 1, max(0, int(round((p / 100.0) * (n - 1)))))
            out[p] = s[idx]
    out["max"] = s[-1]
    out["min"] = s[0]
    return out


def _resampled_pnls(pnls, mode, rng):
    n = len(pnls)
    if mode == "shuffle":
        idx = list(range(n))
        rng.shuffle(idx)
        return [pnls[i] for i in idx]
    # bootstrap (with replacement)
    return [pnls[rng.randrange(n)] for _ in range(n)]


def mc_resample(pnls: Sequence[float], mode: str = "bootstrap",
                n_paths: int = 2000, seed: int = 0) -> List[List[float]]:
    rng = random.Random(seed)
    p = [float(x) for x in pnls]
    return [_resampled_pnls(p, mode, rng) for _ in range(n_paths)]


def mc_drawdown_distribution(pnls: Sequence[float], start_equity: float = 10000.0,
                             mode: str = "bootstrap", n_paths: int = 2000,
                             seed: int = 0) -> Dict:
    """Distribution of the maximum drawdown (absolute and relative) over n_paths resamples."""
    rng = random.Random(seed)
    p = [float(x) for x in pnls]
    mdds, mdd_pcts = [], []
    for _ in range(n_paths):
        rp = _resampled_pnls(p, mode, rng)
        a, r = max_drawdown(equity_curve(rp, start_equity))
        mdds.append(a)
        mdd_pcts.append(r)
    return {"abs": _percentiles(mdds), "pct": _percentiles(mdd_pcts),
            "mode": mode, "n_paths": n_paths}


def risk_of_ruin(pnls: Sequence[float], start_equity: float = 10000.0,
                 ruin_equity: float = 0.0, mode: str = "bootstrap",
                 n_paths: int = 2000, horizon: Optional[int] = None,
                 seed: int = 0) -> float:
    """Empirical probability that equity hits ruin_equity before the end of the horizon.
    bootstrap: i.i.d. draws WITH replacement over 'horizon' steps (default = nb of trades).
    shuffle   : walks ONE permutation of the fixed game (horizon forced to len(pnls))."""
    rng = random.Random(seed)
    p = [float(x) for x in pnls]
    n = len(p)
    ruined = 0
    for _ in range(n_paths):
        if mode == "shuffle":
            seq = _resampled_pnls(p, "shuffle", rng)
        else:
            H = horizon if horizon is not None else n
            seq = [p[rng.randrange(n)] for _ in range(H)]
        eq = start_equity
        hit = False
        for x in seq:
            eq += x
            if eq <= ruin_equity:
                hit = True
                break
        if hit:
            ruined += 1
    return ruined / n_paths


def mc_summary(pnls: Sequence[float], start_equity: float = 10000.0,
               ruin_equity: float = 0.0, mode: str = "bootstrap",
               n_paths: int = 2000, seed: int = 0) -> Dict:
    rng = random.Random(seed)
    p = [float(x) for x in pnls]
    terminals, mdds, mdd_pcts = [], [], []
    ruined = 0
    for _ in range(n_paths):
        rp = _resampled_pnls(p, mode, rng)
        eq = equity_curve(rp, start_equity)
        terminals.append(eq[-1])
        a, r = max_drawdown(eq)
        mdds.append(a)
        mdd_pcts.append(r)
        if min(eq) <= ruin_equity:
            ruined += 1
    profit = sum(1 for t in terminals if t > start_equity) / n_paths
    return {"terminal_equity": _percentiles(terminals),
            "max_drawdown_abs": _percentiles(mdds),
            "max_drawdown_pct": _percentiles(mdd_pcts),
            "prob_profit": profit,
            "risk_of_ruin": ruined / n_paths,
            "mode": mode, "n_paths": n_paths}


# --------------------------------------------------------------------------- #
# Golden vector: gambler's ruin                            #
# --------------------------------------------------------------------------- #
def risk_of_ruin_analytic_bernoulli(p: float, units: int) -> float:
    """Infinite-horizon ruin for a +1/-1 game: (q/p)^units if p>0.5, else 1.0
    (unfavorable or fair game -> certain ruin at infinite horizon). Feller."""
    q = 1.0 - p
    if p <= 0.5:
        return 1.0
    return (q / p) ** units


def risk_of_ruin_bernoulli_mc(p: float, units: int, horizon: int = 3000,
                              n_paths: int = 4000, seed: int = 0) -> float:
    """MC simulator of a +1/-1 game (unit stake). Should approach the analytical formula
    for a long horizon."""
    rng = random.Random(seed)
    ruined = 0
    for _ in range(n_paths):
        eq = units
        for _ in range(horizon):
            eq += 1 if rng.random() < p else -1
            if eq <= 0:
                ruined += 1
                break
    return ruined / n_paths
