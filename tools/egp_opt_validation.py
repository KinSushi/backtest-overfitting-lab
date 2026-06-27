"""
egp_opt_validation.py - Closes the OPTIMIZATION -> VALIDATION loop.

Problem solved: the battery (`acceptance_decision`) already accepts `perf_matrix` (PBO) and
`loss_diff_matrix` (White's Reality Check, Hansen SPA), but NOTHING assembled these matrices
from several tested configurations. Without that, PBO/RC/SPA stay `null` (cf. `run_from_deals`).

This module:
  1. `align_to_grid`: N configs -> per-period returns on a COMMON date GRID
     (each config has its own set of deals/trades; aligned by day, gaps = 0).
  2. `build_matrices`: -> perf_matrix (T x C) and loss_diff_matrix (perf - benchmark).
  3. `validate_optimization`: picks the BEST config and runs the full battery with
     `n_trials = number of configs` (the HONEST trial count), `sr_trials_variance` = variance
     of the Sharpes across configs. Returns PBO/RC/SPA/DSR + a SELECTION overfitting diagnostic.

Convention (verified in egp_accept_gate):
  - perf_matrix[t][c]      = return of config c at instant t.
  - loss_diff_matrix[t][c] = perf_c(t) - benchmark(t)  (>0 = beats the benchmark; default benchmark=0).

Pure-Python. Depends on egp_accept_gate (already tested).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import egp_accept_gate as G


def _date_of(ts: object) -> str:
    """Extracts the date 'YYYY.MM.DD' from an MT5 timestamp 'YYYY.MM.DD HH:MM:SS' (or returns str)."""
    s = str(ts)
    return s[:10] if len(s) >= 10 else s


def align_to_grid(config_trades: Dict[str, Sequence[Dict]], *, pnl_key: str = "net_pnl",
                  time_key: str = "exit_time") -> Tuple[List[str], Dict[str, List[float]]]:
    """N configs -> (sorted date grid, {config: aligned returns}).
    Each config = list of trades (dict with `net_pnl` and `exit_time`). The P&L is summed per
    day; any date missing from a config is filled with 0 (the config did not trade that day)."""
    per_cfg_daily: Dict[str, Dict[str, float]] = {}
    all_dates = set()
    for cfg, trades in config_trades.items():
        daily: Dict[str, float] = {}
        for t in trades:
            d = _date_of(t.get(time_key))
            daily[d] = daily.get(d, 0.0) + float(t.get(pnl_key, 0.0))
        per_cfg_daily[cfg] = daily
        all_dates.update(daily.keys())
    grid = sorted(all_dates)
    aligned = {cfg: [per_cfg_daily[cfg].get(d, 0.0) for d in grid] for cfg in config_trades}
    return grid, aligned


def build_matrices(config_returns: Dict[str, Sequence[float]],
                   benchmark: Optional[Sequence[float]] = None
                   ) -> Tuple[List[List[float]], List[List[float]], List[str]]:
    """{config: returns (same length T)} -> (perf_matrix TxC, loss_diff_matrix TxC, config_ids)."""
    cfg_ids = list(config_returns.keys())
    lengths = {len(config_returns[c]) for c in cfg_ids}
    if len(lengths) != 1:
        raise ValueError(f"returns of different lengths: {lengths} (align first via align_to_grid)")
    T = lengths.pop()
    if benchmark is None:
        benchmark = [0.0] * T
    if len(benchmark) != T:
        raise ValueError("benchmark of wrong length")
    perf = [[float(config_returns[c][t]) for c in cfg_ids] for t in range(T)]
    loss = [[float(config_returns[c][t]) - float(benchmark[t]) for c in cfg_ids] for t in range(T)]
    return perf, loss, cfg_ids


def _sharpe(xs: Sequence[float]) -> float:
    return G.sharpe_ratio(list(xs))


def sr_trials_variance(config_returns: Dict[str, Sequence[float]]) -> float:
    """Variance of the Sharpes across configs (per-period scale) -> feeds the DSR deflation."""
    srs = [_sharpe(v) for v in config_returns.values()]
    srs = [s for s in srs if s == s]                       # drop NaN
    if len(srs) < 2:
        return 0.0
    m = sum(srs) / len(srs)
    return sum((s - m) ** 2 for s in srs) / (len(srs) - 1)


def validate_optimization(config_returns: Dict[str, Sequence[float]], *,
                          benchmark: Optional[Sequence[float]] = None,
                          thresholds: Optional[Dict] = None,
                          select_by: str = "sharpe") -> Dict:
    """Validates an OPTIMIZATION (several aligned configs). Picks the best and runs the
    full battery with n_trials = number of configs.
    select_by: 'sharpe' (default) or 'total' (sum of returns)."""
    cfg_ids = list(config_returns.keys())
    if len(cfg_ids) < 2:
        raise ValueError("at least 2 configs required for PBO/RC/SPA")
    perf, loss, ids = build_matrices(config_returns, benchmark)
    n_trials = len(ids)
    V = sr_trials_variance(config_returns)
    # IN-SAMPLE selection (this is THE choice whose overfitting we measure)
    if select_by == "total":
        best = max(ids, key=lambda c: sum(config_returns[c]))
    else:
        best = max(ids, key=lambda c: (_sharpe(config_returns[c]) if _sharpe(config_returns[c]) == _sharpe(config_returns[c]) else -1e9))
    gate = G.acceptance_decision(list(config_returns[best]), V, n_trials,
                                 perf_matrix=perf, loss_diff_matrix=loss, thresholds=thresholds)
    return {
        "n_configs": n_trials,
        "T_periods": len(perf),
        "sr_trials_variance": V,
        "selected_config": best,
        "select_by": select_by,
        "per_config_sharpe": {c: _sharpe(config_returns[c]) for c in ids},
        "gate": gate,                       # contains dsr, pbo, rc_pvalue, spa_pvalue, decision, reasons
        "selection_overfit": (gate.get("pbo") is not None and gate["pbo"] > (thresholds or {}).get("pbo_max", 0.20)),
    }
