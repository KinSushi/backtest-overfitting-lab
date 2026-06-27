"""
egp_strategy_backtest.py - Concurrency-aware backtest of the SIZED strategy -> gate.

Closes the loop: instead of evaluating the gate (egp_accept_gate) on raw labels, we
evaluate the REAL output of the meta-labeling + bet sizing pipeline. For each bet:
  size m = prob_to_size(calibrated proba)  (egp_bet_sizing / egp_calibration, AFML ch.10)
  effective size = max(0, m)               (meta-labeling: bet only if conviction > 1/K)
  net return = side_return - cost          (cost in return units, egp_costs/triple_barrier)
  bet P&L = effective_size * net_return * notional
The P&L series feeds the deflated gate (Sharpe/DSR/PBO) and the Monte-Carlo (risk of ruin,
drawdown). Concurrency (simultaneously active bets) is measured via egp_sample_weights.

THIS MODULE CONTAINS NO NEW FORMULA: it orchestrates ALREADY-VERIFIED building blocks
(egp_bet_sizing, egp_calibration, egp_costs, egp_triple_barrier, egp_sample_weights,
egp_accept_gate, egp_montecarlo). Pure-Python.

CAVEAT: the per-bet returns must come from a REAL backtest (P0 lock: series of MT5
deals). Here, they are provided as input (from the triple-barrier on synthetic data for the
tests). The gate requires sr_trials_variance + n_trials (number of MHO trials) for deflation.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)

import egp_bet_sizing as BS
import egp_accept_gate as AG
import egp_montecarlo as MC
import egp_sample_weights as SW
import egp_triple_barrier as TB


def _effective_size(prob: float, step_size: float, num_classes: int) -> float:
    """Effective size of a bet (meta-labeling): max(0, prob_to_size) optionally discretized."""
    m = BS.prob_to_size(prob, num_classes)
    if step_size > 0:
        m = BS.discretize_signal(m, step_size)
    return max(0.0, m)


def backtest_sized_strategy(bets: Sequence[Dict], *, cost_model=None, notional: float = 1.0,
                            step_size: float = 0.0, calibrator=None, num_classes: int = 2,
                            n_bars: Optional[int] = None, start_equity: float = 10000.0,
                            min_prob: float = 0.0, max_gross_exposure: Optional[float] = None,
                            sr_trials_variance: float = 0.0, n_trials: int = 1,
                            thresholds: Optional[Dict] = None, mc_paths: int = 1000,
                            seed: int = 0) -> Dict:
    """Backtest of the sized strategy. `bets` = list of dicts
    {prob, side_return, t0, t1, price?, lots?}. Returns P&L, equity, concurrency, gate, Monte-Carlo.
    min_prob: conviction threshold (bet only if calibrated proba >= min_prob).
    max_gross_exposure: cap on aggregated GROSS exposure at entry (risk budgeting, AFML ch.10);
        if opening a bet would exceed the cap given the still-open bets,
        its size is reduced to fit (causal rule, decided at entry)."""
    # 1) desired size per bet (calibrated proba, conviction threshold)
    desired = []
    for b in bets:
        prob = b["prob"]
        if calibrator is not None:
            prob = calibrator.predict([prob])[0]
        size = 0.0 if prob < min_prob else _effective_size(prob, step_size, num_classes)
        if cost_model is not None and "price" in b:
            cost = TB.cost_in_return_units(cost_model, b.get("lots", 1.0), b["price"])
        else:
            cost = b.get("cost", 0.0)
        desired.append({"size": size, "cost": cost, "t0": b["t0"], "t1": b["t1"],
                        "side_return": b["side_return"]})

    # 2) gross exposure cap at entry (processed in t0 order)
    if max_gross_exposure is not None:
        order = sorted(range(len(desired)), key=lambda i: desired[i]["t0"])
        open_bets = []  # (t1, capped_size)
        for i in order:
            d = desired[i]
            t0 = d["t0"]
            open_bets = [ob for ob in open_bets if ob[0] > t0]      # still open at t0
            current_gross = sum(s for _, s in open_bets)
            allowed = max(0.0, max_gross_exposure - current_gross)
            capped = min(d["size"], allowed)
            d["size"] = capped
            if capped > 0:
                open_bets.append((d["t1"], capped))

    # 3) P&L per bet from the final sizes
    pnls: List[float] = []
    rets: List[float] = []
    active_events = []
    n_active = 0
    for d in desired:
        ret = d["size"] * (d["side_return"] - d["cost"])
        pnls.append(ret * notional)
        rets.append(ret)
        if d["size"] > 0:
            n_active += 1
            active_events.append((d["t0"], d["t1"]))


    # concurrency of the bets actually taken
    if active_events:
        nb = n_bars if n_bars is not None else max(t1 for _, t1 in active_events) + 1
        conc = SW.concurrency(active_events, nb)
        nonzero = [c for c in conc if c > 0]
        conc_avg = sum(nonzero) / len(nonzero) if nonzero else 0.0
        conc_max = max(conc) if conc else 0
    else:
        conc_avg, conc_max = 0.0, 0

    equity = [start_equity]
    for p in pnls:
        equity.append(equity[-1] + p)

    # deflated gate + Monte-Carlo (verified building blocks)
    gate = AG.acceptance_decision(rets, sr_trials_variance, n_trials, thresholds=thresholds) \
        if len(rets) >= 2 else {"decision": "INSUFFICIENT_DATA"}
    mc = MC.mc_summary(pnls, start_equity=start_equity, n_paths=mc_paths, seed=seed) \
        if len(pnls) >= 2 else {}

    return {
        "n_bets": len(bets),
        "n_active": n_active,
        "pnl_total": sum(pnls),
        "equity_final": equity[-1],
        "equity_curve": equity,
        "concurrency": {"avg": conc_avg, "max": conc_max},
        "gate": gate,
        "montecarlo": mc,
    }


def bets_from_triple_barrier(tb_results: Sequence[Dict], probs: Sequence[float],
                             prices: Optional[Sequence[float]] = None) -> List[Dict]:
    """Builds the `bets` list from the triple-barrier outputs + meta probas (+ entry price)."""
    bets = []
    for i, r in enumerate(tb_results):
        bet = {"prob": probs[i], "side_return": r["side_return"], "t0": r["t0"], "t1": r["touch_idx"]}
        if prices is not None:
            bet["price"] = prices[r["t0"]]
        bets.append(bet)
    return bets


def sweep_sizing(bets: Sequence[Dict], *, min_probs: Sequence[float] = (0.5, 0.55, 0.6, 0.65),
                 step_sizes: Sequence[float] = (0.0, 0.1, 0.2), cost_model=None, notional: float = 1.0,
                 calibrator=None, num_classes: int = 2, n_bars: Optional[int] = None,
                 max_gross_exposure: Optional[float] = None, sr_trials_variance: float = 0.0,
                 n_trials: int = 1, thresholds: Optional[Dict] = None, mc_paths: int = 200,
                 seed: int = 0) -> Dict:
    """Sweep (conviction threshold x discretization step) optimized on the DSR (DEFLATED Sharpe),
    not on the raw Sharpe -> avoids re-overfitting at the sizing layer.

    The sweep's number of trials increases multiplicity: it is added to n_trials passed to the gate
    (honest deflation). Returns {table, best}; best = combination with maximal DSR."""
    combos = [(mp, ss) for mp in min_probs for ss in step_sizes]
    n_sweep = len(combos)
    table = []
    best = None
    for mp, ss in combos:
        bt = backtest_sized_strategy(
            bets, cost_model=cost_model, notional=notional, step_size=ss, calibrator=calibrator,
            num_classes=num_classes, n_bars=n_bars, min_prob=mp, max_gross_exposure=max_gross_exposure,
            sr_trials_variance=sr_trials_variance, n_trials=n_trials + n_sweep, thresholds=thresholds,
            mc_paths=mc_paths, seed=seed)
        gate = bt["gate"]
        dsr = gate.get("dsr", float("nan"))
        row = {"min_prob": mp, "step_size": ss, "dsr": dsr,
               "sharpe": gate.get("sharpe"), "decision": gate.get("decision"),
               "pnl_total": bt["pnl_total"], "n_active": bt["n_active"]}
        table.append(row)
        if (best is None) or (dsr == dsr and dsr > best["dsr"]):   # ignore NaN
            best = row
    return {"table": table, "best": best, "n_combos": n_sweep}
