"""
egp_real_gate.py - REAL MT5 deals -> AFML verdict (deflated gate + Monte-Carlo + costs).

This is the CULMINATION of the P0 lock: applies the validation arsenal to the REAL output of the
strategy (the deals exported by EGP_MHO_DealsExport.mqh), and not synthetic data.

Chain: deals CSV -> parse (egp_mt5_deals.parse_deals_csv) -> trades by position (net = profit
+ commission + swap + fee) -> per-trade P&L series -> deflated gate (egp_accept_gate:
Sharpe/DSR/PBO) + Monte-Carlo (egp_montecarlo: risk of ruin, drawdown) + cost decomposition.

Bonus (merging the optimizer<->validation worlds): `rank_candidates_by_real_gate` ranks
MHO by their REAL gate (deflated Sharpe, net P&L), not by an aggregate PF -> more honest selection.

THIS MODULE INTRODUCES NO FORMULA: it orchestrates VERIFIED bricks (egp_mt5_deals,
egp_accept_gate, egp_montecarlo). Pur-Python.
CAVEAT: the sign convention (signed commission/swap/fee vs magnitudes) MUST be confirmed on
a real export (`signed` parameter); gate critical values to calibrate; sr_trials_variance +
n_trials (MHO trials) required for the deflation.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Union

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)

import egp_mt5_deals as MD
import egp_accept_gate as AG
import egp_montecarlo as MC


def gate_from_trade_returns(returns: Sequence[float], *, initial_deposit: float = 10000.0,
                            sr_trials_variance: float = 0.0, n_trials: int = 1,
                            thresholds: Optional[Dict] = None, mc_paths: int = 2000,
                            ruin_equity: float = 0.0, seed: int = 0) -> Dict:
    """AFML verdict from a per-trade P&L series (already net): deflated gate + Monte-Carlo."""
    if len(returns) < 2:
        return {"decision": "INSUFFICIENT_DATA", "n_trades": len(returns)}
    gate = AG.acceptance_decision(returns, sr_trials_variance, n_trials, thresholds=thresholds)
    mc = MC.mc_summary(returns, start_equity=initial_deposit, ruin_equity=ruin_equity,
                       n_paths=mc_paths, seed=seed)
    return {"n_trades": len(returns), "net_total": sum(returns),
            "equity_final": initial_deposit + sum(returns), "gate": gate, "montecarlo": mc}


def gate_from_deals(deals: Sequence[Dict], *, initial_deposit: float = 10000.0, signed: bool = True,
                    sr_trials_variance: float = 0.0, n_trials: int = 1,
                    thresholds: Optional[Dict] = None, mc_paths: int = 2000,
                    ruin_equity: float = 0.0, seed: int = 0) -> Dict:
    """AFML verdict from canonical deals: groups by position, computes net returns,
    then applies gate + Monte-Carlo. Adds the gross/net/costs decomposition."""
    trades = MD.trades_from_deals(deals, signed=signed)
    net = MD.pnl_series(trades, net=True)
    gross = MD.pnl_series(trades, net=False)
    res = gate_from_trade_returns(net, initial_deposit=initial_deposit,
                                  sr_trials_variance=sr_trials_variance, n_trials=n_trials,
                                  thresholds=thresholds, mc_paths=mc_paths, ruin_equity=ruin_equity,
                                  seed=seed)
    total_gross = sum(gross)
    total_net = sum(net)
    res.update({"gross_total": total_gross, "total_costs": total_gross - total_net,
                "n_trades": len(trades)})
    return res


def gate_from_deals_csv(csv_text_or_path: str, *, colmap: Optional[Dict] = None,
                        delimiter: Optional[str] = None, is_path: bool = False, **kwargs) -> Dict:
    """AFML verdict directly from a CSV exported by EGP_MHO_DealsExport.mqh."""
    deals = MD.parse_deals_csv(csv_text_or_path, colmap=colmap, delimiter=delimiter, is_path=is_path)
    return gate_from_deals(deals, **kwargs)


def _rank_key(r: Dict):
    """Sort key: ACCEPT first, then descending DSR, then descending net P&L."""
    g = r.get("gate", {})
    accepted = 0 if g.get("decision") == "ACCEPT" else 1
    dsr = g.get("dsr")
    dsr = dsr if (dsr is not None and dsr == dsr) else -9.0
    return (accepted, -dsr, -r.get("net_total", 0.0))


def rank_candidates_by_real_gate(candidate_deals: Dict[Union[str, int], Union[Sequence[Dict], str]],
                                 *, parse_csv: bool = False, **kwargs) -> Dict:
    """Ranks MHO candidates by their REAL gate. `candidate_deals`: {id -> deals|CSV}.
    parse_csv=True if values are CSV text. Connects the optimizer to the validation arsenal."""
    results = {}
    for cid, deals in candidate_deals.items():
        d = MD.parse_deals_csv(deals) if parse_csv else deals
        results[cid] = gate_from_deals(d, **kwargs)
    ranked = sorted(results.keys(), key=lambda cid: _rank_key(results[cid]))
    return {"ranked": ranked, "best": ranked[0] if ranked else None, "results": results}
