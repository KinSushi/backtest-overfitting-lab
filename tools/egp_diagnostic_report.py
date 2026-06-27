"""
egp_diagnostic_report.py - CONSOLIDATED diagnostic orchestrator (single auditable report).

Assembles into ONE report (JSON-serializable dict + markdown render) the validation bricks
already built and tested:
  - GATE deflate         : egp_accept_gate.acceptance_decision (Sharpe, DSR, PBO, RC, SPA)
  - MC ROBUSTNESS        : egp_montecarlo.mc_summary (drawdown, risk of ruin, P(profit))
  - COUTS                : egp_costs.net_metrics / cost_stress / breakeven_cost_multiplier
  - REGIME/SESSION       : egp_regime.performance_by_session / edge_concentration (fragilite)
  - LABELING             : egp_triple_barrier.triple_barrier_from_signals / label_summary

Produces a CONSOLIDATED PASS/REVIEW VERDICT with the list of failed checks. No value
invented: all the numbers come from the called modules; the THRESHOLDS are parameters
explicit, to calibrate. Pure-Python. Input: per-trade pnls (+ optional timestamped trades,
prices+signals, cost model).

CAVEAT: default thresholds (ror_max, cost-survival factor, dsr_min) = leads to calibrate;
the report is an aggregator, not a guarantee.
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
import egp_accept_gate as G
import egp_montecarlo as MC
import egp_costs as C
import egp_regime as R
import egp_triple_barrier as TB


DEFAULT_THRESHOLDS = {
    "ror_max": 0.05,              # maximum tolerated risk of ruin
    "cost_survival_factor": 1.5,  # must stay profitable at +50% costs
    "dsr_min": 0.95,  # minimal deflated Sharpe (the gate also decides)
}


def _clean(obj):
    """Renders a JSON-serializable object : tuples->lists, inf/nan->None (recursive)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def build_report(pnls: Sequence[float], *, sr_trials_variance: float, n_trials: int,
                 lots: float = 0.1, cost_model: Optional["C.CostModel"] = None,
                 trades: Optional[List[Dict]] = None,
                 prices: Optional[Sequence[float]] = None,
                 signal_idx: Optional[Sequence[int]] = None,
                 signal_sides: Optional[Sequence[int]] = None,
                 start_equity: float = 10000.0, mc_paths: int = 2000, seed: int = 0,
                 thresholds: Optional[Dict] = None) -> Dict:
    """Builds the consolidated report. Optional sections are omitted if the input is missing."""
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    cm = cost_model or C.CostModel()
    report: Dict = {"n_trades": len(pnls), "thresholds": thr, "sections": {}, "checks": {}}

    # 1) DEFLATED GATE (Sharpe on the per-trade P&L series)
    gate = G.acceptance_decision(list(pnls), sr_trials_variance=sr_trials_variance,
                                 n_trials=n_trials, thresholds={"dsr_min": thr["dsr_min"]})
    report["sections"]["gate"] = gate
    report["checks"]["gate"] = {"pass": gate["decision"] == "ACCEPT",
                                "detail": f"decision={gate['decision']} DSR={gate['dsr']:.3f}"}

    # 2) ROBUSTESSE Monte-Carlo
    mc = MC.mc_summary(list(pnls), start_equity=start_equity, mode="bootstrap",
                       n_paths=mc_paths, seed=seed)
    report["sections"]["montecarlo"] = mc
    ror_ok = mc["risk_of_ruin"] <= thr["ror_max"]
    report["checks"]["risk_of_ruin"] = {"pass": ror_ok,
                                        "detail": f"RoR={mc['risk_of_ruin']:.3f} <= {thr['ror_max']}"}

    # 3) COUTS
    nm = C.net_metrics(list(pnls), lots, cm)
    stress = C.cost_stress(list(pnls), lots, cm)
    be = C.breakeven_cost_multiplier(list(pnls), lots, cm)
    report["sections"]["costs"] = {"net_metrics": nm,
                                   "stress": [list(t) for t in stress],
                                   "breakeven_multiplier": be}
    cost_ok = be >= thr["cost_survival_factor"]
    report["checks"]["costs"] = {"pass": cost_ok,
                                 "detail": f"breakeven x{be:.2f} >= x{thr['cost_survival_factor']}"}

    # 4) REGIME/SESSION (optionnel)
    if trades:
        by = R.performance_by_session(trades)
        ec = R.edge_concentration(by)
        report["sections"]["regime"] = {"by_session": by, "edge_concentration": ec}
        report["checks"]["regime"] = {"pass": not ec["fragile"],
                                      "detail": f"fragile={ec['fragile']} ({ec['reason']})"}

    # 5) triple-barrier LABELING (optional, diagnostic)
    if prices is not None and signal_idx is not None and signal_sides is not None:
        cost_ret = TB.cost_in_return_units(cm, lots, prices[signal_idx[0]] if signal_idx else 1.0)
        res = TB.triple_barrier_from_signals(prices, signal_idx, signal_sides, cost=cost_ret)
        report["sections"]["labels"] = TB.label_summary(res)

    # CONSOLIDATED VERDICT
    failed = [k for k, v in report["checks"].items() if not v["pass"]]
    report["verdict"] = {
        "overall": "PASS" if not failed else "REVIEW",
        "failed_checks": failed,
        "n_checks": len(report["checks"]),
        "reasons": [report["checks"][k]["detail"] for k in failed],
    }
    return report


def render_markdown(report: Dict) -> str:
    """Readable markdown render of the consolidated report."""
    def _f(x, fmt="{:.3f}"):
        return fmt.format(x) if isinstance(x, (int, float)) and x is not None else "n/a"

    v = report["verdict"]
    lines = ["# Consolidated diagnostic report", ""]
    lines.append(f"**Verdict: {v['overall']}** - {v['n_checks']} checks, "
                 f"{len(v['failed_checks'])} failure(s).")
    if v["failed_checks"]:
        lines.append("")
        lines.append("Failed checks: " + ", ".join(v["failed_checks"]))
        for r in v["reasons"]:
            lines.append(f"- {r}")
    lines += ["", "## Checks", "", "| Check | Result | Detail |", "|---|---|---|"]
    for k, c in report["checks"].items():
        lines.append(f"| {k} | {'PASS' if c['pass'] else 'FAIL'} | {c['detail']} |")

    g = report["sections"]["gate"]
    lines += ["", "## Deflated gate", "",
              f"- Sharpe={_f(g['sharpe'])} | DSR={_f(g['dsr'])} | PBO={_f(g['pbo'])} | "
              f"decision={g['decision']}"]
    if g.get("reasons"):
        lines.append(f"- gate reasons: {'; '.join(g['reasons'])}")

    mc = report["sections"]["montecarlo"]
    dd_med = mc["max_drawdown_pct"].get(50) if isinstance(mc["max_drawdown_pct"], dict) else mc["max_drawdown_pct"]
    eq_med = mc["terminal_equity"].get(50) if isinstance(mc["terminal_equity"], dict) else mc["terminal_equity"]
    dd_pct = dd_med * 100 if isinstance(dd_med, (int, float)) else dd_med
    lines += ["", "## Monte-Carlo robustness", "",
              f"- DD median={_f(dd_pct, '{:.1f}')}% | "
              f"RoR={_f(mc['risk_of_ruin'])} | P(profit)={_f(mc['prob_profit'], '{:.2f}')} | "
              f"final equity med={_f(eq_med, '{:.0f}')}"]

    co = report["sections"]["costs"]
    nm = co["net_metrics"]
    lines += ["", "## Costs", "",
              f"- net total={_f(nm['net_total'], '{:.1f}')} | "
              f"PF net={_f(nm['net_profit_factor'], '{:.2f}')} | "
              f"break threshold x{_f(co['breakeven_multiplier'], '{:.2f}')}"]
    lines.append("- cost stress (factor, net, survives): " +
                 "; ".join(f"x{f}->{_f(net, '{:.0f}')}/{'yes' if s else 'no'}"
                           for f, net, s in co["stress"]))

    if "regime" in report["sections"]:
        ec = report["sections"]["regime"]["edge_concentration"]
        lines += ["", "## Regime / session", "",
                  f"- profitable segments={ec['n_profitable']}/{ec['n_total']} | "
                  f"best={ec['best_segment']} ({_f(ec.get('best_share', 0), '{:.0%}')}) | "
                  f"HHI={_f(ec['hhi'], '{:.2f}')} | fragile={ec['fragile']}"]

    if "labels" in report["sections"]:
        ls = report["sections"]["labels"]
        lines += ["", "## Triple-barrier labeling", "",
                  f"- n={ls['n']} | profit={ls['pt']} stop={ls['sl']} time={ls['time']} | "
                  f"meta+={ls.get('meta_positive', 0)} (taux {_f(ls.get('meta_rate', 0), '{:.0%}')})"]

    lines += ["", "_Thresholds to calibrate; aggregator report, not a guarantee._"]
    return "\n".join(lines)


def write_report(report: Dict, out_dir: str, basename: str = "diagnostic_report") -> Dict[str, str]:
    """Writes the report as JSON (cleaned) and markdown. Returns the paths."""
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, basename + ".json")
    md_path = os.path.join(out_dir, basename + ".md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_clean(report), f, indent=2, ensure_ascii=False, allow_nan=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(report))
    return {"json": json_path, "markdown": md_path}
