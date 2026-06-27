"""
egp_full_report.py - THE single command that runs the whole battery and produces ONE report.

Fills the "no single all-in-one command" gap. Assembles, depending on what is provided:
  - real deals             -> gate (DSR, MinTRL) + Monte-Carlo (risk_of_ruin) + costs    (ALWAYS)
  - optimization results (N configs) -> PBO + White's Reality Check + Hansen SPA          (IF provided)
  - real features          -> MODEL validation (sized vs raw gate)                         (IF provided)

Returns a structured dict + a readable/savable markdown rendering (`render_markdown`).
Pure-Python. Depends on egp_runner, egp_opt_validation, egp_real_pipeline (all tested).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import egp_runner as R
import egp_opt_validation as OV

try:
    import egp_real_pipeline as RP
except Exception:                                          # pragma: no cover
    RP = None


def full_report(deals_csv_or_path: str, *, is_path: bool = False,
                initial_deposit: float = 10000.0, signed: bool = True,
                features_csv: Optional[str] = None,
                opt_config_trades: Optional[Dict[str, Sequence[Dict]]] = None,
                opt_config_returns: Optional[Dict[str, Sequence[float]]] = None,
                thresholds: Optional[Dict] = None) -> Dict:
    """Run the full available battery and return {sections, summary}."""
    sections: Dict[str, Dict] = {}

    # 1) deals -> gate + Monte-Carlo + costs (always)
    deals = R.run_from_deals(deals_csv_or_path, is_path=is_path,
                             initial_deposit=initial_deposit, signed=signed)
    sections["deals_gate"] = deals

    # 2) optimization -> PBO / RC / SPA (if provided)
    if opt_config_trades is not None:
        _, aligned = OV.align_to_grid(opt_config_trades)
        sections["optimization"] = OV.validate_optimization(aligned, thresholds=thresholds)
    elif opt_config_returns is not None:
        sections["optimization"] = OV.validate_optimization(opt_config_returns, thresholds=thresholds)

    # 3) features -> model (if provided)
    if features_csv is not None and RP is not None:
        try:
            sections["model"] = RP.run_real_pipeline(features_csv, deals_csv_or_path,
                                                     is_path=is_path, signed=signed)
        except Exception as e:                              # pragma: no cover
            sections["model"] = {"error": str(e)}

    return {"sections": sections, "summary": _summarize(sections)}


def _summarize(sections: Dict) -> Dict:
    s: Dict = {}
    dg = sections.get("deals_gate", {}).get("manifest", {})
    s["n_trades"] = dg.get("n_trades")
    s["net_total"] = dg.get("net_total")
    s["dsr"] = dg.get("dsr")
    s["risk_of_ruin"] = dg.get("risk_of_ruin")
    s["deals_decision"] = dg.get("gate_decision")
    if "optimization" in sections:
        g = sections["optimization"]["gate"]
        s["opt_n_configs"] = sections["optimization"]["n_configs"]
        s["opt_pbo"] = g.get("pbo")
        s["opt_rc_pvalue"] = g.get("rc_pvalue")
        s["opt_spa_pvalue"] = g.get("spa_pvalue")
        s["opt_decision"] = g.get("decision")
    if "model" in sections and "model_validation" in sections["model"]:
        mv = sections["model"]["model_validation"]
        s["model_adds_value"] = mv.get("model_adds_value")
    return s


def render_markdown(report: Dict) -> str:
    """Readable/savable rendering of the report."""
    sec = report["sections"]
    out: List[str] = ["# EGP validation report", ""]

    dg = sec.get("deals_gate", {})
    man = dg.get("manifest", {})
    ver = dg.get("verdict", {})
    gate = ver.get("gate", {})
    mc = ver.get("montecarlo", {})
    out += ["## 1. Real deals - gate, risk, costs",
            f"- trades: {man.get('n_trades')}",
            f"- net_total: {man.get('net_total')}",
            f"- Sharpe: {gate.get('sharpe')}  |  DSR: {man.get('dsr')}  |  MinTRL: {gate.get('mintrl')}",
            f"- risk_of_ruin: {man.get('risk_of_ruin')}  |  prob_profit: {mc.get('prob_profit')}",
            f"- **decision: {man.get('gate_decision')}**", ""]

    if "optimization" in sec:
        o = sec["optimization"]; g = o["gate"]
        out += ["## 2. Optimization - selection overfitting",
                f"- configs tested (n_trials): {o['n_configs']}  |  periods: {o['T_periods']}",
                f"- selected config: {o['selected_config']} (by {o['select_by']})",
                f"- PBO: {g.get('pbo')}  (SELECTION overfitting)",
                f"- White's RC p-value: {g.get('rc_pvalue')}  |  Hansen SPA p-value: {g.get('spa_pvalue')}",
                f"- DSR (n_trials={o['n_configs']}): {g.get('dsr')}",
                f"- **decision: {g.get('decision')}**"]
        if g.get("reasons"):
            out += [f"- reasons: {'; '.join(g['reasons'])}"]
        out += [""]

    if "model" in sec:
        m = sec["model"]
        if "error" in m:
            out += ["## 3. Model (features)", f"- error: {m['error']}", ""]
        else:
            mv = m.get("model_validation", {})
            out += ["## 3. Model (real features) - does the ML filter add value?",
                    f"- features: {m.get('n_features')}  |  matched trades: {m.get('n_matched')}",
                    f"- raw Sharpe: {mv.get('raw_gate', {}).get('sharpe')}  |  "
                    f"Sharpe with model: {mv.get('sized_gate', {}).get('sharpe')}",
                    f"- **model adds value: {mv.get('model_adds_value')}**", ""]

    sm = report["summary"]
    out += ["## Summary", "```", str(sm), "```"]
    return "\n".join(out)
