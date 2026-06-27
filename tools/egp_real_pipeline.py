"""
egp_real_pipeline.py - ML MODEL validation on REAL data (features<->deals join).

After P0 (real deals -> gate on the P&L curve), this module closes the SECONDARY GAP: running the
MODEL (forest + calibration + MDA + sizing) on REAL data. It joins the features CSV
(exported by EGP_MHO_FeaturesExport.mqh, key = Position) to the deals CSV (egp_mt5_deals), builds
X (features), y (1 if the position is a net winner) and the net return per position, then:
  - trains a random forest, OOB probabilities (no leakage);
  - calibrates the probabilities (purged CV);
  - MDA importance of the REAL features;
  - sizes the bets (bet sizing) and compares the DEFLATED gate sized vs unsized
    (does the model ADD value?).

THIS MODULE INTRODUCES NO FORMULA: orchestration of VERIFIED blocks (egp_mt5_deals,
egp_tree_forest, egp_calibration, egp_cv_importance, egp_strategy_backtest, egp_accept_gate). Pure-Python.
CAVEAT: temporal alignment of concurrent positions simplified (sequential events); quality depends
entirely on the exported real features; sign convention = that of the export.
"""
from __future__ import annotations

import csv
import io
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)

import egp_mt5_deals as MD
import egp_tree_forest as TF
import egp_calibration as CAL
import egp_cv_importance as CV
import egp_strategy_backtest as SBT
import egp_accept_gate as AG


def parse_features_csv(text_or_path: str, delimiter: Optional[str] = None,
                       is_path: bool = False) -> Tuple[List[Dict], List[str]]:
    """Parses the features CSV (header Position,Time,Side,<names>). Returns (rows, feature_names).
    Each row = {position, time, side, features: {name: val}}."""
    text = open(text_or_path, encoding="utf-8", errors="replace").read() if is_path else text_or_path
    if delimiter is None:
        first = text.splitlines()[0] if text.splitlines() else ""
        delimiter = "\t" if "\t" in first else (";" if ";" in first else ",")
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    fixed = {"Position", "Time", "Side"}
    feat_names = [c for c in (reader.fieldnames or []) if c not in fixed]
    rows = []
    for r in reader:
        feats = {}
        for name in feat_names:
            feats[name] = MD._num(r.get(name))
        rows.append({"position": r.get("Position"), "time": r.get("Time"),
                     "side": MD._num(r.get("Side")), "features": feats})
    return rows, feat_names


def align_features_deals(feature_rows: Sequence[Dict], trades: Sequence[Dict], feat_names: Sequence[str]
                         ) -> Tuple[List[List[float]], List[int], List[float], List[Tuple[int, int]]]:
    """Joins features and trades by Position identifier. Returns (X, y, net_returns, events).
    y = 1 if net > 0; events = sequential events (trade order) for the purged CV."""
    # One features row per Position = the ENTRY-time row (no look-ahead). A messy export
    # (e.g. accumulated across optimization passes, or an EA logging more than once) can hold
    # several rows for the same Position; keeping the LAST would import features from a bar AFTER
    # entry. MT5 timestamps 'YYYY.MM.DD HH:MM:SS' are zero-padded and year-first, so a plain string
    # compare orders them chronologically -> pick the earliest.
    feat_by_pos: Dict[str, Dict] = {}
    for fr in feature_rows:
        pid = str(fr["position"])
        prev = feat_by_pos.get(pid)
        if prev is None or str(fr.get("time") or "") < str(prev.get("time") or ""):
            feat_by_pos[pid] = fr
    X, y, net = [], [], []
    for tr in trades:
        pid = str(tr["pos_id"])
        if pid not in feat_by_pos:
            continue
        fr = feat_by_pos[pid]
        X.append([fr["features"][n] for n in feat_names])
        net.append(float(tr["net_pnl"]))
        y.append(1 if tr["net_pnl"] > 0 else 0)
    events = [(i, i + 1) for i in range(len(X))]            # sequential (positions processed in order)
    return X, y, net, events


def _oob_proba(rf: "TF.RandomForestClassifier", X) -> List[float]:
    out = []
    for i in range(len(X)):
        acc = [0.0] * len(rf.classes_)
        cnt = 0
        for b, tree in enumerate(rf.trees_):
            if i not in rf.in_bag_[b]:
                p = tree.predict_proba([X[i]])[0]
                for c in range(len(p)):
                    acc[c] += p[c]
                cnt += 1
        row = [a / cnt for a in acc] if cnt else rf.predict_proba([X[i]])[0]
        out.append(row[1] if len(row) > 1 else row[0])
    return out


def model_gate_from_real(X, y, net_returns, events, feat_names, *, forest_params: Optional[Dict] = None,
                         n_splits: int = 5, embargo_pct: float = 0.0, calibration_method: str = "isotonic",
                         step_size: float = 0.1, sr_trials_variance: float = 0.0, n_trials: int = 1,
                         thresholds: Optional[Dict] = None, mc_paths: int = 1000, seed: int = 0) -> Dict:
    """Trains the model on real (X,y), OOB probabilities, calibration, MDA, then compares the deflated gate
    SIZED (size ~ probability) vs UN-SIZED (raw returns): does the model add value?"""
    n = len(X)
    if n < max(8, 2 * n_splits):
        return {"decision": "INSUFFICIENT_DATA", "n": n}
    fp = forest_params or {"n_estimators": 20, "max_depth": 5, "min_samples_leaf": 3}
    n_bars = n + 1

    rf = TF.RandomForestClassifier(n_estimators=fp["n_estimators"], max_depth=fp["max_depth"],
                                   min_samples_leaf=fp["min_samples_leaf"], bootstrap="uniform", seed=seed)
    rf.fit(X, y)
    proba1 = _oob_proba(rf, X)
    oob = rf.oob_score(X, y)

    # purged-CV calibration
    cv_cal = CAL.calibrate_cv(proba1, y, events, method=calibration_method, n_splits=n_splits,
                              embargo_pct=embargo_pct, n_bars=n_bars)
    calibrator = cv_cal["calibrator"]

# MDA importance of the real features
    try:
        mda = CV.mda_importance(X, y, events, n_splits=n_splits, embargo_pct=embargo_pct,
                                seed=seed, n_bars=n_bars)
        importance = {feat_names[j]: mda.get(j, {}).get("mean", 0.0) for j in range(len(feat_names))}
    except Exception as e:
        importance = {"error": str(e)}

    # SIZED backtest: size ~ calibrated probability, applied to the REAL net return per position
    bets = [{"prob": proba1[i], "side_return": net_returns[i], "t0": i, "t1": i + 1} for i in range(n)]
    sized = SBT.backtest_sized_strategy(bets, notional=1.0, step_size=step_size, calibrator=calibrator,
                                        n_bars=n_bars, sr_trials_variance=sr_trials_variance,
                                        n_trials=n_trials, thresholds=thresholds, mc_paths=mc_paths, seed=seed)

    # UN-SIZED reference: take all trades at size 1 (the raw strategy)
    raw_gate = AG.acceptance_decision(net_returns, sr_trials_variance, n_trials, thresholds=thresholds)

    sized_sharpe = sized["gate"].get("sharpe")
    raw_sharpe = raw_gate.get("sharpe")
    return {
        "n": n,
        "oob_accuracy": oob["oob_accuracy"],
        "feature_importance": importance,
        "calibration_oof_brier": cv_cal["oof_brier"],
        "sized_gate": sized["gate"],
        "raw_gate": raw_gate,
        "model_adds_value": (sized_sharpe is not None and raw_sharpe is not None
                             and sized_sharpe > raw_sharpe),
        "sized_montecarlo": sized["montecarlo"],
    }


def run_real_pipeline(features_csv: str, deals_csv: str, *, is_path: bool = False, signed: bool = True,
                      config: Optional[Dict] = None) -> Dict:
    """End-to-end on REAL data: parses features + deals, joins by Position, validates the model."""
    cfg = config or {}
    feat_rows, feat_names = parse_features_csv(features_csv, is_path=is_path)
    deals = MD.parse_deals_csv(deals_csv, is_path=is_path)
    trades = MD.trades_from_deals(deals, signed=signed)
    X, y, net, events = align_features_deals(feat_rows, trades, feat_names)
    result = model_gate_from_real(
        X, y, net, events, feat_names,
        forest_params=cfg.get("forest_params"), n_splits=cfg.get("n_splits", 5),
        embargo_pct=cfg.get("embargo_pct", 0.0), calibration_method=cfg.get("calibration_method", "isotonic"),
        step_size=cfg.get("step_size", 0.1), sr_trials_variance=cfg.get("sr_trials_variance", 0.0),
        n_trials=cfg.get("n_trials", 1), mc_paths=cfg.get("mc_paths", 500), seed=cfg.get("seed", 0))
    return {"n_features": len(feat_names), "feature_names": feat_names, "n_matched": len(X),
            "n_trades": len(trades), "model_validation": result}
