"""
egp_runner.py - Config-driven AFML pipeline orchestrator + reproducible manifest.

SINGLE entry point that chains the already-verified building blocks, in AFML order, and produces an
auditable run MANIFEST (config hash + data hash + code hash + seeds + metrics + chosen d*).
Meets the reproducibility / traceability / observability requirements.

Steps:
  1. Triple-barrier labeling               (egp_triple_barrier, ch.3)
  2. d* selection via ADF + FFD features    (egp_adf + egp_fracdiff, ch.5)
  3. Meta-labels + feature matrix
  4. Uniqueness weights                     (egp_sample_weights, ch.4)
  5. Random forest (sequential bootstrap) + OOB probabilities  (egp_tree_forest + egp_seq_bootstrap, ch.6)
  6. MDA importance                         (egp_cv_importance, ch.8)
  7. Probability calibration (hold-out)     (egp_calibration)
  8. Bet sizing + concurrency-aware backtest -> deflated gate + Monte-Carlo
                                            (egp_bet_sizing + egp_strategy_backtest, ch.10)
  9. Hash manifest.

THIS MODULE INTRODUCES NO FORMULA: pure plumbing of verified modules. Pure-Python (hashlib/json).
CAVEAT: on synthetic data for the tests; in production, feed the REAL deal series (P0 lock). The gate
thresholds and the MHO trial count (n_trials) are config inputs.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)

import egp_triple_barrier as TB
import egp_adf as ADF
import egp_fracdiff as FD
import egp_sample_weights as SW
import egp_tree_forest as TF
import egp_cv_importance as CV
import egp_calibration as CAL
import egp_strategy_backtest as SBT
import egp_features as FEAT
import egp_cluster_importance as CI
import egp_cpcv_model as CM
import egp_real_gate as RG


def default_config() -> Dict:
    """Default config (all stages enabled). To be overridden by the caller."""
    return {
        "seed": 0,
        "triple_barrier": {"span": 20, "pt_mult": 1.5, "sl_mult": 1.5, "max_horizon": 12, "cost": 0.0003},
        "fracdiff": {"d_grid": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0], "thresh": 1e-4,
                     "regression": "c", "level": 5},
        "features": {"use_feature_stack": True, "windows": [5, 20, 60], "fracdiff_d": [0.3, 0.5],
                     "fracdiff_thresh": 1e-4, "ewma_span": 20, "include_skew_kurt": True,
                     "include_acf": True},
        "forest": {"n_estimators": 20, "max_depth": 5, "min_samples_leaf": 5,
                   "bootstrap": "sequential"},
        "cv": {"n_splits": 5, "embargo_pct": 0.01},
        "calibration": {"method": "isotonic", "cross_validated": True},
        "importance": {"clustered": False, "n_clusters": None, "threshold": None},
        "cpcv": {"enabled": False, "n_groups": 6, "k_test": 2, "purge": 0, "embargo": 0},
        "bet_sizing": {"step_size": 0.1, "notional": 1000.0, "min_prob": 0.0,
                       "max_gross_exposure": None},
        "gate": {"sr_trials_variance": 0.5, "n_trials": 50, "thresholds": None},
        "montecarlo": {"paths": 500},
    }


def _sha16(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def _hash_obj(obj) -> str:
    return _sha16(json.dumps(obj, sort_keys=True, default=str).encode())


def _hash_data(prices, idx, sides) -> str:
    return _sha16(repr((list(prices), list(idx), list(sides))).encode())


def _hash_code(tools_dir: str) -> str:
    h = hashlib.sha256()
    for f in sorted(os.listdir(tools_dir)):
        if f.endswith(".py"):
            with open(os.path.join(tools_dir, f), "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()[:16]


def _oob_proba(rf: "TF.RandomForestClassifier", X) -> List[List[float]]:
    """OOB probabilities per observation (each obs predicted by the trees that did not draw it)."""
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
        out.append([a / cnt for a in acc] if cnt else rf.predict_proba([X[i]])[0])
    return out


def run_pipeline(prices: Sequence[float], signal_idx: Sequence[int], signal_sides: Sequence[int],
                 config: Optional[Dict] = None) -> Dict:
    """Runs the full pipeline and returns {manifest, stages}."""
    cfg = config or default_config()
    seed = cfg.get("seed", 0)
    n_bars = len(prices)
    t_start = time.time()

    # 1) triple-barrier labeling
    tbp = cfg["triple_barrier"]
    tb = TB.triple_barrier_from_signals(prices, signal_idx, signal_sides, span=tbp["span"],
                                        pt_mult=tbp["pt_mult"], sl_mult=tbp["sl_mult"],
                                        max_horizon=tbp["max_horizon"], cost=tbp["cost"])
    events = [(r["t0"], r["touch_idx"]) for r in tb]

    # 2) d* selection (ADF) + fracdiff feature at d*
    fdp = cfg["fracdiff"]
    sel = ADF.min_ffd_order(prices, d_grid=fdp["d_grid"], thresh=fdp["thresh"],
                            regression=fdp["regression"], level=fdp["level"])
    d_star = sel["chosen_d"] if sel["chosen_d"] is not None else 0.5
    ffd = FD.fracdiff_feature(prices, [r["t0"] for r in tb], d=d_star, thresh=fdp["thresh"])

    # 3) feature matrix + meta-labels (1 if the bet in the primary direction wins)
    y = [1 if tb[i]["side_return"] > 0 else 0 for i in range(len(tb))]
    fcfg = cfg.get("features", {})
    if fcfg.get("use_feature_stack", False):
        feat_config = {"windows": fcfg["windows"], "fracdiff_d": fcfg["fracdiff_d"],
                       "fracdiff_thresh": fcfg["fracdiff_thresh"], "ewma_span": fcfg["ewma_span"],
                       "include_skew_kurt": fcfg["include_skew_kurt"],
                       "include_acf": fcfg["include_acf"]}
        X, feat_names = FEAT.build_features(prices, [r["t0"] for r in tb], feat_config)
    else:
        X = [[ffd[i], float(tb[i]["side"]), tb[i]["sigma"]] for i in range(len(tb))]
        feat_names = ["fracdiff", "side", "sigma"]

    # 4) uniqueness weights
    weights = SW.average_uniqueness(events, n_bars)

    # 5) random forest (sequential draw) + OOB probabilities
    fp = cfg["forest"]
    rf = TF.RandomForestClassifier(n_estimators=fp["n_estimators"], max_depth=fp["max_depth"],
                                   min_samples_leaf=fp["min_samples_leaf"],
                                   bootstrap=fp["bootstrap"], seed=seed)
    if fp["bootstrap"] == "sequential":
        rf.fit(X, y, events=events, n_bars=n_bars, sample_weight=weights)
    else:
        rf.fit(X, y, sample_weight=weights)
    oob_p = _oob_proba(rf, X)
    proba1 = [p[1] for p in oob_p]                 # probability of the "winning bet" class
    oob = rf.oob_score(X, y)

    # 6) importance: MDA, or clustered cMDA (robust to the substitution effect)
    icfg = cfg.get("importance", {})
    try:
        if icfg.get("clustered", False):
            cm = CI.clustered_mda(X, y, events, n_clusters=icfg.get("n_clusters"),
                                  threshold=icfg.get("threshold"), n_splits=cfg["cv"]["n_splits"],
                                  embargo_pct=cfg["cv"]["embargo_pct"], seed=seed, n_bars=n_bars)
            importance = {f"cluster_{ci}": {"mean": d["mean"],
                                            "members": [feat_names[m] for m in d["members"]]}
                          for ci, d in cm["importance"].items()}
        else:
            mda = CV.mda_importance(X, y, events, n_splits=cfg["cv"]["n_splits"],
                                    embargo_pct=cfg["cv"]["embargo_pct"], seed=seed, n_bars=n_bars)
            importance = {feat_names[j]: mda.get(j, {}).get("mean", 0.0)
                          for j in range(len(feat_names))}
    except Exception as e:
        importance = {"error": str(e)}

    # 7) OOB probability calibration: purged CV (default) or simple hold-out
    ccfg = cfg["calibration"]
    cal_method = ccfg["method"]
    if ccfg.get("cross_validated", False):
        cv_cal = CAL.calibrate_cv(proba1, y, events, method=cal_method,
                                  n_splits=cfg["cv"]["n_splits"],
                                  embargo_pct=cfg["cv"]["embargo_pct"], n_bars=n_bars)
        calibrator = cv_cal["calibrator"]
        cal_gain = CAL.calibration_gain(proba1, y, calibrator)
        cal_gain["oof_brier"] = cv_cal["oof_brier"]
    else:
        half = max(2, len(proba1) // 2)
        calibrator = CAL.calibrate(proba1[:half], y[:half], method=cal_method)
        cal_gain = CAL.calibration_gain(proba1, y, calibrator)

    # 8) bet sizing + concurrency-aware backtest -> gate + Monte-Carlo
    bsp = cfg["bet_sizing"]
    bets = SBT.bets_from_triple_barrier(tb, proba1)
    backtest = SBT.backtest_sized_strategy(
        bets, notional=bsp["notional"], step_size=bsp["step_size"], calibrator=calibrator,
        n_bars=n_bars, min_prob=bsp.get("min_prob", 0.0),
        max_gross_exposure=bsp.get("max_gross_exposure"),
        sr_trials_variance=cfg["gate"]["sr_trials_variance"],
        n_trials=cfg["gate"]["n_trials"], thresholds=cfg["gate"]["thresholds"],
        mc_paths=cfg["montecarlo"]["paths"], seed=seed)

    # 8b) optional CPCV gate (out-of-sample paths distribution)
    cpcvc = cfg.get("cpcv", {})
    cpcv_gate = None
    if cpcvc.get("enabled", False):
        try:
            cpcv_gate = CM.cpcv_model_gate(
                X, y, bets, events, n_groups=cpcvc["n_groups"], k_test=cpcvc["k_test"],
                purge=cpcvc["purge"], embargo=cpcvc["embargo"],
                forest_params={"n_estimators": fp["n_estimators"], "max_depth": fp["max_depth"],
                               "min_samples_leaf": fp["min_samples_leaf"]},
                step_size=bsp["step_size"], n_trials=cfg["gate"]["n_trials"], seed=seed)
        except Exception as e:
            cpcv_gate = {"error": str(e)}

    # 9) manifeste
    manifest = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t_start, 3),
        "config_hash": _hash_obj(cfg),
        "data_hash": _hash_data(prices, signal_idx, signal_sides),
        "code_hash": _hash_code(_HERE),
        "seed": seed,
        "n_events": len(tb),
        "n_features": len(feat_names),
        "chosen_d": d_star,
        "oob_accuracy": oob["oob_accuracy"],
        "calibration_brier_raw": cal_gain["brier_raw"],
        "calibration_brier_calibrated": cal_gain["brier_calibrated"],
        "calibration_oof_brier": cal_gain.get("oof_brier"),
        "gate_decision": backtest["gate"].get("decision"),
        "cpcv_gate_decision": (cpcv_gate or {}).get("gate", {}).get("decision") if cpcv_gate else None,
        "pnl_total": backtest["pnl_total"],
        "risk_of_ruin": backtest["montecarlo"].get("risk_of_ruin"),
    }
    return {
        "manifest": manifest,
        "labeling": {"n_events": len(tb), "events": events},
        "feature_selection": {"chosen_d": d_star, "adf_table": sel["table"],
                              "feature_names": feat_names},
        "model": {"oob": oob, "feature_importance": importance},
        "calibration": cal_gain,
        "backtest": backtest,
        "cpcv_gate": cpcv_gate,
    }


def run_from_deals(deals_csv_or_path: str, config: Optional[Dict] = None, *,
                   is_path: bool = False, initial_deposit: float = 10000.0, signed: bool = True,
                   colmap: Optional[Dict] = None, delimiter: Optional[str] = None) -> Dict:
    """REAL-DATA entry point (P0 lock lifted): ingests a deals CSV exported by
    EGP_MHO_DealsExport.mqh and produces the AFML verdict (deflated gate + Monte-Carlo + costs) on the
    REAL deals series, with a hashed manifest. To contrast with run_pipeline (synthetic)."""
    cfg = config or default_config()
    seed = cfg.get("seed", 0)
    t_start = time.time()
    text = open(deals_csv_or_path, encoding="utf-8", errors="replace").read() if is_path \
        else deals_csv_or_path
    verdict = RG.gate_from_deals_csv(
        text, colmap=colmap, delimiter=delimiter, initial_deposit=initial_deposit, signed=signed,
        sr_trials_variance=cfg["gate"]["sr_trials_variance"], n_trials=cfg["gate"]["n_trials"],
        thresholds=cfg["gate"]["thresholds"], mc_paths=cfg["montecarlo"]["paths"], seed=seed)
    manifest = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t_start, 3),
        "source": "REAL_DEALS",
        "config_hash": _hash_obj(cfg),
        "data_hash": _sha16(text.encode("utf-8", "replace")),
        "code_hash": _hash_code(_HERE),
        "seed": seed,
        "n_trades": verdict.get("n_trades"),
        "net_total": verdict.get("net_total"),
        "total_costs": verdict.get("total_costs"),
        "gate_decision": verdict.get("gate", {}).get("decision"),
        "dsr": verdict.get("gate", {}).get("dsr"),
        "risk_of_ruin": verdict.get("montecarlo", {}).get("risk_of_ruin"),
    }
    return {"manifest": manifest, "verdict": verdict}


def write_manifest(result: Dict, path: str) -> None:
    """Writes the JSON manifest (auditable / reproducible)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result["manifest"], f, indent=2, default=str)
