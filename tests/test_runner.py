"""Offline tests for the config runner + manifest (orchestration of verified modules)."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_runner as RUN


def _synthetic(seed=11, n=600):
    rng = random.Random(seed)
    prices = [2000.0]
    for _ in range(n):
        prices.append(prices[-1] * (1 + rng.gauss(0.0003, 0.008)))
    idx = list(range(0, n - 60, 6))
    sides = [1 if rng.random() > 0.45 else -1 for _ in idx]
    return prices, idx, sides


def test_pipeline_runs_end_to_end():
    prices, idx, sides = _synthetic()
    out = RUN.run_pipeline(prices, idx, sides)
    m = out["manifest"]
    # all sections produced
    assert m["n_events"] > 0
    assert m["chosen_d"] is not None
    assert 0.0 <= m["oob_accuracy"] <= 1.0
    assert "decision" in out["backtest"]["gate"]
    assert "risk_of_ruin" in out["backtest"]["montecarlo"]


def test_manifest_has_all_hashes():
    prices, idx, sides = _synthetic()
    out = RUN.run_pipeline(prices, idx, sides)
    m = out["manifest"]
    for k in ("config_hash", "data_hash", "code_hash", "seed", "timestamp_utc"):
        assert k in m
    assert len(m["config_hash"]) == 16 and len(m["data_hash"]) == 16 and len(m["code_hash"]) == 16


def test_same_config_same_hashes():
    prices, idx, sides = _synthetic()
    a = RUN.run_pipeline(prices, idx, sides)["manifest"]
    b = RUN.run_pipeline(prices, idx, sides)["manifest"]
    assert a["config_hash"] == b["config_hash"]
    assert a["data_hash"] == b["data_hash"]
    assert a["code_hash"] == b["code_hash"]
    assert a["chosen_d"] == b["chosen_d"]  # deterministic selection
    assert a["pnl_total"] == b["pnl_total"]  # deterministic backtest (same seed)


def test_different_data_changes_data_hash():
    p1, i1, s1 = _synthetic(seed=1)
    p2, i2, s2 = _synthetic(seed=2)
    h1 = RUN.run_pipeline(p1, i1, s1)["manifest"]["data_hash"]
    h2 = RUN.run_pipeline(p2, i2, s2)["manifest"]["data_hash"]
    assert h1 != h2


def test_different_config_changes_config_hash():
    prices, idx, sides = _synthetic()
    cfg = RUN.default_config()
    h1 = RUN.run_pipeline(prices, idx, sides, cfg)["manifest"]["config_hash"]
    cfg2 = RUN.default_config()
    cfg2["forest"]["n_estimators"] = 99
    h2 = RUN.run_pipeline(prices, idx, sides, cfg2)["manifest"]["config_hash"]
    assert h1 != h2


def test_write_manifest(tmp_path):
    prices, idx, sides = _synthetic()
    out = RUN.run_pipeline(prices, idx, sides)
    path = os.path.join(tmp_path, "manifest.json")
    RUN.write_manifest(out, path)
    assert os.path.exists(path)
    import json
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    assert m["config_hash"] == out["manifest"]["config_hash"]


def test_uniform_bootstrap_path():
    # the non-sequential path must also work
    prices, idx, sides = _synthetic()
    cfg = RUN.default_config()
    cfg["forest"]["bootstrap"] = "uniform"
    out = RUN.run_pipeline(prices, idx, sides, cfg)
    assert out["manifest"]["n_events"] > 0


def test_runner_feature_stack_increases_features():
    # the feature stack (default) must produce well more than 3 features
    prices, idx, sides = _synthetic()
    out = RUN.run_pipeline(prices, idx, sides)
    assert out["manifest"]["n_features"] > 10
    assert "ewma_vol" in out["feature_selection"]["feature_names"]


def test_runner_cv_calibration_reports_oof_brier():
    prices, idx, sides = _synthetic()
    out = RUN.run_pipeline(prices, idx, sides)
    assert out["manifest"]["calibration_oof_brier"] is not None


def test_runner_clustered_importance_option():
    prices, idx, sides = _synthetic()
    cfg = RUN.default_config()
    cfg["importance"]["clustered"] = True
    cfg["importance"]["n_clusters"] = 4
    out = RUN.run_pipeline(prices, idx, sides, cfg)
    imp = out["model"]["feature_importance"]
    assert any(k.startswith("cluster_") for k in imp) or "error" in imp


def test_runner_cpcv_gate_option():
    prices, idx, sides = _synthetic()
    cfg = RUN.default_config()
    cfg["cpcv"]["enabled"] = True
    cfg["cpcv"]["n_groups"] = 5
    out = RUN.run_pipeline(prices, idx, sides, cfg)
    assert out["cpcv_gate"] is not None
    assert "n_paths" in out["cpcv_gate"] or "error" in out["cpcv_gate"]


def test_runner_min_prob_and_exposure_cap():
    prices, idx, sides = _synthetic()
    cfg = RUN.default_config()
    cfg["bet_sizing"]["min_prob"] = 0.6
    cfg["bet_sizing"]["max_gross_exposure"] = 2.0
    out = RUN.run_pipeline(prices, idx, sides, cfg)
    assert out["manifest"]["n_events"] > 0          # runs with threshold + cap


def _deals_csv(n=50, seed=0):
    import egp_deals_export_codegen as GEN
    import random
    rng = random.Random(seed)
    lines = [",".join(GEN.HEADER)]
    tk = 1000
    for pid in range(1, n + 1):
        lines.append(",".join(str(x) for x in [f"2024.01.01 10:00:00", tk, pid, "XAUUSD", 0, 0,
                                               0.1, 2000.0, 0.0, 0.0, 0.0, 0.0])); tk += 1
        lines.append(",".join(str(x) for x in [f"2024.01.01 11:00:00", tk, pid, "XAUUSD", 1, 1,
                                               0.1, 2001.0, -0.7, -0.1, 0.0, rng.gauss(3, 5)])); tk += 1
    return "\n".join(lines)


def test_run_from_deals_real_path():
    out = RUN.run_from_deals(_deals_csv(60, seed=1))
    m = out["manifest"]
    assert m["source"] == "REAL_DEALS"
    assert m["n_trades"] == 60
    assert m["gate_decision"] is not None
    assert m["risk_of_ruin"] is not None
    assert len(m["data_hash"]) == 16 and len(m["code_hash"]) == 16


def test_run_from_deals_deterministic_hashes():
    csv = _deals_csv(40, seed=2)
    a = RUN.run_from_deals(csv)["manifest"]
    b = RUN.run_from_deals(csv)["manifest"]
    assert a["data_hash"] == b["data_hash"]
    assert a["net_total"] == b["net_total"]
