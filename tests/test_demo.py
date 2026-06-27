"""Smoke test of the demo: the synthetic generators + full_report produce a complete report.
(We do not run the slow run_pipeline here; it is validated separately.)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo"))
import run_demo as D
import egp_full_report as FR


def test_demo_generators_feed_full_report(tmp_path):
    truth = D.gen_ground_truth(n=140, seed=7)
    deals = D.write_deals_csv(truth, str(tmp_path / "d.csv"))
    feats = D.write_features_csv(truth, str(tmp_path / "f.csv"))
    opt = D.gen_opt_configs(n_cfg=12, n_trades=60, seed=21)
    r = FR.full_report(deals, is_path=True, features_csv=feats, opt_config_trades=opt)
    sec = r["sections"]
    assert "deals_gate" in sec
    assert "optimization" in sec
    assert "model" in sec
    assert r["summary"]["n_trades"] == 140          # depot exclu
    assert r["summary"]["opt_pbo"] is not None
    assert r["summary"]["opt_n_configs"] == 12


def test_demo_deposit_is_filtered(tmp_path):
    truth = D.gen_ground_truth(n=60, seed=1)
    deals = D.write_deals_csv(truth, str(tmp_path / "d.csv"))
    r = FR.full_report(deals, is_path=True)
    assert r["summary"]["n_trades"] == 60  # 60 trades, not 61 (deposit Type=2 removed)


def test_demo_render_markdown(tmp_path):
    truth = D.gen_ground_truth(n=80, seed=2)
    deals = D.write_deals_csv(truth, str(tmp_path / "d.csv"))
    opt = D.gen_opt_configs(n_cfg=10, n_trades=50, seed=3)
    r = FR.full_report(deals, is_path=True, opt_config_trades=opt)
    md = FR.render_markdown(r)
    assert "EGP validation report" in md
    assert "Optimization" in md
