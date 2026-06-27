"""Offline tests for the consolidated diagnostic orchestrator (deterministic)."""
import json
import math
import os
import random
import sys
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_diagnostic_report as D
import egp_costs as C


def _good_pnls(n=400, seed=0):
    rng = random.Random(seed)
    return [rng.gauss(8.0, 15.0) for _ in range(n)]


def _noise_pnls(n=400, seed=1):
    rng = random.Random(seed)
    return [rng.gauss(0.0, 30.0) for _ in range(n)]


def _spread_trades(seed=2):
    # trades spread across several sessions, profitable in each -> not fragile
    rng = random.Random(seed)
    trades = []
    for h in (3, 10, 14, 19):                      # ASIA, LONDON, OVERLAP, NEW_YORK
        for _ in range(20):
            trades.append({"time": datetime(2024, 1, 8, h, rng.randint(0, 59)),
                           "pnl": rng.gauss(5.0, 8.0)})
    return trades


def test_good_dataset_passes():
    pnls = _good_pnls()
    rep = D.build_report(pnls, sr_trials_variance=0.5/252, n_trials=5, lots=0.01,
                         trades=_spread_trades(), mc_paths=200, seed=0)
    assert rep["sections"]["gate"]["decision"] == "ACCEPT"
    assert rep["checks"]["risk_of_ruin"]["pass"] is True
    assert rep["checks"]["costs"]["pass"] is True
    assert rep["verdict"]["overall"] == "PASS", rep["verdict"]


def test_noise_dataset_reviews():
    pnls = _noise_pnls()
    rep = D.build_report(pnls, sr_trials_variance=0.5/252, n_trials=50, lots=0.1,
                         mc_paths=200, seed=1)
    assert rep["verdict"]["overall"] == "REVIEW"
    assert "gate" in rep["verdict"]["failed_checks"]  # noise -> gate fails (low DSR)


def test_report_is_json_serializable():
    rep = D.build_report(_good_pnls(), sr_trials_variance=0.5/252, n_trials=5,
                         lots=0.01, trades=_spread_trades(), mc_paths=100)
    s = json.dumps(D._clean(rep), allow_nan=False)   # must not raise (inf/nan cleaned)
    assert isinstance(s, str) and len(s) > 100


def test_markdown_has_sections():
    rep = D.build_report(_good_pnls(), sr_trials_variance=0.5/252, n_trials=5,
                         lots=0.01, trades=_spread_trades(), mc_paths=100)
    md = D.render_markdown(rep)
    for header in ("# Consolidated diagnostic report", "## Deflated gate",
                   "## Monte-Carlo robustness", "## Costs", "## Regime / session"):
        assert header in md


def test_write_report_creates_files():
    rep = D.build_report(_good_pnls(), sr_trials_variance=0.5/252, n_trials=5,
                         lots=0.01, mc_paths=100)
    with tempfile.TemporaryDirectory() as d:
        paths = D.write_report(rep, d, basename="diag")
        assert os.path.exists(paths["json"]) and os.path.exists(paths["markdown"])
        with open(paths["json"], encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["verdict"]["overall"] in ("PASS", "REVIEW")


def test_partial_inputs_omit_optional_sections():
    # without trades or prices: no regime/labels section, verdict computed on gate+RoR+costs
    rep = D.build_report(_good_pnls(), sr_trials_variance=0.5/252, n_trials=5,
                         lots=0.01, mc_paths=100)
    assert "regime" not in rep["sections"]
    assert "labels" not in rep["sections"]
    assert set(rep["checks"].keys()) == {"gate", "risk_of_ruin", "costs"}


def test_triple_barrier_section_when_prices_given():
    rng = random.Random(3)
    prices = [2000.0]
    for _ in range(200):
        prices.append(prices[-1] * (1 + rng.gauss(0.0003, 0.01)))
    sig_idx = list(range(0, 180, 5))
    sides = [1] * len(sig_idx)
    rep = D.build_report(_good_pnls(), sr_trials_variance=0.5/252, n_trials=5, lots=0.01,
                         prices=prices, signal_idx=sig_idx, signal_sides=sides, mc_paths=100)
    assert "labels" in rep["sections"]
    assert rep["sections"]["labels"]["n"] == len(sig_idx)
