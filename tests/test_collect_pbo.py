# -*- coding: utf-8 -*-
"""Covers demo/collect_pbo.py: build the TxC matrix from N MT5 deals CSVs and run PBO/RC/SPA.

conftest.py adds tools/ and demo/ to sys.path, so both `collect_pbo` (demo) and the tools modules
import directly.
"""
import csv
import os
import random

from egp_mt5_collect import (build_opt_config_trades, features_rows_per_position,
                             mt5_common_files_dir)
from egp_full_report import full_report

_HEADER = ["Time", "Deal", "Position", "Symbol", "Type", "Direction",
           "Volume", "Price", "Commission", "Swap", "Fee", "Profit"]


def _write_config_csv(path, trades):
    """trades: list of (day_index, profit). Writes one MT5-format deals CSV (with a balance row)."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(_HEADER)
        w.writerow(["2026.01.01 00:00:00", 1, 0, "", 2, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 10000.0])  # balance
        deal, pos = 2, 2
        for di, profit in trades:
            day = f"2026.{1 + di // 28:02d}.{1 + di % 28:02d}"
            w.writerow([f"{day} 08:00:00", deal, pos, "XAUUSD", 0, 0, 0.01, 2000.0, -0.03, 0.0, 0.0, 0.0])
            deal += 1
            w.writerow([f"{day} 16:00:00", deal, pos, "XAUUSD", 1, 1, 0.01, 2000.0 + profit, -0.03, 0.0, 0.0, profit])
            deal += 1
            pos += 2


def _make_configs(folder, n_configs=12, n_days=60, seed=123):
    random.seed(seed)
    for c in range(n_configs):
        trades = [(d, round(random.gauss(0.0, 1.0), 2)) for d in range(n_days)]
        _write_config_csv(os.path.join(folder, f"cfg_{c:03d}.csv"), trades)


def test_build_opt_config_trades_reads_all_configs(tmp_path):
    _make_configs(str(tmp_path), n_configs=8, n_days=40)
    paths = sorted(str(p) for p in tmp_path.glob("*.csv"))
    opt, src = build_opt_config_trades(paths, signed=True)
    assert len(opt) == 8
    assert all(len(trs) == 40 for trs in opt.values())          # 40 trades per config
    for trs in opt.values():
        assert "net_pnl" in trs[0] and "exit_time" in trs[0]    # keys align_to_grid needs


def test_collect_pbo_produces_pbo_rc_spa(tmp_path):
    _make_configs(str(tmp_path), n_configs=12, n_days=60)
    paths = sorted(str(p) for p in tmp_path.glob("*.csv"))
    opt, src = build_opt_config_trades(paths, signed=True)
    champ = max(opt, key=lambda k: sum(t["net_pnl"] for t in opt[k]))
    rep = full_report(src[champ], is_path=True, opt_config_trades=opt)

    assert "optimization" in rep["sections"]
    g = rep["sections"]["optimization"]["gate"]
    assert g["pbo"] is not None
    assert g["rc_pvalue"] is not None
    assert g["spa_pvalue"] is not None
    assert rep["sections"]["optimization"]["n_configs"] == 12
    # pure noise across configs -> the battery must NOT accept
    assert g["decision"] == "REJECT"


def test_collect_pbo_needs_at_least_two_configs(tmp_path):
    _make_configs(str(tmp_path), n_configs=1, n_days=30)
    paths = sorted(str(p) for p in tmp_path.glob("*.csv"))
    opt, src = build_opt_config_trades(paths, signed=True)
    assert len(opt) == 1   # the CLI rejects <2; the builder itself just reads what's there


def _write_features_csv(path, rows):
    """rows: list of (position, time). Writes a minimal EGP_MHO_FeaturesExport-format CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Position", "Time", "Side", "f0"])
        for pos, t in rows:
            w.writerow([pos, t, 1, 0.5])


def test_features_rows_per_position_clean(tmp_path):
    p = os.path.join(str(tmp_path), "feat_clean.csv")
    _write_features_csv(p, [(2, "2026.01.02 06:00:00"), (4, "2026.01.05 03:00:00"), (6, "2026.01.07 15:00:00")])
    n, npos, mx = features_rows_per_position(p)
    assert (n, npos, mx) == (3, 3, 1)   # one row per position -> usable


def test_features_rows_per_position_accumulated(tmp_path):
    p = os.path.join(str(tmp_path), "feat_accum.csv")
    rows = [(2, "2026.01.02 06:00:00")] * 5 + [(4, "2026.01.05 03:00:00")] * 2
    _write_features_csv(p, rows)
    n, npos, mx = features_rows_per_position(p)
    assert n == 7 and npos == 2 and mx == 5   # >1 row/position -> accumulated, join ambiguous


def test_mt5_common_files_dir_found_via_appdata(tmp_path, monkeypatch):
    target = tmp_path / "MetaQuotes" / "Terminal" / "Common" / "Files"
    target.mkdir(parents=True)
    monkeypatch.setenv("APPDATA", str(tmp_path))            # simulate Windows
    assert mt5_common_files_dir() == str(target)


def test_mt5_common_files_dir_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA", str(tmp_path / "does_not_exist"))
    monkeypatch.setenv("HOME", str(tmp_path))               # no Wine prefix under this HOME
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert mt5_common_files_dir() is None
