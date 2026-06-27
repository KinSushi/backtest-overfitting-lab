"""Golden tests for the logic of the 4 EA guards (proof independent of MQL5 compilation)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_risk_guard_logic as RG


# 1) risk-based sizing
def test_risk_lot_golden():
    # risk 100, SL distance 2.0 (price), tick_value 1.0, tick_size 0.01
    # value/unit = 1/0.01 = 100 ; money_per_lot = 2.0*100 = 200 ; lots = 100/200 = 0.5
    assert abs(RG.risk_based_lot(100.0, 2.0, 1.0, 0.01) - 0.5) < 1e-9


def test_risk_lot_gold_like():
    # gold: tick_size 0.01, tick_value 1.0 ; SL distance 5.0$ -> money_per_lot 500 ; risk 50 -> 0.1
    assert abs(RG.risk_based_lot(50.0, 5.0, 1.0, 0.01) - 0.1) < 1e-9


def test_risk_lot_uses_ordercalcprofit_value_if_given():
    # si money_per_lot fourni (via OrderCalcProfit), il prime : 100/250 = 0.4
    assert abs(RG.risk_based_lot(100.0, 5.0, 1.0, 0.01, money_per_lot=250.0) - 0.4) < 1e-9


def test_risk_lot_clamped_and_stepped():
    assert RG.risk_based_lot(1e9, 1.0, 1.0, 0.01, vol_max=2.0) == 2.0      # plafond
    assert RG.risk_based_lot(1e-6, 1.0, 1.0, 0.01, vol_min=0.01) == 0.01   # plancher
    # no 0.1: 0.37 -> 0.4
    assert abs(RG.risk_based_lot(74.0, 2.0, 1.0, 0.01, vol_step=0.1) - 0.4) < 1e-9


def test_risk_lot_zero_when_indeterminate():
    assert RG.risk_based_lot(100.0, 0.0, 1.0, 0.01) == 0.0     # distance nulle
    assert RG.risk_based_lot(100.0, 2.0, 1.0, 0.0) == 0.0      # tick_size nul


# 2) coupe-circuit drawdown
def test_drawdown_pct_golden():
    assert abs(RG.drawdown_pct(9500, 10000) - 0.05) < 1e-12
    assert RG.drawdown_pct(10500, 10000) == 0.0               # above the peak -> 0


def test_should_halt_daily_and_total():
    # daily DD 5% > 4% -> halt (daily reason)
    r = RG.should_halt(9500, 10000, 10000, max_total_dd_pct=0.20, max_daily_dd_pct=0.04)
    assert r["halt"] is True and "journalier" in r["reason"]
    # wide thresholds -> no halt
    r2 = RG.should_halt(9500, 10000, 10000, max_total_dd_pct=0.20, max_daily_dd_pct=0.10)
    assert r2["halt"] is False
    # total DD 25% > 20% -> halt (total reason), peak higher than the day start
    r3 = RG.should_halt(7500, 10000, 8000, max_total_dd_pct=0.20, max_daily_dd_pct=0.50)
    assert r3["halt"] is True and "total" in r3["reason"]


# 3) cap de positions
def test_position_cap():
    assert RG.position_cap_ok(2, 3) is True
    assert RG.position_cap_ok(3, 3) is False
    assert RG.position_cap_ok(99, 0) is True                  # 0 = illimite


# 4) break-even
def test_breakeven_long_triggers_and_improves():
    # long entry 2000, price 2010, atr 5, trigger 1.5*5=7.5 ; profit 10>=7.5 -> SL = 2000+0.2*5=2001
    sl = RG.breakeven_new_sl(2000, 2010, +1, 5, current_sl=1990, trigger_mult=1.5, offset_mult=0.2)
    assert abs(sl - 2001.0) < 1e-9


def test_breakeven_not_triggered_below_threshold():
    assert RG.breakeven_new_sl(2000, 2005, +1, 5, current_sl=1990, trigger_mult=1.5) is None   # 5<7.5


def test_breakeven_never_worsens_sl():
    # current SL already at 2002 (> 2001 proposed) -> do not move back
    assert RG.breakeven_new_sl(2000, 2010, +1, 5, current_sl=2002, trigger_mult=1.5, offset_mult=0.2) is None


def test_breakeven_short():
    # short entry 2000, price 1990, atr 5, profit 10>=7.5 -> SL = 2000-1 = 1999 (< current_sl 2010)
    sl = RG.breakeven_new_sl(2000, 1990, -1, 5, current_sl=2010, trigger_mult=1.5, offset_mult=0.2)
    assert abs(sl - 1999.0) < 1e-9


# 5) blackout news
def test_news_window_in_and_out():
    windows = [{"weekday": 5, "start_min": 14 * 60 + 30, "end_min": 15 * 60}]   # vendredi 14:30-15:00
    assert RG.in_news_window(5, 14, 45, windows) is True
    assert RG.in_news_window(5, 15, 0, windows) is False  # upper bound excluded
    assert RG.in_news_window(4, 14, 45, windows) is False       # other day


def test_news_window_any_day():
    windows = [{"weekday": -1, "start_min": 0, "end_min": 60}]  # every day 00:00-01:00
    assert RG.in_news_window(3, 0, 30, windows) is True
    assert RG.in_news_window(3, 1, 30, windows) is False
