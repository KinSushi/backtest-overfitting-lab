"""Verifies the emitted MQL5 for the 4 guards: verified API + inputs + structure."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_risk_guard_codegen as CG


def test_code_has_verified_api():
    c = CG.generate_code()
    # P0 drawdown
    assert "AccountInfoDouble(ACCOUNT_EQUITY)" in c
    assert "TimeToStruct(TimeCurrent()" in c and "MqlDateTime" in c
    # P0 cap positions
    assert "PositionsTotal()" in c
    assert "PositionGetTicket(i)" in c
    assert "PositionGetInteger(POSITION_MAGIC)" in c
    assert "PositionGetString(POSITION_SYMBOL)" in c
    # P1 gold-safe sizing: OrderCalcProfit first + TickValue/TickSize fallback
    assert "OrderCalcProfit(type, _Symbol, 1.0, open_price, sl_price, profit)" in c
    assert "SYMBOL_TRADE_TICK_SIZE" in c and "SYMBOL_TRADE_TICK_VALUE" in c
    assert "SYMBOL_VOLUME_MIN" in c and "SYMBOL_VOLUME_MAX" in c and "SYMBOL_VOLUME_STEP" in c
    # P2/P3 break-even: SL modification
    assert "egp_rg_trade.PositionModify(" in c
    assert "PositionGetDouble(POSITION_PRICE_OPEN)" in c
    # include CTrade + include guard
    assert "#include <Trade\\Trade.mqh>" in c
    assert "#ifndef EGP_MHO_RISKGUARD_MQH" in c and "#endif" in c


def test_code_has_all_inputs():
    c = CG.generate_code()
    for name in ["RG_Magic", "RG_DD_On", "RG_MaxTotalDD_Pct", "RG_MaxDailyDD_Pct",
                 "RG_MaxPositions", "RG_RiskSizing_On", "RG_RiskPct",
                 "RG_BE_On", "RG_BE_TriggerATR", "RG_BE_OffsetATR",
                 "RG_News_On", "RG_News1_On", "RG_News2_On"]:
        assert name in c, f"missing input: {name}"


def test_code_has_all_four_guard_functions():
    c = CG.generate_code()
    for fn in ["void EGP_RG_Update()", "bool EGP_RG_ShouldHalt()",
               "bool EGP_RG_PositionCapOK()", "double EGP_RG_RiskLots(",
               "void EGP_RG_ApplyBreakeven(", "bool EGP_RG_InNewsBlackout()"]:
        assert fn in c, f"missing function: {fn}"


def test_magic_default_is_parametrized():
    assert "RG_Magic              = 4242" in CG.generate_code(magic_default=4242)


def test_generate_writes_file(tmp_path):
    p = CG.generate(str(tmp_path))
    assert p.exists() and p.name == "EGP_MHO_RiskGuard.mqh"
    assert "EGP_RG_ShouldHalt" in p.read_text(encoding="utf-8")
