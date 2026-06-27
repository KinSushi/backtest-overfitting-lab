"""Tests for the MQL5 deals-export codegen: the emitted code uses the verified API + aligned header."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_deals_export_codegen as GEN
import egp_mt5_deals as MD


def test_emitted_code_uses_verified_api():
    code = GEN.generate_code()
    for token in ["HistorySelect(0, TimeCurrent())", "HistoryDealsTotal()", "HistoryDealGetTicket(i)",
                  "HistoryDealGetInteger(ticket, DEAL_TIME)", "HistoryDealGetDouble (ticket, DEAL_PROFIT)",
                  "DEAL_COMMISSION", "DEAL_SWAP", "DEAL_FEE", "DEAL_ENTRY", "DEAL_POSITION_ID",
                  "DEAL_SYMBOL", "DEAL_VOLUME", "DEAL_PRICE"]:
        assert token in code, f"missing token: {token}"


def test_emitted_header_matches_adapter_colmap():
    # the emitted CSV header must match the adapter's _DEFAULT_COLMAP values
    code = GEN.generate_code()
    expected_cols = set(MD._DEFAULT_COLMAP.values()) - {"Balance"}   # balance not exported here
    for col in expected_cols:
        assert f'"{col}"' in code, f"column missing from the emitted header: {col}"


def test_include_guard_and_close():
    code = GEN.generate_code()
    assert "#ifndef EGP_MHO_DEALSEXPORT_MQH" in code
    assert "#endif" in code
    assert "FileClose(h)" in code
    assert "INVALID_HANDLE" in code              # open-error handling


def test_common_folder_flag_toggles():
    assert "FILE_COMMON" in GEN.generate_code(common_folder=True)
    assert "FILE_COMMON" not in GEN.generate_code(common_folder=False)


def test_generate_writes_file(tmp_path):
    p = GEN.generate(str(tmp_path), default_filename="run42.csv")
    assert p.exists()
    txt = p.read_text(encoding="utf-8")
    assert "run42.csv" in txt and "EGP_ExportDeals" in txt
