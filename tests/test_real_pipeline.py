"""Tests for model validation on REAL data (features<->deals join with exact schema)."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_real_pipeline as RP
import egp_features_export_codegen as FGEN
import egp_deals_export_codegen as DGEN


def _features_csv(rows, names):
    # EXACT header of the features writer: Position,Time,Side,<names>
    header = "Position,Time,Side," + ",".join(names)
    lines = [header]
    for pid, side, vals in rows:
        lines.append(",".join([str(pid), "2024.01.01 10:00:00", str(side)] +
                              [f"{v:.8f}" for v in vals]))
    return "\n".join(lines)


def _deals_csv(pos_profits):
    # one IN + one OUT per position, exact schema of the deals writer
    lines = [",".join(DGEN.HEADER)]
    tk = 1000
    for pid, profit in pos_profits:
        lines.append(",".join(str(x) for x in ["2024.01.01 10:00:00", tk, pid, "XAUUSD", 0, 0,
                                               0.1, 2000.0, 0.0, 0.0, 0.0, 0.0])); tk += 1
        lines.append(",".join(str(x) for x in ["2024.01.01 11:00:00", tk, pid, "XAUUSD", 1, 1,
                                               0.1, 2001.0, -0.5, 0.0, 0.0, profit])); tk += 1
    return "\n".join(lines)


def test_parse_features_csv():
    csv_text = _features_csv([(1, 1, [0.3, 0.7]), (2, -1, [0.5, 0.1])], ["f0", "f1"])
    rows, names = RP.parse_features_csv(csv_text)
    assert names == ["f0", "f1"]
    assert len(rows) == 2
    assert abs(rows[0]["features"]["f1"] - 0.7) < 1e-9
    assert rows[1]["side"] == -1


def test_alignment_joins_on_position():
    names = ["f0"]
    feats = _features_csv([(1, 1, [0.9]), (2, 1, [0.1]), (3, 1, [0.5])], names)
    deals = _deals_csv([(1, 5.0), (2, -3.0)])  # position 3 absent from the deals -> not joined
    frows, fnames = RP.parse_features_csv(feats)
    import egp_mt5_deals as MD
    trades = MD.trades_from_deals(MD.parse_deals_csv(deals))
    X, y, net, events = RP.align_features_deals(frows, trades, fnames)
    assert len(X) == 2                                  # only positions 1 and 2 joined
    assert y == [1, 0]                                  # pos1 winner (5-0.5>0), pos2 loser


def test_alignment_picks_entry_row_no_lookahead():
    # Two features rows for the SAME position: the entry-time row carries 0.9, a LATER bar carries 0.1.
    # The join must keep the ENTRY row (earliest timestamp), never the later one (look-ahead).
    # The entry row is placed FIRST so the old "last-wins" join would have wrongly returned 0.1.
    fnames = ["f0"]
    frows = [
        {"position": "1", "time": "2026.01.01 09:00:00", "side": 1, "features": {"f0": 0.9}},  # entry
        {"position": "1", "time": "2026.01.05 10:00:00", "side": 1, "features": {"f0": 0.1}},  # later
    ]
    trades = [{"pos_id": "1", "net_pnl": 5.0}]
    X, y, net, events = RP.align_features_deals(frows, trades, fnames)
    assert len(X) == 1
    assert abs(X[0][0] - 0.9) < 1e-9                     # entry-time feature, not the later 0.1


def test_run_real_pipeline_end_to_end_informative_features():
    # feature f0 predicts the outcome: f0>0.5 -> winning trade; the model must add value
    rng = random.Random(0)
    names = ["f0", "f1"]
    frows = []
    pos_profits = []
    for pid in range(1, 121):
        f0 = rng.random()
        win = f0 > 0.5
        frows.append((pid, 1, [f0, rng.random()]))
        profit = (abs(rng.gauss(4, 1)) if win else -abs(rng.gauss(4, 1)))
        pos_profits.append((pid, profit))
    feats_csv = _features_csv(frows, names)
    deals_csv = _deals_csv(pos_profits)
    out = RP.run_real_pipeline(feats_csv, deals_csv, config={"mc_paths": 200, "n_splits": 4})
    mv = out["model_validation"]
    assert out["n_matched"] == 120
    assert mv["oob_accuracy"] > 0.7                     # features informatives -> modele apprend
    assert mv["feature_importance"]["f0"] > mv["feature_importance"]["f1"]
    assert "sized_gate" in mv and "raw_gate" in mv


def test_features_export_codegen_uses_file_api():
    code = FGEN.generate_code()
    for token in ["FileOpen(", "FileWrite(h", "FileSeek(h, 0, SEEK_END)", "FileSize(h)",
                  "ArraySize(values)", "EGP_LogFeatures", "Position,Time,Side"]:
        assert token in code, f"token manquant: {token}"
    assert "#ifndef EGP_MHO_FEATURESEXPORT_MQH" in code and "#endif" in code


def test_features_export_codegen_writes_file(tmp_path):
    p = FGEN.generate(str(tmp_path), default_filename="feat99.csv")
    assert p.exists() and "feat99.csv" in p.read_text(encoding="utf-8")
