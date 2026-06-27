"""Tests merging real deals -> AFML verdict, on a CSV with the EXACT SCHEMA of the MQL5 writer (co-design)."""
import io
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_real_gate as RG
import egp_deals_export_codegen as GEN


def _make_deals_csv(positions, symbol="XAUUSD"):
    """Builds a deals CSV in the EXACT writer format (GEN.HEADER header).
    `positions` = liste de dicts {profit, commission, swap, fee} (couts SIGNES negatifs)."""
    lines = [",".join(GEN.HEADER)]
    ticket = 1000
    for pid, p in enumerate(positions, start=1):
        # ENTRY deal (Direction=0, profit 0)
        lines.append(",".join(str(x) for x in [
            f"2024.01.01 10:0{pid%10}:00", ticket, pid, symbol, 0, 0, 0.10, 2000.0, 0.0, 0.0, 0.0, 0.0]))
        ticket += 1
        # EXIT deal (Direction=1) carrying the P&L and the costs
        lines.append(",".join(str(x) for x in [
            f"2024.01.01 11:0{pid%10}:00", ticket, pid, symbol, 1, 1, 0.10, 2001.0,
            p.get("commission", 0.0), p.get("swap", 0.0), p.get("fee", 0.0), p.get("profit", 0.0)]))
        ticket += 1
    return "\n".join(lines)


def test_roundtrip_writer_schema_to_returns():
    # net per position = profit + commission + swap + fee (signed negative costs)
    positions = [{"profit": 10.0, "commission": -0.7, "swap": -0.1, "fee": 0.0},
                 {"profit": -4.0, "commission": -0.7, "swap": -0.2, "fee": 0.0},
                 {"profit": 6.0, "commission": -0.7, "swap": 0.0, "fee": -0.1}]
    csv_text = _make_deals_csv(positions)
    deals = __import__("egp_mt5_deals").parse_deals_csv(csv_text)
    trades = __import__("egp_mt5_deals").trades_from_deals(deals, signed=True)
    net = __import__("egp_mt5_deals").pnl_series(trades, net=True)
    expected = [p["profit"] + p["commission"] + p["swap"] + p["fee"] for p in positions]
    assert len(net) == 3
    for a, b in zip(sorted(net), sorted(expected)):
        assert abs(a - b) < 1e-9                       # aller-retour writer->reader coherent


def test_gate_from_deals_csv_full_verdict():
    rng = random.Random(0)
    positions = [{"profit": rng.gauss(3.0, 5.0), "commission": -0.7, "swap": -0.1, "fee": 0.0}
                 for _ in range(60)]
    csv_text = _make_deals_csv(positions)
    out = RG.gate_from_deals_csv(csv_text, initial_deposit=10000.0, sr_trials_variance=0.5,
                                 n_trials=50, mc_paths=300, seed=0)
    assert out["n_trades"] == 60
    assert "dsr" in out["gate"] and "decision" in out["gate"]
    assert "risk_of_ruin" in out["montecarlo"]
    assert out["total_costs"] > 0  # costs were charged


def test_costs_decomposition_consistent():
    positions = [{"profit": 5.0, "commission": -0.7, "swap": -0.3, "fee": -0.0} for _ in range(10)]
    out = RG.gate_from_deals_csv(_make_deals_csv(positions))
    # couts totaux = somme |commission+swap+fee| = 10 * 1.0 = 10
    assert abs(out["total_costs"] - 10.0) < 1e-6
    assert abs(out["gross_total"] - 50.0) < 1e-6        # 10 * profit 5
    assert abs(out["net_total"] - 40.0) < 1e-6          # 50 - 10


def test_profitable_strategy_positive_net():
    positions = [{"profit": 8.0, "commission": -0.7, "swap": -0.1, "fee": 0.0} for _ in range(40)]
    out = RG.gate_from_deals_csv(_make_deals_csv(positions))
    assert out["net_total"] > 0
    assert out["equity_final"] > 10000.0


def test_gate_from_trade_returns_direct():
    rets = [0.5, -0.2, 0.8, -0.1, 0.3] * 12
    out = RG.gate_from_trade_returns(rets, sr_trials_variance=0.3, n_trials=20, mc_paths=200)
    assert out["n_trades"] == 60 and "gate" in out and "montecarlo" in out


def test_rank_candidates_by_real_gate():
    rng = random.Random(1)
    # candidat A : rendements positifs reguliers ; candidat B : bruite autour de 0
    good = [{"profit": 5.0 + rng.gauss(0, 0.5), "commission": -0.5, "swap": 0.0, "fee": 0.0}
            for _ in range(60)]
    bad = [{"profit": rng.gauss(0.0, 6.0), "commission": -0.5, "swap": 0.0, "fee": 0.0}
           for _ in range(60)]
    cands = {"A_good": __import__("egp_mt5_deals").parse_deals_csv(_make_deals_csv(good)),
             "B_bad": __import__("egp_mt5_deals").parse_deals_csv(_make_deals_csv(bad))}
    out = RG.rank_candidates_by_real_gate(cands, sr_trials_variance=0.5, n_trials=20, mc_paths=200,
                                          seed=0)
    assert out["best"] == "A_good"                      # the regular candidate dominates at the real gate
    assert out["ranked"][0] == "A_good"
