"""Offline tests for egp_mt5_deals: reconstruction logic (independent of the format)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_mt5_deals as D
import egp_montecarlo as MC
import egp_accept_gate as G


def _deals_two_positions():
    # Position 1 : in (profit 0) puis out (profit +100, commission 7, swap -1)
    # Position 2 : in (profit 0) puis out (profit -40, commission 7, swap 0)
    return [
        {"time": 1, "pos_id": 1, "entry": D.ENTRY_IN, "volume": 0.1, "profit": 0.0,
         "commission": -3.5, "swap": 0.0, "balance": 9996.5},
        {"time": 2, "pos_id": 1, "entry": D.ENTRY_OUT, "volume": 0.1, "profit": 100.0,
         "commission": -3.5, "swap": -1.0, "balance": 10092.0},
        {"time": 3, "pos_id": 2, "entry": D.ENTRY_IN, "volume": 0.1, "profit": 0.0,
         "commission": -3.5, "swap": 0.0, "balance": 10088.5},
        {"time": 4, "pos_id": 2, "entry": D.ENTRY_OUT, "volume": 0.1, "profit": -40.0,
         "commission": -3.5, "swap": 0.0, "balance": 10045.0},
    ]


def test_trades_net_pnl_reconstruction():
    trades = D.trades_from_deals(_deals_two_positions())
    assert len(trades) == 2
    # Position 1 : net = 100 - (3.5+3.5) - 1 = 92.0 ; gross = 100 ; costs = 8.0
    t1 = trades[0]
    assert abs(t1["gross_pnl"] - 100.0) < 1e-9
    assert abs(t1["costs"] - 8.0) < 1e-9
    assert abs(t1["net_pnl"] - 92.0) < 1e-9
    # Position 2 : net = -40 - 7 - 0 = -47.0
    assert abs(trades[1]["net_pnl"] - (-47.0)) < 1e-9


def test_trades_ordered_by_exit_time():
    deals = list(reversed(_deals_two_positions()))   # ordre melange
    trades = D.trades_from_deals(deals)
    assert [t["pos_id"] for t in trades] == [1, 2]  # re-ordered by exit


def test_pnl_series_feeds_downstream():
    trades = D.trades_from_deals(_deals_two_positions())
    series = D.pnl_series(trades, net=True)
    assert series == [92.0, -47.0]
    assert abs(sum(series) - 45.0) < 1e-9             # net total = 92 - 47


def test_equity_curve_matches_cumulative_net():
    deals = _deals_two_positions()
    eq = D.equity_from_deals(deals, initial_deposit=10000.0)
    # cumulative net results : -3.5, +95.5, -3.5, -43.5 -> 10000,9996.5,10092,10088.5,10045
    assert abs(eq[-1] - 10045.0) < 1e-9
    # via the balance column provided by MT5
    eqb = D.equity_from_deals(deals, initial_deposit=10000.0, use_balance_field=True)
    assert abs(eqb[-1] - 10045.0) < 1e-9


def test_parse_deals_csv_synthetic():
    # Synthetic fixture mimicking the documented STRUCTURE (header to confirm on a real export).
    csv_text = (
        "Time,Deal,Symbol,Type,Direction,Volume,Price,Commission,Swap,Profit,Balance,Position\n"
        "2024.01.01 10:00,1,XAUUSD,buy,in,0.10,2000.0,-3.5,0.0,0.0,9996.5,1\n"
        "2024.01.01 11:00,2,XAUUSD,sell,out,0.10,2010.0,-3.5,-1.0,100.0,10092.0,1\n"
    )
    deals = D.parse_deals_csv(csv_text)
    assert len(deals) == 2
    assert deals[0]["entry"] == D.ENTRY_IN and deals[1]["entry"] == D.ENTRY_OUT
    assert abs(deals[1]["profit"] - 100.0) < 1e-9
    assert abs(deals[1]["commission"] - (-3.5)) < 1e-9
    trades = D.trades_from_deals(deals)
    assert abs(trades[0]["net_pnl"] - 92.0) < 1e-9    # 100 - 7 - 1


def test_full_chain_to_montecarlo_and_gate():
    # End-to-end demonstration on synthetic data: deals -> P&L -> MC + gate.
    deals = []
    import random
    rng = random.Random(0)
    bal = 10000.0
    for i in range(300):
        pnl = rng.gauss(5.0, 50.0)        # petit edge positif, volatil
        bal += pnl
        deals.append({"time": 2 * i, "pos_id": i, "entry": D.ENTRY_IN, "volume": 0.1,
                      "profit": 0.0, "commission": 0.0, "swap": 0.0})
        deals.append({"time": 2 * i + 1, "pos_id": i, "entry": D.ENTRY_OUT, "volume": 0.1,
                      "profit": pnl, "commission": 0.0, "swap": 0.0, "balance": bal})
    series = D.pnl_series(D.trades_from_deals(deals))
    assert len(series) == 300
    # feeds the Monte-Carlo (risk of ruin) and the gate (Sharpe) without error
    ror = MC.risk_of_ruin(series, start_equity=10000.0, ruin_equity=0.0, n_paths=500, seed=1)
    assert 0.0 <= ror <= 1.0
    sr = G.sharpe_ratio(series)
    assert isinstance(sr, float)
