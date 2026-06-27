"""Offline tests for the concurrency-aware backtest of the sized strategy."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_strategy_backtest as SBT


def _bet(prob, side_return, t0, t1, cost=0.0):
    return {"prob": prob, "side_return": side_return, "t0": t0, "t1": t1, "cost": cost}


def test_profitable_strategy_equity_rises():
    bets = [_bet(0.8, 0.01, i, i + 2) for i in range(20)]
    out = SBT.backtest_sized_strategy(bets, notional=1000.0, sr_trials_variance=0.0, n_trials=1)
    assert out["pnl_total"] > 0
    assert out["equity_final"] > out["equity_curve"][0]
    assert out["n_active"] == 20


def test_size_scales_pnl():
    # same return, higher probability -> larger size -> larger P&L
    low = SBT.backtest_sized_strategy([_bet(0.55, 0.01, 0, 2)], notional=1000.0)["pnl_total"]
    high = SBT.backtest_sized_strategy([_bet(0.95, 0.01, 0, 2)], notional=1000.0)["pnl_total"]
    assert high > low > 0


def test_no_bet_when_prob_not_convincing():
    # prob <= 0.5 -> effective size 0 -> no bet, zero P&L
    out = SBT.backtest_sized_strategy([_bet(0.5, 0.02, 0, 2), _bet(0.4, 0.02, 1, 3)], notional=1000.0)
    assert out["n_active"] == 0
    assert out["pnl_total"] == 0.0


def test_cost_reduces_pnl():
    no_cost = SBT.backtest_sized_strategy([_bet(0.8, 0.01, 0, 2, cost=0.0)], notional=1000.0)["pnl_total"]
    with_cost = SBT.backtest_sized_strategy([_bet(0.8, 0.01, 0, 2, cost=0.005)], notional=1000.0)["pnl_total"]
    assert with_cost < no_cost


def test_concurrency_measured():
    # overlapping bets -> mean concurrency > 1
    bets = [_bet(0.8, 0.01, 0, 10), _bet(0.8, 0.01, 2, 12), _bet(0.8, 0.01, 4, 14)]
    out = SBT.backtest_sized_strategy(bets, notional=1000.0, n_bars=20)
    assert out["concurrency"]["avg"] > 1.0
    assert out["concurrency"]["max"] >= 2


def test_gate_and_mc_integration_runs():
    bets = [_bet(0.75, 0.008 if i % 2 == 0 else -0.003, i, i + 2) for i in range(40)]
    out = SBT.backtest_sized_strategy(bets, notional=1000.0, sr_trials_variance=0.5, n_trials=50,
                                      mc_paths=200, seed=0)
    assert "decision" in out["gate"]
    assert "risk_of_ruin" in out["montecarlo"]


def test_deterministic_with_seed():
    bets = [_bet(0.7, 0.01 if i % 3 else -0.005, i, i + 2) for i in range(30)]
    a = SBT.backtest_sized_strategy(bets, notional=1000.0, mc_paths=200, seed=42)
    b = SBT.backtest_sized_strategy(bets, notional=1000.0, mc_paths=200, seed=42)
    assert a["montecarlo"]["risk_of_ruin"] == b["montecarlo"]["risk_of_ruin"]


def test_bets_from_triple_barrier_builder():
    tb = [{"side_return": 0.01, "t0": 0, "touch_idx": 3},
          {"side_return": -0.02, "t0": 1, "touch_idx": 5}]
    bets = SBT.bets_from_triple_barrier(tb, probs=[0.8, 0.6], prices=[2000.0, 2001.0, 2002.0, 2003.0,
                                                                      2004.0, 2005.0])
    assert bets[0]["prob"] == 0.8 and bets[0]["t1"] == 3 and bets[0]["price"] == 2000.0


def test_min_prob_threshold_filters_bets():
    # threshold 0.7 -> bets at probability 0.6 are not taken
    bets = [_bet(0.6, 0.01, i, i + 2) for i in range(10)]
    out = SBT.backtest_sized_strategy(bets, notional=1000.0, min_prob=0.7)
    assert out["n_active"] == 0
    out2 = SBT.backtest_sized_strategy(bets, notional=1000.0, min_prob=0.5)
    assert out2["n_active"] == 10


def test_gross_exposure_cap_limits_concurrency_size():
    # 5 strongly convinced overlapping bets ; low gross cap -> P&L capped
    bets = [_bet(0.95, 0.01, 0, 10) for _ in range(5)]  # all open simultaneously [0,10]
    uncapped = SBT.backtest_sized_strategy(bets, notional=1000.0, n_bars=12)["pnl_total"]
    capped = SBT.backtest_sized_strategy(bets, notional=1000.0, n_bars=12,
                                         max_gross_exposure=1.0)["pnl_total"]
    assert capped < uncapped  # the cap reduces the aggregate exposure
    assert capped > 0


def test_gross_exposure_cap_value_respected():
    # with a 1.0 cap and 5 simultaneous bets of size ~0.9, the gross exposure taken <= 1.0
    bets = [_bet(0.95, 0.01, 0, 10) for _ in range(5)]
    out = SBT.backtest_sized_strategy(bets, notional=1.0, n_bars=12, max_gross_exposure=1.0)
    # sum of effective sizes (= P&L / (side_return*notional) per bet) bounded by the cap
    total_size = out["pnl_total"] / (0.01 * 1.0)
    assert total_size <= 1.0 + 1e-9


def test_sweep_sizing_returns_best_by_dsr():
    bets = [_bet(0.55 + 0.4 * (i % 2), 0.01 if i % 2 == 0 else -0.004, i, i + 2) for i in range(60)]
    out = SBT.sweep_sizing(bets, min_probs=[0.5, 0.6], step_sizes=[0.0, 0.1], notional=1000.0,
                           sr_trials_variance=0.5, n_trials=10, mc_paths=100, seed=0)
    assert out["n_combos"] == 4
    assert len(out["table"]) == 4
    assert out["best"] in out["table"]
    # the best has the maximal DSR of the table (ignoring NaN)
    valid = [r for r in out["table"] if r["dsr"] == r["dsr"]]
    assert out["best"]["dsr"] == max(r["dsr"] for r in valid)
