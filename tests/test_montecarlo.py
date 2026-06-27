"""Offline tests for egp_montecarlo (deterministic)."""
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_montecarlo as MC


def test_max_drawdown_exact():
    # P&L [10,-30,50,-40] from 100 -> equity [100,110,80,130,90].
    # pics 100,110,110,130,130 ; DD 0,0,30,0,40 -> MDD abs=40, pct=40/130.
    eq = MC.equity_curve([10, -30, 50, -40], start_equity=100)
    assert eq == [100, 110, 80, 130, 90]
    a, r = MC.max_drawdown(eq)
    assert a == 40
    assert abs(r - 40.0 / 130.0) < 1e-12


def test_shuffle_preserves_terminal_varies_drawdown():
    pnls = [5, -3, 8, -10, 4, 7, -6, 2, -1, 9]
    total = sum(pnls)
    # shuffle: invariant sum -> identical terminal equity on all paths
    paths = MC.mc_resample(pnls, mode="shuffle", n_paths=200, seed=1)
    terminals = [sum(pp) for pp in paths]
    assert max(terminals) == min(terminals) == total
    # but the drawdown varies with the order
    dds = [MC.max_drawdown(MC.equity_curve(pp, 1000))[0] for pp in paths]
    assert statistics.pstdev(dds) > 0
    # bootstrap : l'equity terminale, elle, varie
    bpaths = MC.mc_resample(pnls, mode="bootstrap", n_paths=200, seed=1)
    bterm = [sum(pp) for pp in bpaths]
    assert statistics.pstdev(bterm) > 0


def test_risk_of_ruin_decreases_with_more_capital():
    # Slightly positive but volatile game.
    pnls = [100, -90] * 50
    ror_low = MC.risk_of_ruin(pnls, start_equity=150, ruin_equity=0,
                              mode="bootstrap", n_paths=2000, horizon=500, seed=2)
    ror_high = MC.risk_of_ruin(pnls, start_equity=1000, ruin_equity=0,
                               mode="bootstrap", n_paths=2000, horizon=500, seed=2)
    assert ror_high < ror_low


def test_gambler_ruin_mc_matches_analytic():
    # GOLDEN : jeu +1/-1, p=0.55, capital 10 unites -> ruine ~ (0.45/0.55)^10 ~ 0.137.
    p, u = 0.55, 10
    analytic = MC.risk_of_ruin_analytic_bernoulli(p, u)
    mc = MC.risk_of_ruin_bernoulli_mc(p, u, horizon=4000, n_paths=5000, seed=3)
    assert abs(mc - analytic) < 0.03, (mc, analytic)
    # jeu defavorable -> ruine certaine (analytique = 1.0) et MC tres eleve
    assert MC.risk_of_ruin_analytic_bernoulli(0.45, 10) == 1.0
    assert MC.risk_of_ruin_bernoulli_mc(0.45, 10, horizon=4000, n_paths=2000, seed=4) > 0.95
    # more capital -> less ruin (favorable game)
    assert (MC.risk_of_ruin_analytic_bernoulli(0.55, 20)
            < MC.risk_of_ruin_analytic_bernoulli(0.55, 10))


def test_summary_fields_present_and_coherent():
    pnls = [50, -30, 40, -20, 60, -45, 30, -10]
    s = MC.mc_summary(pnls, start_equity=1000, ruin_equity=0,
                      mode="bootstrap", n_paths=1000, seed=5)
    assert 0.0 <= s["prob_profit"] <= 1.0
    assert 0.0 <= s["risk_of_ruin"] <= 1.0
    assert s["max_drawdown_abs"]["max"] >= s["max_drawdown_abs"][50]
    assert "terminal_equity" in s
