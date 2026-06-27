"""Offline tests for egp_costs (deterministic)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_costs as C


def test_costs_reduce_pnl_monotonically():
    m = C.CostModel()
    g = [100.0, -50.0, 80.0, -20.0, 60.0]
    net1 = sum(C.apply_costs(g, 0.1, m, cost_factor=1.0))
    net2 = sum(C.apply_costs(g, 0.1, m, cost_factor=2.0))
    assert net2 < net1 < sum(g)            # more cost => less net


def test_breakeven_multiplier_is_exact():
    # cout lineaire => net_total(f) = gross - f*cost1 ; net(f*)=0.
    m = C.CostModel(spread_price=0.30, commission_per_lot=7.0, slippage_price=0.10,
                    contract_size=100.0, slippage_sides=2)
    g = [50.0] * 20            # gross_total = 1000
    cost1 = sum(m.per_trade_cost(0.1) for _ in g)
    fstar = C.breakeven_cost_multiplier(g, 0.1, m)
    assert abs(fstar - 1000.0 / cost1) < 1e-9
    nt = C.net_metrics(g, 0.1, m, cost_factor=fstar)["net_total"]
    assert abs(nt) < 1e-6  # net zero at breakeven


def test_survives_plus_50pct():
    m = C.CostModel()
    cost1 = m.per_trade_cost(0.1)
    n = 100
    # thick edge: gross/trade = 3x the cost -> survives +50%
    fat = [3.0 * cost1] * n
    assert C.survives_plus_pct(fat, 0.1, m, pct=50.0)
    # thin edge: gross/trade = 1.2x the cost -> does not survive +50%
    thin = [1.2 * cost1] * n
    assert not C.survives_plus_pct(thin, 0.1, m, pct=50.0)


def test_cost_stress_table_monotone():
    m = C.CostModel()
    g = [2.0 * m.per_trade_cost(0.1)] * 50
    table = C.cost_stress(g, 0.1, m, factors=(1.0, 1.5, 2.0, 3.0))
    nets = [nt for _, nt, _ in table]
    assert nets == sorted(nets, reverse=True)  # net decreases with the factor
    # survives f=1.0 and 1.5 (edge=2x cost), not f>=2.0
    surv = {f: s for f, _, s in table}
    assert surv[1.0] and surv[1.5] and not surv[2.0]


def test_almgren_chriss_impact_is_quadratic_and_off_by_default():
    base = C.CostModel()
    assert base.impact_cost(0.1) == 0.0  # disabled by default
    ac = C.CostModel(ac_gamma=1e-6, ac_eta=1e-6, ac_exec_time=1.0)
    c1 = ac.impact_cost(1.0)
    c2 = ac.impact_cost(2.0)
    assert c1 > 0.0
    assert abs(c2 - 4.0 * c1) < 1e-9                   # impact ~ Q^2 -> x2 lots => x4
    # at small lots, impact stays negligible vs the linear term
    assert ac.impact_cost(0.1) < 0.01 * ac.linear_cost(0.1)
