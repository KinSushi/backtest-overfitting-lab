"""Offline tests for the swap/execution extension of egp_costs (deterministic)."""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_costs as C


def test_intraday_scalp_pays_no_swap():
    # Opened and closed the same day before the rollover -> 0 rollover.
    e = datetime(2024, 1, 8, 10, 0)   # lundi 10:00
    x = datetime(2024, 1, 8, 15, 0)   # lundi 15:00
    roll = C.count_rollovers(e, x, rollover_hour=22)
    assert roll["effective"] == 0
    sm = C.SwapModel()
    assert C.position_swap(e, x, 0.1, "long", sm)["swap"] == 0.0


def test_one_night_simple():
    e = datetime(2024, 1, 8, 10, 0)   # lundi 10:00
    x = datetime(2024, 1, 9, 10, 0)   # mardi 10:00 (franchit lundi 22:00)
    roll = C.count_rollovers(e, x, rollover_hour=22, triple_weekday=2)
    assert roll["effective"] == 1 and roll["triple"] == 0


def test_wednesday_triple_swap():
    # Position crossing the WEDNESDAY rollover -> triple.
    e = datetime(2024, 1, 10, 10, 0)  # mercredi 10:00
    x = datetime(2024, 1, 11, 10, 0)  # jeudi 10:00 (franchit mercredi 22:00)
    roll = C.count_rollovers(e, x, rollover_hour=22, triple_weekday=2)
    assert roll["effective"] == 3 and roll["triple"] == 1
    # mardi->jeudi : 1 (mardi) + 3 (mercredi) = 4 nuits effectives
    e2 = datetime(2024, 1, 9, 10, 0)  # mardi
    x2 = datetime(2024, 1, 11, 10, 0) # jeudi
    assert C.count_rollovers(e2, x2, rollover_hour=22, triple_weekday=2)["effective"] == 4


def test_weekend_no_swap_but_flagged():
    # Friday -> Monday: crosses Friday 22:00 (1); Saturday/Sunday = no rollover.
    e = datetime(2024, 1, 12, 10, 0)  # vendredi 10:00
    x = datetime(2024, 1, 15, 10, 0)  # lundi 10:00
    roll = C.count_rollovers(e, x, rollover_hour=22)
    assert roll["effective"] == 1                 # no weekend swap
    assert C.weekend_crossings(e, x) == 1         # but weekend gap exposure flagged


def test_swap_sign_is_a_cost():
    sm = C.SwapModel(swap_long_per_lot=-3.0, swap_short_per_lot=-1.5)
    e = datetime(2024, 1, 10, 10, 0)  # mercredi
    x = datetime(2024, 1, 11, 10, 0)  # jeudi (triple)
    out = C.position_swap(e, x, 1.0, "long", sm)
    assert abs(out["swap"] - (3 * 1.0 * -3.0)) < 1e-9  # = -9.0, a cost
# short cheaper here
    assert C.position_swap(e, x, 1.0, "short", sm)["swap"] == 3 * 1.0 * -1.5


def test_slippage_scales_with_volatility():
    s_calm = C.slippage_vol_scaled(0.10, k_vol=2.0, realized_vol=0.01)
    s_wild = C.slippage_vol_scaled(0.10, k_vol=2.0, realized_vol=0.50)
    assert s_wild > s_calm
    assert abs(s_calm - (0.10 + 2.0 * 0.01)) < 1e-12


def test_total_position_cost_breakdown():
    cm = C.CostModel()
    sm = C.SwapModel(swap_long_per_lot=-3.0)
    e = datetime(2024, 1, 10, 10, 0)  # mercredi
    x = datetime(2024, 1, 11, 10, 0)  # jeudi (triple swap)
    out = C.total_position_cost(e, x, 0.1, "long", cm, sm)
    assert out["swap"] == 3 * 0.1 * -3.0          # -0.9
    # total cost = linear + impact - signed_swap (the swap cost adds to the positive total)
    assert abs(out["total_cost"] - (cm.linear_cost(0.1) + cm.impact_cost(0.1) - out["swap"])) < 1e-9
    assert out["total_cost"] > cm.linear_cost(0.1)  # the swap adds to the cost
    assert out["weekend_crossings"] == 0
