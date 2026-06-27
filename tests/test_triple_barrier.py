"""Offline tests for egp_triple_barrier (AFML ch.3), deterministic/golden."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_triple_barrier as TB
import egp_costs as C


def test_ewma_vol_hand_check():
    # span=3 -> alpha=0.5 ; returns=[0.0, 0.1] : var0=0, var1=0.5*0+0.5*0.01=0.005
    vol = TB.ewma_volatility([0.0, 0.1], span=3)
    assert abs(vol[0] - 0.0) < 1e-12
    assert abs(vol[1] - math.sqrt(0.005)) < 1e-12


def test_ewma_vol_responds_to_amplitude():
    calm = TB.ewma_volatility([0.001, -0.001] * 20, span=10)[-1]
    wild = TB.ewma_volatility([0.05, -0.05] * 20, span=10)[-1]
    assert wild > calm > 0


def test_long_hits_profit_barrier():
    # rising price : a long hits the upper barrier (pt) -> label +1
    prices = [100.0, 100.5, 101.0, 102.0, 103.0]
    ft = TB.first_touch(prices, t0=0, sigma=0.01, pt_mult=1.0, sl_mult=1.0,
                        max_horizon=4, side=1)
    assert ft["outcome"] == "pt" and ft["label"] == 1
    # sigma=0.01, pt_mult=1 -> threshold 1%; p1/p0-1=0.5% (<1%), p2=1% (>=1%) -> touch at idx2
    assert ft["touch_idx"] == 2


def test_long_hits_stop_barrier():
    prices = [100.0, 99.5, 99.0, 98.0]  # falling price
    ft = TB.first_touch(prices, t0=0, sigma=0.01, pt_mult=1.0, sl_mult=1.0,
                        max_horizon=3, side=1)
    assert ft["outcome"] == "sl" and ft["label"] == -1


def test_sideways_hits_time_barrier():
    # weak oscillation inside the barriers -> vertical barrier -> label 0
    prices = [100.0, 100.05, 99.97, 100.03, 99.98, 100.02]
    ft = TB.first_touch(prices, t0=0, sigma=0.02, pt_mult=2.0, sl_mult=2.0,
                        max_horizon=5, side=1)
    assert ft["outcome"] == "time" and ft["label"] == 0
    assert ft["touch_idx"] == 5


def test_higher_sigma_widens_barriers():
    prices = [100.0, 100.5, 101.0, 101.5]
    # sigma faible -> profit touche
    near = TB.first_touch(prices, 0, sigma=0.005, pt_mult=1.0, sl_mult=1.0,
                          max_horizon=3, side=1)
    # strong sigma -> barriers too wide -> timeout
    far = TB.first_touch(prices, 0, sigma=0.05, pt_mult=1.0, sl_mult=1.0,
                         max_horizon=3, side=1)
    assert near["outcome"] == "pt"
    assert far["outcome"] == "time"


def test_short_side_inverts():
    falling = [100.0, 99.5, 99.0, 98.0]
    # short on falling price -> profit (pt), label +1
    ft = TB.first_touch(falling, 0, sigma=0.01, pt_mult=1.0, sl_mult=1.0,
                        max_horizon=3, side=-1)
    assert ft["outcome"] == "pt" and ft["label"] == 1
    assert ft["side_return"] > 0
    rising = [100.0, 100.5, 101.0, 102.0]
    ft2 = TB.first_touch(rising, 0, sigma=0.01, pt_mult=1.0, sl_mult=1.0,
                         max_horizon=3, side=-1)
    assert ft2["outcome"] == "sl" and ft2["label"] == -1


def test_meta_label_net_of_cost():
    assert TB.meta_label(0.01, 0.002) == 1     # gain 1% > cout 0.2%
    assert TB.meta_label(0.001, 0.002) == 0    # gain 0.1% < cout 0.2% -> rejete


def test_meta_label_kills_marginal_profit_touch():
    # the upper barrier is hit (primary +1) but the gain does NOT cover the costs -> meta 0
    prices = [100.0, 100.06, 100.2]  # +0.06% reaches a small barrier
    ft = TB.first_touch(prices, 0, sigma=0.0005, pt_mult=1.0, sl_mult=1.0,
                        max_horizon=2, side=1)
    assert ft["label"] == 1                    # primaire : profit touche
    res = TB.add_meta_labels([ft], cost=0.002)  # cout 0.2% > gain ~0.05-0.06%
    assert res[0]["meta_label"] == 0           # net de cout : non rentable


def test_cost_in_return_units():
    cm = C.CostModel()                          # linear_cost(0.1)=5.7
    cru = TB.cost_in_return_units(cm, lots=0.1, price=2000.0)
    # notionnel = 2000*100*0.1 = 20000 -> 5.7/20000
    assert abs(cru - 5.7 / 20000.0) < 1e-12


def test_end_to_end_and_summary():
    prices = [100.0]
    for _ in range(30):
        prices.append(prices[-1] * 1.01)  # steady uptrend
    res = TB.triple_barrier_from_signals(prices, signal_idx=[0, 5, 10],
                                         signal_sides=[1, 1, 1], span=10,
                                         pt_mult=1.0, sl_mult=2.0, max_horizon=5,
                                         cost=0.001)
    summ = TB.label_summary(res)
    assert summ["n"] == 3
    assert summ["pt"] >= 1  # the uptrend hits the profit
    assert 0.0 <= summ["meta_rate"] <= 1.0
