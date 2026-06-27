"""Offline tests for bet sizing AFML ch.10 (golden vectors via NormalDist)."""
import os
import sys
from statistics import NormalDist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_bet_sizing as BS

_PHI = NormalDist()


def test_prob_half_gives_zero_size():
    # p=0.5 -> z=0 -> 2*Phi(0)-1 = 0 (no conviction, no bet)
    assert abs(BS.prob_to_size(0.5) - 0.0) < 1e-12


def test_prob_to_size_golden():
    # p=0.75 : z=(0.75-0.5)/sqrt(0.75*0.25)=0.25/0.4330127=0.57735 ; 2*Phi(z)-1
    p = 0.75
    z = (p - 0.5) / (p * (1 - p)) ** 0.5
    expected = 2 * _PHI.cdf(z) - 1
    assert abs(BS.prob_to_size(p) - expected) < 1e-12
    assert abs(expected - 0.43624) < 1e-4  # expected numerical value


def test_size_monotone_in_prob():
    sizes = [BS.prob_to_size(p) for p in (0.5, 0.6, 0.7, 0.8, 0.9, 0.99)]
    assert all(sizes[i] < sizes[i + 1] for i in range(len(sizes) - 1))
    assert sizes[-1] < 1.0 and sizes[0] == 0.0  # upper bound < 1, p=0.5 -> 0


def test_prob_below_half_negative():
    # p<0.5 -> negative size (bet against / abstain)
    assert BS.prob_to_size(0.3) < 0.0


def test_meta_side_flips_sign():
    s_long = BS.bet_size(0.8, side=1)
    s_short = BS.bet_size(0.8, side=-1)
    assert s_long > 0 and s_short < 0
    assert abs(s_long + s_short) < 1e-12            # symetriques


def test_discretize_golden():
    # 0.46 / 0.25 = 1.84 -> round=2 -> 0.5
    assert abs(BS.discretize_signal(0.46, 0.25) - 0.5) < 1e-12
    # bounds
    assert BS.discretize_signal(1.3, 0.25) == 1.0
    assert BS.discretize_signal(-1.3, 0.25) == -1.0
    # step<=0 -> bound only
    assert abs(BS.discretize_signal(0.37, 0.0) - 0.37) < 1e-12


def test_avg_active_signals_golden():
    # bet A active [0,5) signal 0.4 ; bet B active [3,8) signal 0.8
    sig = [0.4, 0.8]
    t0s = [0, 3]
    t1s = [5, 8]
    avg = BS.avg_active_signals(sig, t0s, t1s, time_points=[1, 4, 6])
    assert abs(avg[1] - 0.4) < 1e-12                # seul A actif
    assert abs(avg[4] - 0.6) < 1e-12  # A and B -> mean 0.6
    assert abs(avg[6] - 0.8) < 1e-12                # seul B actif


def test_avg_active_no_bet_is_zero():
    avg = BS.avg_active_signals([0.5], [2], [4], time_points=[0, 5])
    assert avg[0] == 0.0 and avg[5] == 0.0  # outside window -> 0


def test_end_to_end_meta_bet_sizes():
    probs = [0.9, 0.55, 0.7]
    sides = [1, -1, 1]
    t0s = [0, 1, 2]
    t1s = [3, 4, 6]
    out = BS.bet_sizes_from_meta(probs, sides, t0s, t1s, step_size=0.1)
    assert len(out["per_bet_signal"]) == 3
    assert out["per_bet_signal"][0] > 0             # long forte conviction
    assert out["per_bet_signal"][1] < 0             # short
    # discretizes bounded in [-1,1]
    assert all(-1.0 <= v <= 1.0 for v in out["discretized"].values())


def test_size_to_lots():
    assert abs(BS.size_to_lots(0.5, max_lots=2.0) - 1.0) < 1e-12
    assert abs(BS.size_to_lots(-1.5, max_lots=2.0) - (-2.0)) < 1e-12  # bound
