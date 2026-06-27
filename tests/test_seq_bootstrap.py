"""Offline tests for the sequential bootstrap AFML ch.4.5 (golden + Monte-Carlo property)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_seq_bootstrap as SB


def test_avg_uniqueness_empty_phi_all_ones():
    # empty phi -> concurrency 0 everywhere -> avgU = 1.0 for all (1st uniform draw)
    au = SB.avg_uniqueness_given_drawn([(0, 1), (1, 2), (5, 6)], n_bars=7, phi=[])
    assert all(abs(u - 1.0) < 1e-12 for u in au)


def test_avg_uniqueness_given_drawn_golden():
    # events L0=[0,1], L1=[1,2], L2=[5,6] ; phi=[0]
    # L0 redraw: bars 0,1 conc=1(L0)+1=2 -> 1/2,1/2 -> 0.5
    # L1: bar1 conc=1+1=2, bar2 conc=0+1=1 -> mean(0.5,1)=0.75
    # L2: bars 5,6 conc=0+1=1 -> 1.0
    au = SB.avg_uniqueness_given_drawn([(0, 1), (1, 2), (5, 6)], n_bars=7, phi=[0])
    assert abs(au[0] - 0.5) < 1e-12
    assert abs(au[1] - 0.75) < 1e-12
    assert abs(au[2] - 1.0) < 1e-12


def test_seq_bootstrap_length_and_range():
    events = [(i, i + 2) for i in range(10)]
    phi = SB.seq_bootstrap(events, n_bars=13, s_length=8, seed=1)
    assert len(phi) == 8
    assert all(0 <= p < len(events) for p in phi)


def test_seq_bootstrap_deterministic():
    events = [(i, i + 2) for i in range(10)]
    a = SB.seq_bootstrap(events, n_bars=13, seed=42)
    b = SB.seq_bootstrap(events, n_bars=13, seed=42)
    assert a == b  # same seed -> same draw


def test_disjoint_label_favored_after_draw():
    # after forcing the draw of L0, the next probability favors the disjoint L2 > L1 > L0
    events = [(0, 1), (1, 2), (5, 6)]
    au = SB.avg_uniqueness_given_drawn(events, n_bars=7, phi=[0])
    probs = [u / sum(au) for u in au]
    assert probs[2] > probs[1] > probs[0]           # disjointe > partielle > redondante


def test_seq_more_unique_than_uniform():
    # strongly overlapping labels -> the sequential must be MORE unique than the uniform
    events = [(i, i + 8) for i in range(40)]         # large recouvrement
    cmp = SB.compare_bootstraps(events, n_bars=50, n_runs=40, seed=0)
    assert cmp["seq_mean_uniqueness"] > cmp["uniform_mean_uniqueness"]


def test_sample_avg_uniqueness_bounds():
    events = [(i, i + 2) for i in range(10)]
    sample = SB.seq_bootstrap(events, n_bars=13, seed=3)
    u = SB.sample_avg_uniqueness(events, n_bars=13, sample=sample)
    assert 0.0 < u <= 1.0
