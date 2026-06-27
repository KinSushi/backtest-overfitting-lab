"""Offline tests for egp_sobol: validation against the Ishigami function (analytical)."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_sobol as SB


def _ishigami(x, a=7.0, b=0.1):
    return math.sin(x[0]) + a * math.sin(x[1]) ** 2 + b * (x[2] ** 4) * math.sin(x[0])


BOUNDS = [(-math.pi, math.pi)] * 3
# Analytical values (a=7, b=0.1) confirmed on 3 concordant sources
# (scipy.stats.sobol_indices, sensitivity R, arXiv 0802.1008) :
S_ANALYTIC = [0.3139, 0.4424, 0.0]
ST_ANALYTIC = [0.558, 0.442, 0.244]


def test_ishigami_first_order_matches_analytic():
    idx = SB.sobol_indices(_ishigami, BOUNDS, n_base=65536, seed=1)
    for i in range(3):
        assert abs(idx["first"][i] - S_ANALYTIC[i]) < 0.03, (i, idx["first"][i])


def test_ishigami_total_order_matches_analytic():
    idx = SB.sobol_indices(_ishigami, BOUNDS, n_base=65536, seed=1)
    for i in range(3):
        assert abs(idx["total"][i] - ST_ANALYTIC[i]) < 0.03, (i, idx["total"][i])


def test_x3_has_no_first_but_nonzero_total():
    # Ishigami signature: x3 with no direct effect (S3~0) but interaction with x1 (ST3~0.244).
    idx = SB.sobol_indices(_ishigami, BOUNDS, n_base=65536, seed=2)
    assert idx["first"][2] < 0.05
    assert idx["total"][2] > 0.15


def test_rank_and_select():
    idx = SB.sobol_indices(_ishigami, BOUNDS, n_base=32768, seed=3)
    ranked = SB.rank_importance(idx, names=["x1", "x2", "x3"])
    # x1 (direct effect + interaction) has the strongest total index
    assert ranked[0][0] == "x1"
    part = SB.select_by_total_index(idx, ["x1", "x2", "x3"], threshold=0.01)
    assert set(part["keep"]) == {"x1", "x2", "x3"}  # all > 0.01 here (ST3~0.24)
    # with a high threshold, x3 (zero direct) does not drop because its TOTAL is ~0.24
    assert "x1" in part["keep"] and "x2" in part["keep"]
