"""Offline tests for the CPCV->optimization->gate pipeline (synthetic objective)."""
import math
import os
import random
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "tools"))
import egp_cpcv_pipeline as P
from egp_mho_hybrid import OptDim

# --- objectif synthetique : 3 dimensions, optimum cache vec_opt ---
N_OBS = 250
DIMS = [OptDim("a", -2.0, 2.0, False, kind="cont"),
        OptDim("b", -2.0, 2.0, False, kind="cont"),
        OptDim("c", -2.0, 2.0, False, kind="cont")]
VEC_OPT = [1.0, -0.5, 0.3]


def _closeness(vec):
    d2 = sum((v - o) ** 2 for v, o in zip(vec, VEC_OPT))
    return math.exp(-d2)               # in (0,1], peak at vec_opt


def make_returns_on(alpha):
    def returns_on(vec, idx):
        c = _closeness(vec)
        rng = random.Random(hash(tuple(round(v, 2) for v in vec)) % 10_000_019)
        return [c * alpha[i] + 0.05 * rng.gauss(0, 1) for i in idx]
    return returns_on


def test_counts_and_path_lengths():
    rng = random.Random(0)
    alpha = [0.5 + 0.1 * rng.gauss(0, 1) for _ in range(N_OBS)]   # edge positif partout
    run = P.cpcv_run(DIMS, N_OBS, make_returns_on(alpha), n_groups=5, k_test=2,
                     inner_evals=120, seed=1)
    assert run["n_splits"] == 10 and run["n_paths"] == 4
    paths = P.assemble_path_returns(run)
    assert len(paths) == 4
    assert all(len(p) == N_OBS for p in paths)        # each path covers the whole timeline


def test_generalizing_edge_accepts():
    # alpha with positive mean EVERYWHERE -> the optimum generalizes -> positive OOS Sharpe -> ACCEPT.
    rng = random.Random(1)
    alpha = [0.6 + 0.1 * rng.gauss(0, 1) for _ in range(N_OBS)]
    gate = P.cpcv_walk_forward_gate(DIMS, N_OBS, make_returns_on(alpha), n_groups=5,
                                    k_test=2, inner_evals=200, seed=2, n_trials=30)
    assert gate["median_oos_sharpe"] > 0
    assert gate["frac_positive_paths"] >= 0.5
    assert gate["decision"] == "ACCEPT", gate


def test_pure_noise_rejects():
    # alpha with ZERO mean -> no real edge -> OOS Sharpe ~0 -> REJECT.
    rng = random.Random(2)
    alpha = [rng.gauss(0, 1) for _ in range(N_OBS)]
    gate = P.cpcv_walk_forward_gate(DIMS, N_OBS, make_returns_on(alpha), n_groups=5,
                                    k_test=2, inner_evals=200, seed=3, n_trials=30)
    assert gate["decision"] == "REJECT", gate
    assert gate["dsr"] < 0.95


def test_overfit_gap_reported():
    rng = random.Random(3)
    alpha = [rng.gauss(0, 1) for _ in range(N_OBS)]   # noise: no real edge
    gate = P.cpcv_walk_forward_gate(DIMS, N_OBS, make_returns_on(alpha), n_groups=5,
                                    k_test=2, inner_evals=200, seed=4)
    assert "mean_overfit_gap" in gate and "mean_oos_metric" in gate
    # gap and OOS metric reported and finite (the point of the test = traceability)
    assert math.isfinite(gate["mean_overfit_gap"])
    assert math.isfinite(gate["mean_oos_metric"])


def test_purge_embargo_passthrough():
    rng = random.Random(4)
    alpha = [0.5 for _ in range(N_OBS)]
    run = P.cpcv_run(DIMS, N_OBS, make_returns_on(alpha), n_groups=6, k_test=2,
                     purge=3, embargo=2, inner_evals=80, seed=5)
    # with purge/embargo, the splits stay valid and produce complete paths
    assert run["n_splits"] == 15 and run["n_paths"] == 5
    paths = P.assemble_path_returns(run)
    assert len(paths) == 5 and all(len(p) == N_OBS for p in paths)
