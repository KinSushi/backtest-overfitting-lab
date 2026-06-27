"""Offline tests for egp_ensemble (deterministic)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_ensemble as E


def test_robust_selection_prefers_consistent_over_lucky():
    # cfg0 = good everywhere (1.0); cfg1 = excellent on 1 segment (5.0) but bad elsewhere.
    perf = [
        [1.0, 1.0, 1.0, 1.0],     # cfg0 : robuste
        [5.0, -1.0, -1.0, -1.0],  # cfg1 : lucky on one segment
    ]
    # Robust criterion (median) -> cfg0 ; the raw mean would favor... let's check
    sel_med = E.select_ensemble(perf, k=1, method="median")
    assert sel_med["selected_idx"] == [0]
    sel_worst = E.select_ensemble(perf, k=1, method="worst")
    assert sel_worst["selected_idx"] == [0]
    # The raw mean of the two: cfg0=1.0 vs cfg1=0.5 -> cfg0 here too, but the worst-case
    # discriminates more clearly (cfg1 worst=-1.0).
    assert E.robust_score(perf[1], "worst") < E.robust_score(perf[0], "worst")


def test_mean_minus_std_penalizes_dispersion():
    stable = [1.0, 1.1, 0.9, 1.0]
    volatile = [3.0, -1.0, 3.0, -1.0]      # mean 1.0 but very dispersed
    assert (E.robust_score(stable, "mean_minus_std", lam=1.0)
            > E.robust_score(volatile, "mean_minus_std", lam=1.0))


def test_generalization_flag_and_rank_stability():
    perf = [
        [1.0, 1.2, 0.8],     # cfg0 generalise (min>0)
        [2.0, -0.5, 1.0],    # cfg1 does not generalize (min<0)
        [0.5, 0.6, 0.4],     # cfg2 generalise
    ]
    rep = E.generalization_report(perf, config_names=["a", "b", "c"])
    flags = {c["name"]: c["generalizes"] for c in rep["configs"]}
    assert flags["a"] and not flags["b"] and flags["c"]
    assert rep["rank_stability"] is not None

    # Identical ranks across all segments -> stability = 1
    consistent = [[3.0, 3.0, 3.0], [2.0, 2.0, 2.0], [1.0, 1.0, 1.0]]
    assert abs(E.generalization_report(consistent)["rank_stability"] - 1.0) < 1e-9


def test_combine_param_vectors_median_and_mode():
    vectors = [
        [10.0, 1.0, 2.0],
        [12.0, 1.0, 5.0],
        [14.0, 0.0, 2.0],
    ]
    kinds = ["cont", "bool", "int"]
    combined = E.combine_param_vectors(vectors, kinds)
    assert abs(combined[0] - 12.0) < 1e-9      # mediane de 10,12,14
    assert combined[1] == 1.0                  # mode de 1,1,0
    assert combined[2] == 2                     # mediane(2,5,2)=2 arrondie


def test_ensemble_size_respected():
    perf = [[float(i)] * 3 for i in range(10)]
    sel = E.select_ensemble(perf, k=3, method="median")
    assert len(sel["selected_idx"]) == 3
    # the 3 best (indices 9,8,7)
    assert set(sel["selected_idx"]) == {9, 8, 7}
