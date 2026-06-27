"""
egp_cpcv.py - Combinatorial Purged Cross-Validation (CPCV).

Replaces the SINGLE-PATH walk-forward (high variance) with a DISTRIBUTION of out-of-sample
performance. T observations -> N contiguous groups; we pick k groups as TEST (the rest as
TRAIN), for all C(N,k) combinations. We purge (remove train obs whose label horizon overlaps
the test) and embargo (buffer after each test block). We then recombine phi = C(N-1, k-1)
complete out-of-sample PATHS.

Counts (verified against sources):
  - splits (train/test combinations) = C(N, k)
  - backtest paths phi            = C(N-1, k-1) = k*C(N,k)/N
  e.g. N=6,k=2 -> 15 splits, 5 paths; N=10,k=8 -> 45 splits, 36 paths.

Sources:
  - Lopez de Prado, M. (2018), *Advances in Financial Machine Learning*, Wiley, ch.7
    (Purged k-fold, embargo, CPCV). Also "The 10 Reasons Most ML Funds Fail" (groups:
    the first N-1 of size floor(T/N), the last takes the rest).
  - Reference implementations: timeseriescv (sam31415, MIT), skfolio CombinatorialPurgedCV.
Pure-Python.
"""
from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Callable, Dict, List, Optional, Sequence


def make_groups(n_obs: int, n_groups: int) -> List[tuple]:
    """Splits [0, n_obs) into n_groups contiguous blocks: the first n_groups-1 of size
    floor(n_obs/n_groups), the last takes the rest."""
    base = n_obs // n_groups
    groups = []
    s = 0
    for g in range(n_groups):
        e = (s + base) if g < n_groups - 1 else n_obs
        groups.append((s, e))
        s = e
    return groups


def n_splits(n_groups: int, k_test: int) -> int:
    return comb(n_groups, k_test)


def n_paths(n_groups: int, k_test: int) -> int:
    return comb(n_groups - 1, k_test - 1)


def cpcv_splits(n_obs: int, n_groups: int = 6, k_test: int = 2,
                purge: int = 0, embargo: int = 0,
                t1: Optional[Sequence[int]] = None) -> List[Dict]:
    """Generates the C(N,k) splits {test_groups, train_idx, test_idx}, with purge and embargo.
    purge : number of train obs removed BEFORE each test block (forward-looking labels).
    embargo : number of train obs removed AFTER each test block (autocorrelation).
    t1 : optional, per-observation label-horizon end -> EXACT purge by
         overlap [i, t1[i]] vs test block (in addition to the gap purge/embargo)."""
    groups = make_groups(n_obs, n_groups)
    all_idx = set(range(n_obs))
    splits: List[Dict] = []
    for test_g in combinations(range(n_groups), k_test):
        test_blocks = [groups[g] for g in test_g]
        test_idx = sorted(i for (s, e) in test_blocks for i in range(s, e))
        test_set = set(test_idx)
        train_set = all_idx - test_set
        remove = set()
        for (s, e) in test_blocks:
            for i in range(max(0, s - purge), s):          # purge (before)
                remove.add(i)
            for i in range(e, min(n_obs, e + embargo)):     # embargo (after)
                remove.add(i)
        if t1 is not None:
            for i in list(train_set):
                end_i = t1[i]
                for (s, e) in test_blocks:
                    # does [i, end_i] overlap [s, e)?
                    if not (end_i < s or i >= e):
                        remove.add(i)
                        break
        train_idx = sorted(train_set - remove)
        splits.append({"test_groups": test_g, "train_idx": train_idx, "test_idx": test_idx})
    return splits


def assemble_paths(n_groups: int, k_test: int) -> List[Dict[int, int]]:
    """Recombines the phi = C(N-1,k-1) out-of-sample paths. Each path assigns to
    EACH group the index of the split (combination) that provides its test prediction. Each
    (split, group-in-test) pair is used exactly once across all the paths."""
    combos = list(combinations(range(n_groups), k_test))
    phi = comb(n_groups - 1, k_test - 1)
    occ = {g: [] for g in range(n_groups)}
    for ci, combo in enumerate(combos):
        for g in combo:
            occ[g].append(ci)
    return [{g: occ[g][j] for g in range(n_groups)} for j in range(phi)]


def evaluate(score_fn: Callable[[List[int], List[int]], float],
             n_obs: int, n_groups: int = 6, k_test: int = 2,
             purge: int = 0, embargo: int = 0,
             t1: Optional[Sequence[int]] = None) -> Dict:
    """Applies score_fn(train_idx, test_idx) on all splits -> OOS distribution.
    Returns scores, mean, standard deviation, min/max and the number of paths."""
    splits = cpcv_splits(n_obs, n_groups, k_test, purge, embargo, t1)
    scores = [score_fn(sp["train_idx"], sp["test_idx"]) for sp in splits]
    n = len(scores)
    mean = sum(scores) / n if n else 0.0
    var = sum((x - mean) ** 2 for x in scores) / n if n else 0.0
    return {"scores": scores, "mean": mean, "std": var ** 0.5,
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
            "n_splits": n, "n_paths": n_paths(n_groups, k_test)}
