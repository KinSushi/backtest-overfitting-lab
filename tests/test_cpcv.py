"""Offline tests for egp_cpcv (deterministic)."""
import os
import sys
from collections import Counter
from itertools import combinations
from math import comb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_cpcv as CV


def test_split_and_path_counts_golden():
    # Values published by Lopez de Prado.
    assert CV.n_splits(6, 2) == 15 and CV.n_paths(6, 2) == 5
    assert CV.n_splits(10, 8) == 45 and CV.n_paths(10, 8) == 36
    assert CV.n_splits(4, 2) == 6 and CV.n_paths(4, 2) == 3
    sp = CV.cpcv_splits(600, n_groups=6, k_test=2)
    assert len(sp) == 15


def test_train_test_disjoint_and_cover_groups():
    sp = CV.cpcv_splits(600, n_groups=6, k_test=2)
    for s in sp:
        assert set(s["train_idx"]).isdisjoint(set(s["test_idx"]))
    # each group appears in test exactly C(N-1,k-1)=5 times
    cnt = Counter()
    for s in sp:
        cnt.update(s["test_groups"])
    assert all(cnt[g] == 5 for g in range(6))


def test_groups_partition_all_indices():
    # n_obs not divisible : the last group takes the remainder, union = all
    groups = CV.make_groups(605, 6)
    covered = sorted(i for (a, b) in groups for i in range(a, b))
    assert covered == list(range(605))
    assert groups[-1][1] - groups[-1][0] >= groups[0][1] - groups[0][0]


def test_purge_removes_indices_before_test_blocks():
    purge = 7
    sp = CV.cpcv_splits(600, n_groups=6, k_test=2, purge=purge)
    groups = CV.make_groups(600, 6)
    for s in sp:
        tr = set(s["train_idx"])
        for g in s["test_groups"]:
            start = groups[g][0]
            for i in range(max(0, start - purge), start):
                assert i not in tr            # zone de purge vide cote train


def test_embargo_removes_indices_after_test_blocks():
    embargo = 5
    sp = CV.cpcv_splits(600, n_groups=6, k_test=2, embargo=embargo)
    groups = CV.make_groups(600, 6)
    for s in sp:
        tr = set(s["train_idx"])
        for g in s["test_groups"]:
            end = groups[g][1]
            for i in range(end, min(600, end + embargo)):
                assert i not in tr            # zone d'embargo vide cote train


def test_t1_overlap_purging():
    # Long labels: each obs i has a horizon [i, i+30]. A train obs just before
    # a test block (label extending into the test) must be purged.
    n = 600
    horizon = 30
    t1 = [min(n - 1, i + horizon) for i in range(n)]
    sp = CV.cpcv_splits(n, n_groups=6, k_test=2, t1=t1)
    groups = CV.make_groups(n, 6)
    for s in sp:
        tr = set(s["train_idx"])
        for g in s["test_groups"]:
            start = groups[g][0]
            # an obs at start-1 has a label [start-1, start-1+30] that overlaps the test
            if start - 1 >= 0 and (start - 1) not in set(s["test_idx"]):
                assert (start - 1) not in tr


def test_assemble_paths_structure():
    N, k = 6, 2
    paths = CV.assemble_paths(N, k)
    assert len(paths) == CV.n_paths(N, k)            # 5 paths
    for p in paths:
        assert set(p.keys()) == set(range(N))         # each path covers all groups
    # each (split, group-in-test) pair used exactly once
    used = Counter()
    for p in paths:
        for g, ci in p.items():
            used[(ci, g)] += 1
    combos = list(combinations(range(N), k))
    total_pairs = sum(len(c) for c in combos)         # = k*C(N,k)
    assert sum(used.values()) == total_pairs
    assert all(v == 1 for v in used.values())


def test_evaluate_returns_distribution():
    # score_fn returning the test size -> distribution over the 15 splits.
    res = CV.evaluate(lambda tr, te: float(len(te)), 600, n_groups=6, k_test=2)
    assert res["n_splits"] == 15 and res["n_paths"] == 5
    assert len(res["scores"]) == 15
    assert res["min"] is not None and res["max"] is not None
