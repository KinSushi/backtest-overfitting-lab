"""Offline tests for cMDA (AFML MLAM ch.6): distance correlation, clustering, substitution effect."""
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_cluster_importance as CI


def test_corr_distance_golden():
    # 2 features identiques -> rho=1 -> d=0 ; opposees -> rho=-1 -> d=1
    X = [[v, v, -v] for v in [-2.0, -1.0, 0.0, 1.0, 2.0]]
    D = CI.corr_distance_matrix(X)
    assert abs(D[0][1] - 0.0) < 1e-9          # identiques
    assert abs(D[0][2] - 1.0) < 1e-9          # opposees
    assert abs(D[0][0]) < 1e-12               # diagonale nulle


def test_clustering_groups_correlated_features():
    # f0,f1 correles ; f2,f3 correles (groupe distinct) -> 2 clusters {0,1} et {2,3}
    rng = random.Random(0)
    X = []
    for _ in range(200):
        a = rng.gauss(0, 1)
        b = rng.gauss(0, 1)
        X.append([a, a + rng.gauss(0, 0.05), b, b + rng.gauss(0, 0.05)])
    D = CI.corr_distance_matrix(X)
    clusters = CI.hierarchical_clusters(D, n_clusters=2)
    sets = sorted(sorted(c) for c in clusters)
    assert sets == [[0, 1], [2, 3]]


def test_clustering_threshold_path():
    rng = random.Random(1)
    X = [[rng.gauss(0, 1) for _ in range(4)] for _ in range(150)]
    D = CI.corr_distance_matrix(X)
    clusters = CI.hierarchical_clusters(D, threshold=0.1)   # features ~independantes -> peu fusionnees
    assert len(clusters) >= 1


def test_cmda_informative_cluster_dominates():
    # informative cluster (f0,f1 ~ determine y) vs noise cluster (f2,f3)
    rng = random.Random(2)
    X, y = [], []
    for _ in range(300):
        s = rng.gauss(0, 1)
        f0 = s + rng.gauss(0, 0.05)
        f1 = s + rng.gauss(0, 0.05)
        f2 = rng.gauss(0, 1)
        f3 = rng.gauss(0, 1)
        X.append([f0, f1, f2, f3])
        y.append(1 if s > 0 else 0)
    events = [(i, i + 2) for i in range(len(X))]
    res = CI.clustered_mda(X, y, events, n_clusters=2, n_splits=4, n_bars=len(X) + 3, seed=0)
    imp = res["importance"]
    # identify the cluster containing the informative feature 0
    info_ci = next(ci for ci, d in imp.items() if 0 in d["members"])
    noise_ci = next(ci for ci, d in imp.items() if 0 not in d["members"])
    assert imp[info_ci]["mean"] > imp[noise_ci]["mean"]


def test_cmda_handles_substitution_effect():
    # TWO IDENTICAL informative features -> the cluster groups them and shows high importance
    rng = random.Random(3)
    X, y = [], []
    for _ in range(300):
        s = rng.gauss(0, 1)
        dup = s + rng.gauss(0, 0.02)
        X.append([dup, dup, rng.gauss(0, 1)])  # f0==f1 (substitutes), f2 noise
        y.append(1 if s > 0 else 0)
    events = [(i, i + 2) for i in range(len(X))]
    res = CI.clustered_mda(X, y, events, n_clusters=2, n_splits=4, n_bars=len(X) + 3, seed=0)
    imp = res["importance"]
    info_ci = next(ci for ci, d in imp.items() if 0 in d["members"])
    # the substitute cluster is deemed important (substitution does not dilute it to 0)
    assert imp[info_ci]["mean"] > 0.05


def test_correlation_matrix_diagonal_one():
    rng = random.Random(4)
    X = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(100)]
    C = CI.correlation_matrix(X)
    for i in range(3):
        assert abs(C[i][i] - 1.0) < 1e-9
