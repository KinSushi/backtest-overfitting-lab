"""
egp_cluster_importance.py - Clustered Feature Importance / cMDA (Lopez de Prado, MLAM ch.6).

PROBLEM (substitution effect): with correlated/redundant features, standard MDA makes BOTH appear
unimportant (each judges the other redundant), even if the group is critical. Same for MDI, which
DILUTES importance (two identical features -> importance / 2).

SOLUTION (CFI): (1) project the features into a metric space via a DEPENDENCE distance (here
correlation), (2) group by HIERARCHICAL clustering, (3) compute MDA on the CLUSTERS (we jointly
permute ALL the columns of a cluster) -> robust to substitution.

Formulas VERIFIED against concordant sources:
  - Correlation distance: d_ij = sqrt(0.5 * (1 - rho_ij))  (Mantegna 1999; HRP/MLAM Lopez de Prado).
    rho=1 -> d=0 (identical); rho=-1 -> d=1; rho=0 -> d=0.707. It is a true metric.
  - Agglomerative hierarchical clustering, average linkage: dist(A,B) = mean of d_ij, i in A, j in B.
  - cMDA: like MDA (permuting a feature's values across observations to break the link with y), but we
    permute the WHOLE CLUSTER with the SAME permutation (preserves the intra-cluster structure, breaks
    the cluster<->y link). importance(cluster) = base_score - permuted_score (in purged CV).

Reference: M. Lopez de Prado, *Machine Learning for Asset Managers* (2020), ch.6 "Clustered Feature
Importance"; Mantegna (1999) "Hierarchical structure in financial markets". Pure-Python.
RESERVATION: number of clusters determined by threshold/k (the book's ONC algorithm is not
implemented). Correlation-based distance (information-theoretic variant not implemented).
"""
from __future__ import annotations

import math
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)

import egp_cv_importance as CV

Event = Tuple[int, int]


# --------------------------------------------------------------------------- correlation / distance
def correlation_matrix(X: Sequence[Sequence[float]]) -> List[List[float]]:
    """Pearson correlation matrix between columns (features)."""
    n = len(X)
    m = len(X[0]) if n else 0
    means = [sum(X[i][j] for i in range(n)) / n for j in range(m)]
    var = [sum((X[i][j] - means[j]) ** 2 for i in range(n)) for j in range(m)]
    C = [[0.0] * m for _ in range(m)]
    for a in range(m):
        for b in range(m):
            if var[a] <= 0 or var[b] <= 0:
                C[a][b] = 1.0 if a == b else 0.0
                continue
            cov = sum((X[i][a] - means[a]) * (X[i][b] - means[b]) for i in range(n))
            C[a][b] = cov / math.sqrt(var[a] * var[b])
    return C


def corr_distance_matrix(X: Sequence[Sequence[float]]) -> List[List[float]]:
    """Correlation distance d_ij = sqrt(0.5*(1-rho_ij)) (Mantegna 1999)."""
    C = correlation_matrix(X)
    m = len(C)
    D = [[0.0] * m for _ in range(m)]
    for a in range(m):
        for b in range(m):
            D[a][b] = math.sqrt(max(0.0, 0.5 * (1.0 - C[a][b])))
    return D


# --------------------------------------------------------------------------- hierarchical clustering
def hierarchical_clusters(dist: Sequence[Sequence[float]], n_clusters: Optional[int] = None,
                          threshold: Optional[float] = None) -> List[List[int]]:
    """Agglomerative clustering (average linkage). Stops at n_clusters, or when the smallest
    inter-cluster distance exceeds `threshold`. Returns the list of clusters (feature indices)."""
    m = len(dist)
    clusters: List[List[int]] = [[i] for i in range(m)]

    def cluster_dist(A, B):
        return sum(dist[i][j] for i in A for j in B) / (len(A) * len(B))

    while len(clusters) > 1:
        if n_clusters is not None and len(clusters) <= n_clusters:
            break
        best = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                d = cluster_dist(clusters[a], clusters[b])
                if best is None or d < best[0]:
                    best = (d, a, b)
        if best is None:
            break
        if threshold is not None and best[0] > threshold and (n_clusters is None):
            break
        _, a, b = best
        clusters[a] = clusters[a] + clusters[b]
        clusters.pop(b)
    return clusters


# --------------------------------------------------------------------------- cMDA
def _permute_columns(X, rows_idx, cols, perm):
    """Returns a copy of X[rows_idx] with the columns `cols` permuted according to `perm` (same perm)."""
    sub = [list(X[r]) for r in rows_idx]
    permuted_vals = [[X[rows_idx[perm[k]]][c] for c in cols] for k in range(len(rows_idx))]
    for k in range(len(rows_idx)):
        for ci, c in enumerate(cols):
            sub[k][c] = permuted_vals[k][ci]
    return sub


def clustered_mda(X: Sequence[Sequence[float]], y: Sequence[int], events: Sequence[Event],
                  clusters: Optional[List[List[int]]] = None, n_clusters: Optional[int] = None,
                  threshold: Optional[float] = None, n_splits: int = 5, embargo_pct: float = 0.0,
                  fit_fn=None, score_fn=None, seed: int = 0, n_bars: Optional[int] = None) -> Dict:
    """cMDA: importance per CLUSTER (joint permutation of the group) in purged CV.
    Returns {clusters, importance:{cluster_id:{mean,std,members}}}. Robust to the substitution effect."""
    fit_fn = fit_fn or CV.default_logreg_fit()
    score_fn = score_fn or CV.accuracy_score
    if clusters is None:
        D = corr_distance_matrix(X)
        clusters = hierarchical_clusters(D, n_clusters=n_clusters, threshold=threshold)
    if n_bars is None:
        n_bars = max(t1 for _, t1 in events) + 1
    folds = CV.purged_kfold(events, n_splits=n_splits, embargo_pct=embargo_pct, n_bars=n_bars)
    rng = random.Random(seed)

    drops: Dict[int, List[float]] = {ci: [] for ci in range(len(clusters))}
    for fold in folds:
        tr, te = fold["train_idx"], fold["test_idx"]
        if not tr or not te:
            continue
        Xtr = [X[i] for i in tr]
        ytr = [y[i] for i in tr]
        predict = fit_fn(Xtr, ytr)
        Xte = [X[i] for i in te]
        yte = [y[i] for i in te]
        base = score_fn(predict, Xte, yte)
        for ci, cols in enumerate(clusters):
            perm = list(range(len(te)))
            rng.shuffle(perm)
            Xte_perm = _permute_columns(X, te, cols, perm)
            score_perm = score_fn(predict, Xte_perm, yte)
            drops[ci].append(base - score_perm)             # score drop = importance

    importance = {}
    for ci, vals in drops.items():
        if vals:
            mean = sum(vals) / len(vals)
            std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        else:
            mean, std = 0.0, 0.0
        importance[ci] = {"mean": mean, "std": std, "members": clusters[ci]}
    return {"clusters": clusters, "importance": importance}
