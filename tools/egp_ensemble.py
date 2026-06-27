"""
egp_ensemble.py - Cross-segment/cross-symbol generalization and parameter ensembles.

Two ideas to reduce overfitting and variance:
  1) Do not elect the SINGLE "best" parameter set (often lucky on one segment),
     but an ENSEMBLE of the top-k ROBUST ones (good everywhere) via a criterion penalizing
     dispersion (median, worst-case, or mean - lambda*std).
  2) Check GENERALIZATION: an edge that only holds on one segment/symbol is
     suspicious. We measure mean performance, dispersion, worst-case and RANK STABILITY
     across segments (is the ordering of configs preserved?).

Source: Lopez de Prado, *Advances in Financial Machine Learning* (2018), ch.6 (combinatorial
bagging, ensembles to reduce estimation variance).

NB: real cross-SYMBOL validation (XAUUSD + correlated metals) requires data from
several symbols (deferred). This module provides the ANALYZER (consumes a configs x
segments matrix) and the SELECTION, tested offline on synthetic matrices. The
combination of SIGNALS at execution (vote on the side, sizing average) remains an EA-side
spec.
"""
from __future__ import annotations

from statistics import median, pstdev
from typing import Dict, List, Optional, Sequence, Tuple


def robust_score(perf_row: Sequence[float], method: str = "median",
                 lam: float = 1.0) -> float:
    """Robust score of a config over its segments. 'median' | 'worst' (min) |
    'mean_minus_std' (mean - lambda*std)."""
    vals = [float(v) for v in perf_row]
    if not vals:
        return float("-inf")
    if method == "worst":
        return min(vals)
    if method == "mean_minus_std":
        return sum(vals) / len(vals) - lam * (pstdev(vals) if len(vals) > 1 else 0.0)
    return median(vals)


def select_ensemble(perf_matrix: Sequence[Sequence[float]], k: int = 3,
                    method: str = "median", lam: float = 1.0,
                    config_names: Optional[Sequence[str]] = None) -> Dict:
    """Selects the k configs with the best ROBUST score. perf_matrix: configs x segments.
    Returns indices, names, scores and the full ranking."""
    n = len(perf_matrix)
    names = list(config_names) if config_names is not None else [f"cfg{i}" for i in range(n)]
    scores = [robust_score(perf_matrix[i], method, lam) for i in range(n)]
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    sel = order[:max(1, min(k, n))]
    return {"selected_idx": sel,
            "selected_names": [names[i] for i in sel],
            "scores": scores,
            "ranking": [(names[i], scores[i]) for i in order]}


def _ranks(vals: Sequence[float]) -> List[float]:
    """Average ranks (ties handled)."""
    idx = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and vals[idx[j + 1]] == vals[idx[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[idx[t]] = avg
        i = j + 1
    return ranks


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    ra, rb = _ranks(a), _ranks(b)
    n = len(a)
    ma = sum(ra) / n
    mb = sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((x - mb) ** 2 for x in rb)
    if va <= 0 or vb <= 0:
        return 0.0
    return cov / (va ** 0.5 * vb ** 0.5)


def generalization_report(perf_matrix: Sequence[Sequence[float]],
                          config_names: Optional[Sequence[str]] = None,
                          positive_threshold: float = 0.0) -> Dict:
    """For each config: mean, std, worst-case (min), 'generalizes' (min > threshold).
    Plus the inter-segment RANK STABILITY = mean Spearman over the pairs of segments
    (1 = config ordering identical everywhere; ~0 = unstable)."""
    n = len(perf_matrix)
    if n == 0:
        return {"configs": [], "rank_stability": None}
    nseg = len(perf_matrix[0])
    names = list(config_names) if config_names is not None else [f"cfg{i}" for i in range(n)]
    configs = []
    for i in range(n):
        row = [float(v) for v in perf_matrix[i]]
        configs.append({
            "name": names[i],
            "mean": sum(row) / len(row),
            "std": pstdev(row) if len(row) > 1 else 0.0,
            "worst": min(row),
            "generalizes": min(row) > positive_threshold,
        })
    # rank stability: mean Spearman across all pairs of segments
    cols = [[perf_matrix[i][s] for i in range(n)] for s in range(nseg)]
    pairs = []
    for a in range(nseg):
        for b in range(a + 1, nseg):
            pairs.append(_spearman(cols[a], cols[b]))
    rank_stability = sum(pairs) / len(pairs) if pairs else None
    return {"configs": configs, "rank_stability": rank_stability, "n_segments": nseg}


def combine_param_vectors(vectors: Sequence[Sequence[float]],
                          kinds: Sequence[str]) -> List[float]:
    """Combines k parameter vectors: median for 'cont'/'int', mode for 'enum'/'bool'.
    kinds: per-dimension list ('cont','int','enum','bool')."""
    if not vectors:
        return []
    d = len(vectors[0])
    out = []
    for j in range(d):
        col = [v[j] for v in vectors]
        kind = kinds[j] if j < len(kinds) else "cont"
        if kind in ("enum", "bool"):
            # mode (most frequent value; on a tie, the smallest)
            counts = {}
            for x in col:
                counts[x] = counts.get(x, 0) + 1
            best = max(sorted(counts), key=lambda x: counts[x])
            out.append(best)
        elif kind == "int":
            out.append(round(median(col)))
        else:
            out.append(median(col))
    return out
