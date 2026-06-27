"""
egp_sobol.py - Global sensitivity analysis (Sobol indices).

Attributes the VARIANCE of performance to each parameter:
  - first-order index S_i  : direct effect of parameter i alone;
  - total-order index  S_Ti   : effect of i + all its interactions.
Target use: turn egp_core_select from a HEURISTIC reduction into an EMPIRICAL
reduction - keep parameters with high S_Ti, freeze those with S_Ti ~ 0. This requires a
score function (params -> performance) from REAL BACKTESTS (deferred); the module
is nonetheless VALIDATED offline against known analytical values (Ishigami function).

Sources:
  - Sobol, I. M. (2001), "Global sensitivity indices for nonlinear mathematical models and
    their Monte Carlo estimates", Mathematics and Computers in Simulation 55(1-3):271-280.
  - Saltelli, A. et al. (2010), "Variance based sensitivity analysis of model output. Design
    and estimator for the total sensitivity index", Computer Physics Communications
    181(2):259-270.  -> first-order estimator.
  - Jansen, M. (1999) -> total-order estimator (V_Ti = (1/2N) Σ (f_A - f_AB^i)^2).
Sampling design A / B / A_B^i (N*(d+2) evaluations). Here pseudo-random draws
(a Sobol/LHS sequence would reduce variance; documented).
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Sequence, Tuple


def _matrix(n, bounds, rng):
    d = len(bounds)
    return [[rng.uniform(bounds[j][0], bounds[j][1]) for j in range(d)] for _ in range(n)]


def sobol_indices(func: Callable[[Sequence[float]], float],
                  bounds: Sequence[Tuple[float, float]],
                  n_base: int = 65536, seed: int = 0) -> Dict[str, List[float]]:
    """Returns {'first': [S_i], 'total': [S_Ti], 'var': VarY}. func(point)->scalar;
    bounds = [(lo,hi), ...] per input (inputs assumed INDEPENDENT, Sobol's assumption)."""
    d = len(bounds)
    rng = random.Random(seed)
    A = _matrix(n_base, bounds, rng)
    B = _matrix(n_base, bounds, rng)
    fA = [func(r) for r in A]
    fB = [func(r) for r in B]
    pooled = fA + fB
    mu = sum(pooled) / len(pooled)
    varY = sum((x - mu) ** 2 for x in pooled) / len(pooled)
    if varY <= 0:
        return {"first": [0.0] * d, "total": [0.0] * d, "var": 0.0}
    first: List[float] = []
    total: List[float] = []
    for i in range(d):
        # A_B^i : A with column i replaced by B's
        fABi = []
        for k in range(n_base):
            row = list(A[k]); row[i] = B[k][i]
            fABi.append(func(row))
        # first order (Saltelli 2010) : V_i = (1/N) Σ fB*(fAB^i - fA)
        Vi = sum(fB[k] * (fABi[k] - fA[k]) for k in range(n_base)) / n_base
        # total order (Jansen 1999) : V_Ti = (1/2N) Σ (fA - fAB^i)^2
        VTi = sum((fA[k] - fABi[k]) ** 2 for k in range(n_base)) / (2 * n_base)
        first.append(Vi / varY)
        total.append(VTi / varY)
    return {"first": first, "total": total, "var": varY}


def rank_importance(indices: Dict[str, List[float]], names: Sequence[str] = None
                    ) -> List[Tuple[str, float]]:
    """Sorts parameters by descending TOTAL-order index (= importance with
    interactions). Returns [(name, S_Ti), ...]."""
    tot = indices["total"]
    nm = list(names) if names is not None else [f"x{i}" for i in range(len(tot))]
    return sorted(zip(nm, tot), key=lambda t: t[1], reverse=True)


def select_by_total_index(indices: Dict[str, List[float]], names: Sequence[str],
                          threshold: float = 0.01) -> Dict[str, List[str]]:
    """Partitions parameters into 'keep' (S_Ti >= threshold) and 'freeze' (S_Ti < threshold)."""
    tot = indices["total"]
    keep, freeze = [], []
    for nm, st in zip(names, tot):
        (keep if st >= threshold else freeze).append(nm)
    return {"keep": keep, "freeze": freeze}
