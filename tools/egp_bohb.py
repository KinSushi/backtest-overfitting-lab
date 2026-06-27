"""
egp_bohb.py - BOHB (Bayesian Optimization + HyperBand), multi-fidelity.

Falkner, Klein & Hutter (2018), "BOHB: Robust and Efficient Hyperparameter Optimization
at Scale", ICML, pp.1436-1445. Idea: keep Hyperband's multi-fidelity allocation (robust,
anytime) but REPLACE the random sampling of configurations with a model (KDE / TPE) learned
on past observations -> faster convergence toward the good regions while keeping Hyperband's
guarantees.

Implementation: we REUSE as-is the tested building blocks of egp_meta_algos:
  - hyperband(get_config, score_fn, max_resource, eta, seed)  -> bracket structure
  - SimpleTPE(space, gamma, seed)                              -> KDE/TPE model
We inject a "model-based" get_config: with probability rand_frac (or while the model has too
few observations) we draw at random; otherwise we sample via TPE.suggest().

DOCUMENTED SIMPLIFICATIONS (honesty):
  * a SINGLE KDE over all observations (the original BOHB fits one KDE per budget level
    and picks the highest budget that has enough points);
  * the "resource"/fidelity is ABSTRACT: here a synthetic objective; on the EA side it would
    correspond e.g. to the number of WFO windows or the history length. The real multi-fidelity
    MT5 wiring (variable-budget objective) remains to be done separately.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from egp_meta_algos import SimpleTPE, hyperband  # tested reusable bricks


def space_from_dims(dims) -> Dict[str, dict]:
    """Converts a list of OptDim (egp_mho_hybrid) into a SimpleTPE 'space'.
    int/cont -> bounded float|int; enum/bool -> categorical (choice indices)."""
    space: Dict[str, dict] = {}
    for d in dims:
        kind = getattr(d, "kind", "cont")
        if kind == "bool":
            space[d.name] = {"type": "cat", "choices": [0, 1]}
        elif kind == "enum":
            n = len(getattr(d, "choices", ()) or ())
            space[d.name] = {"type": "cat", "choices": list(range(max(1, n)))}
        elif kind == "int":
            space[d.name] = {"type": "int", "min": int(d.lo), "max": int(d.hi)}
        else:
            space[d.name] = {"type": "float", "min": float(d.lo), "max": float(d.hi)}
    return space


def bohb(objective: Callable[[Dict, float], float], space: Dict[str, dict],
         max_resource: float = 27, eta: int = 3, rand_frac: float = 1.0 / 3.0,
         min_points: int = 8, seed: int = 0) -> Tuple[Dict, float, List[dict]]:
    """
    BOHB. objective(config, budget) -> cost to MINIMIZE (budget = allocated resource).
    Returns (best_config, best_score, brackets). Reuses hyperband + SimpleTPE.

    rand_frac : fraction of configurations drawn at random (exploration, like BOHB);
    min_points : minimum number of observations before activating the TPE model.
    """
    tpe = SimpleTPE(space, seed=seed)
    n_model = [0]
    n_random = [0]

    def score_fn(config: Dict, budget: float) -> float:
        y = objective(config, budget)
        tpe.observe(config, y)            # feeds the model at all budgets
        return y

    def get_config(rng) -> Dict:
        if rng.random() < rand_frac or len(tpe.samples) < min_points:
            n_random[0] += 1
            return tpe.random_candidate()
        n_model[0] += 1
        return tpe.suggest(64)

    best_c, best_s, brackets = hyperband(get_config, score_fn,
                                         max_resource=max_resource, eta=eta, seed=seed)
    for b in brackets:
        b["n_model"] = n_model[0]
        b["n_random"] = n_random[0]
    return best_c, best_s, brackets
