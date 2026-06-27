"""
egp_cpcv_model.py - CPCV backtest at the MODEL level (forest + bet sizing) -> paths -> deflated gate.

The runner (egp_runner) evaluates the model via a simple purged k-fold (a single backtest path).
Combinatorial Purged CV (AFML ch.12) generates SEVERAL backtest paths, hence a DISTRIBUTION of
out-of-sample performance -> a much more robust estimate of generalization (and of PBO at the
config-selection level).

Here we connect the model to the already-verified CPCV infrastructure (egp_cpcv: cpcv_splits,
assemble_paths; egp_cpcv_pipeline.gate_from_cpcv): for each CPCV split, we train the forest on the
train (purge + embargo handled by CPCV), predict on the test groups, SIZE the bets (max(0,
prob_to_size)*(net return)), then ASSEMBLE the paths and pass them to the CPCV deflated gate.

THIS MODULE INTRODUCES NO FORMULA: orchestration of verified building blocks (egp_cpcv,
egp_tree_forest, egp_bet_sizing, egp_cpcv_pipeline). Pure-Python.
RESERVATION: uniform intra-train sampling (the inter-fold purge/embargo is ensured by CPCV);
per-bet returns provided (P0 lock: real MT5 deals). The proper PBO is computed at the multi-config
level (egp_accept_gate.pbo_cscv), not for a single strategy.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)

import egp_cpcv as CP
import egp_cpcv_pipeline as CPP
import egp_tree_forest as TF
import egp_bet_sizing as BS

Event = Tuple[int, int]


def _sized_return(prob: float, side_return: float, cost: float, step_size: float,
                  num_classes: int) -> float:
    m = BS.prob_to_size(prob, num_classes)
    if step_size > 0:
        m = BS.discretize_signal(m, step_size)
    return max(0.0, m) * (side_return - cost)


def cpcv_path_returns(X: Sequence[Sequence[float]], y: Sequence[int], bets: Sequence[Dict],
                      events: Sequence[Event], *, n_groups: int = 6, k_test: int = 2,
                      purge: int = 0, embargo: int = 0, forest_params: Optional[Dict] = None,
                      step_size: float = 0.0, num_classes: int = 2, seed: int = 0
                      ) -> List[List[float]]:
    """Generates the sized returns per CPCV backtest PATH. Each path covers all the
    observations once (predicted out-of-sample)."""
    fp = forest_params or {"n_estimators": 20, "max_depth": 5, "min_samples_leaf": 5}
    n_obs = len(X)
    t1 = [e[1] for e in events]
    groups = CP.make_groups(n_obs, n_groups)
    splits = CP.cpcv_splits(n_obs, n_groups=n_groups, k_test=k_test, purge=purge,
                            embargo=embargo, t1=t1)

    # sized return per (split, observation) for the split's test obs
    per_split_ret: List[Dict[int, float]] = []
    for s in splits:
        tr, te = s["train_idx"], s["test_idx"]
        Xtr = [X[i] for i in tr]
        ytr = [y[i] for i in tr]
        rf = TF.RandomForestClassifier(n_estimators=fp["n_estimators"], max_depth=fp["max_depth"],
                                       min_samples_leaf=fp["min_samples_leaf"],
                                       bootstrap="uniform", seed=seed)
        rf.fit(Xtr, ytr)
        probs = rf.predict_proba([X[i] for i in te])
        d = {}
        for j, i in enumerate(te):
            p1 = probs[j][1] if len(probs[j]) > 1 else probs[j][0]
            d[i] = _sized_return(p1, bets[i]["side_return"], bets[i].get("cost", 0.0),
                                 step_size, num_classes)
        per_split_ret.append(d)

    # path assembly: for each path, each group takes its predictions from the assigned split
    path_maps = CP.assemble_paths(n_groups, k_test)
    paths: List[List[float]] = []
    for pmap in path_maps:
        series: List[float] = []
        for g in range(n_groups):
            split_idx = pmap[g]
            start, end = groups[g]                       # make_groups returns an interval (start, end)
            for i in range(start, end):
                series.append(per_split_ret[split_idx].get(i, 0.0))
        paths.append(series)
    return paths


def cpcv_model_gate(X, y, bets, events, *, n_groups: int = 6, k_test: int = 2, purge: int = 0,
                    embargo: int = 0, forest_params: Optional[Dict] = None, step_size: float = 0.0,
                    num_classes: int = 2, n_trials: Optional[int] = None, dsr_min: float = 0.95,
                    min_frac_positive: float = 0.5, seed: int = 0) -> Dict:
    """Full CPCV backtest -> deflated gate on the distribution of paths.
    Returns {n_paths, path_returns, gate, mean_path_return}."""
    paths = cpcv_path_returns(X, y, bets, events, n_groups=n_groups, k_test=k_test, purge=purge,
                              embargo=embargo, forest_params=forest_params, step_size=step_size,
                              num_classes=num_classes, seed=seed)
    gate = CPP.gate_from_cpcv(paths, n_trials=n_trials, dsr_min=dsr_min,
                              min_frac_positive=min_frac_positive)
    flat_means = [sum(p) / len(p) if p else 0.0 for p in paths]
    return {"n_paths": len(paths), "path_returns": paths, "gate": gate,
            "mean_path_return": sum(flat_means) / len(flat_means) if flat_means else 0.0}
