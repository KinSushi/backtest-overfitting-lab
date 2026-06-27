"""
egp_nested_cv.py - NESTED purged CV for hyperparameter tuning (AFML ch.7).

Problem: choosing hyperparameters by validating on the SAME folds that will serve to evaluate
performance leaks information (optimism). Solution: nested CV. OUTER loop (purged) to
estimate performance; on each outer train, a grid is evaluated by INNER CV (purged)
to choose the hyperparameters; the re-trained model is evaluated on the outer test (never seen
during tuning). The performance estimate is thus NOT BIASED by the selection.

THIS MODULE INTRODUCES NO FORMULA : it orchestrates egp_cv_importance.purged_kfold / cv_score
(purge + embargo VERIFIED in ch.7) and a fit_factory (default: logistic regression). Pure-Python.
Reference : M. Lopez de Prado, *Advances in Financial Machine Learning* (2018), ch.7
(purged k-fold, embargo); standard practice of nested cross-validation.
"""
from __future__ import annotations

import itertools
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)

import egp_cv_importance as CV

Event = Tuple[int, int]


def _param_combos(grid: Dict[str, Sequence]) -> List[Dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]


def grid_search_cv(X, y, events: Sequence[Event], param_grid: Dict[str, Sequence], *,
                   fit_factory: Optional[Callable] = None, n_splits: int = 4,
                   embargo_pct: float = 0.0, score_fn: Optional[Callable] = None,
                   n_bars: Optional[int] = None) -> Dict:
    """Grid search evaluated under purged CV. Returns {best_params, best_score, table}."""
    fit_factory = fit_factory or CV.default_logreg_fit
    score_fn = score_fn or CV.accuracy_score
    table = []
    best = None
    for combo in _param_combos(param_grid):
        fit_fn = fit_factory(**combo)
        res = CV.cv_score(X, y, events, n_splits=n_splits, embargo_pct=embargo_pct,
                          fit_fn=fit_fn, score_fn=score_fn, n_bars=n_bars)
        table.append((combo, res["mean"], res["std"]))
        if best is None or res["mean"] > best[1]:
            best = (combo, res["mean"], res["std"])
    return {"best_params": best[0] if best else {}, "best_score": best[1] if best else float("nan"),
            "table": table}


def nested_cv(X, y, events: Sequence[Event], param_grid: Dict[str, Sequence], *,
              fit_factory: Optional[Callable] = None, outer_splits: int = 4, inner_splits: int = 3,
              embargo_pct: float = 0.0, score_fn: Optional[Callable] = None,
              n_bars: Optional[int] = None) -> Dict:
    """Nested purged CV. Returns {outer_scores, mean_outer_score, chosen_params_per_fold}.
    The mean_outer_score estimate is not biased by the hyperparameter selection."""
    fit_factory = fit_factory or CV.default_logreg_fit
    score_fn = score_fn or CV.accuracy_score
    if n_bars is None:
        n_bars = max(t1 for _, t1 in events) + 1
    outer = CV.purged_kfold(events, n_splits=outer_splits, embargo_pct=embargo_pct, n_bars=n_bars)
    outer_scores: List[float] = []
    chosen: List[Dict] = []
    for fold in outer:
        tr, te = fold["train_idx"], fold["test_idx"]
        if not tr or not te:
            continue
        Xtr = [X[i] for i in tr]
        ytr = [y[i] for i in tr]
        evtr = [events[i] for i in tr]
        gs = grid_search_cv(Xtr, ytr, evtr, param_grid, fit_factory=fit_factory,
                            n_splits=inner_splits, embargo_pct=embargo_pct, score_fn=score_fn,
                            n_bars=n_bars)
        bp = gs["best_params"]
        predict = fit_factory(**bp)(Xtr, ytr)               # re-trains on the whole outer train
        Xte = [X[i] for i in te]
        yte = [y[i] for i in te]
        outer_scores.append(score_fn(predict, Xte, yte))
        chosen.append(bp)
    return {"outer_scores": outer_scores,
            "mean_outer_score": sum(outer_scores) / len(outer_scores) if outer_scores else float("nan"),
            "chosen_params_per_fold": chosen, "n_outer_folds": len(outer_scores)}
