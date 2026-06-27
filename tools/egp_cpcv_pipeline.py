"""
egp_cpcv_pipeline.py - CPCV -> optimization (egp_wfo style) -> deflated gate wiring.

Replaces egp_wfo's SINGLE-PATH walk-forward with an OOS DISTRIBUTION from CPCV:
  1) partition of the timeline into N groups (egp_cpcv);
  2) for EACH split (k groups in test, the rest in PURGED/EMBARGOED train): re-optimization
     of the parameters on the train (via hybrid_minimize, like egp_wfo), then evaluation on the
     test -> series of OOS returns per group;
  3) recombination of the phi = C(N-1,k-1) complete OOS PATHS (egp_cpcv.assemble_paths);
  4) GATE: the distribution of per-path OOS Sharpes feeds the deflated Sharpe
     (egp_accept_gate) -> ACCEPT/REJECT.

Contract provided by the caller:
  returns_on(vec, idx_list) -> list of OOS returns produced by strategy 'vec' on the
  observations idx_list. In production: an MT5 backtest on the corresponding sub-interval,
  from which the per-trade returns series is extracted (cf. egp_mt5_deals). No MT5 here; the
  splitting/optimization/aggregation/gate LOGIC is tested on a synthetic objective.

HONESTY (gate interpretation): CPCV gives the OOS variability of the SAME set retained across
phi paths (variance of "path/luck"). The DSR's multiple-testing deflation additionally requires
the NUMBER OF CONFIGS actually explored during the search -> passed via n_trials (default
= phi, but the caller SHOULD provide the true config count for a correct deflation).

Sources: Lopez de Prado, AFML ch.7 (CPCV); Bailey & LdP (2014) (DSR). Pure-Python.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, Dict, List, Optional, Sequence

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)
from egp_mho_hybrid import hybrid_minimize, decode, OptDim
import egp_cpcv as CV
import egp_accept_gate as G

ReturnsFn = Callable[[Sequence[float], List[int]], List[float]]


def _metric(returns: List[float], kind: str) -> float:
    if not returns:
        return 0.0
    if kind == "mean":
        return sum(returns) / len(returns)
    return G.sharpe_ratio(returns)         # default: non-annualized Sharpe


def cpcv_run(dims: List[OptDim], n_obs: int, returns_on: ReturnsFn,
             n_groups: int = 6, k_test: int = 2, purge: int = 0, embargo: int = 0,
             inner_evals: int = 300, seed: int = 123, opt_metric: str = "sharpe") -> Dict:
    """Optimizes on each split's train, evaluates on the test (per group).
    Returns the per-split results + the OOS returns per (split, group)."""
    bounds = [(d.lo, d.hi) for d in dims]
    groups = CV.make_groups(n_obs, n_groups)
    splits = CV.cpcv_splits(n_obs, n_groups, k_test, purge, embargo)
    per_split = []
    for si, sp in enumerate(splits):
        train_idx = sp["train_idx"]
        res = hybrid_minimize(lambda x: -_metric(returns_on(x, train_idx), opt_metric),
                              bounds, max_evals=inner_evals, seed=seed + si)
        best = [min(hi, max(lo, v)) for v, (lo, hi) in zip(res.best, bounds)]
        is_metric = -res.best_score
        # OOS returns PER test GROUP (needed for path assembly)
        oos_by_group = {}
        for g in sp["test_groups"]:
            a, b = groups[g]
            oos_by_group[g] = returns_on(best, list(range(a, b)))
        all_oos = [r for g in sp["test_groups"] for r in oos_by_group[g]]
        per_split.append({
            "split": si, "test_groups": sp["test_groups"],
            "best": decode(best, dims), "best_vec": best,
            "is_metric": is_metric, "oos_metric": _metric(all_oos, opt_metric),
            "oos_by_group": oos_by_group,
        })
    return {"per_split": per_split, "groups": groups,
            "n_splits": CV.n_splits(n_groups, k_test),
            "n_paths": CV.n_paths(n_groups, k_test),
            "n_groups": n_groups, "k_test": k_test, "opt_metric": opt_metric}


def assemble_path_returns(run: Dict) -> List[List[float]]:
    """Recombines the phi OOS paths: each path concatenates, for each group (temporal
    order), the OOS returns of the split assigned to it. A path's length = n_obs."""
    N, k = run["n_groups"], run["k_test"]
    paths = CV.assemble_paths(N, k)                 # phi paths: group -> split index
    per_split = run["per_split"]
    out = []
    for path in paths:
        series: List[float] = []
        for g in range(N):                          # temporal order of the groups
            si = path[g]
            series.extend(per_split[si]["oos_by_group"][g])
        out.append(series)
    return out


def gate_from_cpcv(path_returns: List[List[float]], n_trials: Optional[int] = None,
                   dsr_min: float = 0.95, min_frac_positive: float = 0.5) -> Dict:
    """Aggregates the paths' OOS distribution into ACCEPT/REJECT.
    - per-path Sharpe -> V[SR] (path variability);
    - pooled OOS returns -> PSR moments;
    - DSR = PSR(SR0(V[SR], N)) with N = n_trials (default = number of paths);
    - ACCEPT decision if DSR>=dsr_min, median OOS Sharpe>0 and fraction of positive paths>=threshold."""
    phi = len(path_returns)
    per_path_sr = [G.sharpe_ratio(r) for r in path_returns]
    srt = sorted(per_path_sr)
    median_sr = srt[len(srt) // 2] if srt else 0.0
    frac_pos = sum(1 for s in per_path_sr if s > 0) / phi if phi else 0.0
    mean = sum(per_path_sr) / phi if phi else 0.0
    v_sr = sum((s - mean) ** 2 for s in per_path_sr) / phi if phi else 0.0
    pooled = [r for series in path_returns for r in series]
    N = n_trials if n_trials is not None else max(2, phi)
    dsr = G.deflated_sharpe(pooled, sr_trials_variance=max(v_sr, 1e-12), n_trials=N)
    reasons = []
    if not (dsr == dsr and dsr >= dsr_min):
        reasons.append(f"DSR={dsr:.3f} < {dsr_min}")
    if median_sr <= 0:
        reasons.append(f"Sharpe OOS median={median_sr:.3f} <= 0")
    if frac_pos < min_frac_positive:
        reasons.append(f"positive paths={frac_pos:.2f} < {min_frac_positive}")
    return {"n_paths": phi, "per_path_sharpe": per_path_sr,
            "median_oos_sharpe": median_sr, "frac_positive_paths": frac_pos,
            "sr_trials_variance": v_sr, "n_trials": N, "dsr": dsr,
            "decision": "ACCEPT" if not reasons else "REJECT", "reasons": reasons}


def cpcv_walk_forward_gate(dims, n_obs, returns_on, n_groups=6, k_test=2,
                           purge=0, embargo=0, inner_evals=300, seed=123,
                           opt_metric="sharpe", n_trials=None, dsr_min=0.95) -> Dict:
    """End to end: CPCV -> per-split optimization -> OOS paths -> deflated gate."""
    run = cpcv_run(dims, n_obs, returns_on, n_groups, k_test, purge, embargo,
                   inner_evals, seed, opt_metric)
    paths = assemble_path_returns(run)
    gate = gate_from_cpcv(paths, n_trials=n_trials, dsr_min=dsr_min)
    oos_metrics = [s["oos_metric"] for s in run["per_split"]]
    is_metrics = [s["is_metric"] for s in run["per_split"]]
    n = max(1, len(oos_metrics))
    gate["mean_oos_metric"] = sum(oos_metrics) / n
    gate["mean_overfit_gap"] = sum(is_metrics) / n - sum(oos_metrics) / n
    gate["n_splits"] = run["n_splits"]
    return gate
