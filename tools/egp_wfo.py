#!/usr/bin/env python3
"""Walk-Forward harness (rolling optimization, anti-overfitting).

Principle (cf. MQL5 doc "Custom Walk Forward optimization", article 3279, and the
native walk-forward of the tester): the history is split into windows
in-sample (IS) -> out-of-sample (OOS). On EACH window we RE-OPTIMIZE the
parameters on the IS (via the hybrid), then the chosen set is EVALUATED on the OOS
(never seen during optimization). The OOS performances are aggregated: this is the
honest measure of generalization. The IS/OOS gap is the diagnostic of
overfitting.

Here : generic engine. `cost_fn(vec, segment_label, window_index) -> float`
(cost to MINIMIZE) is provided by the caller. In production, cost_fn launches an
MT5 backtest over the window date range and returns
egp_mt5_report_parser.robust_score(parsed_report). No backtest is executed
here (no MT5); the splitting and selection logic, however, is tested.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Callable, Optional
from datetime import datetime, timedelta
import sys, json, argparse

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from egp_mho_hybrid import hybrid_minimize, load_opt_dims, decode, OptDim

CostFn = Callable[[List[float], str, int], float]


def make_windows(start: str, end: str, is_days: int, oos_days: int,
                 step_days: Optional[int] = None, fmt: str = '%Y.%m.%d') -> List[Dict[str, str]]:
    """Splits [start, end] into sliding IS->OOS windows (dates inclusive)."""
    step_days = step_days or oos_days
    d0, d1 = datetime.strptime(start, fmt), datetime.strptime(end, fmt)
    out, cur = [], d0
    while True:
        is_from = cur
        is_to = is_from + timedelta(days=is_days - 1)
        oos_from = is_to + timedelta(days=1)
        oos_to = oos_from + timedelta(days=oos_days - 1)
        if oos_to > d1:
            break
        out.append({'is_from': is_from.strftime(fmt), 'is_to': is_to.strftime(fmt),
                    'oos_from': oos_from.strftime(fmt), 'oos_to': oos_to.strftime(fmt)})
        cur = cur + timedelta(days=step_days)
    return out


def walk_forward(dims: List[OptDim], windows: List[Dict[str, str]], cost_fn: CostFn,
                 inner_evals: int = 800, seed: int = 123) -> Dict:
    """Re-optimizes on each window's IS then evaluates on the OOS.
    Returns per-window (best_vec, is_cost, oos_cost) + aggregates."""
    bounds = [(d.lo, d.hi) for d in dims]
    per_window, oos_costs, gaps = [], [], []
    for w, win in enumerate(windows):
        res = hybrid_minimize(lambda x: cost_fn(x, 'IS', w), bounds,
                              max_evals=inner_evals, seed=seed + w)
        best = [min(hi, max(lo, v)) for v, (lo, hi) in zip(res.best, bounds)]
        is_cost = res.best_score
        oos_cost = cost_fn(best, 'OOS', w)
        per_window.append({'window': w, **win, 'best': decode(best, dims),
                           'is_cost': is_cost, 'oos_cost': oos_cost,
                           'gap': oos_cost - is_cost})
        oos_costs.append(oos_cost)
        gaps.append(oos_cost - is_cost)
    n = max(1, len(oos_costs))
    return {'n_windows': len(windows),
            'mean_oos_cost': sum(oos_costs) / n,
            'mean_overfit_gap': sum(gaps) / n,
            'windows': per_window}


def run_from_map(root: str, start: str, end: str, is_days: int, oos_days: int,
                 cost_fn: CostFn, step_days: Optional[int] = None, seed: int = 123) -> Dict:
    dims = load_opt_dims(root)
    windows = make_windows(start, end, is_days, oos_days, step_days)
    rep = walk_forward(dims, windows, cost_fn, seed=seed)
    out = Path(root) / 'VALIDATION' / 'WFO'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'wfo_report.json').write_text(json.dumps(rep, indent=2), encoding='utf-8')
    return rep


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='WFO demo on a synthetic objective (no MT5).')
    ap.add_argument('--root', default='.')
    ap.add_argument('--start', default='2024.01.01')
    ap.add_argument('--end', default='2024.04.30')
    ap.add_argument('--is_days', type=int, default=30)
    ap.add_argument('--oos_days', type=int, default=10)
    a = ap.parse_args()
    dims = load_opt_dims(a.root)

    # Synthetic objective: optimum drifting per window + IS noise (overfit),
    # OOS = true function (no noise). Demonstration of the IS/OOS gap.
    import random

    def cost_fn(vec, seg, w):
        center = [d.lo + (d.hi - d.lo) * (0.4 + 0.05 * w) for d in dims]
        true = sum((v - c) ** 2 for v, c in zip(vec, center))
        if seg == 'IS':
            rng = random.Random(1000 * w + hash(tuple(round(v, 3) for v in vec)) % 997)
            return true + rng.uniform(-0.15, 0.15) * (1 + true)  # noise -> overfitting trap
        return true

    windows = make_windows(a.start, a.end, a.is_days, a.oos_days)
    rep = walk_forward(dims, windows, cost_fn, inner_evals=500, seed=7)
    print(f"windows={rep['n_windows']}  mean_OOS={rep['mean_oos_cost']:.4f}  "
          f"mean_overfit_gap={rep['mean_overfit_gap']:.4f}")
