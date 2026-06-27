"""
egp_core_select.py - Search-space reduction (core).

Goal: go from the 101 optimizable dimensions (CORE+LATER) to a smaller, better-justified
SUBSET, to reduce the overfitting surface and the optimization cost.

Two levers, both DOCUMENTED and traceable:
  1) STATUS FILTER - keep only 'OPTIMIZE_CORE' (SIGNAL_RISK group in the map), freezing
     'OPTIMIZE_LATER' (STRUCTURAL group). This distinction is ALREADY present in
     DOCS/PARAMETER_MAP.csv: optimize the signal/risk core first, the structure later.
     Measured effect: 101 -> 37 dimensions.
  2) TIMEFRAME DOMAIN PRUNING - for an M1 scalper, restrict inputs of type ENUM_TIMEFRAMES
     to a whitelist (PERIOD_CURRENT, M1..H1). Frames > H1 are economically implausible for the
     sub-indicators of an M1 EA and only inflate the combinatorial space. Real reduction of the
     number of enum combinations.

HONESTY / LIMITS:
  * The status filter is a POLICY (judgment encoded in the map), not an empirical importance
    measure.
  * The timeframe whitelist is a HEURISTIC to be confirmed; it is configurable.
  * A reduction by EMPIRICAL IMPORTANCE (Sobol indices, MDA/permutation, ablation) requires real
    backtests and is DEFERRED. This module provides the scaffolding, not the proof of impact.

No dependency; reuses OptDim/load_opt_dims (egp_mho_hybrid) and the CSV map.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from egp_mho_hybrid import OptDim, load_opt_dims

# Default scalper whitelist: from the current tick to H1 inclusive (plausible frames in M1).
DEFAULT_SCALP_TF = (
    "PERIOD_CURRENT", "PERIOD_M1", "PERIOD_M2", "PERIOD_M3", "PERIOD_M4",
    "PERIOD_M5", "PERIOD_M6", "PERIOD_M10", "PERIOD_M12", "PERIOD_M15",
    "PERIOD_M20", "PERIOD_M30", "PERIOD_H1",
)


def _status_map(root: str) -> Dict[str, str]:
    path = Path(root) / "DOCS" / "PARAMETER_MAP.csv"
    return {r["name"]: r.get("status", "") for r in csv.DictReader(path.open(encoding="utf-8"))}


def _is_timeframe_dim(d: OptDim) -> bool:
    names = getattr(d, "choice_names", ()) or ()
    return d.kind == "enum" and len(names) > 0 and all(n.startswith("PERIOD_") for n in names)


def _prune_tf(d: OptDim, whitelist: Sequence[str]) -> Tuple[OptDim, int, int]:
    """Restricts a timeframe OptDim's choices to the whitelist (by name). Returns
    (new OptDim, n_before, n_after)."""
    names = list(getattr(d, "choice_names", ()) or ())
    choices = list(d.choices)
    keep = [(nm, ch) for nm, ch in zip(names, choices) if nm in set(whitelist)]
    if not keep:                      # never empty a domain
        return d, len(choices), len(choices)
    new_names = tuple(nm for nm, _ in keep)
    new_choices = tuple(ch for _, ch in keep)
    nd = OptDim(d.name, 0.0, float(len(new_choices)), False,
                kind="enum", choices=new_choices, choice_names=new_names)
    return nd, len(choices), len(new_choices)


def select_core_dims(root: str, keep_status: Sequence[str] = ("OPTIMIZE_CORE",),
                     prune_timeframes: bool = True,
                     tf_whitelist: Sequence[str] = DEFAULT_SCALP_TF
                     ) -> Tuple[List[OptDim], Dict]:
    """Returns (reduced_dims, report). reduced_dims is usable as-is by
    decode()/hybrid_minimize() in place of load_opt_dims()."""
    all_dims = load_opt_dims(root)
    status = _status_map(root)
    kept: List[OptDim] = []
    dropped_by_status: List[str] = []
    tf_pruned: List[Tuple[str, int, int]] = []
    enum_combos_before = enum_combos_after = 1
    for d in all_dims:
        if status.get(d.name) not in set(keep_status):
            dropped_by_status.append(d.name)
            continue
        if prune_timeframes and _is_timeframe_dim(d):
            nd, n0, n1 = _prune_tf(d, tf_whitelist)
            if n1 != n0:
                tf_pruned.append((d.name, n0, n1))
            kept.append(nd)
        else:
            kept.append(d)
    # combinatorial size of the enum/bool dimensions (discrete-space indicator)
    def discrete_size(dims):
        prod = 1
        for d in dims:
            if d.kind == "enum":
                prod *= max(1, len(d.choices))
            elif d.kind == "bool":
                prod *= 2
        return prod
    report = {
        "n_before": len(all_dims),
        "n_after": len(kept),
        "n_dropped_status": len(dropped_by_status),
        "kept_by_kind": _count_kind(kept),
        "dropped_status_sample": dropped_by_status[:12],
        "timeframe_pruned": tf_pruned,
        "discrete_space_before": discrete_size(all_dims),
        "discrete_space_after": discrete_size(kept),
        "keep_status": list(keep_status),
        "tf_whitelist": list(tf_whitelist),
    }
    return kept, report


def _count_kind(dims) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for d in dims:
        out[d.kind] = out.get(d.kind, 0) + 1
    return out


def format_report(report: Dict) -> str:
    lines = [
        "CORE REDUCTION - report",
        f"  dimensions: {report['n_before']} -> {report['n_after']} "
        f"(frozen by status: {report['n_dropped_status']})",
        f"  by type after reduction: {report['kept_by_kind']}",
        f"  kept statuses: {report['keep_status']}",
    ]
    if report["timeframe_pruned"]:
        lines.append("  pruned timeframe domains (name: before -> after):")
        for nm, a, b in report["timeframe_pruned"]:
            lines.append(f"    - {nm:24} {a} -> {b}")
    lines.append(
        f"  discrete space size (enum/bool): "
        f"{report['discrete_space_before']:.3e} -> {report['discrete_space_after']:.3e}"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    dims, rep = select_core_dims(root)
    print(format_report(rep))
