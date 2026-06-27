"""
egp_fdr.py - False Discovery Rate control (multiple testing).

When testing SEVERAL strategies/symbols/configs, some appear significant
by pure chance. Complementing the DSR (which deflates ONE strategy selected among N trials),
this module controls the whole portfolio of tests:
  - Benjamini-Hochberg (FDR sous independance / dependance positive PRDS) ;
  - Benjamini-Yekutieli (FDR under arbitrary dependence, more conservative);
  - "haircut Sharpe" Harvey-Liu style: by how much a Sharpe is trimmed after correcting
    for M tests?

Sources :
  - Benjamini, Y. & Hochberg, Y. (1995), "Controlling the False Discovery Rate", Journal of
    the Royal Statistical Society B 57(1):289-300.
  - Benjamini, Y. & Yekutieli, D. (2001), "The control of the FDR in multiple testing under
    dependency", Annals of Statistics 29(4):1165-1188.
  - Harvey, C. R. & Liu, Y. (2015), "Backtesting", SSRN ; Harvey, Liu & Zhu (2016),
    "...and the Cross-Section of Expected Returns", Review of Financial Studies 29(1):5-68.
Pure-Python (statistics.NormalDist for the normal approximation of t).
"""
from __future__ import annotations

from statistics import NormalDist
from typing import Dict, List, Sequence

_N = NormalDist()


def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.05) -> Dict:
    """BH procedure. Returns {'reject': [bool], 'n_reject': k, 'threshold': p_(k),
    'adjusted': [q-values]}. k* = max{k : p_(k) <= (k/m) alpha}; reject ranks 1..k*."""
    m = len(pvalues)
    if m == 0:
        return {"reject": [], "n_reject": 0, "threshold": 0.0, "adjusted": []}
    order = sorted(range(m), key=lambda i: pvalues[i])
    ps = [pvalues[i] for i in order]
    kstar = 0
    thr = 0.0
    for k in range(1, m + 1):
        if ps[k - 1] <= (k / m) * alpha:
            kstar = k
            thr = ps[k - 1]
    reject_sorted = [j < kstar for j in range(m)]
    # q-values (BH-adjusted): monotonization by cumulative minimum from the end
    adj_sorted = [0.0] * m
    prev = 1.0
    for k in range(m, 0, -1):
        val = min(prev, ps[k - 1] * m / k)
        adj_sorted[k - 1] = val
        prev = val
    reject = [False] * m
    adjusted = [0.0] * m
    for pos, i in enumerate(order):
        reject[i] = reject_sorted[pos]
        adjusted[i] = min(1.0, adj_sorted[pos])
    return {"reject": reject, "n_reject": kstar, "threshold": thr, "adjusted": adjusted}


def benjamini_yekutieli(pvalues: Sequence[float], alpha: float = 0.05) -> Dict:
    """BH under arbitrary dependence: threshold divided by c(m)=Σ 1/i (more conservative)."""
    m = len(pvalues)
    if m == 0:
        return {"reject": [], "n_reject": 0, "threshold": 0.0, "adjusted": []}
    cm = sum(1.0 / i for i in range(1, m + 1))
    order = sorted(range(m), key=lambda i: pvalues[i])
    ps = [pvalues[i] for i in order]
    kstar = 0
    thr = 0.0
    for k in range(1, m + 1):
        if ps[k - 1] <= (k / (m * cm)) * alpha:
            kstar = k
            thr = ps[k - 1]
    reject_sorted = [j < kstar for j in range(m)]
    adj_sorted = [0.0] * m
    prev = 1.0
    for k in range(m, 0, -1):
        val = min(prev, ps[k - 1] * m * cm / k)
        adj_sorted[k - 1] = val
        prev = val
    reject = [False] * m
    adjusted = [0.0] * m
    for pos, i in enumerate(order):
        reject[i] = reject_sorted[pos]
        adjusted[i] = min(1.0, adj_sorted[pos])
    return {"reject": reject, "n_reject": kstar, "threshold": thr, "adjusted": adjusted}


def sharpe_to_pvalue(sharpe: float, n_obs: int, two_sided: bool = False) -> float:
    """p-value of a NON-annualized Sharpe via the normal approximation of t: t = SR*sqrt(n)."""
    t = sharpe * (n_obs ** 0.5)
    p = 1.0 - _N.cdf(t)
    return 2.0 * min(p, 1.0 - p) if two_sided else p


def haircut_sharpe(sharpe: float, n_obs: int, n_tests: int,
                   method: str = "bonferroni") -> Dict:
    """Harvey-Liu 'Haircut': Sharpe trimmed after correcting for n_tests tests.
    We convert SR -> t -> p (1-sided), adjust p (Bonferroni or Sidak), and convert back to
    t_adj then SR_adj = t_adj/sqrt(n). Returns the adjusted SR and the trimmed fraction."""
    if sharpe <= 0:
        return {"sharpe": sharpe, "haircut_sharpe": sharpe, "haircut_fraction": 0.0,
                "p_raw": sharpe_to_pvalue(sharpe, n_obs), "p_adj": None}
    p = sharpe_to_pvalue(sharpe, n_obs, two_sided=False)
    M = max(1, int(n_tests))
    if method == "sidak":
        p_adj = 1.0 - (1.0 - p) ** M
    else:  # bonferroni
        p_adj = min(1.0, p * M)
    p_adj = min(max(p_adj, 1e-300), 1.0 - 1e-16)
    t_adj = _N.inv_cdf(1.0 - p_adj)
    sr_adj = max(0.0, t_adj / (n_obs ** 0.5))
    frac = 1.0 - sr_adj / sharpe if sharpe > 0 else 0.0
    return {"sharpe": sharpe, "haircut_sharpe": sr_adj,
            "haircut_fraction": frac, "p_raw": p, "p_adj": p_adj}
