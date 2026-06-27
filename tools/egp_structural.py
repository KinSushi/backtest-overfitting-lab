"""
egp_structural.py - Structural break / regime change detection.

Used to decide WHEN to re-optimize: if the distribution of returns (or of an input) breaks,
the parameters tuned on the old regime degrade.

Two tools:
  1) Page CUSUM: sequential detection of a MEAN SHIFT (online alarm).
  2) sup-F (Quandt-Andrews): retrospective test of ONE break at an unknown date; sweeps the
     candidate dates (trimmed), takes the max of the Chow statistic; the location
     is the argmax.

Sources:
  - Page, E. S. (1954), "Continuous Inspection Schemes", Biometrika 41(1/2):100-115 (CUSUM).
  - Quandt, R. (1960); Andrews, D. W. K. (1993), "Tests for Parameter Instability and
    Structural Change with Unknown Change Point", Econometrica 61(4):821-856 (sup-F, critical
    values, non-standard).
  - Multi-break generalization: Bai & Perron (2003), J. Applied Econometrics 18(1):1-22
    (dynamic programming; NOT implemented here).
  - Stationarity/unit root (possible extensions): ADF, KPSS - require tables
    of critical values (not included).
Pure-Python.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence


def page_cusum(stream: Sequence[float], mu0: float = 0.0, k: float = 0.5,
               h: float = 5.0) -> Dict:
    """Two-sided Page CUSUM for a mean shift relative to mu0.
      S+_t = max(0, S+_{t-1} + (x - mu0 - k)),  S-_t = max(0, S-_{t-1} - (x - mu0 + k)).
    Alarm when S+ >= h (up) or S- >= h (down). k = half-amplitude to detect
    (slack), h = threshold. Returns the 1st alarm instant, the direction and the maxima."""
    sp = sm = 0.0
    sp_max = sm_max = 0.0
    alarm = None
    direction = None
    for t, x in enumerate(stream, start=1):
        sp = max(0.0, sp + (x - mu0 - k))
        sm = max(0.0, sm - (x - mu0 + k))
        sp_max = max(sp_max, sp)
        sm_max = max(sm_max, sm)
        if alarm is None and (sp >= h or sm >= h):
            alarm = t
            direction = "+" if sp >= h else "-"
    return {"alarm": alarm, "direction": direction,
            "S_plus_max": sp_max, "S_minus_max": sm_max}


def _ssr_mean(vals):
    n = len(vals)
    if n == 0:
        return 0.0
    m = sum(vals) / n
    return sum((x - m) ** 2 for x in vals)


def supf_break(series: Sequence[float], trim: float = 0.15) -> Dict:
    """sup-F (Quandt-Andrews) test of a single break in the MEAN, unknown date.
    For each cut tau (trimmed), F(tau) = (SSR_r - SSR_u)/1 / (SSR_u/(n-2)),
    SSR_r = sum of squares around the global mean, SSR_u = before + after.
    Returns supF (max), break_index (argmax) and n. Compare supF to the critical values
    from Andrews (1993): ~8.85 at 5% for 1 parameter and 15% trim."""
    x = [float(v) for v in series]
    n = len(x)
    if n < 10:
        return {"supF": 0.0, "break_index": None, "n": n}
    lo = max(1, int(trim * n))
    hi = min(n - 1, int((1.0 - trim) * n))
    ssr_r = _ssr_mean(x)
    best_f = 0.0
    best_i = None
    for tau in range(lo, hi):
        ssr_u = _ssr_mean(x[:tau]) + _ssr_mean(x[tau:])
        denom = ssr_u / (n - 2)
        if denom <= 0:
            continue
        f = (ssr_r - ssr_u) / denom
        if f > best_f:
            best_f = f
            best_i = tau
    return {"supF": best_f, "break_index": best_i, "n": n}
