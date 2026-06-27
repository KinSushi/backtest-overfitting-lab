"""
egp_adf.py - Augmented Dickey-Fuller (ADF) unit-root test, pure-Python.

Used to CHOOSE the minimal d* of the fractional differentiation (egp_fracdiff): the smallest d
such that the FFD series is STATIONARY (unit root rejected) while preserving as much memory
as possible (AFML ch.5). Implements OLS and the ADF statistic by hand (zero statsmodels).

Specification (Stata dfuller; model 'c' by default):
    dX_t = alpha + gamma * X_{t-1} + Sum_{i=1..p} delta_i * dX_{t-i} + eps_t
H0: gamma = 0 (unit root -> NON stationary). Statistic = t-stat of gamma = gamma/se(gamma).
We REJECT H0 (stationary series) if the statistic < critical value (more negative).

ASYMPTOTIC MacKinnon critical values, VERIFIED against concordant sources (Warwick
hand1_trends; Real Statistics):
    nc (no constant)           : 1% -2.56 | 5% -1.94 | 10% -1.62
    c  (constant)              : 1% -3.43 | 5% -2.86 | 10% -2.57
    ct (constant + trend)      : 1% -3.96 | 5% -3.41 | 10% -3.13
Reference: MacKinnon (1994, 2010); Dickey & Fuller (1979, 1981).

RESERVATION: these values are ASYMPTOTIC (large T). In finite samples, MacKinnon uses response
surfaces (slightly more negative values) NOT implemented here -> use on long series (hundreds of
points). Lag selection (AIC/BIC autolag) is available via adf_autolag; adf_test takes the lag as a
parameter (default Schwert rule). For very large-amplitude series (raw prices), centering/rescaling
improves conditioning (FWL: does not change the gamma statistic).
Source: M. Lopez de Prado, *Advances in Financial Machine Learning* (2018), ch.5 (d* selection).
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(__file__)
sys.path.insert(0, _HERE)

# Asymptotic MacKinnon critical values (verified)
ADF_CRIT = {
    "nc": {1: -2.56, 5: -1.94, 10: -1.62},
    "c": {1: -3.43, 5: -2.86, 10: -2.57},
    "ct": {1: -3.96, 5: -3.41, 10: -3.13},
}


def _inv(A: List[List[float]]) -> List[List[float]]:
    """Inverse of a square matrix via Gauss-Jordan with partial pivoting."""
    n = len(A)
    M = [list(A[i]) + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-300:
            raise ValueError("singular matrix (collinearity?)")
        M[col], M[piv] = M[piv], M[col]
        d = M[col][col]
        M[col] = [x / d for x in M[col]]
        for r in range(n):
            if r != col:
                f = M[r][col]
                M[r] = [M[r][j] - f * M[col][j] for j in range(2 * n)]
    return [row[n:] for row in M]


def _ols(Z: List[List[float]], y: List[float]) -> Tuple[List[float], List[float], float]:
    """OLS via normal equations. Returns (beta, se(beta), sigma2)."""
    n = len(Z)
    k = len(Z[0])
    ZtZ = [[sum(Z[r][i] * Z[r][j] for r in range(n)) for j in range(k)] for i in range(k)]
    Zty = [sum(Z[r][i] * y[r] for r in range(n)) for i in range(k)]
    inv = _inv(ZtZ)
    beta = [sum(inv[i][j] * Zty[j] for j in range(k)) for i in range(k)]
    resid = [y[r] - sum(Z[r][j] * beta[j] for j in range(k)) for r in range(n)]
    ssr = sum(e * e for e in resid)
    dof = n - k
    sigma2 = ssr / dof if dof > 0 else float("nan")
    se = [math.sqrt(max(0.0, sigma2 * inv[i][i])) for i in range(k)]
    return beta, se, sigma2


def schwert_lag(n: int) -> int:
    """Schwert rule for the maximum lag: floor(12*(n/100)^0.25)."""
    return int(math.floor(12 * (n / 100.0) ** 0.25))


def adf_test(series: Sequence[float], lag: Optional[int] = None, regression: str = "c",
             level: int = 5) -> Dict:
    """ADF test. regression in {'nc','c','ct'}; level in {1,5,10}. Returns the statistic, the
    critical values, the rejection verdict and the boolean 'stationary'."""
    if regression not in ADF_CRIT:
        raise ValueError("regression must be 'nc', 'c' or 'ct'")
    X = [float(v) for v in series]
    T = len(X)
    if lag is None:
        lag = max(1, min(schwert_lag(T), max(1, (T - 5) // 3)))
    rows: List[List[float]] = []
    y: List[float] = []
    for t in range(lag + 1, T):
        reg: List[float] = []
        if regression in ("c", "ct"):
            reg.append(1.0)
        if regression == "ct":
            reg.append(float(t))
        reg.append(X[t - 1])                              # lagged level (gamma)
        for i in range(1, lag + 1):
            reg.append(X[t - i] - X[t - i - 1])           # dX_{t-i}
        rows.append(reg)
        y.append(X[t] - X[t - 1])
    if len(rows) <= len(rows[0]) if rows else True:
        return {"adf_stat": float("nan"), "lag": lag, "n_obs": len(y),
                "critical_values": ADF_CRIT[regression], "stationary": False,
                "regression": regression, "error": "too few observations"}
    beta, se, _ = _ols(rows, y)
    gi = (1 if regression in ("c", "ct") else 0) + (1 if regression == "ct" else 0)
    gamma, se_gamma = beta[gi], se[gi]
    adf = gamma / se_gamma if se_gamma > 0 else float("nan")
    crit = ADF_CRIT[regression]
    reject = {lv: (adf == adf and adf < crit[lv]) for lv in crit}
    return {"adf_stat": adf, "lag": lag, "n_obs": len(y), "gamma": gamma, "se_gamma": se_gamma,
            "critical_values": crit, "reject": reject,
            "stationary": (adf == adf and adf < crit[level]), "regression": regression}


def min_ffd_order(series: Sequence[float],
                  d_grid: Sequence[float] = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
                                             0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
                  thresh: float = 1e-4, regression: str = "c", lag: Optional[int] = None,
                  level: int = 5) -> Dict:
    """AFML ch.5 selection: smallest d such that the FFD series is stationary (ADF), with the
    retained memory correlation. Returns {chosen_d, table:[(d, adf_stat, width, corr, stat)]}."""
    import egp_fracdiff as FD
    table = []
    chosen = None
    for d in d_grid:
        res = FD.frac_diff_ffd(series, d, thresh)
        vals = res["values"]
        if len(vals) < 20:
            table.append((d, None, res["width"], None, False))
            continue
        adf = adf_test(vals, lag=lag, regression=regression, level=level)
        corr = FD.memory_retention(series, d, thresh)
        st = adf["stationary"]
        table.append((d, adf["adf_stat"], res["width"], corr, st))
        if st and chosen is None:
            chosen = d
    return {"chosen_d": chosen, "level": level, "regression": regression, "table": table}


# --------------------------------------------------------------------------- autolag AIC/BIC
def _adf_regression(X: Sequence[float], lag: int, regression: str, start_t: int):
    """Builds (rows, y) of the ADF regression for t in [start_t, T). start_t common to all
    candidate lags -> same sample -> comparable AIC/BIC (cf. statsmodels/Minitab)."""
    T = len(X)
    rows, y = [], []
    for t in range(start_t, T):
        reg: List[float] = []
        if regression in ("c", "ct"):
            reg.append(1.0)
        if regression == "ct":
            reg.append(float(t))
        reg.append(X[t - 1])
        for i in range(1, lag + 1):
            reg.append(X[t - i] - X[t - i - 1])
        rows.append(reg)
        y.append(X[t] - X[t - 1])
    return rows, y


def adf_autolag(series: Sequence[float], maxlag: Optional[int] = None, regression: str = "c",
                criterion: str = "AIC", level: int = 5) -> Dict:
    """Selects the lag minimizing AIC or BIC (same observations for all lags), then
    returns the ADF test at that lag. IC = T*ln(SSR/T) + penalty*K, penalty=2 (AIC) / ln(T) (BIC)."""
    if criterion.upper() not in ("AIC", "BIC"):
        raise ValueError("criterion must be 'AIC' or 'BIC'")
    X = [float(v) for v in series]
    T = len(X)
    if maxlag is None:
        maxlag = max(1, min(schwert_lag(T), max(1, (T - 5) // 3)))
    start = maxlag + 1                                  # common sample
    table = []
    best = None
    for p in range(0, maxlag + 1):
        rows, y = _adf_regression(X, p, regression, start)
        if len(rows) <= len(rows[0]):
            continue
        _, _, sigma2 = _ols(rows, y)
        n = len(y)
        k = len(rows[0])
        ssr = max(sigma2 * (n - k), 1e-300)
        pen = 2.0 if criterion.upper() == "AIC" else math.log(n)
        ic = n * math.log(ssr / n) + pen * k
        table.append((p, ic, k))
        if best is None or ic < best[1]:
            best = (p, ic, k)
    best_lag = best[0] if best else 0
    res = adf_test(series, lag=best_lag, regression=regression, level=level)
    res.update({"best_lag": best_lag, "criterion": criterion.upper(), "ic_table": table,
                "maxlag": maxlag})
    return res


# --------------------------------------------------------------------------- finite-sample critical values (Monte-Carlo)
def _quantile(sorted_xs: Sequence[float], p: float) -> float:
    """Quantile by linear interpolation (type 7), p in [0,1]. sorted_xs sorted ascending."""
    n = len(sorted_xs)
    if n == 1:
        return sorted_xs[0]
    pos = p * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac


def adf_critical_values_simulated(n: int, regression: str = "c", n_sims: int = 2000,
                                  lag: int = 0, seed: int = 0) -> Dict:
    """Finite-SAMPLE ADF critical values by Monte-Carlo: simulates n_sims random walks
    of length n (H0: unit root), computes the ADF stat, returns the 1/5/10% quantiles.
    Self-verifying: converges to asymptotic MacKinnon as n grows. The ADF stat is left-tailed."""
    import random as _random
    rng = _random.Random(seed)
    stats = []
    for _ in range(n_sims):
        x = [0.0]
        for _ in range(n - 1):
            x.append(x[-1] + rng.gauss(0.0, 1.0))         # random walk (unit root)
        try:
            stats.append(adf_test(x, lag=lag, regression=regression)["adf_stat"])
        except Exception:
            continue
    stats.sort()
    return {1: _quantile(stats, 0.01), 5: _quantile(stats, 0.05), 10: _quantile(stats, 0.10),
            "n": n, "n_sims": len(stats), "regression": regression, "lag": lag}


def adf_test_finite_sample(series: Sequence[float], regression: str = "c", lag: Optional[int] = None,
                           level: int = 5, n_sims: int = 2000, seed: int = 0) -> Dict:
    """ADF test with a finite-SAMPLE critical value (simulated at the series length) instead
    of the asymptotic value. Returns the test enriched with {crit_finite, stationary_finite}."""
    res = adf_test(series, lag=lag, regression=regression, level=level)
    cv = adf_critical_values_simulated(len(series), regression=regression, n_sims=n_sims,
                                       lag=res["lag"], seed=seed)
    crit = cv[level]
    res.update({"crit_finite": crit, "crit_asymptotic": res["critical_values"][level],
                "stationary_finite": res["adf_stat"] < crit, "n_sims": cv["n_sims"]})
    return res
