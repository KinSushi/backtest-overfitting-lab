"""
egp_accept_gate.py - Deflated acceptance gate for strategy selection.

Consumes a SERIES OF RETURNS (per trade or per bar, NON-annualized) and/or
per-configuration performance matrices. Everything is pure-Python (statistics.NormalDist),
without numpy/scipy, to stay portable on the MT5/Windows side.

Functions:
  - sharpe_ratio / summary       : non-annualized SR + moments (skew, NON-excess kurtosis)
  - probabilistic_sharpe (PSR)   : Bailey & Lopez de Prado (2012), J. of Risk 15(2)
  - min_track_record_length      : same (minimum length for significance)
  - expected_maximum_sharpe (SR0): False Strategy Theorem, Bailey & LdP (2014)
  - deflated_sharpe (DSR)        : DSR = PSR(SR0), Bailey & LdP (2014), JPM 40(5):94-107
  - pbo_cscv                     : Probability of Backtest Overfitting via CSCV
                                   (Bailey, Borwein, Lopez de Prado & Zhu)
  - whites_reality_check (RC)    : White (2000), Econometrica 68:1097-1126 (stationary bootstrap)
  - hansen_spa (SPA_c)           : Hansen (2005), JBES (consistent re-centering, more power)
  - acceptance_decision          : aggregates into ACCEPT/REJECT + reasons

CONVENTIONS (verified against primary sources - see each function's docstring):
  * SR and all trial SRs are NON-ANNUALIZED (same frequency as the returns).
  * Kurtosis = NON-excess (normal law => 4.0).
  * Validation vector (Lopez de Prado): annual_SR=2.5 -> sr=2.5/sqrt(252),
    V[SR]=0.5/252, N=100, T=1250, skew=-3, kurt=10  =>  DSR ~ 0.90  (cf. tests).

NOTE (P0 dependency): the current parser (egp_mt5_report_parser) only exposes
AGGREGATE metrics (PF, Sharpe, trades). DSR/PSR/RC/SPA require the SERIES of returns.
The MT5 wiring (extraction of the timestamped deals list) remains to be done separately;
this module is nonetheless fully testable offline on synthetic series.
"""
from __future__ import annotations

import math
import random
from itertools import combinations
from statistics import NormalDist
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_N = NormalDist()
_PHI = _N.cdf            # standard normal CDF
_PHI_INV = _N.inv_cdf    # quantile (inverse CDF)
_GAMMA = 0.5772156649015328606   # Euler-Mascheroni constant
_E = math.e


# --------------------------------------------------------------------------- #
# Moments & Sharpe                                                             #
# --------------------------------------------------------------------------- #
def _moments(r: Sequence[float]) -> Tuple[float, float, float, float]:
    """Return (mean, population standard deviation, skewness, NON-excess kurtosis)."""
    n = len(r)
    if n < 2:
        return (r[0] if n else 0.0), 0.0, 0.0, 3.0
    mu = sum(r) / n
    m2 = sum((x - mu) ** 2 for x in r) / n
    if m2 <= 0.0:
        return mu, 0.0, 0.0, 3.0
    sd = math.sqrt(m2)
    m3 = sum((x - mu) ** 3 for x in r) / n
    m4 = sum((x - mu) ** 4 for x in r) / n
    skew = m3 / (sd ** 3)
    kurt = m4 / (m2 ** 2)            # Pearson, NON-excess (normal = 3)
    return mu, sd, skew, kurt


def sharpe_ratio(returns: Sequence[float], rf: float = 0.0) -> float:
    """Non-annualized SR = (mean - rf) / standard deviation of returns."""
    mu, sd, _, _ = _moments([float(x) for x in returns])
    return 0.0 if sd == 0.0 else (mu - rf) / sd


def summary(returns: Sequence[float], rf: float = 0.0) -> Dict[str, float]:
    r = [float(x) for x in returns]
    mu, sd, skew, kurt = _moments(r)
    sr = 0.0 if sd == 0.0 else (mu - rf) / sd
    return {"n": len(r), "mean": mu, "std": sd, "sharpe": sr,
            "skew": skew, "kurtosis": kurt}


def _sr_stats(returns, rf=0.0):
    s = summary(returns, rf)
    return s["sharpe"], s["skew"], s["kurtosis"], int(s["n"])


# --------------------------------------------------------------------------- #
# PSR / MinTRL / SR0 / DSR                                                     #
# --------------------------------------------------------------------------- #
def probabilistic_sharpe(returns: Optional[Sequence[float]], sr_benchmark: float = 0.0,
                         rf: float = 0.0, *, sr=None, skew=None, kurt=None, n=None) -> float:
    """
    Probabilistic Sharpe Ratio. Bailey & Lopez de Prado (2012), "The Sharpe Ratio
    Efficient Frontier", Journal of Risk 15(2).

        PSR(SR*) = Phi( (SR_hat - SR*) * sqrt(n - 1)
                        / sqrt(1 - g3*SR_hat + ((g4 - 1)/4)*SR_hat^2) )

    g3 = skewness, g4 = NON-excess kurtosis. Returns P(true SR > SR*).
    (The two forms of the denominator, 1 - g3*SR + (g4-1)/4*SR^2 and
     1 + 0.5*SR^2 - g3*SR + (g4-3)/4*SR^2, are algebraically identical.)
    """
    if sr is None:
        sr, skew, kurt, n = _sr_stats(returns, rf)
    if n is None or n < 2:
        return float("nan")
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    if denom <= 0.0:
        return float("nan")
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)
    return _PHI(z)


def min_track_record_length(returns: Optional[Sequence[float]], sr_benchmark: float = 0.0,
                            prob: float = 0.95, rf: float = 0.0,
                            *, sr=None, skew=None, kurt=None) -> float:
    """
    Minimum Track Record Length. Bailey & Lopez de Prado (2012/2014).
        MinTRL = 1 + (1 - g3*SR + ((g4-1)/4)*SR^2) * ( z_prob / (SR - SR*) )^2
    Minimum number of observations for PSR(SR*) > prob. (Direct derivation:
    set PSR = prob in the formula above and solve for n.)
    """
    if sr is None:
        sr, skew, kurt, _ = _sr_stats(returns, rf)
    if sr <= sr_benchmark:
        return float("inf")
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    z = _PHI_INV(prob)
    return 1.0 + denom * (z / (sr - sr_benchmark)) ** 2


def expected_maximum_sharpe(sr_trials_variance: float, n_trials: int) -> float:
    """
    Expected maximum Sharpe under H0 (E[SR]=0) for N independent trials -
    False Strategy Theorem, Bailey & Lopez de Prado (2014), JPM 40(5):94-107.

        SR0 = sqrt(V[SR]) * ( (1 - gamma) * Phi^{-1}(1 - 1/N)
                              + gamma     * Phi^{-1}(1 - 1/(N*e)) )

    gamma = Euler-Mascheroni; V[SR] = variance of the NON-annualized SRs across the N trials.
    (Canonical form confirmed by Wikipedia/DSR and gmarti; a widely circulated Medium
     implementation is wrong - spurious (1-1/N) factors and 1/Z.)
    """
    N = int(n_trials)
    if N < 2 or sr_trials_variance <= 0.0:
        return 0.0
    s = math.sqrt(sr_trials_variance)
    a = (1.0 - _GAMMA) * _PHI_INV(1.0 - 1.0 / N)
    b = _GAMMA * _PHI_INV(1.0 - 1.0 / (N * _E))
    return s * (a + b)


def deflated_sharpe(returns: Optional[Sequence[float]], sr_trials_variance: float,
                    n_trials: int, rf: float = 0.0,
                    *, sr=None, skew=None, kurt=None, n=None) -> float:
    """
    Deflated Sharpe Ratio = PSR(SR0). Bailey & Lopez de Prado (2014).
    Probability that the observed SR comes from a positive-mean distribution, AFTER
    correcting for selection bias (N trials), length, and non-normality.
    """
    if sr is None:
        sr, skew, kurt, n = _sr_stats(returns, rf)
    sr0 = expected_maximum_sharpe(sr_trials_variance, n_trials)
    return probabilistic_sharpe(None, sr_benchmark=sr0, sr=sr, skew=skew, kurt=kurt, n=n)


# --------------------------------------------------------------------------- #
# PBO via CSCV                                                                 #
# --------------------------------------------------------------------------- #
def pbo_cscv(perf_matrix: Sequence[Sequence[float]], n_splits: int = 10,
             metric: Optional[Callable[[List[float]], float]] = None,
             seed: int = 0, max_combos: int = 400) -> float:
    """
    Probability of Backtest Overfitting via Combinatorially-Symmetric Cross-Validation.
    Bailey, Borwein, Lopez de Prado & Zhu, "The Probability of Backtest Overfitting".

    perf_matrix : T rows (instants) x C columns (configurations); value = perf
    of config c at instant t (e.g. return). Procedure:
      1) partition the T rows into S disjoint blocks (S even);
      2) for each combination of S/2 blocks as IS (the rest as OOS):
         n* = argmax(IS perf); relative OOS rank of n*: w = rank_OOS(n*)/(C+1);
         logit lambda = ln(w/(1-w));
      3) PBO = P(lambda <= 0) = frequency at which the IS-best falls BELOW the OOS median.
    NOTE (not a bug): IS and OOS partition ALL rows, so for a given column the OOS perf
    is ~complementary to the IS perf. When all configs are the SAME noise (none has
    skill), selecting the IS-best is pure overfitting and it systematically falls below
    the OOS median -> HIGH PBO (close to 1), which is the correct diagnosis. The paper's
    "PBO~0.5" benchmark assumes IS _|_ OOS (genuinely distinct strategies), an assumption
    that identical columns do not satisfy.
    metric(values of a column over a subset of rows) -> float; default = mean.
    """
    M = [list(map(float, row)) for row in perf_matrix]
    T = len(M)
    if T == 0:
        return float("nan")
    C = len(M[0])
    if C < 2:
        return float("nan")
    if metric is None:
        metric = lambda vals: sum(vals) / len(vals)
    S = n_splits - (n_splits % 2)            # make even
    S = max(2, min(S, T))
    bounds = [round(i * T / S) for i in range(S + 1)]
    blocks = [list(range(bounds[i], bounds[i + 1])) for i in range(S)]
    half = S // 2
    combos = list(combinations(range(S), half))
    rng = random.Random(seed)
    if len(combos) > max_combos:
        combos = rng.sample(combos, max_combos)
    lambdas: List[float] = []
    for IS in combos:
        IS_set = set(IS)
        is_rows = [r for i in IS_set for r in blocks[i]]
        oos_rows = [r for i in range(S) if i not in IS_set for r in blocks[i]]
        if not is_rows or not oos_rows:
            continue
        is_perf = [metric([M[r][c] for r in is_rows]) for c in range(C)]
        oos_perf = [metric([M[r][c] for r in oos_rows]) for c in range(C)]
        nstar = max(range(C), key=lambda c: is_perf[c])
        rank = 1 + sum(1 for c in range(C) if oos_perf[c] < oos_perf[nstar])  # 1=worst..C=best
        w = rank / (C + 1.0)
        w = min(max(w, 1e-9), 1.0 - 1e-9)
        lambdas.append(math.log(w / (1.0 - w)))
    if not lambdas:
        return float("nan")
    return sum(1 for l in lambdas if l <= 0.0) / len(lambdas)


# --------------------------------------------------------------------------- #
# Stationary bootstrap (Politis & Romano 1994) shared by RC and SPA        #
# --------------------------------------------------------------------------- #
def _stationary_boot_colmeans(D, n_boot, block, seed):
    """Return n_boot column-mean vectors under the stationary bootstrap.
    Block length ~ Geometric(1/block); circular resampling."""
    T = len(D)
    K = len(D[0])
    rng = random.Random(seed)
    p = 1.0 / max(1, block)
    out = []
    for _ in range(n_boot):
        cm = [0.0] * K
        i = rng.randrange(T)
        for t in range(T):
            if t == 0 or rng.random() < p:
                i = rng.randrange(T)
            else:
                i = (i + 1) % T
            row = D[i]
            for k in range(K):
                cm[k] += row[k]
        out.append([c / T for c in cm])
    return out


def whites_reality_check(loss_diff_matrix: Sequence[Sequence[float]],
                         n_boot: int = 1000, block: int = 10, seed: int = 0) -> float:
    """
    White (2000) Reality Check, p-value. Econometrica 68:1097-1126.
    loss_diff_matrix : T x K, d[t,k] = perf_k(t) - benchmark(t) (>0 = beats the benchmark).
    H0 : the BEST strategy does not beat the benchmark (max_k E[d_k] <= 0).
      Statistic V = max_k sqrt(T) * mean_t d[t,k];
      re-centered stationary bootstrap: V*_b = max_k sqrt(T)*(mean d*_k - mean d_k);
      p = (#{V*_b >= V} + 1)/(n_boot + 1).
    """
    D = [list(map(float, r)) for r in loss_diff_matrix]
    T = len(D)
    K = len(D[0])
    rootT = math.sqrt(T)
    dbar = [sum(D[t][k] for t in range(T)) / T for k in range(K)]
    V = max(rootT * dbar[k] for k in range(K))
    bm = _stationary_boot_colmeans(D, n_boot, block, seed)
    count = sum(1 for cm in bm
                if max(rootT * (cm[k] - dbar[k]) for k in range(K)) >= V)
    return (count + 1) / (n_boot + 1)


def hansen_spa(loss_diff_matrix: Sequence[Sequence[float]],
               n_boot: int = 1000, block: int = 10, seed: int = 0) -> float:
    """
    Hansen (2005) Superior Predictive Ability, consistent variant SPA_c. JBES.
    Studentizes each d_k by its bootstrap standard deviation omega_k:
      T_SPA = max_k [ sqrt(T)*dbar_k / omega_k ]_+ ,   [x]_+ = max(0,x).
    CONSISTENT re-centering: a strategy contributes to the null only if
      sqrt(T)*dbar_k/omega_k >= -sqrt(2*ln ln T)  (otherwise re-centered to 0),
    which removes the bad models and increases power vs RC.
      T_SPA*_b = max_k [ sqrt(T)*(dbar*_k - g_k)/omega_k ]_+ ;
      p = (#{T_SPA*_b >= T_SPA} + 1)/(n_boot + 1).
    """
    D = [list(map(float, r)) for r in loss_diff_matrix]
    T = len(D)
    K = len(D[0])
    rootT = math.sqrt(T)
    dbar = [sum(D[t][k] for t in range(T)) / T for k in range(K)]
    bm = _stationary_boot_colmeans(D, n_boot, block, seed)
    # omega_k = bootstrap standard deviation of sqrt(T)*mean_k
    omega = []
    for k in range(K):
        vals = [rootT * cm[k] for cm in bm]
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)
        omega.append(math.sqrt(var) if var > 0 else 1e-12)
    Tobs = max([0.0] + [rootT * dbar[k] / omega[k] for k in range(K)])
    thr = math.sqrt(2.0 * math.log(math.log(T))) if T > math.e + 1 else 0.0
    g = [dbar[k] if (rootT * dbar[k] / omega[k]) >= -thr else 0.0 for k in range(K)]
    count = 0
    for cm in bm:
        stat = max([0.0] + [rootT * (cm[k] - g[k]) / omega[k] for k in range(K)])
        if stat >= Tobs:
            count += 1
    return (count + 1) / (n_boot + 1)


# --------------------------------------------------------------------------- #
# Aggregate decision                                                            #
# --------------------------------------------------------------------------- #
_DEFAULT_TH = {
    "dsr_min": 0.95,        # required P(true SR>0 after deflation)
    "pbo_max": 0.20,        # maximum tolerated overfitting probability
    "rc_alpha": 0.05,       # Reality Check p-value threshold
    "spa_alpha": 0.05,      # SPA p-value threshold
    "mintrl_max": None,     # if provided: MinTRL <= available horizon
}


def acceptance_decision(returns: Sequence[float], sr_trials_variance: float, n_trials: int,
                        perf_matrix: Optional[Sequence[Sequence[float]]] = None,
                        loss_diff_matrix: Optional[Sequence[Sequence[float]]] = None,
                        thresholds: Optional[Dict] = None, rf: float = 0.0) -> Dict:
    """Aggregates the AVAILABLE tests into ACCEPT/REJECT. Applies only those for which
    data is provided (returns mandatory; perf_matrix -> PBO;
    loss_diff_matrix -> RC & SPA). 'reasons' lists each failure."""
    th = dict(_DEFAULT_TH)
    if thresholds:
        th.update(thresholds)
    sr, skew, kurt, n = _sr_stats(returns, rf)
    dsr = deflated_sharpe(None, sr_trials_variance, n_trials, sr=sr, skew=skew, kurt=kurt, n=n)
    psr0_ok = (dsr == dsr) and dsr >= th["dsr_min"]
    res = {"sharpe": sr, "skew": skew, "kurtosis": kurt, "n": n,
           "sr0": expected_maximum_sharpe(sr_trials_variance, n_trials),
           "dsr": dsr, "mintrl": min_track_record_length(None, prob=th["dsr_min"],
                                                         sr=sr, skew=skew, kurt=kurt),
           "pbo": None, "rc_pvalue": None, "spa_pvalue": None}
    reasons = []
    if not psr0_ok:
        reasons.append(f"DSR={dsr:.3f} < {th['dsr_min']}")
    if th.get("mintrl_max") is not None and res["mintrl"] > th["mintrl_max"]:
        reasons.append(f"MinTRL={res['mintrl']:.0f} > {th['mintrl_max']}")
    if perf_matrix is not None:
        res["pbo"] = pbo_cscv(perf_matrix)
        if res["pbo"] == res["pbo"] and res["pbo"] > th["pbo_max"]:
            reasons.append(f"PBO={res['pbo']:.3f} > {th['pbo_max']}")
    if loss_diff_matrix is not None:
        res["rc_pvalue"] = whites_reality_check(loss_diff_matrix)
        res["spa_pvalue"] = hansen_spa(loss_diff_matrix)
        if res["rc_pvalue"] > th["rc_alpha"]:
            reasons.append(f"RC p={res['rc_pvalue']:.3f} > {th['rc_alpha']}")
        if res["spa_pvalue"] > th["spa_alpha"]:
            reasons.append(f"SPA p={res['spa_pvalue']:.3f} > {th['spa_alpha']}")
    res["decision"] = "ACCEPT" if not reasons else "REJECT"
    res["reasons"] = reasons
    return res
