"""
egp_sequential.py - "always-valid" sequential tests for live monitoring.

In production (champion/challenger), results are watched CONTINUOUSLY. Classical
tests (p-value at fixed n) are invalid under this "peeking": the type I error explodes.
Anytime-valid sequential tests control the error at ANY stopping time.

Two tools :
  1) Wald SPRT: sequential test of two simple hypotheses (theta0 vs theta1), with
     bounds A = log((1-beta)/alpha) (accept H1) and B = log(beta/(1-alpha)) (accept H0).
  2) Betting martingale (supermartingale test): for H0 "mean <= 0" on increments
     BOUNDED, M_t = Π (1 + lambda * x_t) is a supermartingale under H0; by Ville's
     inequality, P(∃ t : M_t >= 1/alpha) <= alpha. We reject when M_t >= 1/alpha. The p-value
     anytime-valid equals 1/max_s M_s. No peeking cost: valid at any stopping time.

Sources :
  - Wald, A. (1945), "Sequential Tests of Statistical Hypotheses", Annals of Mathematical
    Statistics 16(2):117-186.
  - Ville, J. (1939), "Etude critique de la notion de collectif" (Ville's inequality).
  - Howard, Ramdas, McAuliffe & Sekhon (2021), "Time-uniform, nonparametric, nonasymptotic
    confidence sequences", Annals of Statistics 49(2):1055-1080.
Pur-Python.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence


def wald_boundaries(alpha: float = 0.05, beta: float = 0.10):
    """Bounds (B_inf, A_sup) of the SPRT in LOG-likelihood ratio."""
    A = math.log((1.0 - beta) / alpha)      # crossed from above => accept H1
    B = math.log(beta / (1.0 - alpha))      # crossed from below  => accept H0
    return B, A


def sprt_gaussian(stream: Sequence[float], theta0: float, theta1: float,
                  sigma: float, alpha: float = 0.05, beta: float = 0.10) -> Dict:
    """SPRT for Gaussian observations N(theta, sigma^2), known variance.
    LLR_t += (theta1-theta0)*(x - (theta0+theta1)/2)/sigma^2. Returns the decision
    ('H1','H0','continue'), the stopping time and the final LLR."""
    B, A = wald_boundaries(alpha, beta)
    s2 = sigma * sigma
    llr = 0.0
    for t, x in enumerate(stream, start=1):
        llr += (theta1 - theta0) * (x - 0.5 * (theta0 + theta1)) / s2
        if llr >= A:
            return {"decision": "H1", "stop": t, "llr": llr, "A": A, "B": B}
        if llr <= B:
            return {"decision": "H0", "stop": t, "llr": llr, "A": A, "B": B}
    return {"decision": "continue", "stop": len(stream), "llr": llr, "A": A, "B": B}


def betting_martingale_test(stream: Sequence[float], alpha: float = 0.05,
                            lam: float = 0.5, bound: float = 1.0) -> Dict:
    """Anytime-valid test of H0: mean <= 0 (increments in [-bound, bound]).
    M_t = Π (1 + lambda * x_t/bound); reject as soon as M_t >= 1/alpha (Ville).
    lambda in (0,1] to keep 1 + lambda*x/bound >= 0. Returns reject, time,
    max wealth and anytime p-value = 1/max M."""
    thr = 1.0 / alpha
    m = 1.0
    m_max = 1.0
    stop = None
    for t, x in enumerate(stream, start=1):
        xb = max(-bound, min(bound, x)) / bound
        m *= (1.0 + lam * xb)
        if m > m_max:
            m_max = m
        if stop is None and m >= thr:
            stop = t
    return {"reject": stop is not None, "stop": stop, "max_wealth": m_max,
            "anytime_pvalue": min(1.0, 1.0 / m_max) if m_max > 0 else 1.0,
            "threshold": thr}
