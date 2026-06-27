"""
egp_sizing.py - Robust position sizing (fractional Kelly + volatility targeting).

OFFLINE layer: produces a size MULTIPLIER (leverage) to apply to the lot. It does not
modify the EA, which stays deterministic; the multiplier plugs in upstream (choice
of the .set LOT) or as an external factor. Pure-Python, no dependency.

Sources:
  - Kelly, J. L. (1956). "A New Interpretation of Information Rate." Bell System Technical
    Journal 35(4):917-926.  (log-optimal growth criterion)
  - Thorp, E. O. (2006). "The Kelly Criterion in Blackjack, Sports Betting, and the Stock
    Market."  (continuous form f* = mu/sigma^2; fractional Kelly)
  - MacLean, Thorp & Ziemba (2011). "The Kelly Capital Growth Investment Criterion." World
    Scientific.  (properties of fractional Kelly: less volatility, robustness to
    estimation error)

Warning: full Kelly assumes mu and sigma KNOWN; in practice they are estimated
with error -> fractional Kelly (c=1/2 or 1/4) is the recommended robust form, and
vol targeting makes the size adaptive to the regime.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence


def kelly_fraction_discrete(p: float, b: float) -> float:
    """Kelly fraction for a binary bet. p = win probability, b = net gain per unit
    staked on a win (net odds). f* = (b*p - (1-p)) / b = p - (1-p)/b.
    Kelly (1956). Can be negative (=> do not bet / bet the other way)."""
    if b <= 0:
        return 0.0
    q = 1.0 - p
    return (b * p - q) / b


def kelly_fraction_gaussian(mu: float, sigma: float) -> float:
    """Log-optimal growth leverage in continuous time / Gaussian approximation:
    f* = mu / sigma^2  (mu = mean return per period, sigma = standard deviation per period).
    Thorp (2006). Maximizes g(f) = mu*f - 0.5*sigma^2*f^2 (second-order expansion)."""
    if sigma <= 0:
        return 0.0
    return mu / (sigma * sigma)


def kelly_fraction_from_sharpe(sharpe: float, sigma: float) -> float:
    """Equivalent: f* = SR/sigma (since SR = mu/sigma => mu/sigma^2 = SR/sigma)."""
    if sigma <= 0:
        return 0.0
    return sharpe / sigma


def fractional_kelly(f_star: float, c: float = 0.5, cap: Optional[float] = None) -> float:
    """Fractional Kelly: c * f*, clamped to [0, cap] (negative fractions -> 0;
    no position is taken if the estimated edge is negative). c in (0,1], typically
    1/2 or 1/4. cap = maximum allowed leverage (None = no cap)."""
    f = max(0.0, c * f_star)
    if cap is not None:
        f = min(f, cap)
    return f


def realized_vol(returns: Sequence[float]) -> float:
    """Population standard deviation of the returns."""
    r = [float(x) for x in returns]
    n = len(r)
    if n < 2:
        return 0.0
    mu = sum(r) / n
    m2 = sum((x - mu) ** 2 for x in r) / n
    return math.sqrt(m2) if m2 > 0 else 0.0


def vol_target_leverage(sigma_realized: float, sigma_target: float,
                        cap: Optional[float] = None) -> float:
    """Volatility-targeting leverage: L = sigma_target / sigma_realized, so that
    the scaled position targets an ex-ante volatility = sigma_target. Clamped to cap."""
    if sigma_realized <= 0:
        return 0.0
    L = sigma_target / sigma_realized
    if cap is not None:
        L = min(L, cap)
    return L


def position_multiplier(mu: float, sigma: float, *, kelly_c: float = 0.5,
                        sigma_target: Optional[float] = None,
                        max_leverage: float = 1.0) -> Dict[str, float]:
    """Robust multiplier combining fractional Kelly and (optional) vol targeting.
    Documented policy: start from fractional Kelly (edge-aware), then if a vol target is
    given take the MINIMUM of the two leverages (the more prudent), then cap
    at max_leverage. Returns the detail for audit."""
    f_star = kelly_fraction_gaussian(mu, sigma)
    f_frac = fractional_kelly(f_star, c=kelly_c, cap=max_leverage)
    out = {"f_star": f_star, "fractional_kelly": f_frac, "vol_target": None,
           "multiplier": f_frac}
    if sigma_target is not None:
        L = vol_target_leverage(sigma, sigma_target, cap=max_leverage)
        out["vol_target"] = L
        out["multiplier"] = max(0.0, min(f_frac, L, max_leverage))
    else:
        out["multiplier"] = max(0.0, min(f_frac, max_leverage))
    return out
