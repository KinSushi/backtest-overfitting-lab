"""
egp_bet_sizing.py - Bet sizing, AFML ch.10 (Lopez de Prado, 2018).

"A model can be very accurate; if you do not size your bets properly, the strategy loses
money." This module turns the secondary model's PROBABILITY (probability that the meta-label
= 1, i.e. that the bet wins) into a POSITION SIZE in [-1, 1], with a concurrency correction
(average of active bets) and discretization (anti-overtrading).

Formulas VERIFIED against sources (snippet 10.1 getSignal, official bonus PDF via Studylib;
binary case confirmed by CanerIrfanoglu/advances_in_ml):
  - OvR t-value:  z = (p - 1/numClasses) / sqrt(p*(1-p))
  - size      :  size = 2*Phi(z) - 1     (Phi = standard normal CDF)
  - signal    :  signal = pred * size    (pred = predicted side); in meta-labeling, * side (side of
                  the primary model). Conservative: p~0.5 -> size ~0 (no bet).
  - average of active bets (snippet 10.2 avgActiveSignals): at each instant, average of the
    signals of the ACTIVE bets (issued at/before t AND t < ending t1, or t1 unknown).
  - discretization (snippet 10.4 discreteSignal): signal = round(signal/stepSize)*stepSize,
    clamped to [-1, 1].

AFML sections: 10.3 Bet Sizing from Predicted Probabilities; 10.4 Averaging Active Bets;
10.5 Size Discretization. Phi via statistics.NormalDist (pure-Python, standard deviation 1).
Source: M. Lopez de Prado, *Advances in Financial Machine Learning* (2018), ch.10.

RESERVATION: dynamic sizing on forecast price (10.6, sigmoid/power) and the Gaussian-mixture
adjustment (EF3M, exercise 10.4) are NOT implemented here (out of scope; they would need a price
forecast / EF3M). This module covers 10.3-10.5, the core probability->size.
"""
from __future__ import annotations

from statistics import NormalDist
from typing import Dict, List, Optional, Sequence, Tuple

_PHI = NormalDist()           # N(0, 1)
_EPS = 1e-9


def prob_to_size(prob: float, num_classes: int = 2) -> float:
    """Raw size in [-1, 1] from the probability (snippet 10.1): 2*Phi(z)-1,
    z = (p - 1/numClasses)/sqrt(p(1-p)). Probability clamped to (eps, 1-eps) to avoid division by 0."""
    p = min(1.0 - _EPS, max(_EPS, prob))
    z = (p - 1.0 / num_classes) / (p * (1.0 - p)) ** 0.5
    return 2.0 * _PHI.cdf(z) - 1.0


def bet_size(prob: float, side: int = 1, num_classes: int = 2) -> float:
    """Signal = side * size. In meta-labeling, side = side of the primary model (+1 long / -1 short).
    Returns a value in [-1, 1]."""
    return side * prob_to_size(prob, num_classes)


def discretize_signal(signal: float, step_size: float) -> float:
    """Discretizes (snippet 10.4): round(signal/stepSize)*stepSize, clamped to [-1, 1]."""
    if step_size <= 0:
        return max(-1.0, min(1.0, signal))
    s = round(signal / step_size) * step_size
    return max(-1.0, min(1.0, s))


def avg_active_signals(signals: Sequence[float], t0s: Sequence[int], t1s: Sequence[int],
                       time_points: Optional[Sequence[int]] = None) -> Dict[int, float]:
    """Average of the ACTIVE bets at each instant (snippet 10.2). A bet i is active at t if
    t0_i <= t < t1_i. Returns {t: mean_signal}. If no bet is active -> 0."""
    if time_points is None:
        pts = sorted(set(list(t0s) + list(t1s)))
    else:
        pts = sorted(set(time_points))
    out: Dict[int, float] = {}
    for t in pts:
        active = [signals[i] for i in range(len(signals)) if t0s[i] <= t < t1s[i]]
        out[t] = sum(active) / len(active) if active else 0.0
    return out


def bet_sizes_from_meta(probs: Sequence[float], sides: Sequence[int], t0s: Sequence[int],
                        t1s: Sequence[int], step_size: float = 0.0, num_classes: int = 2,
                        average_active: bool = True) -> Dict:
    """End to end (meta-labeling): probability -> per-bet signal -> average of active -> discretization.
    Returns {per_bet_signal, active_avg (if requested), discretized}."""
    per_bet = [bet_size(probs[i], sides[i], num_classes) for i in range(len(probs))]
    out: Dict = {"per_bet_signal": per_bet}
    if average_active:
        avg = avg_active_signals(per_bet, t0s, t1s)
        out["active_avg"] = avg
        out["discretized"] = {t: discretize_signal(v, step_size) for t, v in avg.items()}
    else:
        out["discretized"] = [discretize_signal(v, step_size) for v in per_bet]
    return out


def size_to_lots(size: float, max_lots: float) -> float:
    """Converts a size [-1,1] into signed lots (|lots| <= max_lots). Bridge to execution."""
    s = max(-1.0, min(1.0, size))
    return s * max_lots
