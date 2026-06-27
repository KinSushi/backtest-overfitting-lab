"""
egp_sample_weights.py - Sample weights AFML ch.4 (Lopez de Prado, 2018).

Problem: the triple-barrier produces OVERLAPPING labels in time -> NON-IID
samples -> the secondary model (egp_meta_model) over-weights redundant information and overfits.
AFML ch.4 solution: weight each label by its UNIQUENESS and the magnitude of its return,
with an optional time decay.

Definitions VERIFIED against sources:
  - Concurrency c_t = number of active labels at bar t (AFML 4.3).
  - Instantaneous uniqueness = 1/c_t (AFML 4.4; CanerIrfanoglu/advances_in_ml).
  - Average uniqueness of label i: u_bar_i = mean of (1/c_t) over its lifespan [t0_i, t1_i]
    (AFML 4.4; whitepaper "10 reasons ML funds fail", Lopez de Prado: "the average uniqueness of
    label i is the average u_{t,i} over the label's lifespan").
  - Return-attribution weights (AFML snippet 4.10): w_i = |Sum_{t in [t0_i,t1_i]} r_t/c_t|,
    r_t = log-return; then normalized so Sum(w) = I (number of labels).
  - Time-decay (AFML snippet 4.11, VERBATIM, verified against the official PDF + Studylib):
        clfW = cumsum(u_bar sorted by time)
        if clfLastW>=0: slope=(1-clfLastW)/clfW[-1]  else: slope=1/((clfLastW+1)*clfW[-1])
        const = 1 - slope*clfW[-1] ; clfW = const + slope*clfW ; clfW[clfW<0]=0
    The decay applies to the CUMULATIVE UNIQUENESS x in [0, Sum u_bar], not chronological time.

Input convention: an event = (t0, t1) INCLUSIVE bar indices (t0 = signal, t1 = barrier
touched, cf. egp_triple_barrier). Events are assumed in TEMPORAL ORDER (increasing t0).
Pure-Python. Source: M. Lopez de Prado, *Advances in Financial Machine Learning* (2018), ch.4.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

Event = Tuple[int, int]


def concurrency(events: Sequence[Event], n_bars: int) -> List[int]:
    """c_t = number of active events at each bar t (AFML 4.3)."""
    c = [0] * n_bars
    for (a, b) in events:
        lo, hi = max(0, a), min(n_bars - 1, b)
        for t in range(lo, hi + 1):
            c[t] += 1
    return c


def average_uniqueness(events: Sequence[Event], n_bars: int) -> List[float]:
    """u_bar_i = mean of 1/c_t over the lifespan of label i (AFML 4.4)."""
    c = concurrency(events, n_bars)
    out: List[float] = []
    for (a, b) in events:
        lo, hi = max(0, a), min(n_bars - 1, b)
        span = hi - lo + 1
        if span <= 0:
            out.append(0.0)
            continue
        s = sum(1.0 / c[t] for t in range(lo, hi + 1) if c[t] > 0)
        out.append(s / span)
    return out


def return_attribution_weights(events: Sequence[Event], log_returns: Sequence[float],
                               n_bars: int, normalize: bool = True) -> List[float]:
    """w_i = |Sum_{t in [t0_i,t1_i]} r_t/c_t| (AFML snippet 4.10), normalized to Sum(w)=I."""
    c = concurrency(events, n_bars)
    w: List[float] = []
    for (a, b) in events:
        lo, hi = max(0, a), min(n_bars - 1, b)
        s = sum(log_returns[t] / c[t] for t in range(lo, hi + 1) if c[t] > 0 and t < len(log_returns))
        w.append(abs(s))
    if normalize:
        tot = sum(w)
        if tot > 0:
            I = len(w)
            w = [x * I / tot for x in w]
    return w


def time_decay(avg_uniqueness_time_ordered: Sequence[float], clf_last_w: float = 1.0) -> List[float]:
    """Piecewise-linear decay over the CUMULATIVE uniqueness (AFML snippet 4.11, verbatim).
    Input: average uniqueness in TEMPORAL ORDER (oldest first).
    clf_last_w=1 -> no decay; 0 -> the oldest tends to 0;
    <0 -> a fraction of the oldest receives zero weight."""
    cum: List[float] = []
    s = 0.0
    for u in avg_uniqueness_time_ordered:
        s += u
        cum.append(s)
    if not cum:
        return []
    last = cum[-1]
    if last <= 0:
        return [1.0 for _ in cum]
    if clf_last_w >= 0:
        slope = (1.0 - clf_last_w) / last
    else:
        slope = 1.0 / ((clf_last_w + 1.0) * last)
    const = 1.0 - slope * last
    return [max(0.0, const + slope * x) for x in cum]


def combined_weights(events: Sequence[Event], log_returns: Sequence[float], n_bars: int,
                     clf_last_w: float = 1.0, use_return_attr: bool = True,
                     use_time_decay: bool = True, normalize: bool = True) -> List[float]:
    """Final AFML ch.4 weight = (return attribution OR average uniqueness) * time-decay,
    normalized to Sum(w)=I. AFML multiplies the sample weight by the decay factor."""
    if use_return_attr:
        base = return_attribution_weights(events, log_returns, n_bars, normalize=False)
    else:
        base = average_uniqueness(events, n_bars)
    if use_time_decay:
        decay = time_decay(average_uniqueness(events, n_bars), clf_last_w)
        w = [base[i] * decay[i] for i in range(len(base))]
    else:
        w = list(base)
    if normalize:
        tot = sum(w)
        if tot > 0:
            I = len(w)
            w = [x * I / tot for x in w]
    return w


def weights_from_barriers(barrier_results: List[Dict], log_returns: Sequence[float], n_bars: int,
                          clf_last_w: float = 1.0, use_return_attr: bool = True,
                          use_time_decay: bool = True, normalize: bool = True) -> List[float]:
    """Helper: builds the events (t0, touch_idx) from egp_triple_barrier then weights them."""
    events = [(r["t0"], r["touch_idx"]) for r in barrier_results]
    return combined_weights(events, log_returns, n_bars, clf_last_w,
                            use_return_attr, use_time_decay, normalize)


def uniqueness_report(events: Sequence[Event], n_bars: int) -> Dict:
    """Diagnostic: global average uniqueness and 'effective size' of the sample.
    A low average uniqueness signals a lot of redundancy (little independent info)."""
    au = average_uniqueness(events, n_bars)
    n = len(au)
    mean_u = sum(au) / n if n else 0.0
    return {"n_events": n, "mean_avg_uniqueness": mean_u,
            "effective_sample_size": mean_u * n,   # ~ number of independent observations
            "min_uniqueness": min(au) if au else 0.0,
            "max_uniqueness": max(au) if au else 0.0}
