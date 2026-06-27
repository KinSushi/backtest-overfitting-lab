"""
egp_seq_bootstrap.py - Sequential bootstrap AFML ch.4.5 (Lopez de Prado, 2018).

Alternative to WEIGHTING (egp_sample_weights) for bagging: instead of drawing samples
uniformly (which over-represents the redundant info of overlapping labels), we draw
SEQUENTIALLY favoring labels that are UNIQUE relative to what is already drawn. The resulting
draw is closer to IID -> better bagging / random forest.

Algorithm VERIFIED verbatim (snippets 4.5 getAvgUniqueness / 4.6 seqBootstrap, official bonus PDF;
mechanism confirmed by the MQL5 article "Blueprint Part 5"):
  - getAvgUniqueness(indM): c = column sums (per-bar concurrency over the labels in
    indM); u = indM/c; avgU = mean of u where u>0 (per label).
  - seqBootstrap: phi=[]; while |phi|<sLength: for each candidate i, avgU_i = average
    uniqueness of i IF added to phi (concurrency = drawn + candidate i); prob = avgU/sum; draw 1.
  c[t] = number of active observations at t, INCLUDING the already-drawn ones AND candidate i.

Effect (AFML/MQL5 example): after drawing an observation, re-drawing it becomes less likely,
an observation with zero overlap becomes the most likely, a partial overlap intermediate.

Convention: an event = (t0, t1) INCLUSIVE indices (cf. egp_triple_barrier). Pure-Python.
Source: M. Lopez de Prado, *Advances in Financial Machine Learning* (2018), ch.4.5.
"""
from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

Event = Tuple[int, int]


def _concurrency(events: Sequence[Event], n_bars: int, label_multiset: Sequence[int]) -> List[int]:
    """Per-bar concurrency from a MULTISET of labels (multiplicity counted)."""
    conc = [0] * n_bars
    for i in label_multiset:
        a, b = events[i]
        for t in range(max(0, a), min(n_bars - 1, b) + 1):
            conc[t] += 1
    return conc


def avg_uniqueness_given_drawn(events: Sequence[Event], n_bars: int,
                               phi: Sequence[int]) -> List[float]:
    """avgU_i = average uniqueness of candidate i IF added to the already-drawn phi (snippet 4.5).
    At i's bars: concurrency = conc_phi[t] + 1. empty phi -> all 1.0 (1st draw uniform)."""
    conc_phi = _concurrency(events, n_bars, phi)
    out: List[float] = []
    for i in range(len(events)):
        a, b = events[i]
        lo, hi = max(0, a), min(n_bars - 1, b)
        span = hi - lo + 1
        if span <= 0:
            out.append(0.0)
            continue
        s = sum(1.0 / (conc_phi[t] + 1) for t in range(lo, hi + 1))
        out.append(s / span)
    return out


def seq_bootstrap(events: Sequence[Event], n_bars: int, s_length: Optional[int] = None,
                  seed: int = 0) -> List[int]:
    """Draws a sample by sequential bootstrap (snippet 4.6). Returns the label indices
    drawn (with replacement). Draw probability proportional to the conditional uniqueness."""
    n = len(events)
    if s_length is None:
        s_length = n
    rng = random.Random(seed)
    phi: List[int] = []
    conc = [0] * n_bars                       # concurrency of the already-drawn (= conc_phi)
    while len(phi) < s_length:
        avgU = []
        for i in range(n):
            a, b = events[i]
            lo, hi = max(0, a), min(n_bars - 1, b)
            span = hi - lo + 1
            if span <= 0:
                avgU.append(0.0)
                continue
            s = sum(1.0 / (conc[t] + 1) for t in range(lo, hi + 1))
            avgU.append(s / span)
        tot = sum(avgU)
        probs = [u / tot for u in avgU] if tot > 0 else [1.0 / n] * n
        d = rng.choices(range(n), weights=probs)[0]
        phi.append(d)
        a, b = events[d]
        for t in range(max(0, a), min(n_bars - 1, b) + 1):
            conc[t] += 1
    return phi


def sample_avg_uniqueness(events: Sequence[Event], n_bars: int, sample: Sequence[int]) -> float:
    """Average uniqueness of a drawn sample (concurrency computed on the sample, multiplicity
    counted). Measures the draw's effective independence: the higher, the closer to IID."""
    conc = _concurrency(events, n_bars, sample)
    us = []
    for i in sample:
        a, b = events[i]
        lo, hi = max(0, a), min(n_bars - 1, b)
        span = hi - lo + 1
        if span <= 0:
            continue
        us.append(sum(1.0 / conc[t] for t in range(lo, hi + 1)) / span)
    return sum(us) / len(us) if us else 0.0


def uniform_bootstrap(n_labels: int, s_length: Optional[int] = None, seed: int = 0) -> List[int]:
    """Standard bootstrap (uniform draw with replacement) - comparison baseline."""
    if s_length is None:
        s_length = n_labels
    rng = random.Random(seed)
    return [rng.randrange(n_labels) for _ in range(s_length)]


def compare_bootstraps(events: Sequence[Event], n_bars: int, n_runs: int = 50,
                       s_length: Optional[int] = None, seed: int = 0) -> dict:
    """Monte-Carlo experiment (AFML 4.5.4): average uniqueness of sequential vs uniform."""
    seq_us, uni_us = [], []
    for r in range(n_runs):
        seq = seq_bootstrap(events, n_bars, s_length, seed=seed + r)
        uni = uniform_bootstrap(len(events), s_length, seed=seed + r)
        seq_us.append(sample_avg_uniqueness(events, n_bars, seq))
        uni_us.append(sample_avg_uniqueness(events, n_bars, uni))
    return {"seq_mean_uniqueness": sum(seq_us) / n_runs,
            "uniform_mean_uniqueness": sum(uni_us) / n_runs,
            "n_runs": n_runs}
