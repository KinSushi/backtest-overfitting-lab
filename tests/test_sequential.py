"""Offline tests for egp_sequential (deterministic, by seeded replications)."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_sequential as SQ


def test_sprt_decides_h1_under_alternative():
    rng = random.Random(0)
    stream = [rng.gauss(0.5, 1.0) for _ in range(2000)]    # vrai theta=0.5
    out = SQ.sprt_gaussian(stream, theta0=0.0, theta1=0.5, sigma=1.0, alpha=0.05, beta=0.10)
    assert out["decision"] == "H1"


def test_sprt_decides_h0_under_null():
    rng = random.Random(1)
    stream = [rng.gauss(0.0, 1.0) for _ in range(2000)]    # vrai theta=0
    out = SQ.sprt_gaussian(stream, theta0=0.0, theta1=0.5, sigma=1.0, alpha=0.05, beta=0.10)
    assert out["decision"] == "H0"


def test_sprt_type1_error_controlled():
    # On null streams, the rate of false H1 decisions stays close to/below alpha.
    alpha = 0.05
    false_h1 = 0
    R = 400
    for s in range(R):
        rng = random.Random(1000 + s)
        stream = [rng.gauss(0.0, 1.0) for _ in range(3000)]
        out = SQ.sprt_gaussian(stream, 0.0, 0.5, 1.0, alpha=alpha, beta=0.10)
        if out["decision"] == "H1":
            false_h1 += 1
    assert false_h1 / R <= alpha + 0.03


def test_betting_martingale_anytime_type1_le_alpha():
    # Key property (Ville) : under H0 (mean 0), P(reject on a day) <= alpha,
    # EVEN when looking at each step. Increments bounded in [-1,1].
    alpha = 0.05
    rejects = 0
    R = 400
    for s in range(R):
        rng = random.Random(7000 + s)
        stream = [max(-1.0, min(1.0, rng.gauss(0.0, 0.5))) for _ in range(500)]
        out = SQ.betting_martingale_test(stream, alpha=alpha, lam=0.5, bound=1.0)
        if out["reject"]:
            rejects += 1
    assert rejects / R <= alpha + 0.03  # anytime control of type I


def test_betting_martingale_has_power_under_alternative():
    # Under H1 (mean > 0), the test rejects in the vast majority of cases.
    alpha = 0.05
    rejects = 0
    R = 200
    for s in range(R):
        rng = random.Random(9000 + s)
        stream = [max(-1.0, min(1.0, rng.gauss(0.3, 0.5))) for _ in range(500)]
        out = SQ.betting_martingale_test(stream, alpha=alpha, lam=0.5, bound=1.0)
        if out["reject"]:
            rejects += 1
    assert rejects / R > 0.80


def test_anytime_pvalue_is_inverse_max_wealth():
    rng = random.Random(3)
    stream = [max(-1.0, min(1.0, rng.gauss(0.3, 0.5))) for _ in range(300)]
    out = SQ.betting_martingale_test(stream, alpha=0.05, lam=0.5, bound=1.0)
    assert abs(out["anytime_pvalue"] - min(1.0, 1.0 / out["max_wealth"])) < 1e-12
