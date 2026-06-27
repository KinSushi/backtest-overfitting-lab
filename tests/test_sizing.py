"""Offline tests for egp_sizing (deterministic)."""
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_sizing as S


def test_kelly_discrete_matches_log_growth_argmax():
    # g(f) = p*ln(1+f*b) + q*ln(1-f) is maximized at f* = (b*p - q)/b (Kelly 1956).
    p, b = 0.6, 1.0  # even bet, edge 60/40 -> f* = 0.2
    f_star = S.kelly_fraction_discrete(p, b)
    assert abs(f_star - 0.2) < 1e-9

    def g(f):
        return p * math.log(1 + f * b) + (1 - p) * math.log(1 - f)

    grid = [i / 1000.0 for i in range(1, 1000)]
    f_grid = max(grid, key=g)
    assert abs(f_grid - f_star) < 0.01  # the numerical argmax matches the formula


def test_kelly_gaussian_formula_and_fractional():
    mu, sigma = 0.001, 0.02
    f_star = S.kelly_fraction_gaussian(mu, sigma)
    assert abs(f_star - mu / sigma ** 2) < 1e-12
    # forme via Sharpe : f* = SR/sigma
    sr = mu / sigma
    assert abs(S.kelly_fraction_from_sharpe(sr, sigma) - f_star) < 1e-12
    # fractional halve, floor at 0 for negative edge, cap respected
    assert abs(S.fractional_kelly(f_star, c=0.5) - 0.5 * f_star) < 1e-12
    assert S.fractional_kelly(-5.0, c=0.5) == 0.0
    assert S.fractional_kelly(10.0, c=1.0, cap=2.0) == 2.0


def test_kelly_gaussian_maximizes_second_order_growth():
    # g(f) = mu*f - 0.5*sigma^2*f^2 ; argmax analytique = mu/sigma^2.
    mu, sigma = 0.0008, 0.015
    f_star = S.kelly_fraction_gaussian(mu, sigma)
    g = lambda f: mu * f - 0.5 * sigma ** 2 * f ** 2
    grid = [i * 0.001 for i in range(0, 20000)]
    assert abs(max(grid, key=g) - f_star) < 0.01 * max(1.0, f_star)


def test_vol_target_leverage_achieves_target():
    # Scaling a series by L = sigma_target/sigma_realized gives a vol ~ target.
    rng = random.Random(0)
    r = [rng.gauss(0, 0.03) for _ in range(5000)]
    sig = S.realized_vol(r)
    target = 0.01
    L = S.vol_target_leverage(sig, target)
    scaled = [x * L for x in r]
    assert abs(S.realized_vol(scaled) - target) < 1e-3
    # garde-fous
    assert S.vol_target_leverage(0.0, 0.01) == 0.0
    assert S.vol_target_leverage(0.001, 1.0, cap=5.0) == 5.0


def test_position_multiplier_is_conservative_and_bounded():
    # Combination fractional Kelly + vol target : takes the minimum, caps at max_leverage,
    # plancher a 0.
    out = S.position_multiplier(mu=0.002, sigma=0.01, kelly_c=0.5,
                                sigma_target=0.008, max_leverage=1.0)
    assert 0.0 <= out["multiplier"] <= 1.0
    assert out["multiplier"] <= out["fractional_kelly"] + 1e-12
    assert out["multiplier"] <= out["vol_target"] + 1e-12
    # edge negatif -> 0
    neg = S.position_multiplier(mu=-0.002, sigma=0.01, kelly_c=0.5, max_leverage=1.0)
    assert neg["multiplier"] == 0.0
