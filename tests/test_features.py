"""Offline tests for the feature stack: shape, sign, causality (no look-ahead), golden moments."""
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_features as FE


def test_skew_kurt_golden():
    # symmetric series -> skew ~ 0; excess kurtosis of a discrete uniform < 0 (platykurtic)
    sym = [-2, -1, 0, 1, 2]
    assert abs(FE._skew(sym)) < 1e-9
    assert FE._kurt(sym) < 0.0


def test_acf1_positive_for_trend():
    # strongly autocorrelated series (trending walk) -> high lag-1 ACF
    x = [0.0]
    for i in range(100):
        x.append(x[-1] + 0.1)  # deterministic trend
    assert FE._acf1(x) > 0.9


def test_acf1_near_zero_for_white_noise():
    rng = random.Random(0)
    x = [rng.gauss(0, 1) for _ in range(2000)]
    assert abs(FE._acf1(x)) < 0.1


def test_feature_matrix_shape_and_names():
    rng = random.Random(1)
    prices = [2000.0]
    for _ in range(300):
        prices.append(prices[-1] * (1 + rng.gauss(0, 0.005)))
    idx = list(range(70, 290, 5))
    X, names = FE.build_features(prices, idx)
    assert len(X) == len(idx)
    assert all(len(row) == len(names) for row in X)     # matrice rectangulaire
    assert "ewma_vol" in names and "fracdiff_0.3" in names and "vol_20" in names


def test_momentum_sign():
    prices = [100.0 * (1.01 ** i) for i in range(200)]  # monotone rise
    X, names = FE.build_features(prices, [100, 150])
    j = names.index("mom_20")
    assert all(row[j] > 0 for row in X)  # positive momentum in an uptrend


def test_no_lookahead():
    # modifying a FUTURE price must not change a feature computed at an earlier index
    rng = random.Random(2)
    prices = [2000.0]
    for _ in range(200):
        prices.append(prices[-1] * (1 + rng.gauss(0, 0.004)))
    idx = [120]
    X1, names = FE.build_features(prices, idx)
    prices2 = list(prices)
    prices2[180] *= 1.5                                   # choc strictement apres l'evenement 120
    X2, _ = FE.build_features(prices2, idx)
    assert X1[0] == X2[0]                                 # features inchangees (causalite)


def test_insufficient_history_zero_filled():
    prices = [2000.0 + i for i in range(50)]
    X, names = FE.build_features(prices, [2])             # t=2 < all windows
    j = names.index("vol_20")
    assert X[0][j] == 0.0  # window unavailable -> 0
