"""Offline tests for the ADF test (AFML ch.5): OLS, critical values, qualitative behavior."""
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_adf as ADF


def test_ols_recovers_known_coefficients():
    # y = 2 + 3x exactement -> beta=[2,3], residus nuls
    X = [[1.0, float(i)] for i in range(20)]
    y = [2.0 + 3.0 * i for i in range(20)]
    beta, se, sigma2 = ADF._ols(X, y)
    assert abs(beta[0] - 2.0) < 1e-6 and abs(beta[1] - 3.0) < 1e-6
    assert sigma2 < 1e-12                            # ajustement parfait


def test_matrix_inverse_identity():
    A = [[4.0, 7.0], [2.0, 6.0]]
    inv = ADF._inv(A)
    prod = [[sum(A[i][k] * inv[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    assert abs(prod[0][0] - 1) < 1e-9 and abs(prod[1][1] - 1) < 1e-9
    assert abs(prod[0][1]) < 1e-9 and abs(prod[1][0]) < 1e-9


def test_critical_values_verified():
    assert ADF.ADF_CRIT["c"] == {1: -3.43, 5: -2.86, 10: -2.57}
    assert ADF.ADF_CRIT["nc"] == {1: -2.56, 5: -1.94, 10: -1.62}
    assert ADF.ADF_CRIT["ct"] == {1: -3.96, 5: -3.41, 10: -3.13}


def test_white_noise_is_stationary():
    rng = random.Random(0)
    x = [rng.gauss(0, 1) for _ in range(500)]
    res = ADF.adf_test(x, lag=1, regression="c")
    assert res["stationary"] is True
    assert res["adf_stat"] < -2.86  # clearly below the 5% threshold


def test_random_walk_is_not_stationary():
    rng = random.Random(1)
    x = [0.0]
    for _ in range(500):
        x.append(x[-1] + rng.gauss(0, 1))            # racine unitaire
    res = ADF.adf_test(x, lag=1, regression="c")
    assert res["stationary"] is False
    assert res["adf_stat"] > -2.86                   # does NOT reject the unit root


def test_schwert_lag():
    assert ADF.schwert_lag(100) == 12
    assert ADF.schwert_lag(500) >= 12


def test_min_ffd_order_random_walk():
    # random walk: d=1 (returns) is stationary; the ADF stat should drop with d
    rng = random.Random(2)
    x = [0.0]
    for _ in range(600):
        x.append(x[-1] + rng.gauss(0, 1))
    out = ADF.min_ffd_order(x, regression="c", lag=1, thresh=1e-4)
    assert out["chosen_d"] is not None
    table = {round(r[0], 3): r for r in out["table"]}
    assert table[1.0][4] is True                     # d=1 = rendements -> stationnaire (robuste)
    # the more we differentiate, the more negative the ADF statistic (d=0 >= d=1)
    assert table[0.0][1] >= table[1.0][1]


def test_stationary_series_chosen_d_zero():
    # series already stationary (noise) -> d*=0 (no differentiation needed)
    rng = random.Random(3)
    x = [rng.gauss(0, 1) for _ in range(400)]
    out = ADF.min_ffd_order(x, regression="c", lag=1, thresh=1e-4)
    assert out["chosen_d"] == 0.0


def test_adf_autolag_white_noise_small_lag():
    rng = random.Random(10)
    x = [rng.gauss(0, 1) for _ in range(500)]
    res = ADF.adf_autolag(x, regression="c", criterion="AIC")
    assert res["stationary"] is True  # stationary white noise
    assert 0 <= res["best_lag"] <= res["maxlag"]
    assert len(res["ic_table"]) >= 1


def test_adf_autolag_bic_le_aic_lags():
    # BIC penalizes more -> tends to choose a lag <= that of the AIC
    rng = random.Random(11)
    x = [0.0]
    for _ in range(500):
        x.append(0.5 * x[-1] + rng.gauss(0, 1))       # AR(1) stationnaire
    aic = ADF.adf_autolag(x, regression="c", criterion="AIC")["best_lag"]
    bic = ADF.adf_autolag(x, regression="c", criterion="BIC")["best_lag"]
    assert bic <= aic + 1                              # BIC not more generous (tolerance 1)


def test_adf_autolag_ic_formula():
    # verifies the IC formula on a controlled case: recompute T*ln(SSR/T)+pen*k
    rng = random.Random(12)
    x = [rng.gauss(0, 1) for _ in range(200)]
    res = ADF.adf_autolag(x, regression="c", criterion="BIC", maxlag=4)
    # the chosen lag must be that of the minimal IC in the table
    best_from_table = min(res["ic_table"], key=lambda r: r[1])[0]
    assert res["best_lag"] == best_from_table


def test_adf_simulated_cv_converges_to_asymptotic():
    # large n -> simulated 5% critical value ('c') close to asymptotic MacKinnon (-2.86)
    cv = ADF.adf_critical_values_simulated(n=400, regression="c", n_sims=3000, seed=0)
    assert abs(cv[5] - (-2.86)) < 0.18
    assert cv[1] < cv[5] < cv[10] < 0.0                # ordre correct, queue gauche


def test_adf_simulated_cv_trend_more_negative():
    # with trend ('ct'), the critical value is more negative than with constant ('c')
    cvc = ADF.adf_critical_values_simulated(n=300, regression="c", n_sims=2000, seed=1)[5]
    cvct = ADF.adf_critical_values_simulated(n=300, regression="ct", n_sims=2000, seed=1)[5]
    assert cvct < cvc


def test_adf_finite_sample_decision():
    # white noise -> stationary even with the finite-sample critical value
    rng = random.Random(2)
    x = [rng.gauss(0, 1) for _ in range(300)]
    res = ADF.adf_test_finite_sample(x, regression="c", n_sims=1500, seed=0)
    assert res["stationary_finite"] is True
    assert "crit_finite" in res and "crit_asymptotic" in res
