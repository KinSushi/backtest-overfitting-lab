"""Offline tests for probability calibration: golden Brier/PAVA + Brier improvement."""
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_calibration as CAL


def test_brier_golden():
    assert CAL.brier_score([1.0, 0.0, 1.0, 0.0], [1, 0, 1, 0]) == 0.0   # parfait
    assert abs(CAL.brier_score([0.5, 0.5], [1, 0]) - 0.25) < 1e-12      # mean(0.25,0.25)


def test_pava_golden():
    # [1,0,1] -> violates monotonicity -> pool the first two (mean 0.5) -> [0.5,0.5,1]
    out = CAL._pava([1.0, 0.0, 1.0], [1.0, 1.0, 1.0])
    assert abs(out[0] - 0.5) < 1e-12 and abs(out[1] - 0.5) < 1e-12 and abs(out[2] - 1.0) < 1e-12


def test_pava_already_monotone_unchanged():
    out = CAL._pava([0.1, 0.2, 0.7], [1, 1, 1])
    assert out == [0.1, 0.2, 0.7]


def test_pava_output_is_monotone():
    rng = random.Random(0)
    y = [rng.randint(0, 1) for _ in range(50)]
    out = CAL._pava(y, [1.0] * 50)
    assert all(out[i] <= out[i + 1] + 1e-12 for i in range(len(out) - 1))


def _temp_data(n, seed, T=2.5):
    # logit z ; vraie proba sigma(z/T) (douce) ; proba BRUTE surconfiante sigma(z) (T=1) ; y~Bernoulli(vraie)
    rng = random.Random(seed)
    logits, raw_prob, y = [], [], []
    for _ in range(n):
        z = rng.uniform(-5, 5)
        p_true = 1.0 / (1.0 + math.exp(-z / T))
        logits.append(z)
        raw_prob.append(1.0 / (1.0 + math.exp(-z)))      # surconfiant
        y.append(1 if rng.random() < p_true else 0)
    return logits, raw_prob, y


def test_platt_improves_over_overconfident():
    lo_f, _, y_f = _temp_data(2500, 1)
    lo_t, raw_t, y_t = _temp_data(2500, 2)
    plt = CAL.PlattScaling(lr=0.1, epochs=2000).fit(lo_f, y_f)
    b_raw = CAL.brier_score(raw_t, y_t)                  # proba brute surconfiante
    b_cal = CAL.brier_score(plt.predict(lo_t), y_t)      # recalibree
    assert b_cal < b_raw
    assert plt.A < 0  # higher score -> higher probability


def test_platt_predictions_monotone():
    lo_f, _, y_f = _temp_data(1500, 3)
    plt = CAL.PlattScaling(lr=0.1, epochs=1000).fit(lo_f, y_f)
    ps = plt.predict([-3, -1, 0, 1, 3])
    assert all(ps[i] <= ps[i + 1] for i in range(len(ps) - 1))  # monotone increasing
    assert all(0.0 <= p <= 1.0 for p in ps)


def test_isotonic_improves_over_overconfident():
    lo_f, _, y_f = _temp_data(2500, 4)
    lo_t, raw_t, y_t = _temp_data(2500, 5)
    iso = CAL.IsotonicRegression().fit(lo_f, y_f)
    b_raw = CAL.brier_score(raw_t, y_t)
    b_cal = CAL.brier_score(iso.predict(lo_t), y_t)
    assert b_cal < b_raw


def test_isotonic_predictions_monotone():
    lo_f, _, y_f = _temp_data(1500, 6)
    iso = CAL.IsotonicRegression().fit(lo_f, y_f)
    xs = sorted(random.Random(7).uniform(-4, 4) for _ in range(20))
    ps = iso.predict(xs)
    assert all(ps[i] <= ps[i + 1] + 1e-9 for i in range(len(ps) - 1))  # non-decreasing


def test_calibration_gain_with_prob_scores():
    # scores already in [0,1] but poorly calibrated (squared, monotone distortion) -> isotonic improves
    rng = random.Random(9)
    sc, y = [], []
    for _ in range(1500):
        p = rng.random()
        sc.append(p * p)
        y.append(1 if rng.random() < p else 0)
    iso = CAL.IsotonicRegression().fit(sc, y)
    gain = CAL.calibration_gain(sc, y, iso)
    assert gain["improved"] is True


def test_calibrate_factory():
    lo, _, y = _temp_data(500, 8)
    p = CAL.calibrate(lo, y, method="platt", epochs=300)
    i = CAL.calibrate(lo, y, method="isotonic")
    assert hasattr(p, "predict") and hasattr(i, "predict")


def test_calibrate_cv_ensemble_and_oof_brier():
    import egp_calibration as CAL
    lo, raw, y = _temp_data(1200, 21)
    events = [(i, i + 2) for i in range(len(lo))]
    out = CAL.calibrate_cv(lo, y, events, method="isotonic", n_splits=5, n_bars=len(lo) + 3)
    assert out["n_folds"] >= 2
    assert hasattr(out["calibrator"], "predict")
    # the out-of-fold Brier must beat the overconfident raw probability
    b_raw = CAL.brier_score(raw, y)
    assert out["oof_brier"] < b_raw


def test_ensemble_calibrator_predictions_bounded():
    import egp_calibration as CAL
    lo, _, y = _temp_data(800, 22)
    events = [(i, i + 2) for i in range(len(lo))]
    out = CAL.calibrate_cv(lo, y, events, method="platt", epochs=300, n_splits=4, n_bars=len(lo) + 3)
    ps = out["calibrator"].predict([-3, 0, 3])
    assert all(0.0 <= p <= 1.0 for p in ps)
