"""Offline tests for egp_structural (deterministic)."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_structural as ST


def test_supf_detects_and_localizes_break():
    rng = random.Random(0)
    series = [rng.gauss(0, 1) for _ in range(200)] + [rng.gauss(1.5, 1) for _ in range(200)]
    out = ST.supf_break(series, trim=0.15)
    assert out["supF"] > 50.0  # clean break
    assert abs(out["break_index"] - 200) <= 10      # localisation precise


def test_supf_small_without_break():
    rng = random.Random(1)
    series = [rng.gauss(0, 1) for _ in range(400)]
    out = ST.supf_break(series, trim=0.15)
    assert out["supF"] < 20.0                       # no break => low statistic


def test_supf_discriminates():
    rng = random.Random(2)
    brk = [rng.gauss(0, 1) for _ in range(200)] + [rng.gauss(1.5, 1) for _ in range(200)]
    rng = random.Random(3)
    noise = [rng.gauss(0, 1) for _ in range(400)]
    assert ST.supf_break(brk)["supF"] > 10 * ST.supf_break(noise)["supF"]


def test_cusum_alarms_on_shift_and_silent_on_noise():
    # Clean scenario (low noise, slack k=1.0): clear upward alarm, no
    # false alarm on the noise.
    rng = random.Random(4)
    shift = [rng.gauss(0, 0.3) for _ in range(200)] + [rng.gauss(2.0, 0.3) for _ in range(200)]
    out = ST.page_cusum(shift, mu0=0.0, k=1.0, h=5.0)
    assert out["alarm"] is not None and out["direction"] == "+"
    assert out["alarm"] >= 190  # after the shift (index 200)

    rng = random.Random(5)
    noise = [rng.gauss(0, 0.3) for _ in range(400)]
    out_n = ST.page_cusum(noise, mu0=0.0, k=1.0, h=5.0)
    assert out_n["alarm"] is None  # silent on centered noise


def test_cusum_detects_downward_shift():
    rng = random.Random(6)
    down = [rng.gauss(0, 0.3) for _ in range(150)] + [rng.gauss(-2.0, 0.3) for _ in range(150)]
    out = ST.page_cusum(down, mu0=0.0, k=1.0, h=5.0)
    assert out["alarm"] is not None and out["direction"] == "-"
