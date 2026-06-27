"""Offline tests for egp_fdr (deterministic)."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_fdr as F


def test_bh_classic_needleman_example():
    # Exemple canonique de Benjamini & Hochberg (1995), 15 p-values, alpha=0.05 -> 4 rejets.
    p = [0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
         0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000]
    out = F.benjamini_hochberg(p, alpha=0.05)
    assert out["n_reject"] == 4, out["n_reject"]
    # the 4 smallest p are rejected, not the following ones
    assert out["reject"][:4] == [True, True, True, True]
    assert not any(out["reject"][4:])


def test_bh_controls_fdr_empirically():
    # m0 true nulls (p~U), m1 false nulls (p very small). Empirical FDR <= alpha (with margin).
    rng = random.Random(0)
    alpha = 0.10
    fdrs = []
    for _ in range(40):
        nulls = [rng.random() for _ in range(900)]          # vrais nuls
        alts = [rng.random() * 0.001 for _ in range(100)]  # false nulls (significant)
        p = nulls + alts
        is_null = [True] * 900 + [False] * 100
        out = F.benjamini_hochberg(p, alpha=alpha)
        rej = out["reject"]
        R = sum(rej)
        V = sum(1 for i in range(len(p)) if rej[i] and is_null[i])
        fdrs.append(V / R if R > 0 else 0.0)
    assert sum(fdrs) / len(fdrs) <= alpha + 0.02


def test_adjusted_pvalues_monotone_and_ge_raw():
    p = [0.001, 0.02, 0.03, 0.5, 0.8]
    out = F.benjamini_hochberg(p, alpha=0.05)
    adj = out["adjusted"]
    # each q-value >= corresponding raw p
    assert all(adj[i] >= p[i] - 1e-12 for i in range(len(p)))


def test_by_more_conservative_than_bh():
    p = [0.001, 0.008, 0.02, 0.03, 0.04, 0.06, 0.1, 0.3, 0.5, 0.9]
    bh = F.benjamini_hochberg(p, alpha=0.05)
    by = F.benjamini_yekutieli(p, alpha=0.05)
    assert by["n_reject"] <= bh["n_reject"]


def test_haircut_reduces_sharpe_more_with_more_tests():
    sr, n = 0.15, 1000
    h1 = F.haircut_sharpe(sr, n, n_tests=1)
    h10 = F.haircut_sharpe(sr, n, n_tests=10)
    h1000 = F.haircut_sharpe(sr, n, n_tests=1000)
    # 1 test ~ no haircut; more tests => lower adjusted Sharpe
    assert h1["haircut_sharpe"] >= h10["haircut_sharpe"] >= h1000["haircut_sharpe"]
    assert h1000["haircut_fraction"] > h10["haircut_fraction"] > 0.0
    assert h1["haircut_fraction"] < 1e-9
    # negative edge -> no haircut
    assert F.haircut_sharpe(-0.1, n, 10)["haircut_fraction"] == 0.0
