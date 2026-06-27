import unittest, math, random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from egp_mho_hybrid import (
    hybrid_minimize, op_es_self_adaptive, op_sbx, op_de_current_to_pbest,
    cmaes_csa_correct, inv_sqrt_C, jacobi_eig,
    load_opt_dims, OptDim, decode, evolve_generation, reverse_test_protected,
)
from egp_meta_algos import cmaes_minimize  # original version (CSA without C^{-1/2})
import pytest

PKG_ROOT = str(ROOT)


def sphere(x): return sum(v * v for v in x)
def rastrigin(x): return 10 * len(x) + sum(v * v - 10 * math.cos(2 * math.pi * v) for v in x)
def ellipsoid(x):  # mal conditionnee : conditionnement 1e6
    n = len(x)
    return sum((1000 ** (i / max(1, n - 1)) * xi) ** 2 for i, xi in enumerate(x))


class TestHybridEngine(unittest.TestCase):

    def test_hybrid_beats_random_on_rugged(self):
        bounds = [(-5.0, 5.0)] * 5
        res = hybrid_minimize(rastrigin, bounds, max_evals=1500, seed=11)
        rng = random.Random(11)
        rand_best = min(rastrigin([rng.uniform(-5, 5) for _ in range(5)]) for _ in range(1500))
        self.assertLess(res.best_score, rand_best)  # hybrid > random search
        self.assertLess(res.best_score, 8.0)

    def test_hybrid_converges_on_sphere(self):
        res = hybrid_minimize(sphere, [(-5.0, 5.0)] * 6, max_evals=1500, seed=3)
        self.assertLess(res.best_score, 1e-2)

    def test_aos_distribution_is_valid_and_learned(self):
        res = hybrid_minimize(sphere, [(-5.0, 5.0)] * 5, max_evals=1200, seed=7)
        self.assertEqual(len(res.op_prob), 4)
        self.assertAlmostEqual(sum(res.op_prob.values()), 1.0, places=6)
        self.assertTrue(all(p >= 0 for p in res.op_prob.values()))
        # learning: the distribution is not perfectly uniform (0.25 each)
        self.assertGreater(max(res.op_prob.values()) - min(res.op_prob.values()), 1e-6)

    def test_es_self_adaptation_shrinks_sigma_and_converges(self):
        # (1+1)-ES using ONLY the self-adaptive ES operator
        rng = random.Random(5)
        x = [4.0, -4.0, 3.0]
        sigma = [2.0, 2.0, 2.0]
        f = sphere(x)
        for _ in range(4000):
            child, csig = op_es_self_adaptive(x, sigma, rng)
            child = [max(-5, min(5, v)) for v in child]
            fc = sphere(child)
            if fc <= f:
                x, sigma, f = child, csig, fc
        self.assertLess(f, 1e-3)                                   # converge
        self.assertLess(sum(sigma) / 3, 0.5)                       # the step auto-reduced

    def test_inv_sqrt_C_is_mathematically_correct(self):
        # random SPD C ; checks C^{-1/2} C C^{-1/2} = I
        rng = random.Random(1)
        n = 4
        A = [[rng.uniform(-1, 1) for _ in range(n)] for _ in range(n)]
        C = [[sum(A[i][k] * A[j][k] for k in range(n)) + (1.0 if i == j else 0.0) for j in range(n)] for i in range(n)]
        S = inv_sqrt_C(C)
        SC = [[sum(S[i][k] * C[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
        SCS = [[sum(SC[i][k] * S[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
        err = max(abs(SCS[i][j] - (1.0 if i == j else 0.0)) for i in range(n) for j in range(n))
        self.assertLess(err, 1e-6)

    def test_corrected_cmaes_converges(self):
        st = cmaes_csa_correct(sphere, [(-3.0, 3.0)] * 4, max_evals=700, seed=2)
        self.assertLess(st.best_score, 1e-3)

    def test_corrected_cmaes_optimizes_illconditioned(self):
        # Honest: we prove the corrected version OPTIMIZES (beats random),
        # NOT that it outperforms the naive version (not demonstrated at short budget).
        b = [(-5.0, 5.0)] * 4
        st = cmaes_csa_correct(ellipsoid, b, max_evals=1500, seed=1)
        self.assertTrue(math.isfinite(st.best_score))
        rng = random.Random(1)
        rand_best = min(ellipsoid([rng.uniform(-5, 5) for _ in range(4)]) for _ in range(1500))
        self.assertLess(st.best_score, rand_best)

    # ----- a-la-carte parameter wiring (fixes MHO-0) + policy ----- #
    def test_load_opt_dims_real_map(self):
        pytest.skip("Requires private EA artifacts (parameter map / real EA), excluded from the public demo repo.")
        dims = load_opt_dims(PKG_ROOT)
        self.assertGreater(len(dims), 0)
        for d in dims:
            self.assertLess(d.lo, d.hi)

    def test_closed_loop_improves_over_generations(self):
        # synthetic dims + real target : generation 1 (wired) beats generation 0
        dims = [OptDim(f'p{i}', -5.0, 5.0, False) for i in range(4)]
        target = sphere
        gen0 = evolve_generation(dims, None, None, family='TPE_LSHADE', n=10, seed=123)
        s0 = [target(v) for v in gen0]
        gen1 = evolve_generation(dims, gen0, s0, family='TPE_LSHADE', n=10, seed=123, inner_evals=200)
        s1 = [target(v) for v in gen1]
        med0 = sorted(s0)[len(s0) // 2]
        self.assertLess(min(s1), med0)  # the propose/observe loop improves
        for v in gen1:                                  # candidates within the OPT bounds
            self.assertTrue(all(-5.0 <= xi <= 5.0 for xi in v))

    def test_gen0_is_space_filling_lhs(self):
        dims = [OptDim(f'p{i}', 0.0, 1.0, False) for i in range(3)]
        gen0 = evolve_generation(dims, None, None, n=10, seed=42)
        self.assertEqual(len(gen0), 10)
        # LHS: each dimension covers the 10 strata [k/10,(k+1)/10)
        for j in range(3):
            strata = sorted(int(v[j] * 10) for v in gen0)
            self.assertEqual(strata, list(range(10)))

    def test_decode_and_reverse_test_respect_policy(self):
        pytest.skip("Requires private EA artifacts (parameter map / real EA), excluded from the public demo repo.")
        dims = load_opt_dims(PKG_ROOT)[:6]
        vec = [(d.lo + d.hi) / 2 for d in dims]
        produced = decode(vec, dims)
        self.assertEqual(reverse_test_protected(PKG_ROOT, produced), [])   # 0 violation
        # injection of a protected key -> must be flagged
        produced_bad = dict(produced); produced_bad['In_META_Enable'] = 'true'
        self.assertIn('In_META_Enable', reverse_test_protected(PKG_ROOT, produced_bad))

    def test_determinism(self):
        dims = [OptDim(f'p{i}', -2.0, 2.0, False) for i in range(3)]
        a = evolve_generation(dims, None, None, n=8, seed=99)
        b = evolve_generation(dims, None, None, n=8, seed=99)
        self.assertEqual(a, b)
        r1 = hybrid_minimize(sphere, [(-5.0, 5.0)] * 4, max_evals=400, seed=77).best_score
        r2 = hybrid_minimize(sphere, [(-5.0, 5.0)] * 4, max_evals=400, seed=77).best_score
        self.assertEqual(r1, r2)


if __name__ == '__main__':
    unittest.main()
