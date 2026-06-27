import unittest, sys, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from egp_meta_algos import cmaes_minimize
from egp_mho_hybrid import inv_sqrt_C


class TestCmaesCSACorrection(unittest.TestCase):
    def test_inv_sqrt_identity(self):
        C = [[2.0, 0.3, 0.0], [0.3, 1.5, 0.1], [0.0, 0.1, 1.0]]
        Ci = inv_sqrt_C(C); n = 3
        CiC = [[sum(Ci[i][k] * C[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
        prod = [[sum(CiC[i][k] * Ci[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(n):
                self.assertAlmostEqual(prod[i][j], 1.0 if i == j else 0.0, places=4)

    def test_cmaes_converges_ill_conditioned(self):
        n, shift = 4, 1.5
        coef = [1000.0 ** (i / (n - 1)) for i in range(n)]    # conditionnement ~1000
        f = lambda x: sum(c * (xi - shift) ** 2 for c, xi in zip(coef, x))
        b = [(-5, 5)] * n
        f_start = f([0.0] * n)                                 # start at the center of the bounds
        st = cmaes_minimize(f, b, max_evals=2000, seed=3)
        self.assertTrue(math.isfinite(st.best_score))
        self.assertLess(st.best_score, 0.1 * f_start)  # net progress (corrected CSA path)


if __name__ == '__main__':
    unittest.main()
