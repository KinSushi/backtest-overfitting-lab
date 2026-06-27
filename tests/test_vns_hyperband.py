"""P3 - documentary honesty: VNS and Hyperband genuinely carry the semantics
of their name (several neighborhoods / several brackets)."""
import unittest, sys, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from egp_meta_algos import (variable_neighborhood_search, _local_search_1flip,
                            hyperband, tabu_vns_optimize, successive_halving)


# 2-flip trap on {0,1}^4: global optimum (1,1,1,1)=0; (1,1,0,0)=1 is an optimum
# local 1-flip (all its 1-flip neighbors equal 2), only a 2-flip leads to the optimum.
def trap(x):
    t = tuple(int(v) for v in x)
    if t == (1, 1, 1, 1):
        return 0.0
    if t == (1, 1, 0, 0):
        return 1.0
    if t in {(0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 1, 0), (1, 1, 0, 1)}:
        return 2.0
    return 3.0


class TestVNS(unittest.TestCase):
    def test_one_flip_is_trapped(self):
        # a descent with a SINGLE neighborhood (1-flip) stays stuck at 1.0
        _, s, _ = _local_search_1flip((1, 1, 0, 0), trap, set(), 1000)
        self.assertEqual(s, 1.0)

    def test_vns_escapes_with_second_neighborhood(self):
        # VNS (neighborhoods k=1 AND k=2) reaches the global optimum 0.0
        best, bs = variable_neighborhood_search((1, 1, 0, 0), trap, k_max=2,
                                                max_evals=300, seed=0)
        self.assertEqual(bs, 0.0)
        self.assertEqual(tuple(best), (1, 1, 1, 1))

    def test_vns_onemax_still_works(self):
        # sanity: minimize the number of 0s -> all to 1
        f = lambda x: sum(1 - v for v in x)
        best, bs = variable_neighborhood_search([0] * 8, f, k_max=3, max_evals=400, seed=1)
        self.assertEqual(bs, 0.0)


class TestHyperband(unittest.TestCase):
    @staticmethod
    def _f_true(c):
        return (c - 3.0) ** 2

    def _score(self, c, budget):
        noise = (abs(hash((round(c, 4), int(budget)))) % 1000) / 1000.0
        return self._f_true(c) + (5.0 / budget) * noise

    def test_multiple_brackets_and_schedule(self):
        get_cfg = lambda rng: rng.uniform(0, 10)
        best_c, best_s, brackets = hyperband(get_cfg, self._score,
                                             max_resource=27, eta=3, seed=1)
        # s_max = floor(log_3 27) = 3  -> 4 brackets
        self.assertEqual(len(brackets), 1 + int(math.floor(math.log(27, 3))))
        self.assertEqual(len(brackets), 4)
        b3 = [b for b in brackets if b['s'] == 3][0]
        b0 = [b for b in brackets if b['s'] == 0][0]
        self.assertEqual(len(b3['rounds']), 4)            # s+1 rounds de halving
        self.assertEqual(len(b0['rounds']), 1)
        # bracket exploratoire (s=3) : beaucoup de configs a petit budget initial
        self.assertGreater(b3['rounds'][0][0], b0['rounds'][0][0])
        self.assertLess(b3['rounds'][0][1], b0['rounds'][0][1])

    def test_finds_good_region(self):
        get_cfg = lambda rng: rng.uniform(0, 10)
        best_c, best_s, _ = hyperband(get_cfg, self._score, max_resource=27, eta=3, seed=1)
        self.assertLess(self._f_true(best_c), 1.5)        # proche de l'optimum c*=3


if __name__ == '__main__':
    unittest.main()
