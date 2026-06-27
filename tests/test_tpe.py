import unittest, sys, math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from egp_mho_hybrid import tpe_sample, evolve_generation, OptDim


class TestTPE(unittest.TestCase):
    def _data(self):
        bounds = [(0, 10), (0, 10)]
        pv = [[1.1, 0.9], [0.8, 1.2], [1.0, 1.0], [7.9, 8.1], [8.2, 7.8], [8.0, 8.0]]
        ps = [0.1, 0.15, 0.05, 5.0, 5.2, 5.1]               # bas = bon (centre ~ (1,1))
        return bounds, pv, ps

    def test_concentrates_on_good(self):
        bounds, pv, ps = self._data()
        out = tpe_sample(bounds, pv, ps, n=12, seed=1)
        cx = [sum(v[0] for v in out) / len(out), sum(v[1] for v in out) / len(out)]
        self.assertLess(math.dist(cx, [1.0, 1.0]), math.dist(cx, [8.0, 8.0]))

    def test_bounds_and_count(self):
        bounds, pv, ps = self._data()
        out = tpe_sample(bounds, pv, ps, n=8, seed=3)
        self.assertEqual(len(out), 8)
        for v in out:
            self.assertTrue(0 <= v[0] <= 10 and 0 <= v[1] <= 10)

    def test_determinism(self):
        bounds, pv, ps = self._data()
        self.assertEqual(tpe_sample(bounds, pv, ps, n=8, seed=5),
                         tpe_sample(bounds, pv, ps, n=8, seed=5))

    def test_route_via_evolve_generation(self):
        bounds, pv, ps = self._data()
        dims = [OptDim('a', 0, 10, False), OptDim('b', 0, 10, False)]
        g = evolve_generation(dims, pv, ps, family='TPE_LSHADE', n=6, seed=2)
        self.assertEqual(len(g), 6)
        for v in g:
            self.assertTrue(0 <= v[0] <= 10 and 0 <= v[1] <= 10)


if __name__ == '__main__':
    unittest.main()
