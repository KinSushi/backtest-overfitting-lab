import unittest, sys, math, json, tempfile
from pathlib import Path
from datetime import datetime
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import egp_ontester_codegen as cg
import egp_wfo as wfo
from egp_mho_hybrid import local_search, memetic_minimize, load_opt_dims, decode
import pytest


SYNTH_ROWS = [
    {'name': 'p_int', 'type': 'int', 'default': '', 'status': 'OPTIMIZE_CORE', 'min': '5', 'max': '60', 'step': '1'},
    {'name': 'p_dbl', 'type': 'double', 'default': '', 'status': 'OPTIMIZE_LATER', 'min': '5', 'max': '60', 'step': '1'},
    {'name': 'p_str', 'type': 'string', 'default': '', 'status': 'OPTIMIZE_CORE', 'min': '', 'max': '', 'step': ''},
    {'name': 'p_unbounded', 'type': 'int', 'default': '', 'status': 'OPTIMIZE_CORE', 'min': '', 'max': '', 'step': ''},
    {'name': 'p_lock_int', 'type': 'int', 'default': '7', 'status': 'LOCK_DEFAULT', 'min': '', 'max': '', 'step': ''},
    {'name': 'p_lock_dbl', 'type': 'double', 'default': '1.00', 'status': 'LOCK_DEFAULT', 'min': '', 'max': '', 'step': ''},
    {'name': 'p_lock_bool', 'type': 'bool', 'default': 'false', 'status': 'MHO_ONLY', 'min': '', 'max': '', 'step': ''},
    {'name': 'p_lock_enum', 'type': 'enum_timeframes', 'default': 'PERIOD_M1', 'status': 'CUSTOM_INDICATOR_INPUT', 'min': '', 'max': '', 'step': ''},
]


class TestCodegen(unittest.TestCase):
    def setUp(self):
        self.code = cg.generate_code(SYNTH_ROWS)

    def test_structure(self):
        self.assertIn('double OnTester()', self.code)
        self.assertIn('int OnTesterInit()', self.code)
        self.assertIn('return(INIT_SUCCEEDED);', self.code)
        self.assertIn('#ifndef EGP_MHO_ONTESTER_MQH', self.code)
        self.assertEqual(self.code.count('{'), self.code.count('}'))

    def test_range_vs_lock_counts(self):
        self.assertEqual(self.code.count(', true,'), 2)    # p_int, p_dbl
        self.assertEqual(self.code.count(', false,'), 3)   # p_lock_int, p_lock_dbl, p_lock_bool

    def test_excludes_nonnumeric(self):
        self.assertNotIn('ParameterSetRange("p_str"', self.code)        # string
        self.assertNotIn('ParameterSetRange("p_unbounded"', self.code)  # OPT without bounds
        self.assertNotIn('ParameterSetRange("p_lock_enum"', self.code)  # non-numeric default

    def test_literal_types(self):
        lines = {ln.split('"')[1]: ln for ln in self.code.splitlines() if 'ParameterSetRange("' in ln}
        self.assertIn('true, 32, 5, 1, 60', lines['p_int'])              # int -> integers
        self.assertIn('true, 32.5, 5.0, 1.0, 60.0', lines['p_dbl'])      # double -> decimals
        self.assertIn('false, 0, 0, 1, 0', lines['p_lock_bool'])         # bool false -> 0

    def test_generate_on_real_map(self):
        pytest.skip("Requires private EA artifacts (parameter map / real EA), excluded from the public demo repo.")
        p = cg.generate(str(ROOT), out=str(Path(tempfile.gettempdir()) / '_egp_ot_test.mqh'))
        t = Path(p).read_text()
        self.assertEqual(t.count('{'), t.count('}'))
        self.assertEqual(t.count(', true,'), 24)          # 24 OPT bounds (verified)
        self.assertIn('STAT_PROFIT_FACTOR', t)


class TestWalkForward(unittest.TestCase):
    def test_make_windows(self):
        ws = wfo.make_windows('2024.01.01', '2024.02.20', 20, 10, 10)
        self.assertGreater(len(ws), 0)
        for w in ws:
            isf, ist = datetime.strptime(w['is_from'], '%Y.%m.%d'), datetime.strptime(w['is_to'], '%Y.%m.%d')
            of, ot = datetime.strptime(w['oos_from'], '%Y.%m.%d'), datetime.strptime(w['oos_to'], '%Y.%m.%d')
            self.assertEqual((of - ist).days, 1)           # OOS starts right after IS
            self.assertEqual((ot - of).days, 9)            # OOS = 10 days inclusive
            self.assertEqual((ist - isf).days, 19)         # IS = 20 days inclusive
            self.assertLessEqual(ot, datetime.strptime('2024.02.20', '%Y.%m.%d'))

    def _cost(self):
        dims = load_opt_dims(str(ROOT))

        def cost_fn(vec, seg, w):
            center = [d.lo + (d.hi - d.lo) * (0.4 + 0.05 * w) for d in dims]
            true = sum((v - c) ** 2 for v, c in zip(vec, center))
            noise = 0.1 * math.sin(sum(vec) * 7.0 + w) * (1 + true) if seg == 'IS' else 0.0
            return true + noise
        return dims, cost_fn

    def test_walk_forward_mechanics(self):
        pytest.skip("Requires private EA artifacts (parameter map / real EA), excluded from the public demo repo.")
        dims, cost_fn = self._cost()
        windows = wfo.make_windows('2024.01.01', '2024.03.01', 30, 10)
        rep = wfo.walk_forward(dims, windows, cost_fn, inner_evals=200, seed=3)
        self.assertEqual(rep['n_windows'], len(windows))
        self.assertTrue(math.isfinite(rep['mean_oos_cost']))
        for w in rep['windows']:
            self.assertTrue(math.isfinite(w['oos_cost']))

    def test_walk_forward_determinism(self):
        pytest.skip("Requires private EA artifacts (parameter map / real EA), excluded from the public demo repo.")
        dims, cost_fn = self._cost()
        windows = wfo.make_windows('2024.01.01', '2024.02.20', 25, 8)
        r1 = wfo.walk_forward(dims, windows, cost_fn, inner_evals=150, seed=11)
        r2 = wfo.walk_forward(dims, windows, cost_fn, inner_evals=150, seed=11)
        self.assertEqual([w['best'] for w in r1['windows']], [w['best'] for w in r2['windows']])


class TestMemetic(unittest.TestCase):
    def test_local_search_monotone_and_converges(self):
        c = [1.5, -2.0, 0.7, 3.1]
        f = lambda x: sum((xi - ci) ** 2 for xi, ci in zip(x, c))
        b = [(-5, 5)] * 4
        x0 = [-4.0, 4.0, -3.0, 0.0]
        r = local_search(f, x0, b, iters=80)
        self.assertLessEqual(r.best_score, f(x0))          # monotone
        self.assertLess(r.best_score, 1e-2)                # converges near the optimum

    def test_local_search_deterministic(self):
        f = lambda x: sum(v * v for v in x)
        b = [(-5, 5)] * 3
        r1 = local_search(f, [1.0, 2.0, 3.0], b, iters=40)
        r2 = local_search(f, [1.0, 2.0, 3.0], b, iters=40)
        self.assertEqual(r1.best, r2.best)

    def test_memetic_finite(self):
        f = lambda x: sum(v * v for v in x)
        res = memetic_minimize(f, [(-5, 5)] * 5, max_evals=400, seed=1)
        self.assertTrue(math.isfinite(res.best_score))
        self.assertTrue(hasattr(res, 'best'))


if __name__ == '__main__':
    unittest.main()
