import unittest, sys, csv, json, math, tempfile
from pathlib import Path
from unittest import mock
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import egp_mho_generate as gen
from egp_bt_cache import (canonical_key, read_opt_key, BacktestCache, default_cache_path, plan)
from egp_mt5_report_parser import enrich_batch_results

MAP_HDR = "name,type,default,status,group,risk_level,min,max,step,active_if,source_file,source_line,notes\n"
SMALL = ["p_core_1,float,0,OPTIMIZE_CORE,G,LOW,-5,5,0.5,,,,\n",
         "p_core_2,int,10,OPTIMIZE_CORE,G,LOW,1,50,1,,,,\n",
         "p_later_1,float,0.5,OPTIMIZE_LATER,G,LOW,0,1,0.05,,,,\n",
         "p_prot_1,double,42,LOCK_SAFETY,G,LOW,,,,,,,\n"]
REPORT = "<tr><td>Profit Factor:</td><td>1.33</td></tr><tr><td>Total Trades:</td><td>245</td></tr><tr><td>Balance Drawdown Maximal:</td><td>800 (8.50%)</td></tr>"


def make_root(tmp):
    r = Path(tmp)
    (r / 'DOCS').mkdir(parents=True, exist_ok=True); (r / 'CONFIG').mkdir(parents=True, exist_ok=True)
    (r / 'DOCS' / 'PARAMETER_MAP.csv').write_text(MAP_HDR + ''.join(SMALL), encoding='utf-8')
    (r / 'CONFIG' / 'optimizer_config.json').write_text(json.dumps({"families": ["TPE_LSHADE"]}), encoding='utf-8')
    return r


class TestCanonicalKey(unittest.TestCase):
    def test_stable_and_order_independent(self):
        self.assertEqual(canonical_key({'a': '1', 'b': '2'}), canonical_key({'b': '2', 'a': '1'}))

    def test_sensitive_to_values(self):
        self.assertNotEqual(canonical_key({'a': '1'}), canonical_key({'a': '2'}))


class TestCacheRoundTrip(unittest.TestCase):
    def test_put_save_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'cache.json'
            c = BacktestCache(p)
            self.assertNotIn('K', c)
            c.put('K', {'profit_factor': 1.5, 'trades': 200}, {'cid': 'x'})
            c.save()
            c2 = BacktestCache(p)                         # rechargement persistant
            self.assertIn('K', c2)
            self.assertEqual(c2.metrics('K')['profit_factor'], 1.5)

    def test_read_opt_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            sp = Path(tmp) / 'c.set'
            sp.write_text('; candidate_id=x\n; opt_key=ABC123\np_core_1=1\n', encoding='ascii')
            self.assertEqual(read_opt_key(sp), 'ABC123')
            self.assertIsNone(read_opt_key(Path(tmp) / 'nope.set'))


class TestDedup(unittest.TestCase):
    def test_duplicates_collapsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = make_root(tmp)
            # evolve returns the SAME valid vector 6 times -> same OPT key -> collapse
            with mock.patch.object(gen, 'evolve_generation', lambda *a, **k: [[0.0, 10, 0.5]] * 6):
                run = gen.generate_generation(str(r), 0, family='TPE_LSHADE', count=6, seed=1)
            mani = json.loads((run / 'generation_manifest.json').read_text())
            self.assertEqual(mani['unique_candidates'], 1)
            self.assertGreater(mani['duplicates_collapsed'], 0)
            self.assertEqual(mani['status'], 'PASS')       # candidat valide
            self.assertEqual(len(list((run / 'sets').glob('*.set'))), 1)

    def test_opt_key_in_set_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = make_root(tmp)
            run = gen.generate_generation(str(r), 0, family='TPE_LSHADE', count=4, seed=1)
            sset = sorted((run / 'sets').glob('*.set'))[0]
            self.assertIsNotNone(read_opt_key(sset))  # key written into the .set
            srow = next(csv.DictReader((run / 'candidate_summary.csv').open(encoding='utf-8')))
            self.assertIn('opt_key', srow); self.assertIn('cached', srow)


class TestCacheHitOnRegen(unittest.TestCase):
    def test_cache_hit_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = make_root(tmp)
            run = gen.generate_generation(str(r), 0, family='TPE_LSHADE', count=6, seed=9)
            keys = [row['opt_key'] for row in csv.DictReader((run / 'candidate_summary.csv').open(encoding='utf-8'))]
            # we inject a known key into the cache
            c = BacktestCache(default_cache_path(r))
            c.put(keys[0], {'profit_factor': 1.4, 'trades': 100, 'balance_dd_pct': 7.0}, {})
            c.save()
            run2 = gen.generate_generation(str(r), 0, family='TPE_LSHADE', count=6, seed=9)  # same vectors
            mani = json.loads((run2 / 'generation_manifest.json').read_text())
            self.assertGreaterEqual(mani['cache_hits'], 1)
            cached_flags = {row['opt_key']: row['cached'] for row in csv.DictReader((run2 / 'candidate_summary.csv').open(encoding='utf-8'))}
            self.assertEqual(cached_flags[keys[0]], '1')


class TestEnrichFeedsCacheAndPlan(unittest.TestCase):
    def test_enrich_puts_then_plan_separates(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = make_root(tmp)
            run = r / 'MHO' / 'RUN_000'; (run / 'sets').mkdir(parents=True); (run / 'results').mkdir(parents=True)
            # two candidates: c0 (will be backtested+cached) and c1 (never tested)
            (run / 'sets' / 'c0.set').write_text('; opt_key=KEY0\np_core_1=1\n', encoding='ascii')
            (run / 'sets' / 'c1.set').write_text('; opt_key=KEY1\np_core_1=2\n', encoding='ascii')
            rep = run / 'results' / 'c0.htm'; rep.write_text(REPORT, encoding='utf-8')
            with (run / 'results' / 'batch_results.csv').open('w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=['candidate_id', 'set_file', 'report'])
                w.writeheader()
                w.writerow({'candidate_id': 'c0', 'set_file': str(run / 'sets' / 'c0.set'), 'report': str(rep)})
            # enrich -> must feed the cache with KEY0
            enrich_batch_results(str(run))
            c = BacktestCache(default_cache_path(r))
            self.assertIn('KEY0', c)
            self.assertAlmostEqual(c.metrics('KEY0')['profit_factor'], 1.33, places=2)
            # candidate_summary for plan : c0 (cached) + c1 (to test)
            with (run / 'candidate_summary.csv').open('w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=['candidate_id', 'set_file', 'opt_key'])
                w.writeheader()
                w.writerow({'candidate_id': 'c0', 'set_file': 'MHO/RUN_000/sets/c0.set', 'opt_key': 'KEY0'})
                w.writerow({'candidate_id': 'c1', 'set_file': 'MHO/RUN_000/sets/c1.set', 'opt_key': 'KEY1'})
            res = plan(str(run), str(r))
            self.assertEqual(res['cached'], 1)
            self.assertEqual(res['to_backtest'], 1)
            self.assertIn('c1.set', (run / 'results' / 'to_backtest.txt').read_text())
            self.assertTrue((run / 'results' / 'cached_results.csv').exists())


class TestMergeCached(unittest.TestCase):
    def test_merge_unions_rows_and_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            from egp_bt_cache import merge_cached
            res = Path(tmp) / 'results'; res.mkdir()
            with (res / 'batch_results.csv').open('w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=['candidate_id', 'trades', 'profit_factor']); w.writeheader()
                w.writerow({'candidate_id': 'c1', 'trades': 200, 'profit_factor': 1.5})
            with (res / 'cached_results.csv').open('w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=['candidate_id', 'trades', 'profit_factor', 'cached']); w.writeheader()
                w.writerow({'candidate_id': 'c0', 'trades': 100, 'profit_factor': 1.2, 'cached': '1'})
            n = merge_cached(str(Path(tmp)))
            rows = list(csv.DictReader((res / 'batch_results.csv').open(encoding='utf-8')))
            self.assertEqual(n, 1)
            self.assertEqual(sorted(r['candidate_id'] for r in rows), ['c0', 'c1'])
            self.assertIn('cached', rows[0].keys())          # colonne fusionnee


if __name__ == '__main__':
    unittest.main()
