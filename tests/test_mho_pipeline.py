import unittest, sys, csv, json, math, random, shutil, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from egp_mt5_report_parser import (parse_report_text, robust_score, robust_score_oos,
                                   enrich_batch_results, PENALTY)
import egp_mho_generate as gen


SYNTH_REPORT = """<html><body><table>
<tr><td>Total Net Profit:</td><td><b>1 234.56</b></td><td>Gross Profit:</td><td>5 000.00</td></tr>
<tr><td>Gross Loss:</td><td>-3 765.44</td><td>Profit Factor:</td><td><b>1.33</b></td></tr>
<tr><td>Expected Payoff:</td><td>2.47</td><td>Recovery Factor:</td><td>0.85</td></tr>
<tr><td>Sharpe Ratio:</td><td>0.91</td></tr>
<tr><td>Balance Drawdown Maximal:</td><td>800.00 (8.50%)</td></tr>
<tr><td>Equity Drawdown Maximal:</td><td>950.00 (9.80%)</td></tr>
<tr><td>Total Trades:</td><td>245</td></tr>
</table></body></html>"""

SMALL_MAP = (
    "name,status,type,min,max,default\n"
    "p_core_1,OPTIMIZE_CORE,float,-5,5,0\n"
    "p_core_2,OPTIMIZE_CORE,int,1,50,10\n"
    "p_later_1,OPTIMIZE_LATER,float,0,1,0.5\n"
    "p_prot_1,LOCK_SAFETY,float,0,100,42\n"
    "LOT,LOCK_DEFAULT,double,0.01,1,0.10\n"
    "In_META_Enable,MHO_ONLY,bool,,,true\n"
)
CFG = json.dumps({"families": ["TPE_LSHADE", "NSGA2_CMAES"], "initial_count": 6})


def make_root(tmp):
    r = Path(tmp)
    (r / 'DOCS').mkdir(parents=True, exist_ok=True)
    (r / 'CONFIG').mkdir(parents=True, exist_ok=True)
    (r / 'DOCS' / 'PARAMETER_MAP.csv').write_text(SMALL_MAP, encoding='utf-8')
    (r / 'CONFIG' / 'optimizer_config.json').write_text(CFG, encoding='utf-8')
    return r


class TestParser(unittest.TestCase):
    def test_parse_metrics(self):
        m = parse_report_text(SYNTH_REPORT)
        self.assertAlmostEqual(m['profit'], 1234.56, places=2)
        self.assertAlmostEqual(m['profit_factor'], 1.33, places=2)
        self.assertAlmostEqual(m['expected_payoff'], 2.47, places=2)
        self.assertAlmostEqual(m['sharpe'], 0.91, places=2)
        self.assertAlmostEqual(m['balance_dd_pct'], 8.50, places=2)
        self.assertAlmostEqual(m['equity_dd_pct'], 9.80, places=2)
        self.assertEqual(m['trades'], 245.0)
        self.assertAlmostEqual(m['gross_loss'], -3765.44, places=2)

    def test_parse_malformed_no_crash(self):
        for bad in ('', '<html><b>garbage', '<<>>', 'Total Net Profit: not_a_number'):
            m = parse_report_text(bad)
            self.assertIsInstance(m, dict)
            self.assertIn('profit_factor', m)

    def test_robust_score_finite_and_ordered(self):
        good = {'trades': 245, 'profit_factor': 1.33, 'balance_dd_pct': 8.5}
        few = {'trades': 5, 'profit_factor': 2.0, 'balance_dd_pct': 3.0}
        losing = {'trades': 100, 'profit_factor': 0.8, 'balance_dd_pct': 20.0}
        self.assertLess(robust_score(good), 0.0)            # bon candidat : cout negatif
        self.assertGreater(robust_score(few), 1e5)          # trop peu de trades : penalise
        self.assertGreater(robust_score(losing), 1e5)       # PF<=1 : penalise
        self.assertLess(robust_score(good), robust_score(losing))

    def test_robust_score_fuzz_finite(self):
        rng = random.Random(0)
        vals = [None, float('nan'), float('inf'), -float('inf'), 0.0, 1e9, -1e9]
        for _ in range(3000):
            m = {k: rng.choice(vals + [rng.uniform(-5, 5)]) for k in
                 ('trades', 'profit_factor', 'balance_dd_pct', 'profit')}
            s = robust_score(m)
            self.assertTrue(math.isfinite(s))

    def test_robust_oos_penalizes_gap(self):
        is_m = {'trades': 200, 'profit_factor': 2.0, 'balance_dd_pct': 5.0}
        oos_good = {'trades': 200, 'profit_factor': 1.9, 'balance_dd_pct': 6.0}
        oos_bad = {'trades': 200, 'profit_factor': 1.1, 'balance_dd_pct': 6.0}
        self.assertLess(robust_score_oos(is_m, oos_good), robust_score_oos(is_m, oos_bad))

    def test_enrich_batch_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'MHO' / 'RUN_000'
            (run / 'results').mkdir(parents=True)
            rep = run / 'results' / 'G000_C0000.htm'
            rep.write_text(SYNTH_REPORT, encoding='utf-8')
            with (run / 'results' / 'batch_results.csv').open('w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=['candidate_id', 'report', 'trades', 'profit_factor'])
                w.writeheader(); w.writerow({'candidate_id': 'G000_C0000', 'report': str(rep), 'trades': '', 'profit_factor': ''})
            rows = enrich_batch_results(str(run))
            self.assertEqual(int(float(rows[0]['trades'])), 245)
            self.assertAlmostEqual(float(rows[0]['profit_factor']), 1.33, places=2)
            self.assertLess(float(rows[0]['robust_score']), 0.0)


class TestPipelineIntegration(unittest.TestCase):
    def test_gen0_lhs_respects_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = make_root(tmp)
            run = gen.generate_generation(str(r), 0, count=6, seed=123)
            mani = json.loads((run / 'generation_manifest.json').read_text())
            self.assertEqual(mani['status'], 'PASS')
            self.assertEqual(mani['mode'], 'LHS_GEN0')
            self.assertEqual(mani['reverse_test_violations'], 0)
            sets = sorted((run / 'sets').glob('*.set'))
            self.assertEqual(len(sets), 6)
            txt = sets[0].read_text()
            self.assertIn('p_prot_1=42', txt)          # protege inchange
            self.assertIn('LOT=0.01', txt)             # overlay securite
            self.assertIn('In_META_Enable=false', txt)  # overlay securite

    def test_gen0_opt_params_actually_vary(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = make_root(tmp)
            run = gen.generate_generation(str(r), 0, count=8, seed=1)
            vals = {}
            with (run / 'candidate_values.csv').open(encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    vals.setdefault(row['name'], set()).add(row['value'])
            self.assertGreater(len(vals['p_core_1']), 1)  # one OPT varies across candidates
            self.assertEqual(vals['p_prot_1'], {'42'})     # a protected one does not vary

    def test_gen1_evolve_from_real_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = make_root(tmp)
            # generation 0
            gen.generate_generation(str(r), 0, family='TPE_LSHADE', count=6, seed=5)
            # simulate backtests: write reports + batch_results for RUN_000
            run0 = r / 'MHO' / 'RUN_000'
            (run0 / 'results').mkdir(parents=True, exist_ok=True)
            cids = sorted({row['candidate_id'] for row in csv.DictReader((run0 / 'candidate_values.csv').open(encoding='utf-8'))})
            br = []
            for k, cid in enumerate(cids):
                pf = 1.0 + 0.1 * k  # PF increasing -> differentiated scores
                rep = run0 / 'results' / f'{cid}.htm'
                rep.write_text(SYNTH_REPORT.replace('1.33', f'{pf:.2f}'), encoding='utf-8')
                br.append({'candidate_id': cid, 'report': str(rep)})
            with (run0 / 'results' / 'batch_results.csv').open('w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=['candidate_id', 'report']); w.writeheader(); w.writerows(br)
            # generation 1: must be driven (EVOLVE) and compliant
            run1 = gen.generate_generation(str(r), 1, family='TPE_LSHADE', count=6, seed=5)
            mani = json.loads((run1 / 'generation_manifest.json').read_text())
            self.assertEqual(mani['mode'], 'EVOLVE')
            self.assertEqual(mani['status'], 'PASS')
            self.assertEqual(mani['reverse_test_violations'], 0)
            self.assertEqual(len(list((run1 / 'sets').glob('*.set'))), 6)

    def test_determinism(self):
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            r1 = make_root(t1); r2 = make_root(t2)
            run1 = gen.generate_generation(str(r1), 0, count=6, seed=42)
            run2 = gen.generate_generation(str(r2), 0, count=6, seed=42)
            def shas(run):
                return [row['sha256'] for row in csv.DictReader((run / 'candidate_summary.csv').open(encoding='utf-8'))]
            self.assertEqual(shas(run1), shas(run2))


if __name__ == '__main__':
    unittest.main()
