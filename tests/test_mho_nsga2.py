import unittest, sys, csv, json, math
from pathlib import Path
from unittest import mock
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import egp_mho_generate as gen
from egp_mho_hybrid import evolve_generation_mo, pareto_front_size, OptDim
from egp_mt5_report_parser import metrics_to_objectives, PENALTY

MAP_HDR = "name,type,default,status,group,risk_level,min,max,step,active_if,source_file,source_line,notes\n"
SMALL = ["p_core_1,float,0,OPTIMIZE_CORE,G,LOW,-5,5,0.5,,,,\n",
         "p_core_2,int,10,OPTIMIZE_CORE,G,LOW,1,50,1,,,,\n",
         "p_later_1,float,0.5,OPTIMIZE_LATER,G,LOW,0,1,0.05,,,,\n",
         "p_prot_1,double,42,LOCK_SAFETY,G,LOW,,,,,,,\n",
         "LOT,double,0.10,LOCK_DEFAULT,G,LOW,,,,,,,\n",
         "In_META_Enable,bool,true,MHO_ONLY,G,LOW,,,,,,,\n"]


def make_root(tmp):
    r = Path(tmp)
    (r / 'DOCS').mkdir(parents=True, exist_ok=True); (r / 'CONFIG').mkdir(parents=True, exist_ok=True)
    (r / 'DOCS' / 'PARAMETER_MAP.csv').write_text(MAP_HDR + ''.join(SMALL), encoding='utf-8')
    (r / 'CONFIG' / 'optimizer_config.json').write_text(json.dumps({"families": ["NSGA2_CMAES"]}), encoding='utf-8')
    return r


def seed_prev_metrics(root):
    """Writes RUN_000/results/batch_results.csv with differentiated metrics (no report)."""
    run0 = root / 'MHO' / 'RUN_000'; (run0 / 'results').mkdir(parents=True, exist_ok=True)
    cids = sorted({row['candidate_id'] for row in csv.DictReader((run0 / 'candidate_values.csv').open(encoding='utf-8'))})
    rows = []
    for k, cid in enumerate(cids):
        rows.append({'candidate_id': cid, 'trades': 50 + 20 * k,
                     'profit_factor': round(1.05 + 0.1 * k, 3), 'balance_dd_pct': round(15 - k, 3)})
    with (run0 / 'results' / 'batch_results.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['candidate_id', 'trades', 'profit_factor', 'balance_dd_pct'])
        w.writeheader(); w.writerows(rows)
    return cids


class TestObjectives(unittest.TestCase):
    def test_vector_and_guard(self):
        self.assertEqual(metrics_to_objectives({'trades': 245, 'profit_factor': 1.33, 'balance_dd_pct': 8.5}),
                         (-1.33, 8.5, -245))
        self.assertEqual(metrics_to_objectives({'trades': 5, 'profit_factor': 9.0, 'balance_dd_pct': 1.0}),
                         (PENALTY, PENALTY, PENALTY))                # trades<min -> domine

    def test_pareto_front(self):
        objs = [(-1.5, 8, -200), (-1.2, 5, -150), (-1.3, 9, -100), (-2.0, 12, -50)]
        self.assertEqual(pareto_front_size(objs), 3)                # 1 seul domine


class TestEngineMO(unittest.TestCase):
    def test_bounds_and_determinism(self):
        dims = [OptDim('a', -5, 5, False), OptDim('b', 1, 50, True), OptDim('c', 0, 1, False)]
        pv = [[0.0, 10, 0.5], [1.0, 20, 0.3], [-2.0, 30, 0.7], [2.0, 5, 0.1]]
        po = [metrics_to_objectives({'trades': 100 + 10 * i, 'profit_factor': 1.1 + 0.1 * i, 'balance_dd_pct': 5 + i}) for i in range(4)]
        g1 = evolve_generation_mo(dims, pv, po, n=6, seed=1)
        g2 = evolve_generation_mo(dims, pv, po, n=6, seed=1)
        self.assertEqual(g1, g2)
        for v in g1:
            self.assertTrue(-5 <= v[0] <= 5 and 1 <= v[1] <= 50 and 0 <= v[2] <= 1)


class TestPipelineNSGA2(unittest.TestCase):
    def test_gen0_lhs_policy(self):
        with __import__('tempfile').TemporaryDirectory() as tmp:
            r = make_root(tmp)
            run = gen.generate_generation(str(r), 0, family='NSGA2_CMAES', count=6, seed=2)
            mani = json.loads((run / 'generation_manifest.json').read_text())
            self.assertEqual(mani['mode'], 'LHS_GEN0')
            self.assertEqual(mani['reverse_test_violations'], 0)
            self.assertEqual(len(list((run / 'sets').glob('*.set'))), 6)
            txt = sorted((run / 'sets').glob('*.set'))[0].read_text()
            self.assertIn('p_prot_1=42', txt); self.assertIn('LOT=0.01', txt); self.assertIn('In_META_Enable=false', txt)

    def test_gen1_nsga2_evolve(self):
        with __import__('tempfile').TemporaryDirectory() as tmp:
            r = make_root(tmp)
            gen.generate_generation(str(r), 0, family='NSGA2_CMAES', count=6, seed=2)
            seed_prev_metrics(r)
            run1 = gen.generate_generation(str(r), 1, family='NSGA2_CMAES', count=6, seed=2)
            mani = json.loads((run1 / 'generation_manifest.json').read_text())
            self.assertEqual(mani['mode'], 'NSGA2_EVOLVE')
            self.assertEqual(mani['status'], 'PASS')
            self.assertEqual(mani['reverse_test_violations'], 0)
            self.assertGreaterEqual(mani['pareto_front0'], 1)
            self.assertEqual(len(list((run1 / 'sets').glob('*.set'))), 6)

    def test_gen1_determinism(self):
        with __import__('tempfile').TemporaryDirectory() as tmp:
            r = make_root(tmp)
            gen.generate_generation(str(r), 0, family='NSGA2_CMAES', count=6, seed=2)
            seed_prev_metrics(r)
            run1 = gen.generate_generation(str(r), 1, family='NSGA2_CMAES', count=6, seed=2)
            shas1 = [row['sha256'] for row in csv.DictReader((run1 / 'candidate_summary.csv').open(encoding='utf-8'))]
            run1b = gen.generate_generation(str(r), 1, family='NSGA2_CMAES', count=6, seed=2)
            shas2 = [row['sha256'] for row in csv.DictReader((run1b / 'candidate_summary.csv').open(encoding='utf-8'))]
            self.assertEqual(shas1, shas2)

    def test_nsga2_forbidden_fails_cleanly(self):
        with __import__('tempfile').TemporaryDirectory() as tmp:
            r = make_root(tmp)
            with mock.patch.object(gen, 'decode', lambda vec, dims: {'p_prot_1': '999'}):
                with self.assertRaises(RuntimeError):
                    gen.generate_generation(str(r), 0, family='NSGA2_CMAES', count=4, seed=1)


if __name__ == '__main__':
    unittest.main()
