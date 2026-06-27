import unittest, sys, csv, math, re, json, subprocess, tempfile
from pathlib import Path
from unittest import mock
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import egp_mho_generate as gen
import egp_ontester_codegen as cg
from egp_mho_hybrid import reverse_test_protected, load_opt_dims
from egp_mt5_report_parser import enrich_batch_results, parse_report_text
import pytest

MAP_HDR = "name,type,default,status,group,risk_level,min,max,step,active_if,source_file,source_line,notes\n"
SYNTH_REPORT = "<tr><td>Profit Factor:</td><td>1.33</td></tr><tr><td>Total Trades:</td><td>245</td></tr><tr><td>Balance Drawdown Maximal:</td><td>800 (8.50%)</td></tr>"


def write_map(root, lines):
    (root / 'DOCS').mkdir(parents=True, exist_ok=True)
    (root / 'CONFIG').mkdir(parents=True, exist_ok=True)
    (root / 'DOCS' / 'PARAMETER_MAP.csv').write_text(MAP_HDR + ''.join(lines), encoding='utf-8')
    (root / 'CONFIG' / 'optimizer_config.json').write_text(json.dumps({"families": ["TPE_LSHADE"]}), encoding='utf-8')


class TestReverseProtected(unittest.TestCase):
    def test_flags_non_opt_key(self):
        pytest.skip("Requires private EA artifacts (parameter map / real EA), excluded from the public demo repo.")
        v = reverse_test_protected(str(ROOT), {'__definitely_not_an_opt_param__': '1'})
        self.assertTrue(v)                                  # forbidden detected

    def test_passes_real_opt_key(self):
        pytest.skip("Requires private EA artifacts (parameter map / real EA), excluded from the public demo repo.")
        dims = load_opt_dims(str(ROOT))
        self.assertTrue(dims)
        v = reverse_test_protected(str(ROOT), {dims[0].name: '1'})
        self.assertEqual(v, [])  # allowed -> no violation


class TestForbiddenFailsCleanly(unittest.TestCase):
    def test_generate_raises_on_protected_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = Path(tmp)
            write_map(r, ["p_core_1,int,10,OPTIMIZE_CORE,G,LOW,5,60,1,,,,\n",
                          "p_prot_1,double,42,LOCK_SAFETY,G,LOW,,,,,,,\n"])
            # inject a forbidden mutation: decode returns a PROTECTED key
            with mock.patch.object(gen, 'decode', lambda vec, dims: {'p_prot_1': '999'}):
                with self.assertRaises(RuntimeError):       # clean failure
                    gen.generate_generation(str(r), 0, count=3, seed=1)
            # the diagnostic must have been written despite the failure
            viol = r / 'MHO' / 'RUN_000' / 'reverse_test_violations.csv'
            self.assertTrue(viol.exists() and viol.read_text().strip() != '')

    def test_generate_raises_on_zero_opt_dims(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = Path(tmp)
            write_map(r, ["p_core_1,int,10,OPTIMIZE_CORE,G,LOW,,,,,,,\n",   # no min/max -> 0 dim
                          "p_prot_1,double,42,LOCK_SAFETY,G,LOW,,,,,,,\n"])
            with self.assertRaises(SystemExit):
                gen.generate_generation(str(r), 0, count=3, seed=1)

    def test_audit_cli_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = Path(tmp)
            write_map(r, ["p_core_1,int,10,OPTIMIZE_CORE,G,LOW,5,60,1,,,,\n",
                          "p_prot_1,double,42,LOCK_SAFETY,G,LOW,,,,,,,\n"])
            run = r / 'MHO' / 'RUN_000'; run.mkdir(parents=True)
            hdr = ['candidate_id', 'generation', 'family', 'name', 'value', 'status']

            def write_cv(prot_value):
                with (run / 'candidate_values.csv').open('w', newline='', encoding='utf-8') as f:
                    w = csv.DictWriter(f, fieldnames=hdr); w.writeheader()
                    w.writerow({'candidate_id': 'G000_C0000', 'generation': 0, 'family': 'X', 'name': 'p_core_1', 'value': '30', 'status': 'OPTIMIZE_CORE'})
                    w.writerow({'candidate_id': 'G000_C0000', 'generation': 0, 'family': 'X', 'name': 'p_prot_1', 'value': prot_value, 'status': 'LOCK_SAFETY'})

            def audit():
                return subprocess.run([sys.executable, str(ROOT / 'tools' / 'egp_set_tools.py'),
                                       'audit', '--root', str(r), '--generation', '0'],
                                      capture_output=True, text=True)
            write_cv('999')                                  # forbidden mutation
            bad = audit()
            self.assertEqual(bad.returncode, 1)              # fails cleanly (exit!=0)
            self.assertIn('FAIL', bad.stdout)
            write_cv('42')                                   # compliant (= default)
            ok = audit()
            self.assertEqual(ok.returncode, 0)
            self.assertIn('PASS', ok.stdout)


class TestRobustnessToGarbage(unittest.TestCase):
    def test_enrich_does_not_crash_on_bad_cell(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / 'MHO' / 'RUN_000'
            (run / 'results').mkdir(parents=True)
            rep = run / 'results' / 'c.htm'; rep.write_text(SYNTH_REPORT, encoding='utf-8')
            with (run / 'results' / 'batch_results.csv').open('w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=['candidate_id', 'report', 'profit_factor', 'sharpe'])
                w.writeheader()
                w.writerow({'candidate_id': 'c', 'report': str(rep), 'profit_factor': 'garbage', 'sharpe': ''})
            rows = enrich_batch_results(str(run))            # must not raise
            self.assertTrue(math.isfinite(float(rows[0]['robust_score'])))

    def test_dd_regex_does_not_cross(self):
        r = "Balance Drawdown Maximal: 800.00   Other: (8.50%)"  # Maximal without %, distant %
        self.assertIsNone(parse_report_text(r)['balance_dd_pct'])


class TestCodegenNoGarbageLiterals(unittest.TestCase):
    NUMLIT = re.compile(r'^[-+]?\d+(\.\d+)?([eE][-+]?\d+)?$')

    def test_every_emitted_line_has_valid_numeric_literals(self):
        pytest.skip("Requires private EA artifacts (parameter map / real EA), excluded from the public demo repo.")
        code = cg.generate(str(ROOT), out=str(Path(tempfile.gettempdir()) / '_egp_lit_test.mqh'))
        text = Path(code).read_text()
        n = 0
        for line in text.splitlines():
            if 'ParameterSetRange("' not in line:
                continue
            n += 1
            after = line[line.index(',') + 1:]              # enable, v, start, step, stop) ;
            toks = [t.strip().rstrip(';').rstrip(')').strip() for t in after.split(',')]
            self.assertIn(toks[0], ('true', 'false'))       # enable
            for tok in toks[1:5]:                            # 4 litteraux numeriques
                self.assertRegex(tok, self.NUMLIT, f'invalid literal {tok!r} in: {line}')
        self.assertGreater(n, 0)


if __name__ == '__main__':
    unittest.main()
