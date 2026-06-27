import unittest, sys, re, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import egp_ontester_codegen as cg
import pytest

SRC = """
input int Period = 14;
sinput double Risk = 1.0;
input group "Section";
input ENUM_TIMEFRAMES TF = PERIOD_M1;
// input bool Commented = true;
/* input bool Blocked = false; */
input bool Live=false;
"""

ROWS = [
    {'name': 'p_in', 'type': 'int', 'default': '', 'status': 'OPTIMIZE_CORE', 'min': '5', 'max': '60', 'step': '1'},
    {'name': 'p_out', 'type': 'int', 'default': '', 'status': 'OPTIMIZE_CORE', 'min': '5', 'max': '60', 'step': '1'},
    {'name': 'prot_in', 'type': 'int', 'default': '7', 'status': 'LOCK_DEFAULT', 'min': '', 'max': '', 'step': ''},
    {'name': 'prot_out', 'type': 'int', 'default': '7', 'status': 'LOCK_DEFAULT', 'min': '', 'max': '', 'step': ''},
]


class TestExtractInputs(unittest.TestCase):
    def test_parses_and_excludes_group_and_comments(self):
        self.assertEqual(cg.extract_inputs(SRC), {'Period', 'Risk', 'TF', 'Live'})


class TestBuildFiltersNotAnInput(unittest.TestCase):
    def test_not_an_input_is_commented_not_emitted(self):
        code, info = cg.build(ROWS, declared_inputs={'p_in', 'prot_in'})
        self.assertEqual(set(info['range']), {'p_in'})
        self.assertEqual(set(info['lock']), {'prot_in'})
        self.assertEqual(set(info['notinput']), {'p_out', 'prot_out'})
        self.assertIn('ParameterSetRange("p_in"', code)
        self.assertIn('ParameterSetRange("prot_in"', code)
        self.assertNotIn('ParameterSetRange("p_out"', code)     # no-op removed
        self.assertNotIn('ParameterSetRange("prot_out"', code)
        self.assertIn('// SKIP(not-an-input) p_out', code)

    def test_without_declared_no_filtering(self):
        code, info = cg.build(ROWS, declared_inputs=None)
        self.assertEqual(len(info['range']), 2)                 # p_in + p_out
        self.assertEqual(len(info['lock']), 2)
        self.assertEqual(info['notinput'], [])


class TestRealMapInvariant(unittest.TestCase):
    def test_every_emitted_name_is_a_real_input(self):
        pytest.skip("Requires private EA artifacts (parameter map / real EA), excluded from the public demo repo.")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'EGP_MHO_OnTester.mqh'
            p = cg.generate(str(ROOT), out=str(out), verify_inputs=True)
            text = Path(p).read_text()
            declared = cg.scan_declared_inputs(ROOT)
            self.assertTrue(declared)                            # real inputs detected
            names = re.findall(r'ParameterSetRange\("([^"]+)"', text)
            self.assertTrue(names)
            for nm in names:                                     # anti no-op INVARIANT
                self.assertIn(nm, declared, f'{nm} emitted but absent from the real inputs')
            # verification report written
            self.assertTrue((Path(tmp) / 'EGP_MHO_OnTester.inputs_report.txt').exists())


if __name__ == '__main__':
    unittest.main()
