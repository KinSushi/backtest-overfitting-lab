import unittest, sys, csv, json, tempfile
from pathlib import Path
from collections import Counter
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from egp_mho_hybrid import decode, encode, load_opt_dims, _parse_choices, OptDim
import egp_mho_generate as gen
import pytest

MAP_HDR = "name,type,default,status,group,risk_level,min,max,step,active_if,source_file,source_line,notes\n"


def make_root(tmp, rows):
    r = Path(tmp)
    (r / 'DOCS').mkdir(parents=True, exist_ok=True); (r / 'CONFIG').mkdir(parents=True, exist_ok=True)
    (r / 'DOCS' / 'PARAMETER_MAP.csv').write_text(MAP_HDR + ''.join(rows), encoding='utf-8')
    (r / 'CONFIG' / 'optimizer_config.json').write_text(json.dumps({"families": ["TPE_LSHADE"]}), encoding='utf-8')
    return r


class TestDecodeEncode(unittest.TestCase):
    def test_roundtrip_all_kinds(self):
        b = OptDim('f', 0, 1, False, kind='bool')
        e = OptDim('m', 0, 3, False, kind='enum', choices=('A', 'B', 'C'))
        i = OptDim('p', 5, 60, True, kind='int')
        c = OptDim('x', -5, 5, False, kind='cont')
        self.assertEqual(decode([0.8], [b]), {'f': 'true'})
        self.assertEqual(decode([0.2], [b]), {'f': 'false'})
        self.assertEqual(encode('true', b), 1.0)
        self.assertEqual(encode('false', b), 0.0)
        self.assertEqual(decode([2.0], [e]), {'m': 'C'})
        self.assertEqual(encode('C', e), 2.0)
        self.assertEqual(decode([10.4], [i]), {'p': '10'})
        self.assertEqual(decode([5.0], [e]), {'m': 'C'})  # bounded index
        # round-trip value -> coord -> value
        for d, val in [(b, 'true'), (b, 'false'), (e, 'B'), (i, '12')]:
            self.assertEqual(decode([encode(val, d)], [d])[d.name], val)


class TestParseChoices(unittest.TestCase):
    def test_from_notes_and_absent(self):
        self.assertEqual(_parse_choices({'notes': 'choices=SMA|EMA|LWMA'}), ['SMA', 'EMA', 'LWMA'])
        self.assertEqual(_parse_choices({'notes': ''}), [])
        self.assertEqual(_parse_choices({'choices': 'X|Y'}), ['X', 'Y'])


class TestLoadOptDims(unittest.TestCase):
    def test_real_map_includes_bools_and_enums(self):
        pytest.skip("Requires private EA artifacts (parameter map / real EA), excluded from the public demo repo.")
        kinds = Counter(d.kind for d in load_opt_dims(str(ROOT)))
        self.assertEqual(kinds['bool'], 21)                     # 21 bool OPT optimisables
        self.assertEqual(kinds.get('enum', 0), 56)              # 56 enum OPT (choix declares & verifies)

    def test_enum_value_format_and_gap(self):
        # NAME:VALUE format -> decode emits the integer VALUE (not the index); handles gaps
        d = OptDim('c', 0, 3, False, kind='enum',
                   choices=('0', '1', '3'), choice_names=('bullish', 'bearish', 'off_1'))
        self.assertEqual(decode([2.0], [d]), {'c': '3'})  # off_1 = 3, value 2 skipped
        self.assertEqual(encode('3', d), 2.0)  # round-trip value -> coord
        self.assertEqual(encode('off_1', d), 2.0)  # tolerates the member NAME
        for emitted in ('0', '1', '3'):
            self.assertEqual(decode([encode(emitted, d)], [d])['c'], emitted)

    def test_synthetic_bool_and_enum(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = make_root(tmp, [
                "p_num,int,10,OPTIMIZE_CORE,G,LOW,1,50,1,,,,\n",
                "p_flag,bool,,OPTIMIZE_CORE,G,LOW,,,,,,,\n",
                "p_enum,ENUM_MA_METHOD,,OPTIMIZE_CORE,G,LOW,,,,,,,choices=SMA|EMA|LWMA\n",
                "p_enum_noch,ENUM_APPLIED_PRICE,,OPTIMIZE_CORE,G,LOW,,,,,,,\n",
            ])
            dims = {d.name: d for d in load_opt_dims(str(r))}
            self.assertEqual(dims['p_flag'].kind, 'bool')
            self.assertEqual(dims['p_enum'].kind, 'enum')
            self.assertEqual(dims['p_enum'].choices, ('SMA', 'EMA', 'LWMA'))
            self.assertNotIn('p_enum_noch', dims)               # enum without choices -> ignored (no invention)


class TestGenerationWithDiscrete(unittest.TestCase):
    ROWS = [
        "p_num,float,0,OPTIMIZE_CORE,G,LOW,-5,5,0.5,,,,\n",
        "p_flag,bool,,OPTIMIZE_CORE,G,LOW,,,,,,,\n",
        "p_enum,ENUM_MA_METHOD,,OPTIMIZE_CORE,G,LOW,,,,,,,choices=SMA|EMA|LWMA\n",
        "p_prot,double,42,LOCK_SAFETY,G,LOW,,,,,,,\n",
    ]

    def test_bool_and_enum_vary_and_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = make_root(tmp, self.ROWS)
            run = gen.generate_generation(str(r), 0, family='TPE_LSHADE', count=10, seed=3)
            mani = json.loads((run / 'generation_manifest.json').read_text())
            self.assertEqual(mani['reverse_test_violations'], 0)
            vals = {}
            for row in csv.DictReader((run / 'candidate_values.csv').open(encoding='utf-8')):
                vals.setdefault(row['name'], set()).add(row['value'])
            self.assertTrue(vals['p_flag'] <= {'true', 'false'})  # valid boolean values
            self.assertGreater(len(vals['p_flag']), 1)  # the bool VARIES across candidates
            self.assertTrue(vals['p_enum'] <= {'SMA', 'EMA', 'LWMA'})  # enum within its choices
            self.assertEqual(vals['p_prot'], {'42'})                 # protege inchange

    def test_gen1_evolve_roundtrips_discrete(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = make_root(tmp, self.ROWS)
            gen.generate_generation(str(r), 0, family='TPE_LSHADE', count=6, seed=3)
            run0 = r / 'MHO' / 'RUN_000'; (run0 / 'results').mkdir(parents=True, exist_ok=True)
            cids = sorted({row['candidate_id'] for row in csv.DictReader((run0 / 'candidate_values.csv').open(encoding='utf-8'))})
            with (run0 / 'results' / 'batch_results.csv').open('w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=['candidate_id', 'trades', 'profit_factor', 'balance_dd_pct'])
                w.writeheader()
                for k, cid in enumerate(cids):
                    w.writerow({'candidate_id': cid, 'trades': 60 + 10 * k, 'profit_factor': round(1.1 + 0.05 * k, 3), 'balance_dd_pct': 8})
            run1 = gen.generate_generation(str(r), 1, family='TPE_LSHADE', count=6, seed=3)
            mani = json.loads((run1 / 'generation_manifest.json').read_text())
            self.assertEqual(mani['mode'], 'EVOLVE')                 # reconstruction (encode) OK
            self.assertEqual(mani['reverse_test_violations'], 0)


if __name__ == '__main__':
    unittest.main()
