#!/usr/bin/env python3
"""Generates the MQL5 module `EGP_MHO_OnTester.mqh` (official self-optimization hook).

Produces, from DOCS/PARAMETER_MAP.csv:
  - `double OnTester()`: robust custom criterion via TesterStatistics()
    (PF, drawdown %, trades). MT5 convention: higher = better (descending sort
    in genetic). Consistent with Python robust_score (criterion = -cost).
    Reference: docs/event_handlers/ontester.
  - `int OnTesterInit()`: ParameterSetRange() to RANGE the OPT bounds (enable=true)
    and LOCK the protected ones (enable=false). Official signature (MQL5 book +
    release notes build 684):
      bool ParameterSetRange(const string name, bool enable, <long|double> value,
                             <long|double> start, <long|double> step, <long|double> stop)
    Callable ONLY from OnTesterInit. Ref: docs/optimization_frames/parametersetrange.

ANTI silent no-op (cross-check B): if the list of the EA's real inputs is provided
(verify_inputs), a name ABSENT from the input/sinput declarations is NOT emitted as a live call
(which would silently return false at runtime): it is COMMENTED OUT and LISTED in a report.

ANTI-invention: only the declared OPT min&max&step are RANGED; only the protected ones with
a numeric default are LOCKED; non-numeric types (string) ignored -> no invented literal.

Official limitation: OnTesterInit() returns int (INIT_SUCCEEDED). Print() inoperative in
optimization. #include ONCE ONLY; the EA must not already define OnTester()/OnTesterInit().
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import csv, re, argparse

OPT = {'OPTIMIZE_CORE', 'OPTIMIZE_LATER'}
PROTECTED = {'LOCK_DEFAULT', 'LOCK_DANGEROUS', 'LOCK_SAFETY', 'STRESS_ONLY',
             'INFRASTRUCTURE_ONLY', 'CUSTOM_INDICATOR_INPUT', 'DEBUG_ONLY',
             'SCRIPT_ONLY', 'AOF_ONLY', 'MHO_ONLY'}


def _kind(typ: str) -> Optional[str]:
    t = (typ or '').strip().lower()
    if t == 'double':
        return 'double'
    if t in ('string',):
        return None
    return 'long'           # int, bool, ulong, datetime, enum_*, combination_*, position*, ...


def _as_long(v: str) -> Optional[int]:
    s = (v or '').strip().lower()
    if s == 'true':
        return 1
    if s == 'false':
        return 0
    try:
        return int(round(float(s)))
    except (ValueError, TypeError):
        return None


def _as_double(v: str) -> Optional[float]:
    try:
        return float((v or '').strip())
    except (ValueError, TypeError):
        return None


def _lit_long(x: int) -> str:
    return str(int(x))


def _lit_double(x: float) -> str:
    s = repr(float(x))
    return s if ('.' in s or 'e' in s or 'E' in s) else s + '.0'


def extract_inputs(text: str):
    """Set of names declared `input`/`sinput` in an MQL5 source (excluding the `group` keyword).
    Comments (// and /* */) are stripped to avoid false positives."""
    text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.S)
    text = re.sub(r'//[^\n]*', ' ', text)
    names = set()
    for m in re.finditer(r'\b(?:input|sinput)\b\s+[A-Za-z_]\w*\s+([A-Za-z_]\w*)\s*[=;]', text):
        if m.group(1) != 'group':
            names.add(m.group(1))
    return names


def build(rows, declared_inputs=None):
    """Returns (code, info). info = range/lock/skip/notinput lists for traceability.
    declared_inputs (set|None): if provided, a name outside this set is commented out
    (not emitted) to eliminate the silent no-op at runtime."""
    set_lines = []
    info = {'range': [], 'lock': [], 'skip': [], 'notinput': []}

    def not_input(name):
        if declared_inputs is not None and name not in declared_inputs:
            set_lines.append(f'   // SKIP(not-an-input) {name} : not among the EA input/sinput declarations')
            info['notinput'].append(name)
            return True
        return False

    for r in rows:
        name, status, typ = r['name'], r['status'], r.get('type', '')
        k = _kind(typ)
        lo, hi, step, default = ((r.get('min') or '').strip(), (r.get('max') or '').strip(),
                                 (r.get('step') or '').strip(), (r.get('default') or '').strip())
        if status in OPT:
            if k is None or not (lo and hi and step):
                set_lines.append(f'   // SKIP(range) {name} : type={typ}, incomplete bounds'); info['skip'].append(name); continue
            if not_input(name):
                continue
            if k == 'long':
                a, b, st = _as_long(lo), _as_long(hi), _as_long(step)
                v = _as_long(default); v = v if v is not None else (a + b) // 2
                if None in (a, b, st):
                    set_lines.append(f'   // SKIP(range) {name} : non-integer bounds'); info['skip'].append(name); continue
                set_lines.append(f'   ParameterSetRange("{name}", true, {_lit_long(v)}, {_lit_long(a)}, {_lit_long(st)}, {_lit_long(b)});')
            else:
                a, b, st = _as_double(lo), _as_double(hi), _as_double(step)
                v = _as_double(default); v = v if v is not None else (a + b) / 2.0
                if None in (a, b, st):
                    set_lines.append(f'   // SKIP(range) {name} : non-numeric bounds'); info['skip'].append(name); continue
                set_lines.append(f'   ParameterSetRange("{name}", true, {_lit_double(v)}, {_lit_double(a)}, {_lit_double(st)}, {_lit_double(b)});')
            info['range'].append(name)
        elif status in PROTECTED:
            if k is None:
                set_lines.append(f'   // SKIP(lock) {name} : type={typ} not rangeable'); info['skip'].append(name); continue
            if not_input(name):
                continue
            if k == 'long':
                v = _as_long(default)
                if v is None:
                    set_lines.append(f'   // SKIP(lock) {name} : non-integer default'); info['skip'].append(name); continue
                set_lines.append(f'   ParameterSetRange("{name}", false, {_lit_long(v)}, {_lit_long(v)}, 1, {_lit_long(v)});')
            else:
                v = _as_double(default)
                if v is None:
                    set_lines.append(f'   // SKIP(lock) {name} : non-numeric default'); info['skip'].append(name); continue
                set_lines.append(f'   ParameterSetRange("{name}", false, {_lit_double(v)}, {_lit_double(v)}, 1.0, {_lit_double(v)});')
            info['lock'].append(name)
    body = '\n'.join(set_lines)
    code = f'''//+------------------------------------------------------------------+
//| EGP_MHO_OnTester.mqh  (auto-generated)                           |
//| Robust optimization criterion + input ranging/locking            |
//| Source of truth: DOCS/PARAMETER_MAP.csv                          |
//| #include ONCE ONLY. The EA must not already define               |
//| OnTester()/OnTesterInit(). ParameterSetRange : OnTesterInit only.|
//+------------------------------------------------------------------+
#ifndef EGP_MHO_ONTESTER_MQH
#define EGP_MHO_ONTESTER_MQH

sinput int    EGP_MHO_MinTrades = 30;    // guard: minimum number of trades
sinput double EGP_MHO_DDWeight  = 0.5;   // weight of the relative drawdown in the criterion

//--- Custom criterion (higher = better; descending sort in genetic)
double OnTester()
  {{
   double trades = TesterStatistics(STAT_TRADES);
   double pf     = TesterStatistics(STAT_PROFIT_FACTOR);
   double profit = TesterStatistics(STAT_PROFIT);
   double ddrel  = TesterStatistics(STAT_EQUITY_DDREL_PERCENT);
   if(trades < (double)EGP_MHO_MinTrades) return(0.0);            // worst: not enough trades
   if(pf == 0.0)                                                  // MT5: PF=0 <=> no loss
      return(profit > 0.0 ? 10.0 - EGP_MHO_DDWeight*(ddrel/100.0) : 0.0);
   if(pf <= 1.0) return(pf*0.001);                                // ordered by PF, heavily penalized
   double criterion = pf - EGP_MHO_DDWeight*(ddrel/100.0);        // = -robust_score (Python)
   if(!MathIsValidNumber(criterion)) return(0.0);
   return(criterion);
  }}

//--- Programmatic definition of the optimization space
int OnTesterInit()
  {{
{body}
   return(INIT_SUCCEEDED);
  }}

#endif // EGP_MHO_ONTESTER_MQH
// generated: range={len(info['range'])} lock={len(info['lock'])} skip={len(info['skip'])} notinput={len(info['notinput'])}
'''
    return code, info


def generate_code(rows, declared_inputs=None) -> str:
    return build(rows, declared_inputs)[0]


def _read_text(fp: Path) -> str:
    raw = fp.read_bytes()
    for enc in ('utf-8-sig', 'utf-8', 'utf-16', 'latin-1'):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode('latin-1', 'ignore')


def scan_declared_inputs(root: Path):
    """Union of the inputs declared in MQL5/Experts/*.mq5 and *.mqh."""
    exp = root / 'MQL5' / 'Experts'
    declared = set()
    if exp.exists():
        for fp in list(exp.glob('*.mq5')) + list(exp.glob('*.mqh')):
            declared |= extract_inputs(_read_text(fp))
    return declared


def generate(root: str, out: Optional[str] = None, verify_inputs: bool = True) -> Path:
    root = Path(root)
    rows = list(csv.DictReader((root / 'DOCS' / 'PARAMETER_MAP.csv').open(encoding='utf-8')))
    declared = None
    note = []
    if verify_inputs:
        declared = scan_declared_inputs(root)
        if not declared:                       # parsing failed -> do not exclude everything
            note.append('VERIFICATION UNAVAILABLE: no input detected in MQL5/Experts.')
            declared = None
    code, info = build(rows, declared)
    out_path = Path(out) if out else root / 'MQL5' / 'Experts' / 'EGP_MHO_OnTester.mqh'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(code, encoding='ascii', errors='ignore')
    rep = out_path.parent / (out_path.stem + '.inputs_report.txt')
    lines = [f'declared_inputs={0 if declared is None else len(declared)}',
             f"range={len(info['range'])} lock={len(info['lock'])} skip={len(info['skip'])} notinput={len(info['notinput'])}"]
    lines += note
    lines += ['', '# NOT-AN-INPUT (commented out, not emitted -> no silent no-op):'] + sorted(info['notinput'])
    rep.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return out_path


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--out', default=None)
    ap.add_argument('--no-verify', action='store_true', help='do not cross-check against the EA real inputs')
    a = ap.parse_args()
    p = generate(a.root, a.out, verify_inputs=not a.no_verify)
    print(p, '->', p.read_text().strip().splitlines()[-1])
