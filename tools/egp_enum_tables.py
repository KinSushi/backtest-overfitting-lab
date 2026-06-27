"""VERIFIED enumeration tables to produce correct MT5 .set files.

Empirical finding (mql5/fxDreema forum + official ParameterSetRange(long) signature):
MT5 represents an enum-typed input by its underlying INTEGER VALUE (not the symbolic
name). E.g. PERIOD_H1 appears as 16385, PRICE_CLOSE as 1, off_1 as 3.

Two origins, NO invention:
  - EA_ENUMS  : enums declared IN the EA source code, values read verbatim
                (ordinals 0,1,2,... only where no '=' value is written).
  - BUILTIN   : standard MQL5 enums (not redeclared in the EA). Members+order = doc
                official; integer values = mql5 doc/forum (MA_METHOD 0-based,
                APPLIED_PRICE 1-based, TIMEFRAMES non-contiguous confirmed by MT5 observation).
"""
import re
from pathlib import Path
from typing import Dict, Optional

_ROOT = Path(__file__).resolve().parents[1]
_SOURCES = [
    _ROOT / 'MQL5' / 'Experts' / 'EGP_DemoEA.mq5',
    _ROOT / 'MQL5' / 'Experts' / 'EGP_MHO_RiskGuard.mqh',
]


def _read(p: Path) -> str:
    raw = p.read_bytes()
    for enc in ('utf-16', 'utf-8-sig', 'utf-8', 'latin-1'):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode('latin-1', 'ignore')


def _parse_source_enums(txt: str) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for m in re.finditer(r'enum\s+([A-Za-z_]\w*)\s*\{([^}]*)\}', txt, re.S):
        name = m.group(1)
        body = re.sub(r'//[^\n]*', '', m.group(2))
        body = re.sub(r'/\*.*?\*/', '', body, flags=re.S)
        members: Dict[str, int] = {}
        nxt = 0
        ok = True
        for part in body.split(','):
            part = part.strip()
            if not part:
                continue
            mm = re.match(r'([A-Za-z_]\w*)\s*(?:=\s*(-?\d+))?$', part)
            if not mm:           # literal non-integer value -> we do not invent
                ok = False
                break
            val = int(mm.group(2)) if mm.group(2) is not None else nxt
            members[mm.group(1)] = val
            nxt = val + 1
        if ok and members:
            out[name] = members
    return out


# --- EA (parse verbatim) ---------------------------------------------------- #
EA_ENUMS: Dict[str, Dict[str, int]] = {}
for _p in _SOURCES:
    if _p.exists():
        for k, v in _parse_source_enums(_read(_p)).items():
            EA_ENUMS.setdefault(k, v)

# --- MQL5 built-ins (official doc for members; verified values) -------- #
BUILTIN_ENUMS: Dict[str, Dict[str, int]] = {
    'ENUM_MA_METHOD': {'MODE_SMA': 0, 'MODE_EMA': 1, 'MODE_SMMA': 2, 'MODE_LWMA': 3},
    'ENUM_APPLIED_PRICE': {'PRICE_CLOSE': 1, 'PRICE_OPEN': 2, 'PRICE_HIGH': 3, 'PRICE_LOW': 4,
                           'PRICE_MEDIAN': 5, 'PRICE_TYPICAL': 6, 'PRICE_WEIGHTED': 7},
    'ENUM_TIMEFRAMES': {'PERIOD_CURRENT': 0, 'PERIOD_M1': 1, 'PERIOD_M2': 2, 'PERIOD_M3': 3,
                        'PERIOD_M4': 4, 'PERIOD_M5': 5, 'PERIOD_M6': 6, 'PERIOD_M10': 10,
                        'PERIOD_M12': 12, 'PERIOD_M15': 15, 'PERIOD_M20': 20, 'PERIOD_M30': 30,
                        'PERIOD_H1': 16385, 'PERIOD_H2': 16386, 'PERIOD_H3': 16387,
                        'PERIOD_H4': 16388, 'PERIOD_H6': 16390, 'PERIOD_H8': 16392,
                        'PERIOD_H12': 16396, 'PERIOD_D1': 16408, 'PERIOD_W1': 32769,
                        'PERIOD_MN1': 49153},
}

# EA takes priority on collision (the EA may redeclare), otherwise built-in.
ENUM_TABLES: Dict[str, Dict[str, int]] = dict(BUILTIN_ENUMS)
ENUM_TABLES.update(EA_ENUMS)


def member_value(type_name: str, member: str) -> Optional[int]:
    """Integer value of an enum member, or None if type/member is unknown."""
    t = ENUM_TABLES.get((type_name or '').strip())
    if not t:
        return None
    return t.get((member or '').strip())


def to_set_value(type_name: str, value: str) -> str:
    """Converts a .set value: if 'value' is a member NAME of the enum 'type_name',
    returns the integer (string); otherwise returns 'value' unchanged (already integer, numeric,
    bool, non-enum type, or unknown value -> never invented)."""
    sval = '' if value is None else str(value).strip()
    iv = member_value(type_name, sval)
    return str(iv) if iv is not None else sval


if __name__ == '__main__':
    print('EA_ENUMS:', len(EA_ENUMS), 'BUILTIN:', len(BUILTIN_ENUMS), 'TOTAL:', len(ENUM_TABLES))
    for chk in [('combination_1', 'off_1'), ('ENUM_TIMEFRAMES', 'PERIOD_H1'),
                ('ENUM_APPLIED_PRICE', 'PRICE_CLOSE'), ('lot_cal', 'Fixed')]:
        print(chk, '->', member_value(*chk))
