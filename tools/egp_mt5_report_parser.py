#!/usr/bin/env python3
"""Parser for MetaTrader 5 Strategy Tester HTML reports (offline).

Retrieves a backtest's metrics (Total Net Profit, Profit Factor, Drawdown %,
Sharpe, Total Trades, etc.) from the `.htm` produced by `Report=` in the .ini,
fills `batch_results.csv` (columns left empty by the PowerShell runner), and
provides a ROBUST OBJECTIVE (to minimize) for the MHO loop, with an
in-sample/forward anti-overfitting variant.

No MT5 call. Robust: missing field -> None; malformed HTML -> no exception.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, List
import re, csv, html, math, json, argparse

# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
# tolerant number: sign, thousands separators (space/no-break/comma), decimal
_NUM = r'[-+]?\d[\d \u00a0\u202f,]*\.?\d*'

# MT5 label -> output key
_LABELS = {
    'profit':          r'Total\s+Net\s+Profit',
    'gross_profit':    r'Gross\s+Profit',
    'gross_loss':      r'Gross\s+Loss',
    'profit_factor':   r'Profit\s+Factor',
    'expected_payoff': r'Expected\s+Payoff',
    'recovery':        r'Recovery\s+Factor',
    'sharpe':          r'Sharpe\s+Ratio',
    'trades':          r'Total\s+Trades',
}


def _detag(s: str) -> str:
    s = re.sub(r'(?is)<\s*(script|style).*?<\s*/\s*\1\s*>', ' ', s)
    s = re.sub(r'(?s)<[^>]+>', ' ', s)
    s = html.unescape(s)
    for ch in ('\u00a0', '\u202f'):
        s = s.replace(ch, ' ')
    return re.sub(r'[ \t\r\n]+', ' ', s)


def _to_float(tok: Optional[str]) -> Optional[float]:
    if tok is None:
        return None
    t = tok.strip().replace(' ', '')
    if ',' in t and '.' not in t:          # decimal comma
        t = t.replace(',', '.')
    else:                                   # comma = thousands separator
        t = t.replace(',', '')
    try:
        return float(t)
    except ValueError:
        return None


def parse_report_text(text: str) -> Dict[str, Optional[float]]:
    t = _detag(text)
    out: Dict[str, Optional[float]] = {}
    for key, label in _LABELS.items():
        m = re.search(label + r'\s*:?\s*(' + _NUM + r')', t, re.I | re.S)
        out[key] = _to_float(m.group(1)) if m else None
    # drawdowns in %: "Maximal: V (P%)" then fallback "Relative: P% (V)"
    for key, lab in (('balance_dd_pct', 'Balance'), ('equity_dd_pct', 'Equity')):
        m = re.search(lab + r'\s+Drawdown\s+Maximal\s*:?\s*' + _NUM + r'\s*\(\s*(' + _NUM + r')\s*%', t, re.I)
        if not m:
            m = re.search(lab + r'\s+Drawdown\s+Relative\s*:?\s*(' + _NUM + r')\s*%', t, re.I)
        out[key] = _to_float(m.group(1)) if m else None
    if out.get('trades') is not None:
        out['trades'] = float(int(out['trades']))
    return out


def parse_report_file(path: str) -> Dict[str, Optional[float]]:
    raw = Path(path).read_bytes()
    text = None
    for enc in ('utf-8-sig', 'utf-8', 'utf-16', 'latin-1'):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode('latin-1', 'ignore')
    return parse_report_text(text)


# --------------------------------------------------------------------------- #
# Objectif robuste (a MINIMISER)
# --------------------------------------------------------------------------- #
PENALTY = 1e6


def robust_score(metrics: Dict[str, Optional[float]], min_trades: int = 30,
                 dd_weight: float = 0.5) -> float:
    """Cost to minimize. Guards: insufficient trades or PF<=1 -> heavily penalized.
    Otherwise cost = -PF + dd_weight*(DD%/100). Always finite."""
    def g(k, default=0.0):
        v = metrics.get(k)
        return v if (v is not None and math.isfinite(v)) else default
    trades = g('trades', 0.0)
    pf = g('profit_factor', 0.0)
    dd = metrics.get('balance_dd_pct')
    dd = dd if (dd is not None and math.isfinite(dd)) else 100.0
    if trades < min_trades:
        return PENALTY - trades                 # ordered by number of trades
    if pf <= 1.0:
        return 0.5 * PENALTY - pf               # ordered by PF
    cost = -pf + dd_weight * (dd / 100.0)
    return cost if math.isfinite(cost) else PENALTY


def robust_score_oos(is_metrics: Dict, oos_metrics: Dict, min_trades: int = 30,
                     overfit_weight: float = 0.5) -> float:
    """Anti-overfitting variant: out-of-sample cost + IS/OOS gap penalty."""
    c_is = robust_score(is_metrics, min_trades)
    c_oos = robust_score(oos_metrics, min_trades)
    base = c_oos
    if c_is < PENALTY * 0.4 and c_oos < PENALTY * 0.4:   # both usable
        base = c_oos + overfit_weight * abs(c_oos - c_is)
    return base if math.isfinite(base) else PENALTY


def metrics_to_objectives(metrics: Dict[str, Optional[float]], min_trades: int = 30):
    """Objective vector to MINIMIZE for NSGA-II: (-PF, DD%, -trades).
    Minimizing -PF = maximizing the profit factor; minimizing DD%; minimizing -trades =
    maximizing the number of trades. Guard: trades<min -> dominated (PENALTY values),
    so those candidates are discarded by the non-dominated sort. Always finite.

    Documented limitation: a PF=0 returned by MT5 means 'no loss' (excellent),
    but from the metrics alone it cannot be distinguished from a low PF; this case
    is handled correctly on the MT5 side by OnTester (via STAT_PROFIT), not here.
    """
    def g(k, d):
        v = metrics.get(k)
        return v if (v is not None and math.isfinite(v)) else d
    trades = g('trades', 0.0)
    pf = g('profit_factor', 0.0)
    dd = metrics.get('balance_dd_pct')
    dd = dd if (dd is not None and math.isfinite(dd)) else 100.0
    if trades < min_trades:
        return (PENALTY, PENALTY, PENALTY)
    return (-pf, dd, -trades)


# --------------------------------------------------------------------------- #
# Batch CSV enrichment
# --------------------------------------------------------------------------- #
_METRIC_KEYS = ['trades', 'profit', 'profit_factor', 'expected_payoff', 'recovery',
                'sharpe', 'balance_dd_pct', 'equity_dd_pct', 'gross_profit', 'gross_loss']


def _num(x):
    """Tolerant parse -> float | None (never raises)."""
    s = str(x).strip()
    if s in ('', 'None'):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def enrich_batch_results(run_dir: str, use_cache: bool = True) -> List[dict]:
    """Reads MHO/RUN_xxx/results/batch_results.csv, parses each .htm report and
    fills the metric columns. Rewrites the CSV. Feeds the backtest cache
    (OPT key read from the .set) if available. Returns the enriched rows."""
    csvp = Path(run_dir) / 'results' / 'batch_results.csv'
    if not csvp.exists():
        return []
    with csvp.open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    cache, puts = None, 0
    if use_cache:
        try:
            from egp_bt_cache import read_opt_key, BacktestCache, default_cache_path
            run = Path(run_dir)
            cache = BacktestCache(default_cache_path(run.parent.parent))
        except Exception:
            cache = None
    for r in rows:
        rep = r.get('report')
        if rep and Path(rep).exists():
            m = parse_report_file(rep)
            for k in _METRIC_KEYS:
                v = m.get(k)
                if v is not None:
                    r[k] = (int(v) if k == 'trades' else v)
            if cache is not None and r.get('set_file'):
                key = read_opt_key(r['set_file'])
                if key:
                    cache.put(key, m, {'candidate_id': r.get('candidate_id'), 'report': rep})
                    puts += 1
        r['robust_score'] = robust_score({k: _num(r.get(k)) for k in _METRIC_KEYS})
    fields = list(rows[0].keys()) if rows else []
    for extra in _METRIC_KEYS + ['robust_score']:
        if extra not in fields:
            fields.append(extra)
    with csvp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    if cache is not None and puts:
        cache.save()
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['parse', 'enrich'])
    ap.add_argument('--report')
    ap.add_argument('--run')
    a = ap.parse_args()
    if a.cmd == 'parse':
        print(json.dumps(parse_report_file(a.report), indent=2))
    else:
        rows = enrich_batch_results(a.run)
        print(f'enriched {len(rows)} rows')
