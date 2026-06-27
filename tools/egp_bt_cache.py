#!/usr/bin/env python3
"""Backtest cache + canonical candidate key (deduplication).

An MT5 backtest is expensive: a previously evaluated OPT vector must never be re-tested.
The canonical key is a hash of the OPT values only (the protected inputs and the safety
overlay are constant), so two candidates with the same OPT vector share the same key,
regardless of their generation, id, or comments.

- Intra-generation dedup: collapse candidates with identical keys (generator side).
- Inter-generation dedup: the cache (key -> metrics) avoids re-running a backtest.

No MT5 call. The cache is filled by the parser after a backtest, and queried before.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict
import hashlib, json, csv, argparse, time

_MK = ['trades', 'profit', 'profit_factor', 'expected_payoff', 'recovery',
       'sharpe', 'balance_dd_pct', 'equity_dd_pct']


def canonical_key(opt_values: Dict[str, str]) -> str:
    """Stable hash of the OPT values (independent of order, comments, protected inputs)."""
    items = sorted((str(k), str(v)) for k, v in opt_values.items())
    blob = ';'.join(f'{k}={v}' for k, v in items)
    return hashlib.sha256(blob.encode('utf-8')).hexdigest()


def read_opt_key(set_path) -> Optional[str]:
    """Reads the '; opt_key=...' key written into a .set by the generator (else None)."""
    if not set_path:
        return None
    try:
        for line in Path(set_path).read_text(encoding='ascii', errors='ignore').splitlines():
            s = line.strip()
            if s.startswith('; opt_key='):
                return s.split('=', 1)[1].strip()
    except OSError:
        return None
    return None


def default_cache_path(root) -> Path:
    return Path(root) / 'MHO' / 'CACHE' / 'backtest_cache.json'


class BacktestCache:
    """Persistent cache key -> {metrics, meta, ts}. Tolerant to missing/corrupt files."""

    def __init__(self, path):
        self.path = Path(path)
        self.data: Dict[str, dict] = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding='utf-8'))
            except (json.JSONDecodeError, OSError):
                self.data = {}

    def __contains__(self, key):
        return key in self.data

    def get(self, key):
        return self.data.get(key)

    def metrics(self, key) -> dict:
        e = self.data.get(key) or {}
        return e.get('metrics', {})

    def put(self, key, metrics, meta=None):
        if not key:
            return
        self.data[key] = {'metrics': {k: metrics.get(k) for k in metrics},
                          'meta': meta or {}, 'ts': time.time()}

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding='utf-8')

    def stats(self):
        return {'entries': len(self.data)}


def plan(run_dir, root=None) -> dict:
    """Splits a generation's candidates into (to test) vs (already cached).
    Writes results/to_backtest.txt (names of .set to test) and results/cached_results.csv
    (pre-filled metrics for already-known candidates). Lets the MT5 runner execute only
    the non-cached candidates."""
    run = Path(run_dir)
    root = Path(root) if root else run.parent.parent
    cache = BacktestCache(default_cache_path(root))
    summ = run / 'candidate_summary.csv'
    if not summ.exists():
        return {'cached': 0, 'to_backtest': 0}
    rows = list(csv.DictReader(summ.open(encoding='utf-8')))
    to_bt, cached_rows = [], []
    for r in rows:
        key = r.get('opt_key') or read_opt_key(root / r['set_file'])
        if key and key in cache:
            m = cache.metrics(key)
            cached_rows.append({'candidate_id': r['candidate_id'], 'set_file': r['set_file'],
                                'report': '', 'runtime_hits': '', 'cached': '1',
                                **{k: ('' if m.get(k) is None else m.get(k)) for k in _MK}})
        else:
            to_bt.append(r)
    out = run / 'results'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'to_backtest.txt').write_text(
        '\n'.join(Path(r['set_file']).name for r in to_bt) + ('\n' if to_bt else ''), encoding='utf-8')
    if cached_rows:
        fields = ['candidate_id', 'set_file', 'report', 'runtime_hits', 'cached'] + _MK
        with (out / 'cached_results.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(cached_rows)
    return {'cached': len(cached_rows), 'to_backtest': len(to_bt)}


def merge_cached(run_dir) -> int:
    """Merges results/cached_results.csv into results/batch_results.csv (union of
    columns; adds the missing cached candidates). The gate thus sees cached + tested."""
    run = Path(run_dir); res = run / 'results'
    br, cc = res / 'batch_results.csv', res / 'cached_results.csv'
    if not cc.exists():
        return 0
    cached = list(csv.DictReader(cc.open(encoding='utf-8-sig')))
    existing = list(csv.DictReader(br.open(encoding='utf-8-sig'))) if br.exists() else []
    have = {r.get('candidate_id') for r in existing}
    add = [r for r in cached if r.get('candidate_id') not in have]
    fields = list(existing[0].keys()) if existing else []
    for r in add:
        for k in r:
            if k not in fields:
                fields.append(k)
    with br.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, restval=''); w.writeheader()
        w.writerows(existing + add)
    return len(add)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['plan', 'merge', 'stats'])
    ap.add_argument('--run')
    ap.add_argument('--root', default=None)
    a = ap.parse_args()
    if a.cmd == 'plan':
        print(plan(a.run, a.root))
    elif a.cmd == 'merge':
        print('merged_cached_rows=%d' % merge_cached(a.run))
    else:
        print(BacktestCache(default_cache_path(a.root or '.')).stats())
