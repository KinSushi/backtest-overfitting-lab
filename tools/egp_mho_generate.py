#!/usr/bin/env python3
"""Candidate generation DRIVEN by metaheuristic (fixes MHO-0 in the pipeline).

Replaces the random generation of `egp_set_tools.generate()` with the loop
propose/observe :
  - generation 0 : Latin Hypercube (couverture d'espace).
  - generation g>0: reads the real MT5 metrics of RUN_{g-1} (via the parser),
    computes the robust scores, and has generation g proposed by the kernel of the
    `family` (evolve_generation), WITHIN THE OPT BOUNDS only.

Policy enforced : only OPTIMIZE_CORE/OPTIMIZE_LATER vary ; the protected ones
keep the baseline value; safety overlay forced; reverse-test mandatory.

No MT5 call. Deterministic when seeded.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import csv, json, argparse, sys

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import egp_set_tools as st
from egp_mho_hybrid import (load_opt_dims, decode, encode, evolve_generation, evolve_generation_mo,
                            pareto_front_size, reverse_test_protected, OptDim)
from egp_mt5_report_parser import enrich_batch_results, metrics_to_objectives, _num
from egp_bt_cache import canonical_key, BacktestCache, default_cache_path
from egp_enum_tables import to_set_value as _enum_to_set

SAFETY_OVERLAY = {'Lot_calculate': 'Fixed', 'LOT': '0.01', 'Ratio_Martingle': '1',
                  'number_buy_in': '1', 'number_sell_in': '1', 'In_META_Enable': 'false'}


def _read_prev(root: Path, gen_prev: int, dims: List[OptDim]):
    """Reconstructs (OPT vectors, robust scores, nb_discarded) from RUN_{gen_prev}."""
    run = root / 'MHO' / f'RUN_{gen_prev:03d}'
    cv = run / 'candidate_values.csv'
    if not cv.exists():
        return None, None, 0
    enrich_batch_results(str(run))                       # remplit metriques + robust_score
# scores per candidate
    scores = {}
    br = run / 'results' / 'batch_results.csv'
    if br.exists():
        with br.open(encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                try:
                    scores[r['candidate_id']] = float(r.get('robust_score', ''))
                except (ValueError, KeyError):
                    pass
# OPT vectors per candidate
    vecs = {}
    with cv.open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            vecs.setdefault(r['candidate_id'], {})[r['name']] = r['value']
    prev_vectors, prev_scores = [], []
    for cid, namevals in vecs.items():
        if cid not in scores:
            continue
        vec = [encode(namevals.get(d.name), d) for d in dims]
        prev_vectors.append([max(d.lo, min(d.hi, v)) for v, d in zip(vec, dims)])
        prev_scores.append(scores[cid])
    dropped = sum(1 for cid in vecs if cid not in scores)
    if not prev_vectors:
        return None, None, dropped
    return prev_vectors, prev_scores, dropped


def _read_prev_mo(root: Path, gen_prev: int, dims: List[OptDim]):
    """Like _read_prev but reconstructs OBJECTIVE-VECTORS (multi-objective NSGA-II)."""
    run = root / 'MHO' / f'RUN_{gen_prev:03d}'
    cv = run / 'candidate_values.csv'
    if not cv.exists():
        return None, None, 0
    enrich_batch_results(str(run))
    objs = {}
    br = run / 'results' / 'batch_results.csv'
    if br.exists():
        with br.open(encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                m = {k: _num(r.get(k)) for k in ('trades', 'profit_factor', 'balance_dd_pct')}
                objs[r['candidate_id']] = metrics_to_objectives(m)
    vecs = {}
    with cv.open(encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            vecs.setdefault(r['candidate_id'], {})[r['name']] = r['value']
    prev_vectors, prev_objs = [], []
    for cid, namevals in vecs.items():
        if cid not in objs:
            continue
        vec = [encode(namevals.get(d.name), d) for d in dims]
        prev_vectors.append([max(d.lo, min(d.hi, v)) for v, d in zip(vec, dims)])
        prev_objs.append(objs[cid])
    dropped = sum(1 for cid in vecs if cid not in objs)
    if not prev_vectors:
        return None, None, dropped
    return prev_vectors, prev_objs, dropped


def generate_generation(root: str, generation: int, family: Optional[str] = None,
                        count: int = 20, seed: int = 123, inner_evals: int = 400) -> Path:
    root = Path(root)
    rows = st.load_map(root)
    base = st.baseline_values(rows)
    # MT5 represents an enum input by its INTEGER VALUE (cf. egp_enum_tables). We make
    # EVERY enum value (baseline, non-OPT, overlay) integer-consistent: the generated .set files
    # are then loadable as-is and the protected audit compares integer-vs-integer.
    _type_by_name = {r['name']: (r.get('type') or '') for r in rows}
    _enumize = lambda d: {k: _enum_to_set(_type_by_name.get(k, ''), v) for k, v in d.items()}
    base = _enumize(base)
    _overlay = _enumize(SAFETY_OVERLAY)
    dims = load_opt_dims(str(root))
    if not dims:
        raise SystemExit('NO_OPTIMIZABLE_DIMS: no bounded OPT parameter (min/max) in PARAMETER_MAP.csv')

    cfg_path = root / 'CONFIG' / 'optimizer_config.json'
    families = ['TPE_LSHADE']
    if cfg_path.exists():
        try:
            families = json.loads(cfg_path.read_text(encoding='utf-8')).get('families', families)
        except json.JSONDecodeError:
            pass
    fam = family or (families[generation % len(families)] if families else 'TPE_LSHADE')
    is_mo = 'NSGA2' in fam.upper()
    pareto_front0 = 0

    prev_vectors = prev_objs = prev_scores = None
    if generation == 0:
        mode, dropped = 'LHS_GEN0', 0
    elif is_mo:
        prev_vectors, prev_objs, dropped = _read_prev_mo(root, generation - 1, dims)
        mode = 'NSGA2_EVOLVE' if prev_vectors else 'LHS_FALLBACK_NO_PRIOR_SCORES'
        pareto_front0 = pareto_front_size(prev_objs) if prev_objs else 0
    else:
        prev_vectors, prev_scores, dropped = _read_prev(root, generation - 1, dims)
        mode = 'EVOLVE' if prev_vectors else 'LHS_FALLBACK_NO_PRIOR_SCORES'

    def _propose(s):
        if generation == 0 or not prev_vectors:
            return (evolve_generation_mo(dims, None, None, n=count, seed=s) if is_mo
                    else evolve_generation(dims, None, None, family=fam, n=count, seed=s, inner_evals=inner_evals))
        if is_mo:
            return evolve_generation_mo(dims, prev_vectors, prev_objs, n=count, seed=s)
        return evolve_generation(dims, prev_vectors, prev_scores, family=fam, n=count, seed=s, inner_evals=inner_evals)

    # Intra-generation dedup: keep only vectors with a distinct OPT key (bounded top-up).
    seen, uniq, duplicates, rounds = set(), [], 0, 0
    cur = _propose(seed)
    while True:
        for vec in cur:
            ov = decode(vec, dims)
            key = canonical_key(ov)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key); uniq.append((vec, ov, key))
            if len(uniq) >= count:
                break
        if len(uniq) >= count or rounds >= 4:
            break
        rounds += 1
        cur = _propose(seed + 7919 * rounds)
    uniq = uniq[:count]

    cache = BacktestCache(default_cache_path(root))
    cache_hits = 0

    run = root / 'MHO' / f'RUN_{generation:03d}'
    sets = run / 'sets'
    sets.mkdir(parents=True, exist_ok=True)
    cand_rows, summ, all_violations = [], [], []

    for i, (vec, opt_vals, key) in enumerate(uniq):
        cid = f'G{generation:03d}_C{i:04d}'
        # reverse-test: no decoded key outside OPT
        all_violations += [{'candidate_id': cid, 'name': k, 'violation': 'NON_OPT_KEY'}
                           for k in reverse_test_protected(str(root), opt_vals)]
        # full set: baseline for everything, OPT override, forced safety overlay
        vals = dict(base)
        vals.update(opt_vals)
        vals.update(_overlay)
        cached = key in cache
        if cached:
            cache_hits += 1
        sp = sets / f'{cid}.set'
        with sp.open('w', encoding='ascii', errors='ignore') as f:
            f.write(f'; candidate_id={cid}\n; generation={generation}\n; family={fam}\n; mode={mode}\n; opt_key={key}\n')
            for k, v in vals.items():
                if v != '':
                    f.write(f'{k}={v}\n')
        # audit: a protected parameter must never differ from the baseline (outside overlay)
        for r in rows:
            nm = r['name']
            if r['status'] in st.PROTECTED and nm not in SAFETY_OVERLAY:
                if str(vals.get(nm, '')) != str(base.get(nm, '')):
                    all_violations.append({'candidate_id': cid, 'name': nm, 'violation': 'PROTECTED_MUTATED'})
            cand_rows.append({'candidate_id': cid, 'generation': generation, 'family': fam,
                              'name': nm, 'value': vals.get(nm, ''), 'status': r['status']})
        summ.append({'candidate_id': cid, 'generation': generation, 'family': fam,
                     'set_file': str(sp.relative_to(root)), 'sha256': st.sha256_file(sp),
                     'opt_key': key, 'cached': '1' if cached else '0'})

    with (run / 'candidate_values.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(cand_rows[0].keys())); w.writeheader(); w.writerows(cand_rows)
    with (run / 'candidate_summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)
    (run / 'generation_manifest.json').write_text(json.dumps(
        {'status': 'PASS' if not all_violations else 'FAIL',
         'count': count, 'unique_candidates': len(uniq), 'duplicates_collapsed': duplicates,
         'generation': generation, 'family': fam, 'mode': mode,
         'prior_dropped': dropped, 'pareto_front0': pareto_front0, 'cache_hits': cache_hits,
         'reverse_test_violations': len(all_violations)}, indent=2), encoding='utf-8')
    out = run / 'reverse_test_violations.csv'
    with out.open('w', newline='', encoding='utf-8') as f:
        if all_violations:
            w = csv.DictWriter(f, fieldnames=list(all_violations[0].keys())); w.writeheader(); w.writerows(all_violations)
        else:
            f.write('')
    if all_violations:
        # forbidden -> clean failure (non-zero exit), after writing diagnostics
        raise RuntimeError(f'POLICY_VIOLATION: {len(all_violations)} violation(s) recorded -> {out}')
    return run


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--generation', type=int, default=0)
    ap.add_argument('--family', default=None)
    ap.add_argument('--count', type=int, default=20)
    ap.add_argument('--seed', type=int, default=123)
    a = ap.parse_args()
    run = generate_generation(a.root, a.generation, a.family, a.count, a.seed)
    mani = json.loads((run / 'generation_manifest.json').read_text())
    print(run, '->', mani['status'], 'mode=', mani['mode'], 'violations=', mani['reverse_test_violations'])
