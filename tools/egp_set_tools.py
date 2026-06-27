#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import csv, json, random, hashlib, argparse, shutil

OPT={'OPTIMIZE_CORE','OPTIMIZE_LATER'}
PROTECTED={'LOCK_DEFAULT','LOCK_DANGEROUS','LOCK_SAFETY','STRESS_ONLY','INFRASTRUCTURE_ONLY','CUSTOM_INDICATOR_INPUT','DEBUG_ONLY','SCRIPT_ONLY','AOF_ONLY','MHO_ONLY'}

def load_map(root):
    p=Path(root)/'DOCS/PARAMETER_MAP.csv'
    with p.open(encoding='utf-8') as f: return list(csv.DictReader(f))

def baseline_values(rows):
    return {r['name']:r['default'] for r in rows if r.get('default')!=''}

def sha256_file(p):
    h=hashlib.sha256();
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(65536),b''): h.update(b)
    return h.hexdigest()

def mutate_value(r,val,rng):
    if r['status'] not in OPT: return val
    typ=r['type']; lo=r.get('min'); hi=r.get('max')
    if lo and hi:
        if typ in ['int'] or typ.startswith('ENUM_TIMEFRAMES'):
            return str(rng.randint(int(float(lo)),int(float(hi))))
        try:
            return f"{rng.uniform(float(lo),float(hi)):.6g}"
        except Exception:
            return val
    # bool/enum structural later: keep mostly baseline, small toggles for bool
    if typ=='bool' and rng.random()<0.2:
        return 'false' if str(val).lower()=='true' else 'true'
    return val

def generate(root, count=20, generation=0, seed=123):
    root=Path(root); rng=random.Random(seed); rows=load_map(root); base=baseline_values(rows)
    run=root/'MHO'/f'RUN_{generation:03d}'; sets=run/'sets'; sets.mkdir(parents=True,exist_ok=True)
    cand_rows=[]; summ=[]
    for i in range(count):
        cid=f'G{generation:03d}_C{i:04d}'; vals={}
        for r in rows:
            vals[r['name']]=mutate_value(r,base.get(r['name'],''),rng)
        # enforce safe fixed sizing
        vals.update({'Lot_calculate':'Fixed','LOT':'0.01','Ratio_Martingle':'1','number_buy_in':'1','number_sell_in':'1','In_META_Enable':'false'})
        sp=sets/f'{cid}.set'
        with sp.open('w',encoding='ascii',errors='ignore') as f:
            f.write(f'; candidate_id={cid}\n; generation={generation}\n; family=SEEDED_LOCAL\n')
            for k,v in vals.items():
                if v!='': f.write(f'{k}={v}\n')
        for r in rows: cand_rows.append({'candidate_id':cid,'generation':generation,'family':'SEEDED_LOCAL','name':r['name'],'value':vals.get(r['name'],''),'status':r['status']})
        summ.append({'candidate_id':cid,'generation':generation,'family':'SEEDED_LOCAL','set_file':str(sp.relative_to(root)),'sha256':sha256_file(sp)})
    with (run/'candidate_values.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(cand_rows[0].keys())); w.writeheader(); w.writerows(cand_rows)
    with (run/'candidate_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)
    (run/'generation_manifest.json').write_text(json.dumps({'status':'PASS','count':count,'generation':generation,'family':'SEEDED_LOCAL'},indent=2),encoding='utf-8')
    return run

def audit_run(root, generation=0):
    root=Path(root); rows={r['name']:r for r in load_map(root)}; run=root/'MHO'/f'RUN_{generation:03d}'
    with (run/'candidate_values.csv').open(encoding='utf-8') as _f:
        vals=list(csv.DictReader(_f))
    violations=[]
    for v in vals:
        r=rows.get(v['name']);
        if not r: violations.append({**v,'violation':'UNKNOWN_PARAM'}); continue
        if r['status'] in PROTECTED and v['value'] not in [r['default'],'']:
            # safety overlay exceptions allow fixed-lot safety changes but still report? treat dangerous as violation unless explicit safety overlay
            if v['name'] not in {'Lot_calculate','LOT','Ratio_Martingle','number_buy_in','number_sell_in','In_META_Enable'}:
                violations.append({**v,'violation':'PROTECTED_MUTATED'})
    out=root/'VALIDATION'/'PIPELINE_SELFTEST'; out.mkdir(parents=True,exist_ok=True)
    with (out/'candidate_safety_violations.csv').open('w',newline='',encoding='utf-8') as f:
        if violations:
            w=csv.DictWriter(f,fieldnames=list(violations[0].keys())); w.writeheader(); w.writerows(violations)
        else: f.write('')
    return violations

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('cmd',choices=['generate','audit']); ap.add_argument('--root',default='.'); ap.add_argument('--count',type=int,default=20); ap.add_argument('--generation',type=int,default=0)
    a=ap.parse_args()
    if a.cmd=='generate': print(generate(a.root,a.count,a.generation))
    else:
        v=audit_run(a.root,a.generation); print('PASS' if not v else f'FAIL {len(v)}')
        import sys; sys.exit(0 if not v else 1)
