#!/usr/bin/env python3
from pathlib import Path
import csv,json,argparse

def gate(root, results):
    root=Path(root); rows=list(csv.DictReader(open(results,encoding='utf-8')))
    dec=[]
    for r in rows:
        reason=[]
        def f(k):
            raw=str(r.get(k,'')).strip()
            try: return float(raw.replace(',','.'))
            except Exception:
                if raw not in ('','0'): reason.append(f'PARSE_ERROR:{k}')
                return 0.0
        if f('trades')<1: reason.append('TRADES_BELOW_MIN')
        if f('profit_factor')<1.0: reason.append('PF_BELOW_1')
        if r.get('runtime_hits'): reason.append('RUNTIME_HITS')
        dec.append({**r,'decision':'REJECT' if reason else 'CANDIDATE_ONLY','reason':';'.join(reason)})
    out=root/'VALIDATION/FINAL_GATE'; out.mkdir(parents=True,exist_ok=True)
    with (out/'final_gate_decisions.csv').open('w',newline='',encoding='utf-8') as f:
        if dec:
            w=csv.DictWriter(f,fieldnames=list(dec[0].keys())); w.writeheader(); w.writerows(dec)
    (out/'final_gate_summary.json').write_text(json.dumps({'count':len(dec),'accepted':sum(1 for d in dec if d['decision']!='REJECT')},indent=2),encoding='utf-8')
    print(out/'final_gate_decisions.csv')
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='.'); ap.add_argument('--results',required=True)
    a=ap.parse_args(); gate(a.root,a.results)
