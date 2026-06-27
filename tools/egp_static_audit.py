#!/usr/bin/env python3
from pathlib import Path
import re,csv,json,hashlib,argparse,datetime

def read(p):
    b=p.read_bytes()
    for enc in ['utf-16','utf-8-sig','utf-8','latin-1']:
        try: return b.decode(enc)
        except Exception: pass
    return b.decode('utf-8',errors='ignore')

def sha(p):
    h=hashlib.sha256();
    with p.open('rb') as f:
        for b in iter(lambda:f.read(65536),b''): h.update(b)
    return h.hexdigest()

def audit(root):
    root=Path(root); findings=[]; files=[]; input_names=[]
    markers=re.compile(r'\b(TODO|FIXME|STUB|PLACEHOLDER|TBD)\b',re.I)
    for p in sorted(root.rglob('*')):
        if p.is_file():
            rel=str(p.relative_to(root)).replace('\\','/')
            files.append({'path':rel,'size':p.stat().st_size,'sha256':sha(p)})
            if p.suffix.lower() in ['.mq5','.mqh','.ps1','.py','.md','.csv','.json']:
                text=read(p)
                for m in markers.finditer(text): findings.append({'severity':'FAIL','kind':'DEV_MARKER','file':rel,'detail':m.group(0)})
            if p.suffix.lower() in ['.mq5','.mqh']:
                text=read(p)
                if 'MQL5/Include/EGP' in rel and re.search(r'\b(OrderSend|CTrade|PositionOpen|Buy\(|Sell\()\b',text):
                    findings.append({'severity':'FAIL','kind':'TRADE_CALL_IN_META_INCLUDE','file':rel,'detail':'trade call candidate'})
                if rel.endswith('.mq5') or rel.endswith('.mqh'):
                    for m in re.finditer(r'\binput\s+(?!group\b)\w+\s+([^;]+);',text):
                        for n in re.findall(r'\b([A-Za-z_]\w*)\s*(?:=|,|$)',m.group(1)):
                            if n not in ['true','false']: input_names.append(n)
                for m in re.finditer(r'CopyBuffer\s*\(',text): findings.append({'severity':'INFO','kind':'COPYBUFFER_OCCURRENCE','file':rel,'detail':str(text[:m.start()].count('\n')+1)})
    # parameter map
    pmap=root/'DOCS/PARAMETER_MAP.csv'
    if not pmap.exists(): findings.append({'severity':'FAIL','kind':'PARAMETER_MAP_MISSING','file':'DOCS/PARAMETER_MAP.csv','detail':''}); rows=[]
    else:
        rows=list(csv.DictReader(pmap.open(encoding='utf-8')))
        names=[r['name'] for r in rows]
        dup=sorted({x for x in names if names.count(x)>1})
        for d in dup: findings.append({'severity':'FAIL','kind':'PARAMETER_DUPLICATE','file':'DOCS/PARAMETER_MAP.csv','detail':d})
        for r in rows:
            if not r.get('status'): findings.append({'severity':'FAIL','kind':'PARAMETER_STATUS_EMPTY','file':'DOCS/PARAMETER_MAP.csv','detail':r.get('name','')})
    out=root/'VALIDATION/LOCAL_STATIC_AUDIT'; out.mkdir(parents=True,exist_ok=True)
    (out/'file_manifest.json').write_text(json.dumps(files,indent=2),encoding='utf-8')
    with (out/'findings.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['severity','kind','file','detail']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(findings)
    fail=[x for x in findings if x['severity']=='FAIL']
    summary={'generated_utc':datetime.datetime.utcnow().isoformat()+'Z','status':'PASS' if not fail else 'FAIL','files':len(files),'findings_total':len(findings),'blocking_findings':len(fail),'parameter_map_rows':len(rows) if pmap.exists() else 0,'inputs_extracted_static':len(set(input_names))}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (out/'summary.md').write_text('# Local Static Audit\n\n'+'\n'.join(f'- **{k}**: {v}' for k,v in summary.items())+'\n',encoding='utf-8')
    return summary
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='.')
    a=ap.parse_args(); s=audit(a.root); print(json.dumps(s,indent=2)); raise SystemExit(0 if s['status']=='PASS' else 2)
