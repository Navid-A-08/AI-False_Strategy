import json,sys,re
src,dst=sys.argv[1],sys.argv[2]
d=json.load(open(src)) if src.endswith('.json') else None
txt=d[0]['text'] if d else open(src).read()
recs=[json.loads(l) for l in txt.splitlines() if l.strip().startswith('{"model"')]
with open(dst,'w') as f:
    for r in recs: f.write(json.dumps(r,ensure_ascii=False)+'\n')
print(len(recs))
