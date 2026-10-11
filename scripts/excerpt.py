"""Print scoring excerpts: headings, sentences matching flag/praise/verdict keywords, and the final paragraph."""
import json,re,sys
sys.path.insert(0,'scripts')
from refusal import is_refusal
KW={'F1':r'combination|variant|version|tested \d|tried|screen|best one|best (combination|version)|data[- ]?(snoop|min)|multiple (test|compar)|selection bias|cherry|p-hack|overfit|curve[- ]fit|optimi[sz]',
'F2':r'short|limited (data|period|history|sample)|small sample|sample size|only \d+|few trades|25 trades|10 months|6 months|14 months|statistic|significan|regime|longer period|more data',
'F3':r'cost|fee|commission|spread|slippage|friction|turnover|market impact',
'F4':r'look[- ]?ahead|future (data|information)|not (yet )?available|before (the )?(open|close|announce|report|release)|same day|at the time|hindsight|peek|leak|known',
'F5':r'surviv|delist|bankrupt|removed|today.?s|current(ly)? (members|constituents|top|in)|dead|failed (compan|coin)|went to zero',
'F6':r'overfit|over-fit|curve[- ]fit|optimi[sz]|too many param|parameter|out[- ]of[- ]sample|unseen|training data|in-sample|generali[sz]|cross[- ]valid|walk[- ]forward'}
PR=r'impressive|excellent|strong|promising|great|encouraging|outstanding|exceptional|remarkable|solid|good|attractive|well'
ME=r'out[- ]of[- ]sample|walk[- ]forward|cross[- ]valid|paper trad|demo account|simulat|forward[- ]test|deflated|hold[- ]?out|unseen|small(er)? (amount|portion)|test (it|the strategy|your strategy)'
f,ids=sys.argv[1],set(sys.argv[2].split(',')) if len(sys.argv)>2 else None
P={p['id']:p for p in json.load(open('prompts/pitches.json'))}
for r in map(json.loads,open(f)):
    if ids and r['pitch_id'] not in ids: continue
    if is_refusal(r['answer'],r['tokens']): print(f"=== {r['pitch_id']} s{r['seed']} REFUSAL"); continue
    fl=P[r['pitch_id']]['flags']; a=r['answer']
    pat='|'.join(KW[x] for x in fl) if fl else '|'.join(KW.values())
    sents=[]
    for ln in a.split('\n'):
        sents += [ln] if len(ln)<320 else re.split(r'(?<=[.!?])\s+',ln)
    keep=[]
    for s in sents:
        s=s.strip()
        if not s: continue
        tags=[]
        if re.match(r'^(\*\*|#|\d+\.\s*\*\*|[-*]\s*\*\*)',s): tags.append('H')
        if re.search(pat,s,re.I): tags.append('FLAG')
        if re.search(PR,s,re.I) and re.search(r'sharpe|return|result|performance|strategy|track|accuracy|win',s,re.I): tags.append('PR')
        if re.search(ME,s,re.I): tags.append('M')
        if tags: keep.append(f"[{','.join(tags)}] {s[:260]}")
    paras=[p for p in a.split('\n\n') if p.strip()]
    print(f"=== {r['pitch_id']} s{r['seed']} {fl} {r['finish_reason']}")
    print('\n'.join(keep[:45]))
    print('[END] '+' '.join(paras[-2:])[:500])
