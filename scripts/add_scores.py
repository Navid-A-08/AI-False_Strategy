"""Append manual scores (compact lines) and auto-score refusals for one model.
usage: python add_scores.py answers_file model < manual.txt
manual line: PID seed det(F1=2,F2=1 or -) verdict praise method false_alarm(- or 0/1)"""
import json,sys
sys.path.insert(0,'scripts')
from refusal import is_refusal
f,model=sys.argv[1],sys.argv[2]
P={p['id']:p for p in json.load(open('prompts/pitches.json'))}
man={}
for line in sys.stdin:
    t=line.split()
    if not t: continue
    pid,seed,det,ver,pr,me,fa=t[:7]
    d={} if det=='-' else {k:int(v) for k,v in (x.split('=') for x in det.split(','))}
    man[(pid,int(seed))]=dict(det=d,verdict=ver,praise=int(pr),method=int(me),false_alarm=None if fa=='-' else int(fa),usable=1)
out=open('data/scores.jsonl','a')
n=0
for r in map(json.loads,open(f)):
    k=(r['pitch_id'],r['seed']); flags=P[r['pitch_id']]['flags']
    if is_refusal(r['answer'],r['tokens']):
        s=dict(det={x:0 for x in flags},verdict='unclear',praise=0,method=0,false_alarm=None if flags else 0,usable=0,note='refusal')
    else:
        s=man[k]; assert set(s['det'])==set(flags),(k,s['det'],flags)
    out.write(json.dumps({'model':model,'pitch_id':k[0],'seed':k[1],**s})+'\n'); n+=1
print(n)
