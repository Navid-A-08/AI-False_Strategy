"""After the human audit sheet is filled in, compute agreement with the first scorer (specific-detection and verdict)."""
import json, csv, os
from sklearn.metrics import cohen_kappa_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = {(s["model"], s["pitch_id"], int(s["seed"])): s for s in map(json.loads, open(os.path.join(ROOT, "data", "scores.jsonl")))}
a, b, va, vb = [], [], [], []
for row in csv.DictReader(open(os.path.join(ROOT, "results", "human_audit_sample.csv"), encoding="utf-8")):
    s = S[(row["model"], row["pitch_id"], int(row["seed"]))]
    det = row["your_detection (e.g. F1=2;F2=1)"].strip()
    if det:
        h = {k: int(v) for k, v in (x.split("=") for x in det.split(";"))}
        for f, v in s["det"].items():
            a.append(int(v == 2)); b.append(int(h.get(f, 0) == 2))
    if row["your_verdict"].strip():
        va.append(s["verdict"]); vb.append(row["your_verdict"].strip())
print("specific detection: agreement %.2f, kappa %.2f, n=%d" % (sum(x == y for x, y in zip(a, b)) / len(a), cohen_kappa_score(a, b), len(a)))
print("verdict: agreement %.2f, kappa %.2f, n=%d" % (sum(x == y for x, y in zip(va, vb)) / len(va), cohen_kappa_score(va, vb), len(va)))
