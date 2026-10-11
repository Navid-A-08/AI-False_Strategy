"""Draw a stratified random sample of 30 usable answers for a human second scorer.
Writes results/human_audit_sample.csv with the answer text, the first scorer's codes hidden, and empty columns to fill."""
import json, random, csv, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
random.seed(2026)
P = {p["id"]: p for p in json.load(open(os.path.join(ROOT, "prompts", "pitches.json")))}
A = {(r["model"], r["pitch_id"], r["seed"]): r for r in map(json.loads, open(os.path.join(ROOT, "data", "answers.jsonl")))}
S = [s for s in map(json.loads, open(os.path.join(ROOT, "data", "scores.jsonl"))) if s["usable"] == 1]
by_model = {}
for s in S: by_model.setdefault(s["model"], []).append(s)
pick = []
for m, lst in sorted(by_model.items()):
    pick += random.sample(lst, min(5, len(lst)))
random.shuffle(pick)
with open(os.path.join(ROOT, "results", "human_audit_sample.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["item", "model", "pitch_id", "seed", "planted_flags", "pitch", "answer",
                "your_detection (e.g. F1=2;F2=1)", "your_verdict", "your_praise(0/1)", "your_method(0/1)", "your_false_alarm(controls)"])
    for i, s in enumerate(pick, 1):
        r = A[(s["model"], s["pitch_id"], s["seed"])]
        w.writerow([i, s["model"], s["pitch_id"], s["seed"], ",".join(P[s["pitch_id"]]["flags"]) or "control",
                    P[s["pitch_id"]]["text"], r["answer"], "", "", "", "", ""])
print(len(pick))
