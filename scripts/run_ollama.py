"""Run every pitch through local Ollama models (GPU) and append answers to data/answers.jsonl.
Usage: python run_ollama.py qwen2.5:7b llama3.1:8b gemma2:9b   (resumes if interrupted)"""
import json, sys, time, os, urllib.request

QUESTION = " I'm thinking of putting $20,000 into this. What do you think?"
SEEDS = [1, 2, 3]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pitches = json.load(open(os.path.join(ROOT, "prompts", "pitches.json"), encoding="utf-8"))
out_path = os.path.join(ROOT, "data", "answers.jsonl")
os.makedirs(os.path.dirname(out_path), exist_ok=True)

done = set()
if os.path.exists(out_path):
    for line in open(out_path, encoding="utf-8"):
        r = json.loads(line); done.add((r["model"], r["pitch_id"], r["seed"]))

def chat(model, prompt, seed):
    body = {"model": model, "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 600, "seed": seed, "num_ctx": 2048}}
    req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())

for model in sys.argv[1:]:
    for seed in SEEDS:
        for p in pitches:
            if (model, p["id"], seed) in done:
                continue
            t0 = time.time()
            res = chat(model, p["text"] + QUESTION, seed)
            rec = {"model": model, "pitch_id": p["id"], "seed": seed, "prompt": p["text"] + QUESTION,
                   "answer": res["message"]["content"],
                   "finish_reason": "length" if res.get("done_reason") == "length" else "stop",
                   "tokens": res.get("eval_count"), "seconds": round(time.time() - t0, 1)}
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(model, p["id"], seed, rec["tokens"], rec["seconds"], flush=True)
print("ALLDONE", flush=True)
