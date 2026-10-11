"""Analysis for 'Can AI spot a false strategy?'
Reads prompts/pitches.json, data/answers.jsonl, data/scores.jsonl.
Writes results/*.csv, results/stats.json and figures. Main analyses use the three main models;
the size comparison adds the three smaller models from the same families."""
import json, os
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results"); os.makedirs(R, exist_ok=True)
FLAG_NAMES = {"F1": "Multiple testing", "F2": "Too little data", "F3": "No trading costs",
              "F4": "Look-ahead bias", "F5": "Survivorship bias", "F6": "Overfitted parameters"}
VISIBLE = {"F2", "F3"}
MAIN = ["qwen2.5:7b", "llama3.1:8b", "gemma2:9b"]
SMALL = ["qwen2.5:3b", "llama3.2:3b", "gemma2:2b"]
LARGE = ["qwen2.5:14b"]
LABEL = {"qwen2.5:7b": "Qwen2.5-7B", "llama3.1:8b": "Llama-3.1-8B", "gemma2:9b": "Gemma-2-9B",
         "qwen2.5:3b": "Qwen2.5-3B", "qwen2.5:14b": "Qwen2.5-14B", "llama3.2:3b": "Llama-3.2-3B", "gemma2:2b": "Gemma-2-2B"}
FAMILY = {"qwen2.5:14b": "Qwen", "qwen2.5:7b": "Qwen", "qwen2.5:3b": "Qwen", "llama3.1:8b": "Llama", "llama3.2:3b": "Llama",
          "gemma2:9b": "Gemma", "gemma2:2b": "Gemma"}

pitches = {p["id"]: p for p in json.load(open(os.path.join(ROOT, "prompts", "pitches.json")))}
S = pd.DataFrame([json.loads(l) for l in open(os.path.join(ROOT, "data", "scores.jsonl"))])
A = pd.DataFrame([json.loads(l) for l in open(os.path.join(ROOT, "data", "answers.jsonl"))])
S = S.merge(A[["model", "pitch_id", "seed", "finish_reason", "tokens"]], on=["model", "pitch_id", "seed"], how="left")
S["control"] = S.pitch_id.map(lambda i: len(pitches[i]["flags"]) == 0)
S["sharpe"] = S.pitch_id.map(lambda i: pitches[i]["sharpe"])
S["hype"] = [int(pr == 1 and (len(d) == 0 or max(d.values()) < 2)) for pr, d in zip(S.praise, S.det)]
S["safe_verdict"] = S.verdict.isin(["test_first", "dont"]).astype(int)
S["refusal"] = (S.usable == 0).astype(int)
models_present = [m for m in MAIN + SMALL + LARGE if m in set(S.model)]

def wilson(k, n):
    return proportion_confint(k, n, method="wilson") if n else (np.nan, np.nan)

def flag_rows(df):
    rows = []
    for _, s in df[~df.control].iterrows():
        for f in pitches[s.pitch_id]["flags"]:
            rows.append({"model": s.model, "pitch_id": s.pitch_id, "seed": s.seed, "flag": f, "usable": s.usable,
                         "score": s.det[f], "visible": int(f in VISIBLE), "multi": int(len(pitches[s.pitch_id]["flags"]) > 1)})
    fl = pd.DataFrame(rows)
    fl["specific"] = (fl.score == 2).astype(int); fl["generic"] = (fl.score == 1).astype(int); fl["missed"] = (fl.score == 0).astype(int)
    return fl

FL_ALL = flag_rows(S)
out = {"answers_per_model": S.groupby("model").size().to_dict()}

# ---------- per-model summary (all six models)
summ = []
for m in models_present:
    a = S[S.model == m]; u = a[a.usable == 1]; f = FL_ALL[FL_ALL.model == m]; fu = f[f.usable == 1]
    summ.append({"model": LABEL[m], "family": FAMILY[m], "size": "main" if m in MAIN else ("small" if m in SMALL else "large"),
                 "answers": len(a), "refusal_rate": a.refusal.mean(),
                 "specific_rate_usable": fu.specific.mean() if len(fu) else np.nan,
                 "generic_rate_usable": fu.generic.mean() if len(fu) else np.nan,
                 "specific_rate_all": f.specific.mean(),
                 "hype_rate_flagged_usable": u[~u.control].hype.mean() if len(u) else np.nan,
                 "method_rate_usable": u.method.mean() if len(u) else np.nan,
                 "safe_verdict_flagged_usable": u[~u.control].safe_verdict.mean() if len(u) else np.nan,
                 "safe_verdict_control_usable": u[u.control].safe_verdict.mean() if len(u[u.control]) else np.nan,
                 "invest_or_cautious_flagged_usable": u[~u.control].verdict.isin(["invest", "cautious"]).mean() if len(u) else np.nan,
                 "false_alarm_rate_usable": u[u.control].false_alarm.astype(float).mean() if len(u[u.control]) else np.nan,
                 "usable_controls": int(len(u[u.control]))})
summ = pd.DataFrame(summ); summ.to_csv(os.path.join(R, "summary_by_model.csv"), index=False)

# ---------- main analyses: main models, usable answers
M = S[S.model.isin(MAIN)].copy(); use = M[M.usable == 1].copy()
fl = FL_ALL[FL_ALL.model.isin(MAIN) & (FL_ALL.usable == 1)].copy()
out["main"] = {"n_answers": len(M), "n_usable": int(M.usable.sum()), "refusals": int(M.refusal.sum()),
               "n_truncated": int((M.finish_reason == "length").sum()), "n_flag_rows_usable": len(fl)}

tab = fl.groupby("flag").agg(n=("score", "size"), specific=("specific", "sum"), generic=("generic", "sum"), missed=("missed", "sum")).reset_index()
tab["name"] = tab.flag.map(FLAG_NAMES); tab["specific_rate"] = tab.specific / tab.n
tab[["ci_lo", "ci_hi"]] = [wilson(k, n) for k, n in zip(tab.specific, tab.n)]
tab.to_csv(os.path.join(R, "detection_by_flag.csv"), index=False)
mt = fl.groupby(["model", "flag"]).agg(n=("specific", "size"), rate=("specific", "mean")).reset_index()
mt.to_csv(os.path.join(R, "detection_by_model_flag.csv"), index=False)

# H1
k = [int(fl[fl.visible == 1].specific.sum()), int(fl[fl.visible == 0].specific.sum())]
n = [int((fl.visible == 1).sum()), int((fl.visible == 0).sum())]
z, p_h1 = proportions_ztest(k, n, alternative="larger")
fl["cluster"] = pd.factorize(fl.pitch_id)[0]
try:
    lm = smf.logit("specific ~ visible + multi + C(model)", data=fl).fit(disp=0, cov_type="cluster", cov_kwds={"groups": fl.cluster})
    h1_logit = {"odds_ratio_visible": float(np.exp(lm.params["visible"])), "p_two_sided": float(lm.pvalues["visible"]),
                "ci_or": [float(np.exp(x)) for x in lm.conf_int().loc["visible"]]}
except Exception as e:
    h1_logit = {"error": str(e)}
out["H1"] = {"visible_rate": k[0] / n[0], "statistical_rate": k[1] / n[1], "k": k, "n": n, "z": z, "p_one_sided": p_h1, "logit": h1_logit}

# H2
pl = use[(~use.control) & use.sharpe.notna()].groupby("pitch_id").agg(sharpe=("sharpe", "first"), hype=("hype", "mean"), n=("hype", "size")).reset_index()
rho, p2 = stats.spearmanr(pl.sharpe, pl.hype)
out["H2"] = {"spearman_rho": rho, "p_one_sided": p2 / 2 if rho > 0 else 1 - p2 / 2, "n_pitches": len(pl)}
pl.to_csv(os.path.join(R, "sharpe_vs_hype.csv"), index=False)

# H3
g, s_ = int(fl.generic.sum()), int(fl.specific.sum())
bt = stats.binomtest(g, g + s_, 0.5, alternative="greater")
out["H3"] = {"generic": g, "specific": s_, "missed": int(fl.missed.sum()), "share_generic_among_mentions": g / (g + s_), "p_one_sided": bt.pvalue}

# H4
fa_, ca_ = use[~use.control], use[use.control]
table = [[int(fa_.safe_verdict.sum()), int(len(fa_) - fa_.safe_verdict.sum())], [int(ca_.safe_verdict.sum()), int(len(ca_) - ca_.safe_verdict.sum())]]
orr, p_h4 = stats.fisher_exact(table, alternative="greater")
out["H4"] = {"safe_rate_flagged": fa_.safe_verdict.mean(), "safe_rate_control": ca_.safe_verdict.mean(), "table": table, "odds_ratio": orr, "p_one_sided": p_h4}

ps = [out[h]["p_one_sided"] for h in ["H1", "H2", "H3", "H4"]]
rej, padj, _, _ = multipletests(ps, method="fdr_bh")
for h, pa, r in zip(["H1", "H2", "H3", "H4"], padj, rej):
    out[h]["p_bh"] = float(pa); out[h]["supported"] = bool(r)

vd = use.assign(group=np.where(use.control, "Control", "Flagged")).groupby(["group", "verdict"]).size().unstack(fill_value=0)
vd.to_csv(os.path.join(R, "verdicts.csv"))
mf = fl[fl.multi == 1].groupby(["model", "pitch_id", "seed"]).specific.min()
fl_all_main = FL_ALL[FL_ALL.model.isin(MAIN)]
out["overall_main"] = {"specific_rate": fl.specific.mean(), "generic_rate": fl.generic.mean(), "missed_rate": fl.missed.mean(),
    "specific_rate_all_answers": fl_all_main.specific.mean(),
    "hype_rate_flagged": fa_.hype.mean(), "praise_rate_flagged": fa_.praise.mean(), "praise_rate_control": ca_.praise.mean(),
    "method_rate_flagged": fa_.method.mean(), "method_rate_control": ca_.method.mean(),
    "invest_or_cautious_flagged": fa_.verdict.isin(["invest", "cautious"]).mean(), "dont_rate_flagged": (fa_.verdict == "dont").mean(),
    "false_alarm_rate": ca_.false_alarm.astype(float).mean(), "false_alarm_n": int(ca_.false_alarm.sum()), "control_usable": len(ca_),
    "multi_all_caught": float(mf.mean()) if len(mf) else None, "n_multi_answers": int(len(mf))}

# ---------- size comparison (pairs by family)
size = []
for fam, big, small in [("Qwen", "qwen2.5:7b", "qwen2.5:3b"), ("Llama", "llama3.1:8b", "llama3.2:3b"), ("Gemma", "gemma2:9b", "gemma2:2b")]:
    for m in (big, small):
        if m not in models_present: continue
        f = FL_ALL[FL_ALL.model == m]; fu = f[f.usable == 1]
        size.append({"family": fam, "model": LABEL[m], "size": "larger" if m == big else "smaller",
                     "refusal_rate": S[S.model == m].refusal.mean(), "specific_usable": fu.specific.mean() if len(fu) else np.nan,
                     "n_flag_rows_usable": len(fu), "specific_all": f.specific.mean()})
for m in ["qwen2.5:14b"]:
    if m in models_present:
        f = FL_ALL[FL_ALL.model == m]; fu = f[f.usable == 1]
        size.append({"family": "Qwen", "model": LABEL[m], "size": "largest", "refusal_rate": S[S.model == m].refusal.mean(),
                     "specific_usable": fu.specific.mean(), "n_flag_rows_usable": len(fu), "specific_all": f.specific.mean()})
size = pd.DataFrame(size); size.to_csv(os.path.join(R, "size_comparison.csv"), index=False)
sm = FL_ALL[FL_ALL.model.isin(SMALL) & (FL_ALL.usable == 1)]
if len(sm) and len(fl):
    kk = [int(fl.specific.sum()), int(sm.specific.sum())]; nn = [len(fl), len(sm)]
    zz, pz = proportions_ztest(kk, nn)
    out["size_test"] = {"larger_rate": kk[0] / nn[0], "smaller_rate": kk[1] / nn[1], "n": nn, "p_two_sided_exploratory": pz}

json.dump(out, open(os.path.join(R, "stats.json"), "w"), indent=2, default=float)

# ---------- figures
plt.rcParams.update({"font.size": 9.5, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
C_SPEC, C_GEN, C_MISS = "#1f5f7a", "#9cc3d5", "#e4e4e4"
t = tab.set_index("flag").sort_values("specific_rate")
fig, ax = plt.subplots(figsize=(6.6, 3.0)); y = np.arange(len(t))
ax.barh(y, t.specific / t.n * 100, color=C_SPEC, label="Named the specific problem")
ax.barh(y, t.generic / t.n * 100, left=t.specific / t.n * 100, color=C_GEN, label="Generic caution only")
ax.barh(y, t.missed / t.n * 100, left=(t.specific + t.generic) / t.n * 100, color=C_MISS, label="Not mentioned")
for yi, (r, nn_) in enumerate(zip(t.specific_rate, t.n)):
    ax.text(r * 100 + 1, yi, f"{r*100:.0f}%", va="center", fontsize=8, color="white" if r > 0.12 else "#333")
ax.set_yticks(y, [f"{FLAG_NAMES[f]} (n={int(nn_)})" for f, nn_ in zip(t.index, t.n)])
ax.set_xlim(0, 100); ax.set_xlabel("Share of answers that addressed the flag (%)")
ax.legend(loc="upper center", bbox_to_anchor=(0.38, -0.2), ncol=3, frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(R, "fig1_detection_by_flag.png"), dpi=220); plt.close(fig)

if len(summ):
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    order = [LABEL[m] for m in MAIN + SMALL + LARGE if m in models_present]
    s2 = summ.set_index("model").loc[order]
    x = np.arange(len(order)); w = 0.38
    ax.bar(x - w/2, s2.refusal_rate * 100, w, color="#b5543c", label="Refused to answer")
    ax.bar(x + w/2, s2.specific_rate_all * 100, w, color=C_SPEC, label="Named the planted flaw (all answers)")
    for xi, (a_, b_) in enumerate(zip(s2.refusal_rate, s2.specific_rate_all)):
        ax.text(xi - w/2, a_ * 100 + 1.5, f"{a_*100:.0f}", ha="center", fontsize=7.5)
        ax.text(xi + w/2, b_ * 100 + 1.5, f"{b_*100:.0f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, order, fontsize=8); ax.set_ylabel("% of answers"); ax.set_ylim(0, 105)
    if len(order) > 3: ax.axvline(2.5, color="#999", lw=0.8, ls=":"); ax.text(2.55, 98, "smaller models", fontsize=7.5, color="#666")
    if len(order) > 6: ax.axvline(5.5, color="#999", lw=0.8, ls=":"); ax.text(5.55, 98, "larger", fontsize=7.5, color="#666")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig2_models.png"), dpi=220); plt.close(fig)

order_v = ["invest", "cautious", "test_first", "dont", "unclear"]
labels_v = {"invest": "Invest", "cautious": "Invest cautiously", "test_first": "Test first", "dont": "Don't invest", "unclear": "No clear verdict"}
vcol = {"invest": "#b5543c", "cautious": "#e3a48a", "test_first": "#9cc3d5", "dont": "#1f5f7a", "unclear": "#e4e4e4"}
vv = vd.reindex(columns=order_v, fill_value=0); vv = vv.div(vv.sum(axis=1), axis=0) * 100; vv = vv.reindex(["Flagged", "Control"])
fig, ax = plt.subplots(figsize=(6.6, 2.3)); left = np.zeros(2)
for o in order_v:
    ax.barh([0, 1], vv[o], left=left, color=vcol[o], label=labels_v[o]); left += vv[o].values
ax.set_yticks([0, 1], ["Flawed pitches", "Clean controls"]); ax.invert_yaxis()
ax.set_xlim(0, 100); ax.set_xlabel("Share of usable answers (%)")
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.5), ncol=5, frameon=False, fontsize=7.5)
fig.tight_layout(); fig.savefig(os.path.join(R, "fig3_verdicts.png"), dpi=220); plt.close(fig)

print(json.dumps(out, indent=1, default=float)); print(tab.to_string()); print(summ.T.to_string()); print(size.to_string())
