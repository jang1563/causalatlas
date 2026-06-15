#!/usr/bin/env python3
"""Close the deployable loop: does the BUILDABLE 0.88 competence signal drive near-ORACLE verification allocation,
beating the 0.70 (magnitude+regime) signal and no-signal? (LLM-free: Verify-or-Trust already showed the LLM
follows a supplied signal 94-99%, so we isolate the SIGNAL's value.)

Policy: verify the top-fraction of edges by competence p(FM-wrong); verified edges = correct (cost lambda),
unverified = the FM's call (correct iff FM right). net = accuracy - lambda*verify-rate. Signals compared:
random / 0.70 predictor (|pred|+regime) / 0.88 predictor (+additive-disagreement) / oracle (true FM-wrong).
All competence predictors are cross-fit by perturbation (no leakage) and inference-available.

  python move1_competence_close.py
"""
import re, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

g = pd.read_csv("results/gears_norman/gears_vs_real_all.csv").dropna(subset=["gears_log2FC"]).drop_duplicates(["perturbation","gene"])
m = pd.read_csv("results/gears_norman/labeled_marginal.csv"); m = m[np.isclose(m["delta"], 0.25)]
single = {(r.target, r.response_id): r.log2FC for r in m[m.kind=="single"].itertuples()}
def addlfc(p, gn):
    ps = re.split(r"[+_]", p); v = [single.get((x, gn)) for x in ps]
    return float(np.sum(v)) if len(ps) > 1 and all(x is not None for x in v) else np.nan
c = g[g.regime.str.startswith("combo")].copy()
c["add_lfc"] = [addlfc(p, gn) for p, gn in zip(c.perturbation, c.gene)]
c = c.dropna(subset=["add_lfc"]).copy()
fm = c.gears_log2FC.to_numpy(); ad = c.add_lfc.to_numpy(); DELTA = 0.25
y = (c.gears_call != c.real_call).to_numpy().astype(int)              # FM-wrong (truth)
reg = pd.get_dummies(c.regime).to_numpy().astype(float); groups = c.perturbation.to_numpy()
X70 = np.column_stack([np.abs(fm), reg])
X88 = np.column_stack([np.abs(fm), reg, np.abs(ad), np.abs(fm-ad),
                       (np.sign(fm)!=np.sign(ad)).astype(float), ((np.abs(fm)>=DELTA)!=(np.abs(ad)>=DELTA)).astype(float)])
def crossfit(X):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        p[te] = HistGradientBoostingClassifier(max_depth=3, max_iter=300, random_state=0).fit(X[tr], y[tr]).predict_proba(X[te])[:,1]
    return p
p70, p88 = crossfit(X70), crossfit(X88)
sigs = {"random": np.random.default_rng(0).random(len(y)), "signal-0.70": p70, "signal-0.88": p88, "oracle": y + 1e-6*np.random.default_rng(1).random(len(y))}
print(f"combos {len(c)} | FM-wrong rate {y.mean():.3f} | signal AUC: 0.70={roc_auc_score(y,p70):.3f} 0.88={roc_auc_score(y,p88):.3f}")

def net_at(sig, frac, lam):
    k = int(frac*len(y)); S = np.argsort(-sig)[:k]; mask = np.zeros(len(y), bool); mask[S]=True
    acc = 1 - (y[~mask].sum())/len(y)                                  # verified correct; unverified correct iff FM right
    return acc - lam*frac, (y[mask].sum()/max(1,y.sum()))             # net, vRecall(of FM-wrong)

print(f"\nnet@lambda0.5 (acc - 0.5*verify_rate) by verify-budget:")
print(f"{'verify%':>8s} " + " ".join(f"{s:>12s}" for s in sigs))
for frac in [0.0, 0.1, 0.2, 0.27, 0.4]:
    row = []
    for s, sig in sigs.items():
        nt, _ = net_at(sig, frac, 0.5); row.append(nt)
    print(f"{frac:>8.0%} " + " ".join(f"{v:>12.3f}" for v in row))
print(f"\nvRecall (FM-wrong caught) @ verify-budget = base rate ({y.mean():.0%}):")
for s, sig in sigs.items():
    _, vr = net_at(sig, y.mean(), 0.5); print(f"   {s:12s} vRecall {vr:.2f}")
print("\n(0.88 frontier approaching oracle and beating 0.70/random = a BUILDABLE, inference-available signal"
      "\n drives near-oracle verification allocation -> the deployable loop closes; the #1 lever is engineerable.)")
