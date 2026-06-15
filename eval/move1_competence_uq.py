#!/usr/bin/env python3
"""Dig: can the per-input COMPETENCE signal (predict where the FM is wrong) be pushed above the ~0.70 ceiling
with inference-available features -- specifically the Layer-A additive-disagreement signal?

Hypothesis: GEARS's errors are SYSTEMATIC (shrinkage + off-additive), so an external classifier trained on
held-out errors, fed |gears - additive| and call-disagreement, should beat the basic trust-head (|pred|+regime
~0.70). All features are inference-available (the additive baseline is computed from known single perturbations;
no real-experiment info). Cross-fit by perturbation (no edge leakage). Combos only (additive defined there).

  python move1_competence_uq.py
"""
import re, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss

g = pd.read_csv("results/gears_norman/gears_vs_real_all.csv").dropna(subset=["gears_log2FC"]).drop_duplicates(["perturbation","gene"])
m = pd.read_csv("results/gears_norman/labeled_marginal.csv"); m = m[np.isclose(m["delta"], 0.25)]
single = {(r.target, r.response_id): r.log2FC for r in m[m.kind=="single"].itertuples()}
def addlfc(p, gn):
    ps = re.split(r"[+_]", p); v = [single.get((x, gn)) for x in ps]
    return float(np.sum(v)) if len(ps) > 1 and all(x is not None for x in v) else np.nan
c = g[g.regime.str.startswith("combo")].copy()
c["add_lfc"] = [addlfc(p, gn) for p, gn in zip(c.perturbation, c.gene)]
c = c.dropna(subset=["add_lfc"]).copy()
c["fm_wrong"] = (c.gears_call != c.real_call).astype(int)             # TARGET: FM wrong (the verify signal)
DELTA = 0.25
fm = c.gears_log2FC.to_numpy(); ad = c.add_lfc.to_numpy()
reg_oh = pd.get_dummies(c.regime).to_numpy().astype(float)
F = {
  "base |pred|":            np.column_stack([np.abs(fm)]),
  "+regime":                np.column_stack([np.abs(fm), reg_oh]),
  "+additive-disagreement": np.column_stack([np.abs(fm), reg_oh, np.abs(ad), np.abs(fm-ad),
                                             (np.sign(fm)!=np.sign(ad)).astype(float),
                                             ((np.abs(fm)>=DELTA)!=(np.abs(ad)>=DELTA)).astype(float)]),
}
y = c.fm_wrong.to_numpy(); groups = c.perturbation.to_numpy()
print(f"combos: {len(c)} edges / {c.perturbation.nunique()} perts | FM-wrong base rate {y.mean():.3f}")
print(f"{'feature set':26s} {'GBM AUC':>9s} {'LogReg AUC':>11s} {'Brier(GBM)':>11s}")
gkf = GroupKFold(n_splits=5)
for name, X in F.items():
    for clf_name, mk in [("gbm", lambda: HistGradientBoostingClassifier(max_depth=3, max_iter=300, random_state=0)),
                          ("lr", lambda: LogisticRegression(max_iter=1000))]:
        p = np.zeros(len(y))
        for tr, te in gkf.split(X, y, groups):
            p[te] = mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        if clf_name == "gbm": auc_g, br = roc_auc_score(y, p), brier_score_loss(y, p)
        else: auc_l = roc_auc_score(y, p)
    print(f"{name:26s} {auc_g:>9.3f} {auc_l:>11.3f} {br:>11.3f}")
print("\n(>0.70 on +additive-disagreement = the competence signal IS improvable with inference-available features"
      "\n -> the #1 lever is engineerable, not a wall. theoretical ceiling < 1: cannot perfectly predict errors.)")
