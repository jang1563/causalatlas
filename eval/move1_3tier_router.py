#!/usr/bin/env python3
"""3-tier orchestrator -- the DEPLOYABLE learned add-vs-FM router head.

K1 showed: routing between the two FREE options (additive, FM) has large headroom (oracle-free 0.921 vs
always-additive 0.825), but the naive inference signal |fm_pred-add_pred| FAILS (0.794 < additive). So the
deployable component must be a LEARNED per-edge router. This trains two cross-fit (GroupKFold by perturbation,
no test-edge leakage) classifiers -- P(additive correct) and P(FM correct) from INFERENCE-ONLY features -- and
routes each edge to the higher predicted-correct option. We measure how much of the free headroom it captures.
(If this also fails, the honest conclusion is that which-of-add-or-FM-is-right is not predictable from the
predictions alone -- a real negative for the deployable free-routing tier.)

  python move1_3tier_router.py --cv results/move1/gears_cv_vs_real.csv --marginal results/gears_norman/labeled_marginal.csv
"""
import argparse, re, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

DELTA = 0.25

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", default="results/move1/gears_cv_vs_real.csv")
    ap.add_argument("--marginal", default="results/gears_norman/labeled_marginal.csv")
    a = ap.parse_args()
    cv = pd.read_csv(a.cv).dropna(subset=["gears_log2FC"])
    m = pd.read_csv(a.marginal); m = m[np.isclose(m["delta"], DELTA)]
    single = {(r.target, r.response_id): r.log2FC for r in m[m.kind == "single"].itertuples()}
    def addlfc(p, gn):
        ps = re.split(r"[+_]", p); v = [single.get((x, gn)) for x in ps]
        return float(np.sum(v)) if len(ps) > 1 and all(x is not None for x in v) else np.nan
    cv["add_lfc"] = [addlfc(p, gn) for p, gn in zip(cv.perturbation, cv.gene)]
    cv = cv.dropna(subset=["add_lfc"]).copy()
    real_eff = (cv.real_call == "effect").to_numpy()
    fm = cv.gears_log2FC.to_numpy(); ad = cv.add_lfc.to_numpy()
    fm_correct  = ((np.abs(fm) >= DELTA) == real_eff)
    add_correct = ((np.abs(ad) >= DELTA) == real_eff)
    groups = cv.perturbation.to_numpy()

    # INFERENCE-ONLY features (no real_lfc): the two predictions + derived
    X = np.column_stack([
        np.abs(fm), np.abs(ad), np.abs(fm - ad), fm, ad,
        (np.abs(fm) >= DELTA).astype(float), (np.abs(ad) >= DELTA).astype(float),
        (np.sign(fm) == np.sign(ad)).astype(float),
        ((np.abs(fm) >= DELTA) == (np.abs(ad) >= DELTA)).astype(float),   # calls agree
    ])
    # cross-fit P(add correct) and P(fm correct)
    p_add = np.zeros(len(cv)); p_fm = np.zeros(len(cv))
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, add_correct, groups):
        p_add[te] = HistGradientBoostingClassifier(max_depth=3, max_iter=200, random_state=0
                      ).fit(X[tr], add_correct[tr].astype(int)).predict_proba(X[te])[:, 1]
        p_fm[te]  = HistGradientBoostingClassifier(max_depth=3, max_iter=200, random_state=0
                      ).fit(X[tr], fm_correct[tr].astype(int)).predict_proba(X[te])[:, 1]
    use_fm = p_fm > p_add
    routed_correct = np.where(use_fm, fm_correct, add_correct)

    add_acc = add_correct.mean(); fm_acc = fm_correct.mean()
    oracle_free = (add_correct | fm_correct).mean()
    routed_acc = routed_correct.mean()
    # cluster-bootstrap CI: routed - always-additive, and headroom captured
    P = np.unique(groups); idx = {p: np.where(groups == p)[0] for p in P}
    rng = np.random.default_rng(0); gaps = []
    for _ in range(2000):
        take = np.concatenate([idx[p] for p in rng.choice(P, len(P), replace=True)])
        gaps.append(routed_correct[take].mean() - add_correct[take].mean())
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    print(f"3-tier LEARNED router (cross-fit by perturbation): {len(cv)} edges / {len(P)} combos")
    print(f"  always-additive   {add_acc:.3f}")
    print(f"  trust-all-FM      {fm_acc:.3f}")
    print(f"  LEARNED router    {routed_acc:.3f}   (routes {use_fm.mean():.0%} to FM)")
    print(f"  oracle-free       {oracle_free:.3f}   (ceiling)")
    print(f"  routed - additive  {routed_acc-add_acc:+.3f} [{lo:+.3f},{hi:+.3f}]  -> {'PASS (CI excl 0)' if lo>0 else 'FAIL'}")
    head = oracle_free - add_acc
    print(f"  headroom captured  {(routed_acc-add_acc)/head:.0%} of the {head:.3f} free oracle headroom")
    # how separable is which-is-right? AUC of (p_fm-p_add) for predicting fm-strictly-better on disagreement edges
    from sklearn.metrics import roc_auc_score
    disagree = fm_correct != add_correct
    if disagree.sum() > 10:
        y = fm_correct[disagree].astype(int)   # 1 = FM right (add wrong) on disagreement edges
        auc = roc_auc_score(y, (p_fm - p_add)[disagree])
        print(f"  router AUC on disagreement edges (which one is right): {auc:.3f}  (0.5=unpredictable; n={disagree.sum()})")

if __name__ == "__main__":
    main()
