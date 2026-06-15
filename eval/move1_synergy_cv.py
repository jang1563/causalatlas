#!/usr/bin/env python3
"""Move 1 powering -- decisive recompute: does GEARS's non-additive (synergy) signal GENERALIZE to novel
combos, on the powered leave-combo-out CV held-out set?

Reads the CV held-out predictions (gears_cv_vs_real.csv from move1_gears_cv.py; every combo predicted while
held out, all combo_seen2 = novel combination of seen singles) and recomputes the GEARS-vs-observed-additive
edge-AUROC gap on the synergy subset, with perturbation-cluster bootstrap CI. Compare to the single-split
result that motivated this: held-out synergy gap was +.104 [-.03,+.22] (NS, n=26 perturbations).

  python move1_synergy_cv.py --cv /tmp/gears_cv_vs_real.csv \
      --marginal results/gears_norman/labeled_marginal.csv
"""
import argparse, re, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", default="/tmp/gears_cv_vs_real.csv")
    ap.add_argument("--marginal", default="results/gears_norman/labeled_marginal.csv")
    ap.add_argument("--thr", type=float, default=0.33, help="synergy threshold on |real-additive| (0.33 = the single-split 75th pct, for comparability)")
    ap.add_argument("--delta", type=float, default=0.25)
    a = ap.parse_args()

    cv = pd.read_csv(a.cv).dropna(subset=["gears_log2FC"])
    m = pd.read_csv(a.marginal); m = m[np.isclose(m["delta"], a.delta)]
    single = {(r.target, r.response_id): r.log2FC for r in m[m.kind == "single"].itertuples()}
    creal  = {(r.target, r.response_id): r.log2FC for r in m[m.kind == "combo"].itertuples()}

    def addlfc(p, gn):
        ps = re.split(r"[+_]", p); v = [single.get((x, gn)) for x in ps]
        return float(np.sum(v)) if len(ps) > 1 and all(x is not None for x in v) else np.nan
    cv["add_lfc"]  = [addlfc(p, gn) for p, gn in zip(cv.perturbation, cv.gene)]
    cv["real_lfc"] = [creal.get((p, gn), np.nan) for p, gn in zip(cv.perturbation, cv.gene)]
    cv = cv.dropna(subset=["add_lfc", "real_lfc"]).copy()
    cv["y"] = (cv.real_call == "effect").astype(int)
    cv["inter"] = (cv.real_lfc - cv.add_lfc).abs()

    def auc(sub, col):
        return roc_auc_score(sub.y, sub[col].abs()) if sub.y.nunique() == 2 else np.nan
    def gap_ci(sub, B=2000, seed=0):
        rng = np.random.default_rng(seed); perts = sub.perturbation.unique()
        by = {p: sub[sub.perturbation == p] for p in perts}
        pt = auc(sub, "gears_log2FC") - auc(sub, "add_lfc"); vals = []
        for _ in range(B):
            bs = pd.concat([by[p] for p in rng.choice(perts, len(perts), replace=True)], ignore_index=True)
            g, ad = auc(bs, "gears_log2FC"), auc(bs, "add_lfc")
            if np.isfinite(g) and np.isfinite(ad): vals.append(g - ad)
        return pt, np.percentile(vals, 2.5), np.percentile(vals, 97.5)

    print(f"CV held-out: {cv.perturbation.nunique()} combos (all combo_seen2), {len(cv)} decidable edges")
    print(f"GEARS held-out combo accuracy: {(((cv.gears_log2FC.abs()>=a.delta)==(cv.real_call=='effect')).mean()):.3f}\n")
    print(f"{'subset':30s} {'n':>5s} {'nP':>4s} {'maj':>5s} | {'GEARS_AUROC':>11s} {'ADD_AUROC':>10s} {'G-A gap [95% CI]':>22s}")
    for label, sub in [("ALL held-out combos", cv),
                       ("ADDITIVE edges (|inter|<thr)", cv[cv.inter < a.thr]),
                       ("SYNERGY edges (|inter|>=thr)", cv[cv.inter >= a.thr])]:
        if len(sub) < 20 or sub.y.nunique() < 2:
            print(f"{label:30s} n={len(sub)} too small"); continue
        g, ad = auc(sub, "gears_log2FC"), auc(sub, "add_lfc")
        pt, lo, hi = gap_ci(sub)
        sig = "  CI excl 0" if not (lo < 0 < hi) else "  NS"
        print(f"{label:30s} {len(sub):>5d} {sub.perturbation.nunique():>4d} {max(sub.y.mean(),1-sub.y.mean()):>5.2f} | "
              f"{g:>11.3f} {ad:>10.3f} {pt:>+8.3f} [{lo:+.3f},{hi:+.3f}]{sig}")
    print(f"\n(compare single-split held-out synergy: +.104 [-.027,+.219] NS, n=26 perts)")
    print(f"(synergy threshold |real-additive| >= {a.thr}; also report CV-own 75th pct = {cv.inter.quantile(0.75):.3f})")

if __name__ == "__main__":
    main()
