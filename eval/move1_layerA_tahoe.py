#!/usr/bin/env python3
"""Move 1, Layer A -- Tahoe / Arc-STATE instantiation (Part C, drug->gene, zeroshot held-out).

Same measure as the Norman/GEARS run (move1_layerA.py), 2nd substrate + modality:
- PRIMARY  = edge-level AUROC of |STATE predicted log2FC| discriminating effect vs no-effect.
- SECONDARY = cosine-PDS (rescale-invariant VCC discrimination).
- BASELINES = no-change (predict zero effect; THE clean no-peek simple baseline for drugs, since additive
              is undefined for drug perturbations) + mean (in-sample per-gene mean of the real effect, an
              OPTIMISTIC floor that peeks at test; labelled as such). additive: n/a (drugs).
- UNCERTAINTY = perturbation-cluster bootstrap, optimized (precomputed per-perturbation numpy arrays;
              no pandas concat) for the 2.27M-row table.

Input table (exported Tahoe / Arc-STATE comparison CSV):
  perturbation (drug@dose|cellline), gene (anon g0..), gears_log2FC (= STATE pred), regime (=cell line),
  gears_call, real_label, real_call, gears_correct, raw_log2FC (real), raw_se, raw_q, n_trt, n_cntrl.

  python move1_layerA_tahoe.py --table /tmp/tahoe_vs_real.csv --out results/move1
"""
import os, json, argparse
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

PRIMARY_DELTA = 0.25
B_ACC = 2000
B_AUC = 250

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="/tmp/tahoe_vs_real.csv")
    ap.add_argument("--out", default="results/move1")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t = pd.read_csv(a.table)
    print(f"[tahoe] {t.shape[0]} edges, {t['perturbation'].nunique()} perturbations (zeroshot), "
          f"{t['gene'].nunique()} genes, regime(s)={sorted(t['regime'].unique())[:3]}")

    real_eff = (t["real_call"].to_numpy() == "effect")
    s_state = t["gears_log2FC"].astype(float).abs().to_numpy()
    pred_mean = t.groupby("gene")["raw_log2FC"].transform("mean")          # in-sample per-gene mean (optimistic)
    s_mean = pred_mean.abs().to_numpy()
    call_mean  = s_mean  >= PRIMARY_DELTA
    correct = {
        # STATE uses its OWN call (pred q<0.05 & |log2FC|>=delta = table's gears_call), NOT a naive magnitude
        # threshold -- giving STATE its best foot forward. no-change/mean have no significance filter.
        "STATE":     (t["gears_correct"].astype(str) == "True").to_numpy(),
        "no_change": (~real_eff),                                          # no-change always predicts no_effect
        "mean":      (call_mean  == real_eff),                             # mean: |per-gene mean of real| >= delta
    }

    # perturbation grouping (row indices per perturbation)
    perts, codes = np.unique(t["perturbation"].to_numpy(), return_inverse=True)
    P = len(perts)
    order = np.argsort(codes, kind="stable")
    bnd = np.searchsorted(codes[order], np.arange(P + 1))
    pert_rows = [order[bnd[i]:bnd[i + 1]] for i in range(P)]
    ns = np.array([len(r) for r in pert_rows])

    def boot_acc(corr, B=B_ACC, seed=0):
        cs = np.array([corr[r].sum() for r in pert_rows], float)
        rng = np.random.default_rng(seed)
        point = cs.sum() / ns.sum()
        vals = [cs[i].sum() / ns[i].sum() for i in (rng.integers(0, P, P) for _ in range(B))]
        return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    y_by = [real_eff[r].astype(int) for r in pert_rows]
    def boot_auroc(score, B=B_AUC, seed=0):
        sc_by = [score[r] for r in pert_rows]
        point = roc_auc_score(real_eff.astype(int), score)
        rng = np.random.default_rng(seed); vals = []
        for _ in range(B):
            idx = rng.integers(0, P, P)
            yy = np.concatenate([y_by[i] for i in idx]); ss = np.concatenate([sc_by[i] for i in idx])
            if len(np.unique(yy)) == 2: vals.append(roc_auc_score(yy, ss))
        return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    def cosine_pds(predmat, truemat):
        def unit(M):
            n = np.linalg.norm(M, axis=1, keepdims=True); n[n == 0] = 1.0; return M / n
        dist = 1.0 - unit(predmat) @ unit(truemat).T
        ranks = np.array([int(np.where(np.argsort(dist[i]) == i)[0][0]) + 1 for i in range(dist.shape[0])])
        return float(np.mean(1.0 - (ranks - 1) / (dist.shape[0] - 1)))

    base = t["real_call"].eq("no_effect").mean()
    print(f"\n{'='*92}\nMOVE 1 LAYER-A  |  Arc-STATE vs simple baselines on Tahoe zeroshot ({P} novel drugs, 1 cell line C32)"
          f"\n{'='*92}")
    print(f"no_effect base rate (= no-change accuracy ceiling for a constant predictor): {base:.3f}")
    print(f"\n{'policy':12s} | {'call_acc@0.25 [95% CI]':>26s} | {'edge-AUROC [95% CI]':>26s}")
    report = {"n_pert": int(P), "n_edge": int(len(t)), "no_effect_base_rate": float(base),
              "primary_delta": PRIMARY_DELTA, "acc": {}, "auroc": {}, "pds": {}}
    for name in ["STATE", "no_change", "mean"]:
        ac = boot_acc(correct[name])
        report["acc"][name] = ac
        if name == "no_change":
            au = (0.5, np.nan, np.nan)                                     # constant predictor cannot rank
        else:
            au = boot_auroc(s_state if name == "STATE" else s_mean)
        report["auroc"][name] = au
        fa = f"{ac[0]:.3f} [{ac[1]:.3f},{ac[2]:.3f}]"
        fu = "0.500 (constant)" if name == "no_change" else f"{au[0]:.3f} [{au[1]:.3f},{au[2]:.3f}]"
        print(f"{name:12s} | {fa:>26s} | {fu:>26s}")

    oc = boot_auroc(t["raw_log2FC"].abs().to_numpy())                       # oracle-magnitude AUROC ceiling
    report["auroc"]["|real|_oracle_ceiling"] = oc
    print(f"{'|real|ceil':12s} | {'(oracle magnitude)':>26s} | {oc[0]:.3f} [{oc[1]:.3f},{oc[2]:.3f}]"
          f"   <- even oracle |real| barely beats 0.5: the sceptre label is significance-based, not magnitude")

    # cosine-PDS (dense perturbation x gene matrices)
    pm = t.pivot_table(index="perturbation", columns="gene", values="gears_log2FC", aggfunc="mean").fillna(0.0)
    tm = t.pivot_table(index="perturbation", columns="gene", values="raw_log2FC", aggfunc="mean").fillna(0.0)
    tm = tm.reindex(index=pm.index, columns=pm.columns).fillna(0.0)
    pds_state = cosine_pds(pm.to_numpy(), tm.to_numpy())
    mean_vec = tm.to_numpy().mean(axis=0, keepdims=True).repeat(pm.shape[0], axis=0)   # identical mean profile
    pds_mean = cosine_pds(mean_vec, tm.to_numpy())
    report["pds"] = {"STATE": pds_state, "mean": pds_mean}
    print(f"\ncosine-PDS (1=perfect, ~0.5 random):  STATE={pds_state:.3f}   mean-profile={pds_mean:.3f}")

    print(f"\nVERDICT: STATE call-acc {report['acc']['STATE'][0]:.3f} vs no-change floor {base:.3f} "
          f"-> {'BELOW floor' if report['acc']['STATE'][0] < base else 'above floor'}; "
          f"discrimination AUROC {report['auroc']['STATE'][0]:.3f}.")
    with open(os.path.join(a.out, "layerA_tahoe.json"), "w") as f:
        json.dump(report, f, indent=2, default=lambda o: None if (isinstance(o, float) and not np.isfinite(o)) else o)
    print(f"[saved] {os.path.join(a.out, 'layerA_tahoe.json')}")

if __name__ == "__main__":
    main()
