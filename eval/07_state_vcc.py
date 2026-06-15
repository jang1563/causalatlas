#!/usr/bin/env python3
"""
CausalAtlas Phase 3 - Step 07: assemble the STATE/VCC calibrated-orchestration substrate from STATE inference
output + the VCC real answer-key. Flagship sibling of 06_gears_norman.py. Outputs state_vs_real.csv with
the same column names used by the verification-allocation benchmark (gears_* = "the SFM"). ALSO computes
the per-perturbation STATE reliability gradient + the BIMODAL GATE: if STATE is
uniformly (un)reliable across perts, the calibration eval can't show a win -> stop and rethink (pre-flight review).

Run (post-infer):
  PYTHONNOUSERSITE=1 ~/envs/state/bin/python 07_state_vcc.py \
    --preds ~/CausalAtlas/results/state_vcc_esm/val_preds.h5ad \
    --real  ~/CausalAtlas/results/state_vcc_esm/vcc_val_labeled.csv \
    --outdir ~/CausalAtlas/results/state_vcc_esm --delta 0.25 --pert-col target_gene --control non-targeting
"""
import argparse, os, sys
import numpy as np, pandas as pd

def log(*a): print("[state07]", *a, file=sys.stderr, flush=True)

def get_pred_matrix(ad_obj):
    """Find STATE's predicted-expression matrix: prefer obsm['X_state_pred'], else a non-X obsm, else .X."""
    import scipy.sparse as sp
    for k in ["X_state_pred", "X_pred", "state_pred", "preds"]:
        if k in ad_obj.obsm:
            M = ad_obj.obsm[k]; return (M.toarray() if sp.issparse(M) else np.asarray(M)), f"obsm['{k}']"
    M = ad_obj.X
    return (M.toarray() if sp.issparse(M) else np.asarray(M)), ".X"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="STATE infer output h5ad (val_preds.h5ad)")
    ap.add_argument("--real", required=True, help="vcc_val_labeled.csv (real equivalence-test labels + raw stats)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--delta", type=float, default=0.25)
    ap.add_argument("--pert-col", default="target_gene")
    ap.add_argument("--control", default="non-targeting")
    ap.add_argument("--rel-thresh", type=float, default=0.667, help="reliability split (EV-optimal verify boundary)")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    import anndata as ad
    EPS = 1e-2

    P = ad.read_h5ad(a.preds)
    log("preds:", P.shape, "| obsm keys:", list(P.obsm.keys()), "| layers:", list(P.layers.keys()))
    X, src = get_pred_matrix(P)
    log(f"prediction matrix from {src}: shape={X.shape} max={X.max():.3f} min={X.min():.3f} mean={X.mean():.3f}")
    # detect log1p space (trained on log1p .X) -> expm1 to linear before log2FC, to match real labels (linear)
    is_log = X.max() < 20
    Xl = np.expm1(X) if is_log else X
    log(f"is_log1p={is_log} -> using {'expm1(linear)' if is_log else 'raw'} for log2FC")
    genes = np.array(P.var_names)
    g2i = {g: i for i, g in enumerate(genes)}
    pcol = a.pert_col if a.pert_col in P.obs else next(c for c in ["target_gene","gene"] if c in P.obs)
    tg = P.obs[pcol].astype(str).values
    cmask = tg == a.control
    if cmask.sum() == 0: log("WARN: no control cells in preds; will fall back to overall mean")
    ctrl_mean = Xl[cmask].mean(0) if cmask.sum() else Xl.mean(0)

    real = pd.read_csv(a.real)   # cols: target, response_id, log2FC, se, q, label, n_trt, n_cntrl, delta
    real = real[np.isclose(real["delta"], a.delta)] if "delta" in real else real
    readouts = set(real["response_id"].unique())          # restrict to labeled (HVG) readouts
    real_call = {(r.target, r.response_id): ("effect" if r.label == "POSITIVE" else
                 ("no_effect" if r.label == "TESTED_NEGATIVE" else "untested")) for r in real.itertuples()}
    real_label = {(r.target, r.response_id): r.label for r in real.itertuples()}

    perts = [p for p in pd.unique(tg) if p != a.control]
    rid = [g for g in genes if g in readouts]; ridx = [g2i[g] for g in rid]
    log(f"{len(perts)} perts x {len(rid)} readout genes")

    rows = []; self_kd = []
    for p in perts:
        pm = tg == p
        if pm.sum() == 0: continue
        pmean = Xl[pm].mean(0)
        lfc = np.log2((pmean[ridx] + EPS) / (ctrl_mean[ridx] + EPS))
        for j, g in enumerate(rid):
            key = (p, g)
            if key not in real_call or real_call[key] == "untested":
                continue                                   # only decidable edges (POSITIVE/TESTED_NEGATIVE)
            scall = "effect" if abs(lfc[j]) >= a.delta else "no_effect"
            rows.append((p, g, float(lfc[j]), scall, real_label[key], real_call[key],
                         scall == real_call[key]))
        # STATE self-knockdown sanity (does STATE predict the target gene down?)
        if p in g2i:
            self_kd.append((p, float(np.log2((pmean[g2i[p]] + EPS) / (ctrl_mean[g2i[p]] + EPS)))))

    df = pd.DataFrame(rows, columns=["perturbation","gene","gears_log2FC","gears_call",
                                     "real_label","real_call","gears_correct"])
    # --- per-perturbation reliability gradient (the BIMODAL GATE) ---
    rel = df.groupby("perturbation")["gears_correct"].mean().rename("reliability")
    df = df.merge(rel, on="perturbation")
    df["regime"] = np.where(df["reliability"] < a.rel_thresh, "unreliable", "reliable")
    df.to_csv(os.path.join(a.outdir, "state_vs_real.csv"), index=False)

    # --- diagnostics / gate ---
    log(f"\n=== STATE self-knockdown sanity (target gene should be DOWN) ===")
    if self_kd:
        skd = pd.DataFrame(self_kd, columns=["pert","self_log2FC"])
        log(f"  {len(skd)} testable; median self log2FC={skd.self_log2FC.median():+.3f}; "
            f"fraction negative={ (skd.self_log2FC<0).mean():.0%}")
    log(f"\n=== BIMODAL GATE: per-perturbation STATE reliability distribution ===")
    rq = rel.quantile([0,.1,.25,.5,.75,.9,1.0])
    log("  " + "  ".join(f"p{int(q*100)}={v:.2f}" for q, v in rq.items()))
    n_unrel = (rel < a.rel_thresh).sum(); n_rel = (rel >= a.rel_thresh).sum()
    log(f"  perts reliable(>= {a.rel_thresh}): {n_rel}  | unreliable(< {a.rel_thresh}): {n_unrel}")
    log(f"  overall STATE accuracy: {df['gears_correct'].mean():.1%}  | decidable edges: {len(df)}")
    GATE = (n_rel >= 5 and n_unrel >= 5)
    log(f"\n  >>> BIMODAL GATE {'PASS' if GATE else 'FAIL'}: need >=5 perts in BOTH regimes for a calibration test.")
    if not GATE:
        log("  >>> STATE reliability is ~unimodal -> calibration can't demonstrably beat the best constant policy.")
        log("  >>> Options: lower rel-thresh, get a sharper SFM, or bin by gene/edge difficulty instead of pert.")
    log(f"\nwrote state_vs_real.csv ({len(df)} edges) -> {a.outdir}")

if __name__ == "__main__":
    main()
