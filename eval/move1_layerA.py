#!/usr/bin/env python3
"""Move 1, Layer A -- does the perturbation SFM beat simple baselines under a magnitude-robust
DISCRIMINATION metric on an interpolation-proof held-out? (no LLM; pure model-vs-baseline.)

This is the GEARS / Norman instantiation. The same core metric functions feed
the Tahoe / Arc-STATE instantiation.

Design locks (literature review 2026-06-13, verified primary sources):
- PRIMARY metric  = edge-level AUROC of |predicted log2FC| discriminating effect vs no-effect.
                    Magnitude-robust, maps to our three-state, and avoids Pearson-on-pseudobulk
                    (only a VCC *secondary* metric; the contested "FM works" reading rests on it).
- SECONDARY       = cosine-PDS: the VCC Perturbation Discrimination Score but with COSINE distance.
                    VCC/cell-eval ships L1-PDS, which is rescale-gameable (arXiv 2511.16954);
                    cosine is invariant to global rescaling -> the honest profile-level cross-check.
- BASELINES (field-standard: Ahlmann-Eltze, Huber & Anders, Nat Methods 2025; GEARS, Roohani 2024):
    observed-additive  y_add = y_A + y_B (- y_ctrl, absorbed) from OBSERVED single effects (combos only)
    no-change          predict zero effect everywhere
    mean               predict the mean single-perturbation effect per gene (the unseen-single baseline)
- UNCERTAINTY     = PERTURBATION-CLUSTER bootstrap (resample perturbations, not edges). This is the
                    honest treatment of the small strict interpolation-proof split (combo_seen0, n=4).
- delta sweep {0.1,0.25,0.5,1.0} for the call-accuracy framing (equivalence threshold).

  python move1_layerA.py --gears results/gears_norman/gears_vs_real_all.csv \
                         --marginal results/gears_norman/labeled_marginal.csv --out results/move1
"""
import os, re, json, argparse
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

DELTAS = [0.1, 0.25, 0.5, 1.0]
PRIMARY_DELTA = 0.25
B_BOOT = 2000

# ---------- data ----------
def load(gears_csv, marginal_csv):
    g = pd.read_csv(gears_csv).dropna(subset=["gears_log2FC"]).copy()
    # upstream build doubled a subset of single_* edges (1360 exact (pert,gene) dups; combos have ZERO) -> dedup
    n0 = len(g); g = g.drop_duplicates(["perturbation", "gene"]).copy()
    print(f"[load] dedup (pert,gene): {n0} -> {len(g)} rows ({n0-len(g)} exact dups dropped)", file=__import__("sys").stderr)
    g["y"] = (g["real_call"] == "effect").astype(int)
    m = pd.read_csv(marginal_csv)
    m1 = m[m["delta"] == PRIMARY_DELTA].copy()          # log2FC is delta-independent; delta only labels
    singles = m1[m1["kind"] == "single"]
    single_lookup = {(r.target, r.response_id): r.log2FC for r in singles.itertuples()}
    mean_by_gene = singles.groupby("response_id")["log2FC"].mean().to_dict()
    combo_true = m1[m1["kind"] == "combo"][["target", "response_id", "log2FC"]]
    return g, single_lookup, mean_by_gene, combo_true

def add_predictions(g, single_lookup, mean_by_gene):
    def add_lfc(pert, gene):
        ps = re.split(r"[+_]", pert)
        if len(ps) < 2: return np.nan
        vals = [single_lookup.get((p, gene)) for p in ps]
        if any(v is None for v in vals): return np.nan
        return float(np.sum(vals))
    g = g.copy()
    g["pred_GEARS"]    = g["gears_log2FC"].astype(float)
    g["pred_additive"] = [add_lfc(p, gn) for p, gn in zip(g["perturbation"], g["gene"])]
    g["pred_nochange"] = 0.0
    g["pred_mean"]     = [float(mean_by_gene.get(gn, 0.0)) for gn in g["gene"]]
    return g

# ---------- metrics ----------
def acc_at(sub, predcol, delta=PRIMARY_DELTA):
    pred = sub[predcol].to_numpy(float)
    valid = np.isfinite(pred)
    if not valid.any(): return np.nan
    call_eff = np.abs(pred[valid]) >= delta
    real_eff = (sub["real_call"].to_numpy()[valid] == "effect")
    return float(np.mean(call_eff == real_eff))

def auroc(sub, predcol):
    y = sub["y"].to_numpy()
    s = np.abs(sub[predcol].to_numpy(float))
    msk = np.isfinite(s)
    y, s = y[msk], s[msk]
    if len(y) == 0 or len(np.unique(y)) < 2: return np.nan
    if np.allclose(s, s[0]): return 0.5                 # constant predictor (no-change) cannot discriminate
    return float(roc_auc_score(y, s))

def _ci(df, fn, B=B_BOOT, seed=0):
    """Perturbation-cluster bootstrap CI for a single metric fn(subdf)->scalar."""
    rng = np.random.default_rng(seed)
    perts = df["perturbation"].unique()
    point = fn(df)
    if len(perts) < 2 or not np.isfinite(point): return (point, np.nan, np.nan)
    by = {p: df[df["perturbation"] == p] for p in perts}
    vals = []
    for _ in range(B):
        samp = rng.choice(perts, size=len(perts), replace=True)
        v = fn(pd.concat([by[p] for p in samp], ignore_index=True))
        if np.isfinite(v): vals.append(v)
    return (point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

def _ci_delta(df, fn_a, fn_b, B=B_BOOT, seed=0):
    """Paired cluster-bootstrap CI for fn_a - fn_b under the SAME perturbation resample."""
    rng = np.random.default_rng(seed)
    perts = df["perturbation"].unique()
    point = fn_a(df) - fn_b(df)
    if len(perts) < 2: return (point, np.nan, np.nan)
    by = {p: df[df["perturbation"] == p] for p in perts}
    vals = []
    for _ in range(B):
        samp = rng.choice(perts, size=len(perts), replace=True)
        bs = pd.concat([by[p] for p in samp], ignore_index=True)
        a, b = fn_a(bs), fn_b(bs)
        if np.isfinite(a) and np.isfinite(b): vals.append(a - b)
    return (point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

def cosine_pds(g_sub, combo_true, predcol):
    """VCC-style discrimination with COSINE distance (rescale-invariant). Panel-level secondary.
    Per perturbation: rank the matched TRUE profile among all true profiles by cosine distance to the
    PREDICTED profile. PDS_i = 1 - (rank_i - 1)/(P-1); 1=perfect, ~0.5 random. Mean over perturbations."""
    perts = list(g_sub["perturbation"].unique())
    genes = sorted(g_sub["gene"].unique())
    gi = {gn: i for i, gn in enumerate(genes)}
    P, G = len(perts), len(genes)
    if P < 3: return np.nan
    pred = np.zeros((P, G)); true = np.zeros((P, G))
    tlook = {(r.target, r.response_id): r.log2FC for r in combo_true.itertuples()}
    for ip, p in enumerate(perts):
        for r in g_sub[g_sub["perturbation"] == p].itertuples():
            v = getattr(r, predcol)
            if np.isfinite(v): pred[ip, gi[r.gene]] = v
            t = tlook.get((p, r.gene))
            if t is not None and np.isfinite(t): true[ip, gi[r.gene]] = t
    def unit(M):
        n = np.linalg.norm(M, axis=1, keepdims=True); n[n == 0] = 1.0; return M / n
    dist = 1.0 - (unit(pred) @ unit(true).T)
    pds = []
    for i in range(P):
        rank = int(np.where(np.argsort(dist[i]) == i)[0][0]) + 1
        pds.append(1.0 - (rank - 1) / (P - 1))
    return float(np.mean(pds))

# ---------- report ----------
def fmt(t): return "   nan   " if not np.isfinite(t[0]) else f"{t[0]:.3f}[{t[1]:.2f},{t[2]:.2f}]"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gears", default="results/gears_norman/gears_vs_real_all.csv")
    ap.add_argument("--marginal", default="results/gears_norman/labeled_marginal.csv")
    ap.add_argument("--out", default="results/move1")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    g, single_lookup, mean_by_gene, combo_true = load(a.gears, a.marginal)
    g = add_predictions(g, single_lookup, mean_by_gene)
    report = {"primary_delta": PRIMARY_DELTA, "B_boot": B_BOOT, "combo": {}, "single": {}, "pds": {}, "delta_sweep": {}}

    combo = g[g["regime"].str.startswith("combo")]
    strata = {
        "combo_in_train": g[g["regime"] == "combo_in_train"],
        "combo_seen1":    g[g["regime"] == "combo_seen1"],
        "combo_seen0":    g[g["regime"] == "combo_seen0"],
        "interp_pool(seen1+seen0)": g[g["regime"].isin(["combo_seen1", "combo_seen0"])],
        "all_combos":     combo,
    }
    print(f"\n{'='*108}\nMOVE 1 LAYER-A  |  GEARS vs observed-additive on Norman combos  |  delta={PRIMARY_DELTA}, "
          f"cluster-bootstrap {B_BOOT}x by perturbation\n{'='*108}")
    print(f"{'stratum':26s} {'nP':>3s} {'nE':>5s} | {'GEARS_acc':>16s} {'ADD_acc':>16s} {'ADD-GEARS':>17s} "
          f"| {'GEARS_AUROC':>16s} {'ADD_AUROC':>16s}")
    for name, sub in strata.items():
        sub = sub.dropna(subset=["pred_additive"])
        if len(sub) == 0: continue
        gacc = _ci(sub, lambda d: acc_at(d, "pred_GEARS"))
        aacc = _ci(sub, lambda d: acc_at(d, "pred_additive"))
        dacc = _ci_delta(sub, lambda d: acc_at(d, "pred_additive"), lambda d: acc_at(d, "pred_GEARS"))
        gauc = _ci(sub, lambda d: auroc(d, "pred_GEARS"))
        aauc = _ci(sub, lambda d: auroc(d, "pred_additive"))
        nP, nE = sub["perturbation"].nunique(), len(sub)
        report["combo"][name] = dict(nP=int(nP), nE=int(nE), GEARS_acc=gacc, ADD_acc=aacc,
                                     ADD_minus_GEARS_acc=dacc, GEARS_AUROC=gauc, ADD_AUROC=aauc)
        print(f"{name:26s} {nP:>3d} {nE:>5d} | {fmt(gacc):>16s} {fmt(aacc):>16s} {fmt(dacc):>17s} "
              f"| {fmt(gauc):>16s} {fmt(aauc):>16s}")

    # singles: GEARS vs no-change vs mean (additive n/a)
    print(f"\n{'-'*108}\nSINGLES (additive undefined -> compare GEARS to no-change & mean floor)\n{'-'*108}")
    print(f"{'stratum':26s} {'nP':>3s} {'nE':>5s} | {'GEARS_acc':>16s} {'noChg_acc':>16s} {'mean_acc':>16s} "
          f"| {'GEARS_AUROC':>16s} {'mean_AUROC':>16s}")
    for name, sub in {"single_seen": g[g["regime"] == "single_seen"],
                      "single_unseen": g[g["regime"] == "single_unseen"]}.items():
        gacc = _ci(sub, lambda d: acc_at(d, "pred_GEARS"))
        nacc = _ci(sub, lambda d: acc_at(d, "pred_nochange"))
        macc = _ci(sub, lambda d: acc_at(d, "pred_mean"))
        gauc = _ci(sub, lambda d: auroc(d, "pred_GEARS"))
        mauc = _ci(sub, lambda d: auroc(d, "pred_mean"))
        nP, nE = sub["perturbation"].nunique(), len(sub)
        report["single"][name] = dict(nP=int(nP), nE=int(nE), GEARS_acc=gacc, noChg_acc=nacc,
                                      mean_acc=macc, GEARS_AUROC=gauc, mean_AUROC=mauc)
        print(f"{name:26s} {nP:>3d} {nE:>5d} | {fmt(gacc):>16s} {fmt(nacc):>16s} {fmt(macc):>16s} "
              f"| {fmt(gauc):>16s} {fmt(mauc):>16s}")

    # cosine-PDS (secondary, profile-level) on the combo strata
    print(f"\n{'-'*108}\nSECONDARY: cosine-PDS (rescale-invariant discrimination; 1=perfect, ~0.5 random)\n{'-'*108}")
    print(f"{'stratum':26s} {'nP':>3s} | {'GEARS_PDS':>10s} {'ADD_PDS':>10s} {'mean_PDS':>10s}")
    for name in ["combo_in_train", "combo_seen1", "interp_pool(seen1+seen0)", "all_combos"]:
        sub = strata[name].dropna(subset=["pred_additive"])
        gp = cosine_pds(sub, combo_true, "pred_GEARS")
        ap_ = cosine_pds(sub, combo_true, "pred_additive")
        mp = cosine_pds(sub, combo_true, "pred_mean")
        report["pds"][name] = dict(nP=int(sub["perturbation"].nunique()), GEARS=gp, additive=ap_, mean=mp)
        f = lambda x: f"{x:.3f}" if np.isfinite(x) else "  nan "
        print(f"{name:26s} {sub['perturbation'].nunique():>3d} | {f(gp):>10s} {f(ap_):>10s} {f(mp):>10s}")

    # delta sweep: ADD-GEARS accuracy gap on the interpolation pool
    print(f"\n{'-'*108}\nDELTA SWEEP: accuracy on interp_pool(seen1+seen0)  (ADD - GEARS, paired cluster-boot)\n{'-'*108}")
    pool = strata["interp_pool(seen1+seen0)"].dropna(subset=["pred_additive"])
    print(f"{'delta':>6s} | {'GEARS_acc':>16s} {'ADD_acc':>16s} {'ADD-GEARS':>17s}")
    for d in DELTAS:
        ga = _ci(pool, lambda x, d=d: acc_at(x, "pred_GEARS", d))
        aa = _ci(pool, lambda x, d=d: acc_at(x, "pred_additive", d))
        da = _ci_delta(pool, lambda x, d=d: acc_at(x, "pred_additive", d), lambda x, d=d: acc_at(x, "pred_GEARS", d))
        report["delta_sweep"][str(d)] = dict(GEARS_acc=ga, ADD_acc=aa, ADD_minus_GEARS=da)
        print(f"{d:>6.2f} | {fmt(ga):>16s} {fmt(aa):>16s} {fmt(da):>17s}")

    with open(os.path.join(a.out, "layerA_norman.json"), "w") as f:
        json.dump(report, f, indent=2, default=lambda o: None if (isinstance(o, float) and not np.isfinite(o)) else o)
    print(f"\n[saved] {os.path.join(a.out, 'layerA_norman.json')}")

if __name__ == "__main__":
    main()
