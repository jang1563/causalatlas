#!/usr/bin/env python3
"""Prep for experiment-selection: Replogle K562_essential -> effect matrix + GO/co-expr features.

ESM2 dropped (the saved file covers only 201 genes; too thin for the ~2000-perturbation pool). Features that
cover all perturbation genes and are apt for effect-similarity: GO (gene2go, pathway membership) + co-expression
(computed from CONTROL cells only -> no leak from perturbation effects).

  ~/envs/gears/bin/python move1_replogle_prep.py --out ~/CausalAtlas/results/expsel
Outputs: effects.npy (n_pert x n_readout), F_go.npy, F_coexpr.npy, names.txt, readout.txt
"""
import argparse, os, sys, pickle, numpy as np
def log(*a): print("[prep]", *a, file=sys.stderr, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default=os.path.expanduser("~/data/ReplogleWeissman2022_K562_essential.h5ad"))
    ap.add_argument("--gene2go", default=os.path.expanduser("~/CausalAtlas/gears_data/gene2go_all.pkl"))
    ap.add_argument("--out", default=os.path.expanduser("~/CausalAtlas/results/expsel"))
    ap.add_argument("--nhvg", type=int, default=2000); ap.add_argument("--min-cells", type=int, default=25)
    ap.add_argument("--svd", type=int, default=50)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    import anndata as ad, scanpy as sc, scipy.sparse as sp
    from sklearn.decomposition import TruncatedSVD
    log("loading", a.h5ad)
    adata = ad.read_h5ad(a.h5ad)
    genes = adata.obs["gene"].astype(str)
    uniq = genes.unique()
    ctrl_labels = [g for g in uniq if any(k in g.lower() for k in
                   ["non-targeting", "non_targeting", "nontargeting", "control", "ntc", "safe-harbor", "safe_harbor", "neg"])]
    log("control labels:", ctrl_labels)
    if not ctrl_labels:
        log("NO control label found; obs.gene sample:", list(uniq[:20])); sys.exit(2)
    # normalize to logCPM if X looks like counts
    X = adata.X
    mx = X.max() if not sp.issparse(X) else X.max()
    if mx > 50:
        log("X looks like counts (max %.1f) -> normalize_total + log1p" % mx)
        sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=a.nhvg)
    hvg = adata.var["highly_variable"].values
    readout = list(np.asarray(adata.var_names)[hvg])
    Xr = adata[:, hvg].X; Xr = Xr.toarray() if sp.issparse(Xr) else np.asarray(Xr)
    isctrl = genes.isin(ctrl_labels).values
    ctrl_mean = Xr[isctrl].mean(0)
    log(f"control cells {isctrl.sum()}, readout HVGs {len(readout)}")
    # per-perturbation effect = pseudobulk(logCPM) - control mean
    E, names = [], []
    for g in uniq:
        if g in ctrl_labels: continue
        m = (genes == g).values
        if m.sum() < a.min_cells: continue
        E.append(Xr[m].mean(0) - ctrl_mean); names.append(g)
    E = np.asarray(E); log("E", E.shape, "perturbations", len(names))

    # GO features (gene x GO-term, terms in >=5 perts), SVD-reduced
    g2go = pickle.load(open(a.gene2go, "rb"))
    cnt = {}
    for n in names:
        for t in g2go.get(n, []): cnt[t] = cnt.get(t, 0) + 1
    keep = [t for t, c in cnt.items() if c >= 5]; ti = {t: i for i, t in enumerate(keep)}
    Fgo = np.zeros((len(names), len(keep)), np.float32)
    for i, n in enumerate(names):
        for t in g2go.get(n, []):
            if t in ti: Fgo[i, ti[t]] = 1
    cov = (Fgo.sum(1) > 0).mean()
    Fgo_r = TruncatedSVD(n_components=min(a.svd, max(2, Fgo.shape[1]-1)), random_state=0).fit_transform(Fgo) if Fgo.shape[1] > a.svd else Fgo
    log(f"GO: {len(keep)} terms (>=5), coverage {cov:.0%}, reduced {Fgo_r.shape}")

    # co-expression features from CONTROL cells: corr of each pert gene's control expression to the readout panel
    var = list(adata.var_names); vi = {g: i for i, g in enumerate(var)}
    Xc = adata[isctrl].X; Xc = Xc.toarray() if sp.issparse(Xc) else np.asarray(Xc)
    Xc_read = Xc[:, [vi[g] for g in readout]]
    Zr = (Xc_read - Xc_read.mean(0)) / (Xc_read.std(0) + 1e-8)
    Fco = np.zeros((len(names), len(readout)), np.float32); ncov = 0
    for i, n in enumerate(names):
        if n in vi:
            x = Xc[:, vi[n]]; xz = (x - x.mean()) / (x.std() + 1e-8)
            Fco[i] = (Zr * xz[:, None]).mean(0); ncov += 1
    Fco_r = TruncatedSVD(n_components=a.svd, random_state=0).fit_transform(np.nan_to_num(Fco))
    log(f"co-expr: gene-in-var coverage {ncov}/{len(names)}, reduced {Fco_r.shape}")

    np.save(os.path.join(a.out, "effects.npy"), E.astype(np.float32))
    np.save(os.path.join(a.out, "F_go.npy"), Fgo_r.astype(np.float32))
    np.save(os.path.join(a.out, "F_coexpr.npy"), Fco_r.astype(np.float32))
    open(os.path.join(a.out, "names.txt"), "w").write("\n".join(names))
    open(os.path.join(a.out, "readout.txt"), "w").write("\n".join(readout))
    log("DONE ->", a.out)

if __name__ == "__main__":
    main()
