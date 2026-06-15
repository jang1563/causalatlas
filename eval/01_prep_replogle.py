#!/usr/bin/env python3
"""
CausalAtlas Phase 1 - Step 01: prepare Replogle K562_essential for sceptre.

Reads the scPerturb-harmonized Replogle h5ad with h5py ONLY (no anndata/scanpy
dependency) and writes
the inputs the R/sceptre step needs:
  - matrix.mtx.gz / barcodes.tsv.gz / features.tsv.gz  (10x-style raw counts, genes x cells)
  - cell_covariates.csv        (per cell: guide_id, grna_target, QC covariates)
  - grna_target_data_frame.csv (grna_id=guide_id -> grna_target)

Design (locked 2026-06-09):
  - Low MOI: nperts==1 cells are single-perturbation; nperts==0 = non-targeting controls
    (label "control" in this h5ad). Control cells carry 97 distinct non-targeting guide_ids
    that sceptre uses to build negative-control pairs for the calibration check.
  - grna assignment is guide-level; R step uses grna_integration_strategy="union".

Usage:
  module load anaconda3/2023.09-3
  python 01_prep_replogle.py \
      --h5ad ~/data/ReplogleWeissman2022_K562_essential.h5ad \
      --outdir ~/CausalAtlas/inputs/K562_essential \
      [--subset-cells 20000]   # optional random subset for a fast validation run
"""
import argparse, os, gzip, sys
import numpy as np, pandas as pd
import scipy.io as sio, scipy.sparse as sp
import h5py

def log(*a): print("[prep]", *a, file=sys.stderr, flush=True)

def read_cat(g):
    """read an anndata categorical group -> string array."""
    cats = g["categories"][:]
    cats = np.array([c.decode() if isinstance(c, bytes) else c for c in cats])
    codes = g["codes"][:]
    out = np.where(codes < 0, "NA", cats[np.clip(codes, 0, len(cats)-1)])
    return out

def read_obs_col(obs, name):
    if name not in obs: return None
    item = obs[name]
    if isinstance(item, h5py.Group):           # categorical
        return read_cat(item)
    arr = item[:]
    if arr.dtype.kind == "S": arr = arr.astype(str)
    return arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--subset-cells", type=int, default=0,
                    help="random cell subset (validation; gives FEW cells per perturbation)")
    ap.add_argument("--subset-targets", type=int, default=0,
                    help="keep ALL cells for the top-N targets by cell count + all controls "
                         "(validation with proper per-perturbation power)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    f = h5py.File(args.h5ad, "r")
    obs, var = f["obs"], f["var"]

    # index keys are named via group attrs (e.g. obs._index = "cell_barcode")
    def idx_key(grp, default):
        k = grp.attrs.get("_index", default)
        return k.decode() if isinstance(k, bytes) else k
    obs_idx_key, var_idx_key = idx_key(obs, "_index"), idx_key(var, "_index")

    obs_index = obs[obs_idx_key][:].astype(str)
    pert  = read_obs_col(obs, "perturbation")
    guide = read_obs_col(obs, "guide_id")
    if guide is None: guide = pert

    Xg = f["X"]
    enc = str(Xg.attrs.get("encoding-type", ""))
    if "array" in enc:                       # dense (cells x genes) - this Replogle h5ad
        n_cells, n_genes = Xg.shape
    elif "csr" in enc or "csc" in enc:
        n_cells, n_genes = tuple(int(s) for s in Xg.attrs["shape"])
    else:
        n_cells, n_genes = Xg.shape if hasattr(Xg, "shape") else (len(obs_index), len(var[var_idx_key]))
    log(f"X enc={enc} cells x genes = {n_cells} x {n_genes}")

    # --- optional subset ---
    if args.subset_targets:
        # keep ALL cells of the top-N targets (by cell count) + all controls -> proper power
        is_ctrl = pert == "control"
        vals, cnts = np.unique(pert[~is_ctrl], return_counts=True)
        top = set(vals[np.argsort(-cnts)[:args.subset_targets]].tolist())
        mask = is_ctrl | np.isin(pert, list(top))
        keep = np.where(mask)[0]
        log(f"subset-targets: {len(top)} targets (all their cells) + controls -> {len(keep)} cells")
    elif args.subset_cells and args.subset_cells < n_cells:
        rng = np.random.default_rng(args.seed)
        is_ctrl = pert == "control"
        ctrl = np.where(is_ctrl)[0]; other = np.where(~is_ctrl)[0]
        n_other = max(0, args.subset_cells - len(ctrl))
        keep = np.sort(np.concatenate([ctrl, rng.choice(other, min(n_other, len(other)), replace=False)]))
    else:
        keep = np.arange(n_cells)
    # multiplet filter (N7, 2026-06-11): low-MOI 1:1 attribution requires single-perturbation cells;
    # drop nperts>1 multiplets (keep nperts==1 + controls). The scPerturb-harmonized h5ad concatenates all
    # detected guides for multiplet cells, so unfiltered they bias the cis knockdown (esp. low-n regulators).
    nperts = read_obs_col(obs, "nperts")
    if nperts is not None:
        npf = np.asarray(nperts).astype(float)
        single = (npf == 1) | (pert == "control")
        before = len(keep); keep = keep[single[keep]]
        log(f"multiplet filter: {before} -> {len(keep)} cells (dropped {before-len(keep)} nperts>1 multiplets)")
    sel = keep
    obs_index, pert, guide = obs_index[keep], pert[keep], guide[keep]

    # --- read X in cell-chunks -> sparse CSC (genes x cells), integer counts ---
    keep_set = set(keep.tolist()); keep_pos = {c: i for i, c in enumerate(keep)}
    blocks, CHUNK = [], 20000
    checked = False
    for start in range(0, n_cells, CHUNK):
        stop = min(start + CHUNK, n_cells)
        local = [c for c in range(start, stop) if c in keep_set]
        if not local: continue
        if "array" in enc:
            blk = Xg[start:stop, :]                     # dense chunk (cells x genes)
            blk = blk[[c - start for c in local], :]
        else:                                           # sparse fallback
            data, indices, indptr = Xg["data"], Xg["indices"], Xg["indptr"]
            sub = sp.csr_matrix((data[:], indices[:], indptr[:]), shape=(n_cells, n_genes))[start:stop]
            blk = sub[[c - start for c in local], :].toarray()
        if not checked:
            fi = np.mean(np.isclose(blk[:5], np.round(blk[:5])))
            if fi < 0.99: log("WARNING: X not integer counts (frac=%.3f)" % fi)
            checked = True
        blocks.append(sp.csc_matrix(np.round(blk).astype(np.int32)))
        log(f"  read cells {start}-{stop} (kept {len(local)})")
    Xkeep = sp.vstack(blocks).tocsr()                   # kept_cells x genes
    gc = Xkeep.T.tocsc().astype(np.int32)               # genes x cells, CSC
    gc.sort_indices()
    nr, nc = gc.shape
    n_cells = nc
    nnz = int(gc.nnz)
    # Binary CSC transfer (R reads via readBin -> dgCMatrix; no text MatrixMarket parsing).
    # dgCMatrix p/i slots are 32-bit; nnz must fit int32. scipy CSC genes x cells maps directly:
    #   p = indptr (over cells/columns), i = indices (gene rows, 0-based), x = data.
    if nnz >= 2**31 - 1:
        raise SystemExit(f"nnz={nnz} exceeds int32; dgCMatrix needs 64-bit p (use a chunked path)")
    log("writing binary CSC", (nr, nc), "nnz=", nnz)
    np.asarray(gc.indptr, np.int32).tofile(os.path.join(args.outdir, "X_indptr.i32"))
    np.asarray(gc.indices, np.int32).tofile(os.path.join(args.outdir, "X_indices.i32"))
    np.asarray(gc.data, np.int32).tofile(os.path.join(args.outdir, "X_data.i32"))
    with open(os.path.join(args.outdir, "X_dims.txt"), "w") as fh:
        fh.write(f"{nr} {nc} {nnz}\n")

    # features + barcodes
    def vcol(*names):
        for n in names:
            if n in var:
                a = var[n][:]; return a.astype(str) if a.dtype.kind == "S" else np.asarray(a).astype(str)
        return None
    gene_symbol = var[var_idx_key][:].astype(str)       # var index = gene symbol ("gene_name")
    gene_id     = vcol("ensembl_id", "gene_id")
    if gene_id is None: gene_id = gene_symbol
    with gzip.open(os.path.join(args.outdir, "features.tsv.gz"), "wt") as fh:
        for gid, gs in zip(gene_id, gene_symbol): fh.write(f"{gid}\t{gs}\tGene Expression\n")
    with gzip.open(os.path.join(args.outdir, "barcodes.tsv.gz"), "wt") as fh:
        for bc in obs_index: fh.write(bc + "\n")

    # grna_target per cell
    grna_target = np.where(pert == "control", "non-targeting", pert)
    cov = pd.DataFrame({"cell_barcode": obs_index, "guide_id": guide, "grna_target": grna_target})
    for c in ["ncounts", "ngenes", "percent_mito", "percent_ribo", "nperts"]:
        v = read_obs_col(obs, c)
        if v is not None: cov[c] = np.asarray(v, dtype=float)[sel] if isinstance(sel, np.ndarray) else np.asarray(v, dtype=float)
    for c in ["batch", "gemgroup"]:
        v = read_obs_col(obs, c)
        if v is not None: cov[c] = (v[sel] if isinstance(sel, np.ndarray) else v)
    cov.to_csv(os.path.join(args.outdir, "cell_covariates.csv"), index=False)

    gdf = cov[["guide_id", "grna_target"]].drop_duplicates().rename(columns={"guide_id": "grna_id"})
    gdf.to_csv(os.path.join(args.outdir, "grna_target_data_frame.csv"), index=False)

    n_targets = gdf.loc[gdf.grna_target != "non-targeting", "grna_target"].nunique()
    n_nt = int((gdf.grna_target == "non-targeting").sum())
    with open(os.path.join(args.outdir, "MANIFEST.txt"), "w") as fh:
        fh.write(f"cells\t{n_cells}\ngenes\t{n_genes}\nguide_ids\t{gdf.shape[0]}\n")
        fh.write(f"targets_genes\t{n_targets}\nnon_targeting_guides\t{n_nt}\n")
        fh.write(f"control_cells\t{int((pert=='control').sum())}\n")
    log("done. targets=%d non-targeting guides=%d -> %s" % (n_targets, n_nt, args.outdir))

if __name__ == "__main__":
    main()
