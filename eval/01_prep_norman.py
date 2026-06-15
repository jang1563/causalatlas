#!/usr/bin/env python3
"""
CausalAtlas Phase 2 - Step 01: prepare Norman 2019 (K562 CRISPR-activation GI) for sceptre.

Reads norman_2019.h5ad with h5py ONLY and writes the same sceptre-input bundle as the Phase-1
Replogle prep (binary CSC genes x cells + covariates + grna_target_data_frame), so 02_sceptre
and the downstream classifier reuse the existing readers.

Norman-specific structure (VERIFIED 2026-06-11):
  - LEGACY anndata categoricals: categories live in obs/__categories/<col>, codes in obs/<col>
    (NOT a per-column categorical Group like the Replogle h5ad).
  - Perturbation field = obs/perturbation_name: 105 single genes + 131 dual "A+B" combos + "control"
    (11,835 cells). Every dual has BOTH singles also measured (GI-testable = 131/131).
  - RAW integer counts are in layers/counts (csr). X itself is log-normalized -> unusable for sceptre.
  - var index = "gene_symbols"; obs index = "index". Counts shape (111255, 19018), nnz ~3.6e8.
  - CRISPR-ACTIVATION (overexpression): cis self-edges go UP (positive log2FC). The G1 knockdown gate
    sign-flips to an activation gate downstream in 03; the prep is sign-agnostic.

Each perturbation_name value becomes a grna_target, so sceptre tests every single / combo vs control.
The genetic-interaction residual (combo vs single-A + single-B additive prediction) is computed in 03.

Usage:
  module load anaconda3/2023.09-3
  python 01_prep_norman.py \
      --h5ad ~/data/norman_2019.h5ad \
      --outdir ~/CausalAtlas/inputs/norman_gi \
      [--gi-subset 8]   # validation: 8 highest-cell duals + their singles + control (self-contained GI set)
"""
import argparse, os, gzip, sys
import numpy as np, pandas as pd
import scipy.sparse as sp
import h5py

def log(*a): print("[prep-norman]", *a, file=sys.stderr, flush=True)

def read_legacy_cat(obs, name):
    """Norman legacy categorical: codes in obs/<name>, categories in obs/__categories/<name>."""
    cats = obs["__categories"][name][:]
    cats = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in cats])
    codes = obs[name][:]
    return np.where(codes < 0, "NA", cats[np.clip(codes, 0, len(cats) - 1)])

def read_obs_col(obs, name):
    """Plain array (QC cols) or legacy categorical."""
    if name not in obs:
        return None
    if "__categories" in obs and name in obs["__categories"]:
        return read_legacy_cat(obs, name)
    arr = obs[name][:]
    if arr.dtype.kind == "S":
        arr = arr.astype(str)
    return arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--gi-subset", type=int, default=0,
                    help="keep the top-N duals by cell count + BOTH their singles + control "
                         "(self-contained GI validation set with the additive prediction available)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    f = h5py.File(args.h5ad, "r")
    obs, var = f["obs"], f["var"]

    def idx_key(grp, default):
        k = grp.attrs.get("_index", default)
        return k.decode() if isinstance(k, bytes) else k
    obs_idx_key = idx_key(obs, "index")
    var_idx_key = idx_key(var, "gene_symbols")

    obs_index = obs[obs_idx_key][:].astype(str)
    pert = read_obs_col(obs, "perturbation_name")
    if pert is None:
        sys.exit("perturbation_name not found in obs")
    # control cells carry distinct NegCtrl guide_identity values (e.g. NegCtrl10_NegCtrl0__...);
    # sceptre's calibration check needs >=2 non-targeting gRNAs, so keep them as separate guide_ids
    # (all mapped to grna_target "non-targeting") rather than collapsing to one "control" label.
    gident = read_obs_col(obs, "guide_identity")

    # counts: prefer layers/counts (raw integer); X is log-normalized
    if "layers" in f and "counts" in f["layers"]:
        C = f["layers"]["counts"]
        csrc = "layers/counts"
    else:
        C = f["X"]
        csrc = "X (WARNING: no layers/counts -- verify integer)"
    enc = str(C.attrs.get("encoding-type", ""))
    shape = C.attrs.get("shape", None)
    if shape is not None:
        n_cells, n_genes = (int(shape[0]), int(shape[1]))
    else:
        n_cells, n_genes = len(obs_index), len(var[var_idx_key])
    log(f"counts source={csrc} enc={enc} cells x genes = {n_cells} x {n_genes}")

    # --- optional GI validation subset: N highest-cell duals + their singles + control ---
    cats_all = np.unique(pert)
    is_dual = np.array(["+" in c for c in cats_all])
    if args.gi_subset:
        vals, cnts = np.unique(pert, return_counts=True)
        cnt = dict(zip(vals, cnts))
        duals = sorted([c for c in cats_all if "+" in c], key=lambda x: -cnt.get(x, 0))
        chosen = set()
        for dconf in duals:
            a, b = dconf.split("+")
            if a in cnt and b in cnt:          # both singles measured (always true for Norman)
                chosen.update([dconf, a, b])
            if len(set(x for x in chosen if "+" in x)) >= args.gi_subset:
                break
        chosen.add("control")
        keep = np.where(np.isin(pert, list(chosen)))[0]
        log(f"gi-subset: {sum('+' in x for x in chosen)} duals + {sum('+' not in x and x!='control' for x in chosen)} singles + control -> {len(keep)} cells")
    else:
        keep = np.arange(n_cells)

    obs_index_k, pert_k = obs_index[keep], pert[keep]
    gident_k = gident[keep] if gident is not None else pert_k

    # --- read counts in cell-chunks via indptr -> genes x cells CSC int32 ---
    if "csr" not in enc and enc != "":
        sys.exit(f"expected csr counts, got enc={enc}")
    indptr_full = C["indptr"][:]
    keep_set = set(keep.tolist())
    blocks, CHUNK = [], 20000
    checked = False
    for start in range(0, n_cells, CHUNK):
        stop = min(start + CHUNK, n_cells)
        local = [c for c in range(start, stop) if c in keep_set]
        if not local:
            continue
        lo, hi = int(indptr_full[start]), int(indptr_full[stop])
        data = C["data"][lo:hi]
        indices = C["indices"][lo:hi]
        ip = indptr_full[start:stop + 1] - lo
        blk = sp.csr_matrix((data, indices, ip), shape=(stop - start, n_genes))
        blk = blk[[c - start for c in local], :]
        if not checked:
            sd = blk.data[:2000]
            fi = float(np.mean(np.isclose(sd, np.round(sd)))) if sd.size else 1.0
            if fi < 0.99:
                log(f"WARNING: counts not integer (frac={fi:.3f}) from {csrc}")
            checked = True
        blocks.append(sp.csc_matrix(np.round(blk.toarray()).astype(np.int32)))
        log(f"  read cells {start}-{stop} (kept {len(local)})")
    Xkeep = sp.vstack(blocks).tocsr()                 # kept_cells x genes
    gc = Xkeep.T.tocsc().astype(np.int32)             # genes x cells CSC
    gc.sort_indices()
    nr, nc = gc.shape
    nnz = int(gc.nnz)
    if nnz >= 2**31 - 1:
        raise SystemExit(f"nnz={nnz} exceeds int32; use a chunked dgCMatrix path")
    log("writing binary CSC", (nr, nc), "nnz=", nnz)
    np.asarray(gc.indptr, np.int32).tofile(os.path.join(args.outdir, "X_indptr.i32"))
    np.asarray(gc.indices, np.int32).tofile(os.path.join(args.outdir, "X_indices.i32"))
    np.asarray(gc.data, np.int32).tofile(os.path.join(args.outdir, "X_data.i32"))
    with open(os.path.join(args.outdir, "X_dims.txt"), "w") as fh:
        fh.write(f"{nr} {nc} {nnz}\n")

    # features + barcodes
    gene_symbol = var[var_idx_key][:].astype(str)
    gene_id = var["index"][:].astype(str) if "index" in var else gene_symbol
    with gzip.open(os.path.join(args.outdir, "features.tsv.gz"), "wt") as fh:
        for gid, gs in zip(gene_id, gene_symbol):
            fh.write(f"{gid}\t{gs}\tGene Expression\n")
    with gzip.open(os.path.join(args.outdir, "barcodes.tsv.gz"), "wt") as fh:
        for bc in obs_index_k:
            fh.write(bc + "\n")

    # grna_target per cell: control -> non-targeting; else the perturbation_name (single or "A+B" combo).
    # guide_id: control cells keep their distinct NegCtrl guide_identity (gives >=2 NT gRNAs for the
    # calibration check); perturbed cells use the collapsed perturbation_name identity.
    is_ctrl = pert_k == "control"
    grna_target = np.where(is_ctrl, "non-targeting", pert_k)
    guide_id = np.where(is_ctrl, gident_k, pert_k)
    cov = pd.DataFrame({"cell_barcode": obs_index_k, "guide_id": guide_id, "grna_target": grna_target})
    qc_map = {"ncounts": "total_counts", "ngenes": "n_genes",
              "percent_mito": "pct_counts_mt", "batch": "gemgroup"}
    for out_c, in_c in qc_map.items():
        v = read_obs_col(obs, in_c)
        if v is not None:
            vv = np.asarray(v)[keep]
            cov[out_c] = vv.astype(float) if out_c != "batch" else vv
    cov.to_csv(os.path.join(args.outdir, "cell_covariates.csv"), index=False)

    gdf = (cov[["guide_id", "grna_target"]].drop_duplicates()
           .rename(columns={"guide_id": "grna_id"}))
    gdf.to_csv(os.path.join(args.outdir, "grna_target_data_frame.csv"), index=False)

    n_single = int(sum(("+" not in t) and t != "non-targeting" for t in gdf.grna_target.unique()))
    n_dual = int(sum("+" in t for t in gdf.grna_target.unique()))
    with open(os.path.join(args.outdir, "MANIFEST.txt"), "w") as fh:
        fh.write(f"cells\t{nc}\ngenes\t{n_genes}\n")
        fh.write(f"single_targets\t{n_single}\ndual_targets\t{n_dual}\n")
        fh.write(f"control_cells\t{int((pert_k=='control').sum())}\n")
        fh.write(f"counts_source\t{csrc}\nmodality\tCRISPRa\n")
    log(f"done. singles={n_single} duals={n_dual} control_cells={int((pert_k=='control').sum())} -> {args.outdir}")

if __name__ == "__main__":
    main()
