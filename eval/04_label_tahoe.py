#!/usr/bin/env python3
"""
CausalAtlas Phase 2 Track 2b — label Tahoe-100M drug->gene edges (cross-context three-state).

Tahoe ships DESeq2 pseudobulk DE per (drug, concentration, cell_line, gene): log2FoldChange + explicit
lfcSE + padj + baseMean + n_cells_trt/ctrl. We apply the SAME equivalence-test three-state as Phase 1,
now to DRUG perturbations across many cell-line CONTEXTS (the project's signature cross-context
tested-negative-vs-untested, at 50-line scale, on a DESeq2 framework instead of sceptre).

Edge = (drug A, max concentration) -> gene B, in cell line L. Three-state per delta:
  POSITIVE        : |log2FC| >= delta AND padj < q* AND detectable AND drug-bioactive-in-L
  TESTED_NEGATIVE : |log2FC| + z*lfcSE < delta AND detectable AND drug-bioactive-in-L   (effect >= delta EXCLUDED)
  UNTESTED        : otherwise
G1-analog = DRUG BIOACTIVITY: a drug is active in L iff it has >= BIOACT_MIN_GENES strong, significant DE
genes there. Without this, an inert drug's whole transcriptome would be trivially "tested-negative"; the gate
makes a tested-negative mean "this drug WORKS in L but specifically does not move gene B." lfcSE is explicit
(no recovery). Cross-context = the same (drug,gene) labeled across lines -> POSITIVE in some, TESTED_NEGATIVE
in others = context-dependent edge (the headline).

Usage:
  python 04_label_tahoe.py --de-dir ~/data/tahoe_pb/metadata/pseudobulk_differential_expression \
      --lines A549,MCF-7,HCT116,HT-29,MIA PaCa-2,HepG2/C3A,SK-MEL-2,NCI-H1299 \
      --outdir ~/CausalAtlas/results/tahoe_xctx
"""
import argparse, os, sys, glob, json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

DELTAS = [0.1, 0.25, 0.5, 1.0]
DELTA_PRIMARY = 0.25
QSTAR = 0.05
Z95, Z90 = 1.959964, 1.644854
BASEMEAN_MIN = 1.0       # gene detectably expressed
N_TRT_MIN = 50
N_CNTRL_MIN = 100
BIOACT_MIN_GENES = 5     # drug must move >=5 strong-sig genes in L to count as bioactive there
COLS = ["gene_name", "log2FoldChange", "lfcSE", "pvalue", "padj", "baseMean",
        "n_cells_trt", "n_cells_ctrl", "drug", "concentration", "Cell_Name_Vevo"]


def log(*a): print("[tahoe]", *a, file=sys.stderr, flush=True)


def build_shard_index(de_dir):
    """map cell_line -> [parquet paths] from row-group min/max stats on Cell_Name_Vevo (footers only,
    no data read). Cell lines are stored CONTIGUOUSLY so each shard holds 1 (or a boundary of 2) lines.
    Cached to <de_dir>/_shard_index.json."""
    cache = os.path.join(de_dir, "_shard_index.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    files = sorted(glob.glob(os.path.join(de_dir, "*.parquet")))
    idx = {}
    for n, f in enumerate(files):
        md = pq.ParquetFile(f).metadata
        col = md.schema.names.index("Cell_Name_Vevo")
        lines = set()
        for rg in range(md.num_row_groups):
            st = md.row_group(rg).column(col).statistics
            if st is not None and st.has_min_max:
                lines.add(st.min); lines.add(st.max)
        if not lines:                      # no stats -> read just that column to discover lines
            lines = set(pq.read_table(f, columns=["Cell_Name_Vevo"]).column(0).unique().to_pylist())
        for L in lines:
            idx.setdefault(L, []).append(f)
        if (n + 1) % 200 == 0:
            log(f"  index {n+1}/{len(files)} shards")
    try:
        json.dump(idx, open(cache, "w"))
    except Exception:
        pass
    return idx


def label_line(df, line):
    """Label all (drug->gene) edges for one cell line; df = that line's DETECTABLE max-concentration rows.
    WIDE format: one row per edge with a label_<delta> column each (avoids the 4x row blow-up that OOMs)."""
    lfc = df["log2FoldChange"].to_numpy(float)
    se = df["lfcSE"].to_numpy(float)
    padj = df["padj"].to_numpy(float)
    ntrt = df["n_cells_trt"].to_numpy(float)
    drug = df["drug"].to_numpy()
    alfc = np.abs(lfc)
    sig = (padj < QSTAR) & ~np.isnan(padj)
    # drug bioactivity in L: per drug, count strong+sig DE genes (all rows here are already detectable)
    strong_sig = sig & (alfc >= DELTA_PRIMARY)
    bioact_count = pd.Series(strong_sig).groupby(drug).transform("sum").to_numpy()
    bioactive = bioact_count >= BIOACT_MIN_GENES

    out = pd.DataFrame(dict(cell_line=line, drug=drug, gene=df["gene_name"].to_numpy(),
                            log2FC=lfc, lfcSE=se, padj=padj, baseMean=df["baseMean"].to_numpy(),
                            n_trt=ntrt, n_cntrl=df["n_cells_ctrl"].to_numpy(),
                            concentration=df["concentration"].to_numpy(),
                            drug_bioactive=bioactive, drug_nsig=bioact_count.astype(np.int32)))
    for d in DELTAS:
        pos = sig & (alfc >= d) & bioactive
        tn = (~pos) & bioactive & ~np.isnan(se) & (alfc + Z90 * se < d)
        out[f"label_{d}"] = pd.Categorical(
            np.where(pos, "POSITIVE", np.where(tn, "TESTED_NEGATIVE", "UNTESTED")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--de-dir", required=True)
    ap.add_argument("--lines", required=True, help="comma-separated Cell_Name_Vevo values")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--detected-only", action="store_true",
                    help="write only detectable-gene rows (smaller; recommended)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    lines = [x.strip() for x in args.lines.split(",") if x.strip()]
    log("building shard->line index (footers only)...")
    idx = build_shard_index(args.de_dir)
    log(f"index: {len(idx)} cell lines across shards")

    all_summ = []
    combined_path = os.path.join(args.outdir, "labeled_tahoe.csv")
    first = True
    for L in lines:
        paths = idx.get(L, [])
        if not paths:
            log(f"  {L}: no shards (name mismatch? have e.g. {list(idx)[:3]}) -- skipping"); continue
        log(f"reading {L} from {len(paths)} shards ...")
        # read shard-by-shard, filter to this line + DETECTABLE in arrow, accumulate (peak = 1 shard)
        parts = []
        for p in paths:
            t = pq.read_table(p, columns=COLS)
            m = pc.and_(pc.equal(t["Cell_Name_Vevo"], L),
                pc.and_(pc.greater_equal(t["baseMean"], BASEMEAN_MIN),
                pc.and_(pc.greater_equal(t["n_cells_trt"], N_TRT_MIN),
                        pc.greater_equal(t["n_cells_ctrl"], N_CNTRL_MIN))))
            t = t.filter(m)
            if t.num_rows:
                parts.append(t)
            del t
        df = pa.concat_tables(parts).to_pandas() if parts else pd.DataFrame()
        del parts
        if df.empty:
            log(f"  {L}: no detectable rows -- skipping"); continue
        # one concentration per drug among detectable rows: the max (strongest exposure)
        df = df[df["concentration"] == df.groupby("drug")["concentration"].transform("max")]
        log(f"  {L}: {len(df)} detectable edges at max-conc; {df['drug'].nunique()} drugs")
        lab = label_line(df, L)
        lab.to_csv(combined_path, mode="w" if first else "a", header=first, index=False)
        first = False
        vc = lab["label_0.25"].astype(str).value_counts()
        all_summ.append(dict(cell_line=L, drugs=int(df["drug"].nunique()),
                             POSITIVE=int(vc.get("POSITIVE", 0)),
                             TESTED_NEGATIVE=int(vc.get("TESTED_NEGATIVE", 0)),
                             UNTESTED=int(vc.get("UNTESTED", 0)),
                             bioactive_drugs=int(lab.drop_duplicates("drug")["drug_bioactive"].sum())))
        log(f"  {L} d=0.25: " + " ".join(f"{k}={all_summ[-1][k]}" for k in ("POSITIVE", "TESTED_NEGATIVE", "UNTESTED")))
        del df, lab

    summ = pd.DataFrame(all_summ)
    summ.to_csv(os.path.join(args.outdir, "summary_by_line.csv"), index=False)
    log("per-line summary (delta=0.25):")
    print(summ.to_string(index=False), file=sys.stderr)
    log(f"done -> {combined_path}")


if __name__ == "__main__":
    main()
