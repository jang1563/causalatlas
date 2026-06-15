#!/usr/bin/env python3
"""
CausalAtlas Phase 2 Track 2b — cross-context analysis of Tahoe drug->gene labels.

Pivots labeled_tahoe.csv (label_0.25) to (drug, gene) x cell_line and finds CONTEXT-DEPENDENT edges:
a (drug, gene) that is POSITIVE in >=1 line AND TESTED_NEGATIVE in >=1 line (the drug confirmedly moves
that gene in some contexts and is confirmedly EXCLUDED from moving it in others) -- the cross-context
tested-negative-vs-untested headline, on a drug modality + DESeq2 framework.

Outputs:
  context_summary.csv        : how many edges are line-specific / context-dependent / consistent
  cross_context_edges.csv    : the context-dependent (drug,gene) edges + per-line label + per-line log2FC
  (these also seed the Track-2b eval items)

Usage:
  python 05_tahoe_xctx.py --labeled ~/CausalAtlas/results/tahoe_xctx/labeled_tahoe.csv \
      --outdir ~/CausalAtlas/results/tahoe_xctx --min-detected 4
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.compute as pc

def log(*a): print("[xctx]", *a, file=sys.stderr, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min-detected", type=int, default=4,
                    help="require a (drug,gene) detectable in >=N lines for a fair cross-context call")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    cols = ["drug", "gene", "cell_line", "label_0.25", "log2FC", "lfcSE", "padj", "drug_bioactive"]
    log("reading", a.labeled)
    tbl = pacsv.read_csv(a.labeled,
                         convert_options=pacsv.ConvertOptions(include_columns=cols))
    # dictionary-encode string cols -> pandas category (keeps 70M-row frame light, avoids object-string OOM)
    for c in ["drug", "gene", "cell_line", "label_0.25"]:
        i = tbl.schema.get_field_index(c)
        tbl = tbl.set_column(i, c, pc.dictionary_encode(tbl.column(c)))
    df = tbl.to_pandas()
    del tbl
    log(f"{len(df)} rows; {df.drug.nunique()} drugs, {df.gene.nunique()} genes, {df.cell_line.nunique()} lines")

    # pivot label across lines
    lab = df.pivot_table(index=["drug", "gene"], columns="cell_line", values="label_0.25",
                         aggfunc="first", observed=True)
    lines = list(lab.columns)
    n_pos = lab.eq("POSITIVE").sum(axis=1)
    n_tn = lab.eq("TESTED_NEGATIVE").sum(axis=1)
    n_unt = lab.eq("UNTESTED").sum(axis=1)
    n_det = lab.notna().sum(axis=1)
    log(f"{len(lab)} unique (drug,gene) pairs")

    decided = (n_pos + n_tn)
    enough = n_det >= a.min_detected
    context_dep = enough & (n_pos >= 1) & (n_tn >= 1)         # POS somewhere, confirmed-negative elsewhere
    consistent_pos = enough & (n_pos >= 1) & (n_tn == 0) & (n_unt == 0)
    consistent_tn = enough & (n_tn >= 1) & (n_pos == 0) & (n_unt == 0)

    summary = pd.DataFrame([
        dict(category="context_dependent (POS in some, TESTED_NEG in others)", n=int(context_dep.sum())),
        dict(category="consistent_positive (POS in all decided, no TN)", n=int(consistent_pos.sum())),
        dict(category="consistent_tested_negative (TN in all decided, no POS)", n=int(consistent_tn.sum())),
        dict(category=f"detectable in >={a.min_detected} lines", n=int(enough.sum())),
        dict(category="total (drug,gene) pairs", n=int(len(lab))),
    ])
    summary.to_csv(os.path.join(a.outdir, "context_summary.csv"), index=False)
    log("context summary:")
    print(summary.to_string(index=False), file=sys.stderr)

    # cross-context edges + per-line log2FC (for inspection + eval seeding)
    cc = lab[context_dep].copy()
    lfc = df.pivot_table(index=["drug", "gene"], columns="cell_line", values="log2FC",
                         aggfunc="first", observed=True)
    lfc = lfc.loc[cc.index]
    cc.columns = [f"label::{c}" for c in cc.columns]
    lfc.columns = [f"log2FC::{c}" for c in lfc.columns]
    out = cc.join(lfc).reset_index()
    out["n_pos"] = n_pos[context_dep].values
    out["n_tn"] = n_tn[context_dep].values
    out["n_unt"] = n_unt[context_dep].values
    out.to_csv(os.path.join(a.outdir, "cross_context_edges.csv"), index=False)
    log(f"wrote {len(out)} context-dependent edges")

    # a few striking examples: maximal POS/TN split
    ex = out.assign(split=out.n_pos + out.n_tn).sort_values(["split", "n_pos"], ascending=False).head(12)
    log("examples (drug -> gene; per-line label):")
    labcols = [c for c in out.columns if c.startswith("label::")]
    for _, r in ex.iterrows():
        ls = " ".join(f"{c.split('::')[1]}={str(r[c])[:3]}" for c in labcols if pd.notna(r[c]))
        print(f"  {r['drug'][:22]:22s} -> {r['gene']:10s} | {ls}", file=sys.stderr)

if __name__ == "__main__":
    main()
