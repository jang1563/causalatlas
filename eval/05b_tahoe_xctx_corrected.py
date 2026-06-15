#!/usr/bin/env python3
"""
CausalAtlas Phase 2 Track 2b — review-corrected cross-context count (fixes C3/C4).

Two fixes over 05_tahoe_xctx.py:
  C4: plate replicates are aggregated to a deterministic per-(line,drug,gene) CONSENSUS label
      (POSITIVE/TESTED_NEGATIVE only if ALL plates of that triple agree, else UNTESTED) + median log2FC,
      replacing the order-dependent aggfunc="first".
  C3: report (a) the within-line cross-plate CONFLICT FLOOR (same-context POS-vs-TN rate = pure replicate
      noise), (b) the LOOSE context-dependent count (POS in >=1 line, TN in >=1), and (c) the STRICT
      noise-robust count: >=2 lines POSITIVE with |log2FC| >= STRONG and >=1 line TESTED_NEGATIVE with
      |log2FC| < WEAK (a real separation, immune to single boundary flips).

Usage:
  python 05b_tahoe_xctx_corrected.py --labeled <...>/labeled_tahoe.csv --outdir <...>/tahoe_xctx
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.compute as pc

STRONG = 0.5    # a POSITIVE line must have |log2FC| >= this to anchor a context-dependent edge
WEAK = 0.15     # a TESTED_NEGATIVE line must have |log2FC| < this (genuinely excluded)

def log(*a): print("[xctx2]", *a, file=sys.stderr, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min-detected", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    cols = ["drug", "gene", "cell_line", "label_0.25", "log2FC"]
    log("reading", a.labeled)
    tbl = pacsv.read_csv(a.labeled, convert_options=pacsv.ConvertOptions(include_columns=cols))
    for c in ["drug", "gene", "cell_line", "label_0.25"]:
        i = tbl.schema.get_field_index(c)
        tbl = tbl.set_column(i, c, pc.dictionary_encode(tbl.column(c)))
    df = tbl.to_pandas(); del tbl
    log(f"{len(df)} rows")

    # ---- C4: deterministic plate consensus per (cell_line, drug, gene) -- VECTORIZED built-in aggs ----
    key = ["cell_line", "drug", "gene"]
    df["is_pos"] = df["label_0.25"].eq("POSITIVE")     # category-safe boolean indicators
    df["is_tn"] = df["label_0.25"].eq("TESTED_NEGATIVE")
    df["is_unt"] = df["label_0.25"].eq("UNTESTED")
    # cheap max-aggregations only (NO nunique -- that was the 56M-group bottleneck)
    cons = df.groupby(key, observed=True).agg(
        n_plate=("gene", "size"), log2FC=("log2FC", "median"),
        has_pos=("is_pos", "max"), has_tn=("is_tn", "max"), has_unt=("is_unt", "max")).reset_index()
    # conservative consensus: a triple keeps its label only if ALL plates agree (exactly one of the three
    # indicators set); any disagreement -> UNTESTED.
    cons["label"] = np.where(cons.has_pos & ~cons.has_tn & ~cons.has_unt, "POSITIVE",
                     np.where(cons.has_tn & ~cons.has_pos & ~cons.has_unt, "TESTED_NEGATIVE", "UNTESTED"))
    # plate-conflict FLOOR: among multi-plate triples that are decided on some plate, fraction with BOTH POS & TN
    multi_dec = cons[(cons["n_plate"] > 1) & (cons["has_pos"] | cons["has_tn"])]
    n_conflict = int((multi_dec["has_pos"] & multi_dec["has_tn"]).sum())
    floor = n_conflict / max(len(multi_dec), 1)
    log(f"plate-conflict FLOOR (same-context POS-vs-TN): {n_conflict}/{len(multi_dec)} = {floor:.1%}")

    # ---- pivot consensus label + lfc to (drug,gene) x line ----
    lab = cons.pivot_table(index=["drug", "gene"], columns="cell_line", values="label",
                           aggfunc="first", observed=True)
    lfc = cons.pivot_table(index=["drug", "gene"], columns="cell_line", values="log2FC",
                           aggfunc="first", observed=True)
    n_pos = lab.eq("POSITIVE").sum(axis=1)
    n_tn = lab.eq("TESTED_NEGATIVE").sum(axis=1)
    n_det = lab.notna().sum(axis=1)
    enough = n_det >= a.min_detected
    absl = lfc.abs()
    n_pos_strong = (lab.eq("POSITIVE") & absl.ge(STRONG)).sum(axis=1)
    n_tn_weak = (lab.eq("TESTED_NEGATIVE") & absl.lt(WEAK)).sum(axis=1)

    loose = enough & (n_pos >= 1) & (n_tn >= 1)
    strict = enough & (n_pos_strong >= 2) & (n_tn_weak >= 1)

    summary = pd.DataFrame([
        dict(metric="total (drug,gene) pairs", n=int(len(lab))),
        dict(metric=f"detectable in >={a.min_detected} lines", n=int(enough.sum())),
        dict(metric="LOOSE context-dependent (05 original definition, plate-consensus)", n=int(loose.sum())),
        dict(metric=f"STRICT context-dependent (>=2 POS |lfc|>={STRONG} & >=1 TN |lfc|<{WEAK})", n=int(strict.sum())),
        dict(metric="same-context plate-conflict floor (%)", n=round(100 * floor, 1)),
    ])
    summary.to_csv(os.path.join(a.outdir, "context_summary_corrected.csv"), index=False)
    log("corrected context summary:")
    print(summary.to_string(index=False), file=sys.stderr)

if __name__ == "__main__":
    main()
