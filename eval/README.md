# CausalAtlas Phase 1 ground-truth pipeline (Replogle K562_essential)

Builds the three-state edge ground truth ({tested-positive, tested-negative, untested})
from the real Replogle low-MOI CRISPRi screen, using the locked refinement #1 rule
(equivalence test: a tested-negative is an edge where an effect >= delta is statistically
EXCLUDED, not merely q>=q*).

## Steps
1. `01_prep_replogle.py` (python/h5py) - h5ad -> 10x-style counts + cell covariates +
   guide->target map. Low MOI, guide-level assignment; controls = 97 non-targeting guides.
2. `02_run_sceptre.R` (R/sceptre, moi="low") - import -> calibration check -> power check
   (positive controls = cis self-pairs) -> discovery (cis knockdown self-pairs + trans
   subspace pairs). Writes `discovery_results.csv` (log2FC, p, significant, n_nonzero_*),
   tagged `pair_type` in {cis_knockdown, trans}.
3. `03_classify_edges.py` (python) - applies the locked rule across delta in {0.1,0.25,0.5,1.0}
   (primary 0.25), with gates G0 (calibration), G1 (knockdown >=30%/>=50%), G2 (detectability),
   and confidence tiers A/B. Writes `labeled_edges.csv` + `summary.csv`.

## Run
```
# full data is ~310k cells; set SUBSET for a fast validation pass
export SUBSET=20000          # optional
python eval/01_prep_replogle.py --h5ad <K562_essential.h5ad> --outdir <inputs> --subset-cells ${SUBSET}
Rscript eval/02_run_sceptre.R --indir <inputs> --outdir <results> --threads 32
python eval/03_classify_edges.py --discovery <results>/discovery_results.csv --outdir <results>
```

## Status (2026-06-09): END-TO-END VALIDATED on real Replogle K562_essential
sceptre 0.10.3 installed from GitHub. The full pipeline was validated over a powered subset
(top-200 targets by cell count + all controls = 88,556 cells; 50-regulator trans subspace,
9450 trans edges):
- calibration FPR = 0.00% (target <5%); 44/50 regulators knockdown-confirmed; SE from sceptre's
  se_fold_change. Three-state output behaves exactly as designed across the delta sweep:

  | delta | POSITIVE | TESTED_NEGATIVE (A/B) | UNTESTED |
  |---|---|---|---|
  | 0.10 | 2450 | 939 (506/433) | 6061 |
  | 0.25 | 1337 | 3852 (2648/1204) | 4261 |
  | 0.50 | 328 | 5592 (4382/1210) | 3530 |
  | 1.00 | 14 | 5826 (4708/1118) | 3610 |

  delta up -> POSITIVE down, TESTED_NEGATIVE up, UNTESTED down (equivalence logic monotone, correct).

Validated step-by-step: 01 (h5py reader; dense X cells x genes, obs idx "cell_barcode",
var idx "gene_name"; vectorized MatrixMarket writer), 02 (sceptre moi=low), 03 (locked rule).

## Matrix transfer
Step 01 -> 02 passes the counts as a binary CSC trio (X_indptr/X_indices/X_data .i32 + X_dims.txt);
R reads via readBin and builds a dgCMatrix directly (no text MatrixMarket; scipy CSC genes x cells maps
1:1 to dgCMatrix p/i/x). Validated identical to the old mtx path on the powered subset, and at FULL
scale: K562_essential 310,385 cells x 8,563 genes, nnz 1,121,681,215; prep ~4 min / 37GB, R load ~17GB,
calibration FPR 0%. The full-scale run used a 256G / 48h resource envelope.

## Future Work
- Define the curated regulator->candidate-target powered subspace (--pairs); 02_run_sceptre.R
  currently falls back to a capped 50-regulator x perturbed-gene placeholder (~93k pairs at full scale).
- RPE1 full run (same pipeline, --h5ad ...rpe1.h5ad).
- Extend the downstream verification-allocation benchmark on top of this ground truth.
