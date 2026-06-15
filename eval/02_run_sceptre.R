#!/usr/bin/env Rscript
# =============================================================================
# CausalAtlas Phase 1 - Step 02: sceptre on Replogle K562_essential (low MOI)
# Produces the per-edge test table (log2FC, SE, p, q) that Step 03 turns into
# {tested-positive, tested-negative, untested} labels via the locked equivalence rule.
#
#   cis self-pairs  (gene -> own gene)      = knockdown QC (gate G1) + positive controls
#   trans pairs     (regulator -> target)   = the powered-subspace edges to classify
#
# Usage:
#   module load R/4.4.1
#   R_LIBS_USER=$HOME/R/causalatlas Rscript 02_run_sceptre.R \
#       --indir  ~/CausalAtlas/inputs/K562_essential \
#       --outdir ~/CausalAtlas/results/K562_essential \
#       [--pairs trans_pairs.csv]   # cols: grna_target,response_id  (gene symbols)
#       [--calib-pairs 500] [--threads 32]
# =============================================================================
suppressPackageStartupMessages({
  library(sceptre); library(Matrix); library(data.table)
})
set.seed(42)

# ---- args ----
args <- commandArgs(trailingOnly = TRUE)
getopt <- function(flag, default=NULL) { i <- which(args==flag); if(length(i)) args[i+1] else default }
INDIR  <- getopt("--indir")
OUTDIR <- getopt("--outdir")
PAIRS  <- getopt("--pairs", NA)
CALIB  <- as.integer(getopt("--calib-pairs", "500"))
THREADS<- as.integer(getopt("--threads", "0"))
stopifnot(!is.null(INDIR), !is.null(OUTDIR))
dir.create(OUTDIR, recursive=TRUE, showWarnings=FALSE)
if (THREADS>0) { Sys.setenv(OMP_NUM_THREADS=THREADS, OPENBLAS_NUM_THREADS=THREADS) }

cat("[sceptre] reading inputs from", INDIR, "\n")
# response matrix: genes x cells (raw counts)
# binary CSC transfer from Step 01 (readBin -> dgCMatrix; avoids slow text MatrixMarket)
dims <- scan(file.path(INDIR, "X_dims.txt"), what = integer(), quiet = TRUE)
ng <- dims[1]; nc <- dims[2]; nnz <- dims[3]
p_ <- readBin(file.path(INDIR, "X_indptr.i32"),  integer(), n = nc + 1, size = 4)
i_ <- readBin(file.path(INDIR, "X_indices.i32"), integer(), n = nnz,    size = 4)
x_ <- as.double(readBin(file.path(INDIR, "X_data.i32"), integer(), n = nnz, size = 4))
resp <- new("dgCMatrix", i = i_, p = p_, x = x_, Dim = c(ng, nc))
rm(i_, x_); gc()
feats <- read.delim(gzfile(file.path(INDIR, "features.tsv.gz")), header=FALSE,
                    stringsAsFactors=FALSE)
bcs   <- readLines(gzfile(file.path(INDIR, "barcodes.tsv.gz")))
# use gene SYMBOL as response id so target<->response pairing is by symbol
rownames(resp) <- make.unique(feats$V2)
colnames(resp) <- bcs
cat(sprintf("  response: %d genes x %d cells\n", nrow(resp), ncol(resp)))

cov  <- fread(file.path(INDIR, "cell_covariates.csv"))
gdf  <- fread(file.path(INDIR, "grna_target_data_frame.csv"))   # grna_id, grna_target
stopifnot(all(cov$cell_barcode == bcs))

# ---- build binary grna matrix (guides x cells) from per-cell guide_id (low MOI) ----
guide_levels <- gdf$grna_id
gi <- match(cov$guide_id, guide_levels)
keep <- !is.na(gi)
grna_mat <- sparseMatrix(i = gi[keep], j = which(keep), x = 1L,
                         dims = c(length(guide_levels), ncol(resp)),
                         dimnames = list(guide_levels, bcs))
cat(sprintf("  grna: %d guides x %d cells (assigned cells=%d)\n",
            nrow(grna_mat), ncol(grna_mat), sum(keep)))

# numeric extra covariates for the formula (sceptre adds log umi etc. by default)
extra_cols <- intersect(c("percent_mito","percent_ribo"), names(cov))
extra <- if (length(extra_cols)) as.data.frame(cov[, ..extra_cols]) else data.frame(row.names=bcs)
if (nrow(extra)) rownames(extra) <- bcs

# ---- import + parameters ----
gtdf <- data.frame(grna_id = gdf$grna_id, grna_target = gdf$grna_target, stringsAsFactors=FALSE)
sobj <- import_data(response_matrix = resp,
                    grna_matrix = grna_mat,
                    grna_target_data_frame = gtdf,
                    moi = "low",
                    extra_covariates = extra)

targets <- setdiff(unique(gtdf$grna_target), "non-targeting")
present_targets <- intersect(targets, rownames(resp))
cat(sprintf("  targets present as response genes (cis-testable): %d / %d\n",
            length(present_targets), length(targets)))

# cis self-pairs = knockdown QC + positive controls
self_pairs <- data.frame(grna_target = present_targets, response_id = present_targets,
                         stringsAsFactors = FALSE)

# trans pairs: from --pairs if given, else a default validation subspace
if (!is.na(PAIRS) && file.exists(PAIRS)) {
  trans <- fread(PAIRS); trans <- as.data.frame(trans[, .(grna_target, response_id)])
  trans <- trans[trans$grna_target %in% targets & trans$response_id %in% rownames(resp), ]
  cat(sprintf("  trans pairs from %s: %d\n", PAIRS, nrow(trans)))
} else {
  # DEFAULT (validation): perturbed-gene -> perturbed-gene, capped for a quick run.
  # Replace with the curated regulator->candidate-target subspace (next-steps #2).
  rg <- head(present_targets, 50)
  trans <- expand.grid(grna_target = rg, response_id = present_targets, stringsAsFactors=FALSE)
  trans <- trans[trans$grna_target != trans$response_id, ]
  cat(sprintf("  trans pairs DEFAULT (validation, 50 regulators x %d targets): %d\n",
              length(present_targets), nrow(trans)))
}
discovery_pairs <- unique(rbind(self_pairs, trans))

sobj <- set_analysis_parameters(
  sobj,
  discovery_pairs = discovery_pairs,
  positive_control_pairs = self_pairs,
  side = "both",
  grna_integration_strategy = "union")

# ---- assign (binary indicator -> threshold 1), QC, calibration, power, discovery ----
sobj <- assign_grnas(sobj, method = "thresholding", threshold = 1)
sobj <- run_qc(sobj, n_nonzero_trt_thresh = 7, n_nonzero_cntrl_thresh = 7)

cat("[sceptre] calibration check...\n")
sobj <- run_calibration_check(sobj, n_calibration_pairs = CALIB, calibration_group_size = 1)
calib <- get_result(sobj, "run_calibration_check")
fwrite(calib, file.path(OUTDIR, "calibration_check.csv"))
cat(sprintf("  calibration FPR: %.2f%% (target <5%%)\n",
            100*mean(calib$significant, na.rm=TRUE)))

cat("[sceptre] power check (positive-control = knockdown self-pairs)...\n")
sobj <- run_power_check(sobj)
fwrite(get_result(sobj, "run_power_check"), file.path(OUTDIR, "power_check.csv"))

cat("[sceptre] discovery analysis...\n")
sobj <- run_discovery_analysis(sobj)
disc <- get_result(sobj, "run_discovery_analysis")

# tag cis (knockdown) vs trans so Step 03 can apply G1
disc$pair_type <- ifelse(as.character(disc$grna_target) == as.character(disc$response_id),
                         "cis_knockdown", "trans")
fwrite(disc, file.path(OUTDIR, "discovery_results.csv"))
saveRDS(sobj, file.path(OUTDIR, "sceptre_object.rds"))

cat(sprintf("[sceptre] done. discovery rows=%d (cis=%d trans=%d) -> %s\n",
            nrow(disc), sum(disc$pair_type=="cis_knockdown"),
            sum(disc$pair_type=="trans"), OUTDIR))
