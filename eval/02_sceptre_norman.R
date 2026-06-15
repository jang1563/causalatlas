#!/usr/bin/env Rscript
# =============================================================================
# CausalAtlas Phase 2 - Step 02: sceptre on Norman 2019 (K562 CRISPR-activation, GI)
# Produces the per-edge table (log2FC, SE, p) for the genetic-interaction classifier (03).
#
# Each grna_target is a perturbation IDENTITY: a single gene "A", a dual combo "A+B",
# or non-targeting (control). For every dual A+B and panel gene C the discovery set
# contains the TRIPLET {A->C, B->C, A+B->C}; Step 03 forms the additive prediction
# (A->C)+(B->C) and tests the combo residual for epistasis (three-state).
#
# pair_type tags (for Step 03):
#   cis_activation : single A -> A          (activation QC / G1; CRISPRa => log2FC > 0)
#   trans_single   : single A -> C  (C!=A)  (the additive-prediction components)
#   combo          : "A+B" -> C             (combo effect; C in {A,B} = combo cis-on-constituent)
#
# Usage:
#   module load R/4.4.1
#   R_LIBS_USER=$HOME/R/causalatlas Rscript 02_sceptre_norman.R \
#       --indir  ~/CausalAtlas/inputs/norman_gi_sub8 \
#       --outdir ~/CausalAtlas/results/norman_gi_sub8 \
#       [--response-extra hvg.txt]   # optional: extra response genes (one symbol/line) beyond drivers
#       [--calib-pairs 500] [--threads 32]
# =============================================================================
suppressPackageStartupMessages({
  library(sceptre); library(Matrix); library(data.table)
})
set.seed(42)

args <- commandArgs(trailingOnly = TRUE)
getopt <- function(flag, default=NULL) { i <- which(args==flag); if(length(i)) args[i+1] else default }
INDIR  <- getopt("--indir")
OUTDIR <- getopt("--outdir")
REXTRA <- getopt("--response-extra", NA)
CALIB  <- as.integer(getopt("--calib-pairs", "500"))
THREADS<- as.integer(getopt("--threads", "0"))
stopifnot(!is.null(INDIR), !is.null(OUTDIR))
dir.create(OUTDIR, recursive=TRUE, showWarnings=FALSE)
if (THREADS>0) { Sys.setenv(OMP_NUM_THREADS=THREADS, OPENBLAS_NUM_THREADS=THREADS) }

cat("[sceptre-norman] reading inputs from", INDIR, "\n")
dims <- scan(file.path(INDIR, "X_dims.txt"), what = integer(), quiet = TRUE)
ng <- dims[1]; nc <- dims[2]; nnz <- dims[3]
p_ <- readBin(file.path(INDIR, "X_indptr.i32"),  integer(), n = nc + 1, size = 4)
i_ <- readBin(file.path(INDIR, "X_indices.i32"), integer(), n = nnz,    size = 4)
x_ <- as.double(readBin(file.path(INDIR, "X_data.i32"), integer(), n = nnz, size = 4))
resp <- new("dgCMatrix", i = i_, p = p_, x = x_, Dim = c(ng, nc))
rm(i_, x_); gc()
feats <- read.delim(gzfile(file.path(INDIR, "features.tsv.gz")), header=FALSE, stringsAsFactors=FALSE)
bcs   <- readLines(gzfile(file.path(INDIR, "barcodes.tsv.gz")))
rownames(resp) <- make.unique(feats$V2)   # gene symbol
colnames(resp) <- bcs
cat(sprintf("  response: %d genes x %d cells\n", nrow(resp), ncol(resp)))

cov  <- fread(file.path(INDIR, "cell_covariates.csv"))
gdf  <- fread(file.path(INDIR, "grna_target_data_frame.csv"))   # grna_id, grna_target
stopifnot(all(cov$cell_barcode == bcs))

# ---- binary grna matrix (perturbation-identity x cells); each cell has one identity ----
guide_levels <- gdf$grna_id
gi <- match(cov$guide_id, guide_levels)
keep <- !is.na(gi)
grna_mat <- sparseMatrix(i = gi[keep], j = which(keep), x = 1L,
                         dims = c(length(guide_levels), ncol(resp)),
                         dimnames = list(guide_levels, bcs))
cat(sprintf("  grna: %d identities x %d cells (assigned=%d)\n",
            nrow(grna_mat), ncol(grna_mat), sum(keep)))

extra_cols <- intersect(c("percent_mito","percent_ribo"), names(cov))
extra <- if (length(extra_cols)) as.data.frame(cov[, ..extra_cols]) else data.frame(row.names=bcs)
if (nrow(extra)) rownames(extra) <- bcs

gtdf <- data.frame(grna_id = gdf$grna_id, grna_target = gdf$grna_target, stringsAsFactors=FALSE)
sobj <- import_data(response_matrix = resp, grna_matrix = grna_mat,
                    grna_target_data_frame = gtdf, moi = "low", extra_covariates = extra)

# ---- build GI discovery triplets ----
targets <- setdiff(unique(gtdf$grna_target), "non-targeting")
singles <- targets[!grepl("\\+", targets)]
combos  <- targets[grepl("\\+", targets)]
present_singles <- intersect(singles, rownames(resp))   # driver genes measurable as responses
# response panel = driver genes (+ optional extra HVG list), restricted to measured genes
panel <- present_singles
if (!is.na(REXTRA) && file.exists(REXTRA)) {
  ex <- intersect(readLines(REXTRA), rownames(resp))
  panel <- union(panel, ex)
}
cat(sprintf("  singles=%d (driver-present=%d) combos=%d | response panel=%d\n",
            length(singles), length(present_singles), length(combos), length(panel)))

# cis activation QC (single A -> A) = positive controls + G1
self_pairs <- data.frame(grna_target = present_singles, response_id = present_singles,
                         stringsAsFactors = FALSE)
# all targets (single + combo) x panel, minus the cis selves -> the additive components + combo effects
allpairs <- expand.grid(grna_target = targets, response_id = panel, stringsAsFactors=FALSE)
allpairs <- allpairs[!(allpairs$grna_target == allpairs$response_id), ]   # drop single cis (in self_pairs)
discovery_pairs <- unique(rbind(self_pairs, allpairs))
cat(sprintf("  discovery pairs: %d (cis_activation=%d, trans/combo=%d)\n",
            nrow(discovery_pairs), nrow(self_pairs), nrow(allpairs)))

sobj <- set_analysis_parameters(
  sobj,
  discovery_pairs = discovery_pairs,
  positive_control_pairs = self_pairs,
  side = "both",
  grna_integration_strategy = "union")

sobj <- assign_grnas(sobj, method = "thresholding", threshold = 1)
sobj <- run_qc(sobj, n_nonzero_trt_thresh = 7, n_nonzero_cntrl_thresh = 7)

cat("[sceptre-norman] calibration check...\n")
sobj <- run_calibration_check(sobj, n_calibration_pairs = CALIB, calibration_group_size = 1)
calib <- get_result(sobj, "run_calibration_check")
fwrite(calib, file.path(OUTDIR, "calibration_check.csv"))
cat(sprintf("  calibration FPR: %.2f%% (target <5%%)\n", 100*mean(calib$significant, na.rm=TRUE)))

cat("[sceptre-norman] power check (cis-activation positive controls)...\n")
sobj <- run_power_check(sobj)
fwrite(get_result(sobj, "run_power_check"), file.path(OUTDIR, "power_check.csv"))

cat("[sceptre-norman] discovery analysis...\n")
sobj <- run_discovery_analysis(sobj)
disc <- get_result(sobj, "run_discovery_analysis")

# pair_type for Step 03: cis_activation | trans_single | combo
gt <- as.character(disc$grna_target); rid <- as.character(disc$response_id)
disc$pair_type <- ifelse(grepl("\\+", gt), "combo",
                  ifelse(gt == rid, "cis_activation", "trans_single"))
fwrite(disc, file.path(OUTDIR, "discovery_results.csv"))
saveRDS(sobj, file.path(OUTDIR, "sceptre_object.rds"))

cat(sprintf("[sceptre-norman] done. rows=%d (cis_activation=%d trans_single=%d combo=%d) -> %s\n",
            nrow(disc), sum(disc$pair_type=="cis_activation"),
            sum(disc$pair_type=="trans_single"), sum(disc$pair_type=="combo"), OUTDIR))
