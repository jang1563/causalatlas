# Data Provenance

This repository is a public research release for CausalAtlas. It contains code,
small result artifacts, schemas, and writeups. It does not redistribute raw
third-party single-cell matrices, large regenerated intermediates, model
checkpoints, or run logs.

## Source Families

| source | upstream home | role in this repo | public-release handling |
|---|---|---|---|
| Norman 2019 Perturb-seq | NCBI GEO `GSE133344` | K562 CRISPRa single and pair perturbations; source for the Norman / GEARS layer-A and verification-allocation substrate. | Raw cells are not stored here. Preparation and labeling scripts regenerate derived tables from a local upstream download. |
| GEARS | `https://github.com/snap-stanford/GEARS` | Perturbation foundation-model predictions for Norman. The upstream repository is MIT licensed. | Code calls GEARS from the local scientific environment; GEARS checkpoints and regenerated prediction caches are not stored here. |
| Replogle / Weissman K562 essential screen | public Replogle 2022 Perturb-seq release; local input file expected as `ReplogleWeissman2022_K562_essential.h5ad` | CRISPRi ground-truth pipeline and experiment-selection substrate. | Raw h5ad and regenerated count/covariate matrices are not stored here. |
| Arc STATE / Tahoe | `https://huggingface.co/arcinstitute/ST-HVG-Tahoe` | Drug perturbation substrate for the Tahoe / Arc STATE layer-A check. | Upstream model/data terms apply, including the model license and acceptable-use policy in the Arc repository. Tahoe raw outputs are not redistributed here. |
| Verify-or-Trust benchmark dataset | `https://huggingface.co/datasets/jang1563/verify-or-trust` | Public packaged substrate for the verification-allocation layer. | Hosted separately on Hugging Face with its own dataset card, provenance notes, and Apache-2.0 metadata for the released derived artifacts. |

## Transformation Chain

The public pipeline is organized as a sequence of reproducible transformations:

1. **Input preparation.** `eval/01_prep_replogle.py` and
   `eval/01_prep_norman.py` convert upstream h5ad inputs into count matrices,
   cell covariates, and guide-target tables used by the statistical labeling
   steps.
2. **Ground-truth labeling.** `eval/02_run_sceptre.R`,
   `eval/02_sceptre_norman.R`, `eval/03_classify_edges.py`, and
   `eval/03_gi_classify.py` build three-state labels:
   tested-positive, tested-negative, and untested.
3. **Foundation-model prediction.** `eval/06_gears_norman.py` and
   `eval/06b_gears_predict_all.py` generate GEARS predictions for Norman.
   Tahoe / Arc STATE comparisons are built from upstream Arc released
   prediction and real-DE tables.
4. **Move-1 analyses.** `eval/move1_*.py` scripts aggregate the prepared
   substrates into the released result artifacts under `results/move1/`.
5. **Verification-allocation benchmark.** The packaged benchmark layer lives in
   the linked `verify-or-trust` repository and Hugging Face dataset.

## Released Artifacts

The public repository includes:

- `results/move1/layerA_norman.json`
- `results/move1/layerA_tahoe.json`
- `results/move1/RESULT_layerA.md`
- `results/move1/MOVE1_3TIER_RESULT.md`
- `results/move1/EXPSEL_DESIGN.md`
- `results/move1/MOVE1_SYNTHESIS.md`
- `results/move1/MOVE1_WRITEUP_DRAFT.md`
- `results/move1/move1_arc.svg`
- `artifact_manifest.json`
- `schemas/*.json`

The linked Hugging Face dataset includes the derived GEARS/Norman substrate and
the Norman cell subset needed by the live `run_de` tool in Verify-or-Trust.

## Not Released Here

The following are intentionally excluded from the public GitHub tree:

- Raw upstream h5ad files.
- Regenerated count matrices, covariate tables, and large intermediate CSVs.
- GEARS model checkpoints and training caches.
- Tahoe raw prediction/real-DE downloads governed by Arc terms.
- LLM run logs and local execution traces.

These exclusions keep the repository license-clean, lightweight, and reviewable.
They also avoid repackaging third-party data under the CausalAtlas code license.

## Validation

The public release validator checks that the released tree stays within this
policy:

```bash
python3 scripts/validate_public_release.py
```

Expected output:

```text
strict JSON OK
manifest OK
tracked file set OK
public text OK
```

For reproduction levels and rerun notes, see `docs/REPRODUCIBILITY.md`. For the
claim-by-claim evidence map, see `docs/CLAIMS.md`.
