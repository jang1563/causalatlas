# Reproducibility Notes

This document describes the public-release reproducibility levels for CausalAtlas.
The repository is designed to make three levels easy:

1. Inspect the claims and frozen result artifacts.
2. Validate machine-readable artifacts and schemas.
3. Re-run selected analyses after regenerating inputs from public sources.

Raw third-party single-cell data is not redistributed in this repository. The public
Move-1 result artifacts and the Verify-or-Trust benchmark substrate are hosted
separately on Hugging Face.

## Public Artifacts

Core entrypoints:

- `README.md` -- project overview.
- `docs/CLAIMS.md` -- claim-by-claim evidence map.
- `docs/DATA_PROVENANCE.md` -- source datasets, transformation steps, and redistribution boundaries.
- `docs/ARCHIVAL_RELEASE.md` -- DOI-ready release metadata and archive invariants.
- `artifact_manifest.json` -- machine-readable artifact and claim map.
- `results/move1/layerA_norman.json` -- Norman / GEARS layer-A result artifact.
- `results/move1/layerA_tahoe.json` -- Tahoe / Arc STATE layer-A result artifact.
- `results/move1/MOVE1_WRITEUP_DRAFT.md` -- full written account.

Linked public result artifacts:

```bash
huggingface-cli download jang1563/causalatlas-move1 --repo-type dataset --local-dir causalatlas_move1
```

Python artifact-loading example:

```python
import json
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="jang1563/causalatlas-move1",
    filename="results/move1/layerA_norman.json",
    repo_type="dataset",
)

with open(path) as f:
    layer_a_norman = json.load(f)
```

Linked public benchmark data:

```bash
huggingface-cli download jang1563/verify-or-trust --repo-type dataset --local-dir vot_data
```

Python loading example:

```python
from datasets import load_dataset

ds = load_dataset("jang1563/verify-or-trust", "gears_norman")
```

## Quick Validation

The same checks run in GitHub Actions via `.github/workflows/validate.yml`.
For a local run:

```bash
python3 scripts/validate_public_release.py
```

Validate that the public JSON artifacts parse:

```bash
python3 -m json.tool artifact_manifest.json >/tmp/causalatlas_manifest.json
python3 -m json.tool schemas/artifact_manifest.schema.json >/tmp/causalatlas_manifest_schema.json
python3 -m json.tool schemas/layerA_norman.schema.json >/tmp/layerA_norman_schema.json
python3 -m json.tool schemas/layerA_tahoe.schema.json >/tmp/layerA_tahoe_schema.json
python3 -m json.tool results/move1/layerA_norman.json >/tmp/layerA_norman.json
python3 -m json.tool results/move1/layerA_tahoe.json >/tmp/layerA_tahoe.json
```

Strict JSON check for non-standard constants such as `NaN`:

```bash
python3 -c 'import json, pathlib; [json.loads(p.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))) for p in map(pathlib.Path, ["artifact_manifest.json", "results/move1/layerA_norman.json", "results/move1/layerA_tahoe.json"])]'
```

Expected result: exit code 0.

Optional schema validation, if `jsonschema` is available:

```bash
python3 -m jsonschema schemas/artifact_manifest.schema.json -i artifact_manifest.json
python3 -m jsonschema schemas/layerA_norman.schema.json -i results/move1/layerA_norman.json
python3 -m jsonschema schemas/layerA_tahoe.schema.json -i results/move1/layerA_tahoe.json
```

## Re-running Analyses

The `eval/` directory has two layers:

- `01` to `07` scripts regenerate data-processing and labeling substrates from upstream public data.
- `move1_*.py` scripts compute the Move-1 measurements from prepared substrates and result tables.

The full raw-data pipeline requires downloading the relevant upstream datasets and installing the scientific
stack used by each substrate:

- Python: `numpy`, `pandas`, `scipy`, `scikit-learn`, `anndata` or `h5py` depending on the script.
- R: `sceptre`, `Matrix`, `data.table`.
- Optional LLM run: `eval/move1_3tier_env.py` reads `ANTHROPIC_API_KEY` from the environment.

The released repository intentionally keeps the small public result artifacts and excludes large regenerated
data products. For most readers, `docs/CLAIMS.md`, `artifact_manifest.json`, and `results/move1/*.json` are the
fastest reproducibility surface.

## Expected Result Artifacts

Layer-A Norman:

- Path: `results/move1/layerA_norman.json`
- Schema: `schemas/layerA_norman.schema.json`
- Main fields: `primary_delta`, `B_boot`, `combo`, `single`, `pds`, `delta_sweep`

Layer-A Tahoe:

- Path: `results/move1/layerA_tahoe.json`
- Schema: `schemas/layerA_tahoe.schema.json`
- Main fields: `n_pert`, `n_edge`, `no_effect_base_rate`, `primary_delta`, `acc`, `auroc`, `pds`

Machine-readable manifest:

- Path: `artifact_manifest.json`
- Schema: `schemas/artifact_manifest.schema.json`
- Main fields: `project`, `linked_public_artifacts`, `entrypoints`, `claims`, `result_artifacts`,
  `pipeline_files`, `schema_files`

## Release Invariants

The public repository should keep these invariants:

- One clean public history root for the release branch.
- No raw third-party data redistribution.
- No local private archive.
- No run logs or large regenerated artifacts.
- JSON artifacts are strict JSON.
- The claim map and manifest point to files that exist in the public tree.
- The CI validator passes on `main`.
