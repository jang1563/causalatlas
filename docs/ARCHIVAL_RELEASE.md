# Archival Release Notes

This repository is prepared for DOI-backed archival releases through GitHub
Releases and Zenodo.

## Current Frozen Release

- Version: `0.1.3`
- GitHub release: `https://github.com/jang1563/causalatlas/releases/tag/v0.1.3`
- Canonical revision: Git tag `v0.1.3`
- Companion artifact dataset: `https://huggingface.co/datasets/jang1563/causalatlas-move1`

The GitHub tag is the canonical software snapshot. The Hugging Face dataset
mirrors lightweight Move-1 result artifacts, schemas, and provenance documents;
it is a companion artifact store, not the source-code archive.

## DOI Update Checklist

After Zenodo mints a DOI for a GitHub release:

1. Add the DOI badge to `README.md`.
2. Add a DOI identifier to `CITATION.cff`.
3. Add the DOI to `artifact_manifest.json` under the `release` object.
4. Add the DOI as a related identifier in `.zenodo.json` if a later archive
   snapshot should explicitly reference the previous DOI.
5. Re-run `python3 scripts/validate_public_release.py`.

Do not add a DOI placeholder before the DOI exists.

## Release Invariants

- No raw third-party data is redistributed.
- No run logs or local execution traces are part of the archive.
- Machine-readable JSON artifacts remain strict JSON.
- The manifest points only to public files or public external URLs.
- GitHub Actions must pass on the release commit.
