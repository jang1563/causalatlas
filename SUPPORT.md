# Support

This repository is a public research release. The best support path depends on
the type of request.

## Use GitHub Issues For

- Reproducibility bugs in documented commands.
- Broken paths in `artifact_manifest.json`.
- Strict-JSON or schema-validation problems.
- Documentation, provenance, or citation corrections.
- Questions about release boundaries for included artifacts.

## Use Hugging Face For Downloads

- CausalAtlas Move 1 result artifacts:
  `https://huggingface.co/datasets/jang1563/causalatlas-move1`
- Verify-or-Trust benchmark data:
  `https://huggingface.co/datasets/jang1563/verify-or-trust`

## Before Opening An Issue

Please run:

```bash
python3 scripts/validate_public_release.py
```

Include the command, artifact path, expected behavior, observed behavior, and
environment details. Do not attach raw third-party datasets, regenerated caches,
run logs, access tokens, or unpublished data.
