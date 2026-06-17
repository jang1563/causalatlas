# Contributing

Thanks for your interest in CausalAtlas. This is a public research release, so
contributions should improve correctness, reproducibility, provenance, or the
clarity of the released artifacts.

## Local Checks

Before opening a pull request, run:

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

If you change JSON artifacts, also check that they parse as strict JSON and
match the schemas under `schemas/`.

## Ground Rules

- Keep the release license-clean: do not add raw third-party single-cell data,
  model checkpoints, regenerated caches, run logs, or local execution traces.
- Keep secrets out of the repository. API keys and service tokens must come from
  the environment.
- Keep claims traceable. New or changed claims should be reflected in
  `docs/CLAIMS.md`, `artifact_manifest.json`, and the relevant result artifacts.
- Keep machine-readable artifacts stable. Prefer explicit schemas, strict JSON,
  and deterministic commands.
- Document redistribution boundaries in `docs/DATA_PROVENANCE.md` whenever a
  contribution touches data sources or derived artifacts.

## Adding Result Artifacts

Small, derived result artifacts may be added when they are useful for review and
do not package raw upstream data. For each new machine-readable result:

1. Add the artifact under a stable path.
2. Add or update its schema under `schemas/`.
3. Link it from `artifact_manifest.json`.
4. Add a short interpretation path in `docs/CLAIMS.md` or
   `docs/REPRODUCIBILITY.md`.
5. Run the public validator.

## Scope

Good contributions include reproducibility fixes, schema tightening, clearer
provenance, result-artifact validation, and small documentation improvements.
Large regenerated data products should remain out of the GitHub tree unless a
separate release plan explains why they are license-clean and lightweight.
