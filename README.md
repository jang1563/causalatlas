# CausalAtlas

![license](https://img.shields.io/badge/license-Apache--2.0-green)
![release](https://img.shields.io/github/v/release/jang1563/causalatlas?label=release)
![validation](https://github.com/jang1563/causalatlas/actions/workflows/validate.yml/badge.svg)
[![HF artifacts](https://img.shields.io/badge/HF-artifacts-yellow)](https://huggingface.co/datasets/jang1563/causalatlas-move1)

**A measurement-first research release for asking where causal grounding comes from when an LLM orchestrates a
fallible single-cell perturbation foundation model (FM): the model, the agent, the RL loop, or the per-input
reliability signal.**

The headline is a *regime map*, not a flat verdict, and it ends constructively: the bottleneck is a per-input
reliability/value signal that is **mean-dominated and weak by default — but engineerable**. We measure this across
five settings on real perturbation data (Norman 2019 / GEARS, Replogle, Tahoe / Arc STATE) and real frontier LLMs.

<p align="center">
  <a href="results/move1/move1_arc.svg">
    <img src="results/move1/move1_arc.svg" width="760" alt="CausalAtlas Move 1 visual abstract: five measured limits converge on the per-input competence and value signal as the deployable lever for grounded perturbation-biology orchestration.">
  </a>
</p>
<p align="center">
  <sub><strong>Visual abstract.</strong> Five measured settings converge on one deployable recipe: default to the free additive baseline, flag FM disagreement, and verify the highest-value uncertain cases.</sub>
</p>

## At a glance

| Reader question | Where to go |
|---|---|
| What is the study claiming? | [`docs/CLAIMS.md`](docs/CLAIMS.md) |
| What is the full narrative account? | [`results/move1/MOVE1_WRITEUP_DRAFT.md`](results/move1/MOVE1_WRITEUP_DRAFT.md) |
| What can be checked by a script? | [`artifact_manifest.json`](artifact_manifest.json), [`schemas/`](schemas/) |
| What public artifacts can be downloaded directly? | [CausalAtlas Move 1 HF dataset](https://huggingface.co/datasets/jang1563/causalatlas-move1) |
| Where is the packaged verification benchmark? | [verify-or-trust](https://github.com/jang1563/verify-or-trust) and its [HF dataset](https://huggingface.co/datasets/jang1563/verify-or-trust) |
| What data is redistributed here? | No raw third-party data; see [`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md) |

## Quick validation

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

Download the mirrored lightweight artifacts:

```bash
huggingface-cli download jang1563/causalatlas-move1 --repo-type dataset --local-dir causalatlas_move1
```

## The arc (five measured settings)
1. **Does the FM ground causation?** Only narrowly. On an interpolation-proof held-out with a magnitude-robust
   discrimination metric, the free *observed-additive* baseline beats GEARS in aggregate; the FM wins only the
   non-additive (synergy) minority. (`eval/move1_layerA*.py`, `eval/move1_gears_cv.py`, `eval/move1_synergy_cv.py`)
2. **Can an LLM orchestrate it?** The verify-or-trust question — does the agent verify where the FM is wrong? It
   allocates near-chance; *more capable models over-verify more* (a cost-conditional capability inversion). The
   packaged, public benchmark for this layer is **[verify-or-trust](https://github.com/jang1563/verify-or-trust)**.
3. **The missing free action (3-tier).** Default-to-additive is the largest deployable, safety-aligned move; no
   frontier model discovers it unaided. (`eval/move1_3tier_*.py`)
4. **Can the agent choose the next experiment?** Goal-dependent: informed selection beats random ~2.7× for
   hit-discovery, but loses for uniform map-reconstruction. (`eval/move1_expsel*.py`)
5. **Is the signal a wall? No — it is buildable.** A controlled simulator (`eval/move1_p5_*.py`) isolates the
   lever, and a competence predictor (`eval/move1_competence_*.py`) lifts per-edge FM-error prediction from AUC
   ~0.69 (magnitude + regime) to **0.89** using *inference-available* disagreement with the additive baseline,
   driving near-oracle verification allocation. The central negative becomes constructive.

## Thesis
For perturbation-biology FMs, the binding constraint on grounded/agentic use is the **quality of the per-input
competence/value signal** — weak because effects are mean-dominated — not model scale, agent capability, or naive
RL. Agents *exploit* a supplied signal but cannot *self-generate* it; whether exploiting it helps is
**goal-dependent**; and the signal is **engineerable, not fundamental**.

## Layout
- `docs/CLAIMS.md` — claim-by-claim evidence map for readers and reviewers.
- `docs/REPRODUCIBILITY.md` — validation levels, expected artifacts, and rerun notes.
- `docs/DATA_PROVENANCE.md` — source datasets, transformation steps, and redistribution boundaries.
- `CONTRIBUTING.md`, `SUPPORT.md`, `SECURITY.md`, and `CODE_OF_CONDUCT.md` — contribution and reporting
  guidelines for the public repository.
- `artifact_manifest.json` — machine-readable map of claims, artifacts, linked datasets, and next quality upgrades.
- `schemas/` — JSON Schemas for released machine-readable artifacts.
- `.github/workflows/validate.yml` — CI check for strict JSON, manifest paths, and public-release guards.
- `eval/` — the pipeline (`01`–`07`: prep, sceptre three-state ground truth, GEARS/STATE prediction, labeling) and
  the Move-1 analyses (`move1_*.py`, LLM-free unless noted; the LLM-driven `move1_3tier_env.py` reads
  `ANTHROPIC_API_KEY` from the environment).
- `results/move1/` — the writeups (Layer A, 3-tier, experiment-selection design, synthesis) + small result tables.

## Data
No third-party data is redistributed here. Ground truth is regenerated by the pipeline from public sources:
Norman 2019 (GEO **GSE133344**, via GEARS — MIT), Replogle 2022, and Arc STATE / Tahoe (Arc's Hugging Face repo,
**non-commercial**; downloaded from source). The small Move-1 result artifacts are mirrored on the
**[CausalAtlas Move 1 Hugging Face dataset](https://huggingface.co/datasets/jang1563/causalatlas-move1)**.
The processed GEARS/Norman substrate + a cell subset for the live assay are published, with provenance, on the
**[verify-or-trust Hugging Face dataset](https://huggingface.co/datasets/jang1563/verify-or-trust)**. See
[`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md) for the source-by-source release policy.

## Related
- **[verify-or-trust](https://github.com/jang1563/verify-or-trust)** — the packaged, reproducible benchmark for
  Layer 2 (verification allocation), with an LLM-free value proof and an Arc STATE substrate.
- **[grounding-atlas](https://github.com/jang1563/grounding-atlas)** — the broader content-vs-name grounding
  program (train / retrieve / orchestrate decision map).

## Citation & license
See [`CITATION.cff`](CITATION.cff), [`.zenodo.json`](.zenodo.json), and
[`docs/ARCHIVAL_RELEASE.md`](docs/ARCHIVAL_RELEASE.md) for citation and archival metadata.
Code is Apache-2.0; third-party data retains its own license (see **Data**). A preprint is in preparation.
