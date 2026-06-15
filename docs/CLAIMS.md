# CausalAtlas Claim Map

This document is the human-readable evidence map for the public CausalAtlas release.
It turns the repository into a set of checkable claims: each claim has a short statement,
the main evidence, the exact artifact paths, and the fastest way to inspect or reproduce it.

## How to Read This Repository

- Start with `README.md` for the high-level thesis.
- Read `results/move1/MOVE1_WRITEUP_DRAFT.md` for the narrative account.
- Use this file to jump from each claim to evidence, code, and limitations.
- Use `artifact_manifest.json` for a machine-readable version of the same map.

## Claims

| ID | Claim | Evidence | Primary artifacts |
|---|---|---|---|
| C1 | The perturbation foundation model has a regime map, not a flat win or flat loss. | GEARS loses to observed-additive in aggregate on held-out Norman combos, but wins on the non-additive synergy subset. | `results/move1/RESULT_layerA.md`, `results/move1/layerA_norman.json` |
| C2 | Arc STATE / Tahoe does not beat the no-change floor on novel-drug edge calls under the primary call-accuracy metric. | STATE call accuracy is 0.732, below the no-change floor of 0.805; cosine-PDS is higher, showing a metric-dependent split. | `results/move1/RESULT_layerA.md`, `results/move1/layerA_tahoe.json` |
| C3 | LLM orchestration fails mainly as verification allocation, not as missing biology knowledge. | Verify-or-Trust shows near-chance FM-wrong allocation, over-verification under cost, and strong following of an external reliability signal. | `results/move1/MOVE1_SYNTHESIS.md`, `results/move1/MOVE1_WRITEUP_DRAFT.md`, `https://github.com/jang1563/verify-or-trust` |
| C4 | Adding the free additive baseline creates a three-tier decision problem, but tested frontier models do not discover the default-to-baseline move unaided. | Always-additive beats all tested LLM policies at non-trivial verification cost; models either over-verify or over-trust the FM. | `results/move1/MOVE1_3TIER_RESULT.md`, `eval/move1_3tier_baselines.py`, `eval/move1_3tier_env.py`, `eval/move1_3tier_router.py` |
| C5 | Experiment selection is goal-dependent: informed selection helps hit-discovery but hurts uniform map reconstruction. | Replogle active-learning results show uncertainty/k-center worse than random for uniform mapping, while GO-predicted value beats random for hit-search. | `results/move1/EXPSEL_DESIGN.md`, `eval/move1_expsel.py`, `eval/move1_expsel_checks.py`, `eval/move1_replogle_prep.py` |
| C6 | The per-input competence signal is the main bottleneck, but it is engineerable. | Additive-disagreement features lift FM-error prediction from the weak-signal regime to about 0.88 AUC, enabling near-oracle verification allocation at low budget. | `results/move1/MOVE1_SYNTHESIS.md`, `results/move1/MOVE1_WRITEUP_DRAFT.md`, `eval/move1_competence_uq.py`, `eval/move1_competence_close.py` |

## Claim Details

### C1. Foundation-model grounding is regime-dependent

Statement: GEARS does not beat the observed-additive baseline in aggregate on held-out Norman combo edges,
but it captures a non-additive interaction regime that additive cannot.

Key numbers:
- Held-out combo aggregate: GEARS AUROC 0.790 vs additive AUROC 0.897 on all held-out combos.
- Additive-edge subset: GEARS AUROC 0.755 vs additive AUROC 0.945.
- Synergy-edge subset: GEARS AUROC 0.749 vs additive AUROC 0.606; GEARS-additive gap +0.142 with 95% CI [+0.050, +0.233].

Primary files:
- `results/move1/RESULT_layerA.md`
- `results/move1/layerA_norman.json`
- `eval/move1_layerA.py`
- `eval/move1_gears_cv.py`
- `eval/move1_synergy_cv.py`

Limitations:
- Observed-additive uses measured singles and is a strong field-standard baseline.
- The result is scoped to public-data-rich Norman/GEARS perturbation settings.

### C2. Tahoe / Arc STATE shows metric-dependent grounding

Statement: On Tahoe novel-drug edge calls, STATE falls below a no-change baseline under call accuracy,
while cosine-PDS gives a more favorable perturbation-identity signal.

Key numbers:
- STATE call accuracy: 0.732 [0.728, 0.736].
- No-change call accuracy: 0.805 [0.800, 0.809].
- STATE edge-AUROC: 0.439.
- STATE cosine-PDS: 0.793.

Primary files:
- `results/move1/RESULT_layerA.md`
- `results/move1/layerA_tahoe.json`
- `eval/move1_layerA_tahoe.py`

Limitations:
- Tahoe / Arc STATE artifacts are governed by their upstream non-commercial terms.
- The edge-AUROC metric is compressed on this substrate; call accuracy and PDS should be read together.

### C3. LLM orchestration is limited by reliability allocation

Statement: The LLM can follow a supplied reliability signal but does not reliably infer where the FM is wrong
from the raw instance on its own.

Evidence summary:
- Verify-or-Trust measures trust-vs-verify allocation against held-out experiment truth.
- More capable models verify more under cost but do not allocate verification accurately enough.
- External reliability signals are followed strongly, indicating that the bottleneck is signal availability and quality.

Primary files:
- `results/move1/MOVE1_SYNTHESIS.md`
- `results/move1/MOVE1_WRITEUP_DRAFT.md`
- `https://github.com/jang1563/verify-or-trust`
- `https://huggingface.co/datasets/jang1563/verify-or-trust`

Limitations:
- The benchmark is a controlled orchestration task, not a general claim about all LLM tool use.

### C4. The additive baseline creates a missing third action

Statement: Once the free additive baseline is available, the best simple policy is often to use it by default.
Frontier LLMs tested in the three-tier setting do not discover this unaided.

Key numbers:
- Always-additive free default: 0.802 accuracy, net 0.802.
- Trust-all-FM: 0.661 accuracy, net 0.661.
- Haiku/Sonnet/Opus at lambda 0.5: net 0.725, 0.722, 0.730, all below always-additive.

Primary files:
- `results/move1/MOVE1_3TIER_RESULT.md`
- `eval/move1_3tier_baselines.py`
- `eval/move1_3tier_router.py`
- `eval/move1_3tier_env.py`

Limitations:
- Single neutral prompt family.
- The result is cost-dependent; near-free verification changes the policy ranking.

### C5. Experiment selection depends on the goal

Statement: Informed selection is not generically better than random. It helps when the goal is hit-discovery,
but hurts when the goal is representative map reconstruction.

Key numbers:
- Uniform map-reconstruction: informed uncertainty/k-center selection is worse than random, with AULC gap about -0.018.
- Hit-discovery: GO-predicted effect-magnitude selection beats random by about 2.7x in recall at budget.

Primary files:
- `results/move1/EXPSEL_DESIGN.md`
- `eval/move1_expsel.py`
- `eval/move1_expsel_checks.py`
- `eval/move1_replogle_prep.py`

Limitations:
- Offline selection is corpus-bound; a true discovery loop requires new experiments or a validated simulator.

### C6. The competence signal is engineerable

Statement: The weak competence-signal ceiling is not fundamental. A feature based on disagreement with the
additive baseline gives a much stronger, inference-available FM-error predictor.

Key numbers:
- Prior weak-signal regime: about 0.67 to 0.73 AUC across trust heads, probes, and free routers.
- Additive-disagreement competence predictor: about 0.88 AUC.
- Signal-gated verification with the stronger predictor approaches oracle allocation at low verification budget.

Primary files:
- `results/move1/MOVE1_SYNTHESIS.md`
- `results/move1/MOVE1_WRITEUP_DRAFT.md`
- `eval/move1_competence_uq.py`
- `eval/move1_competence_close.py`

Limitations:
- The strongest feature is combo-specific; other perturbation settings need analogous inference-available features.

## Reproducibility Pointers

The repository does not redistribute third-party raw data. The code regenerates substrates from public sources
described in `README.md`, and small released result artifacts are stored under `results/move1/`.

Fast checks:

```bash
python3 -m json.tool artifact_manifest.json >/tmp/causalatlas_manifest.json
python3 -m json.tool results/move1/layerA_norman.json >/tmp/layerA_norman.json
python3 -m json.tool results/move1/layerA_tahoe.json >/tmp/layerA_tahoe.json
```

Public benchmark data:

```python
from datasets import load_dataset

ds = load_dataset("jang1563/verify-or-trust", "gears_norman")
```

## Machine-Readable Companion

See `artifact_manifest.json` for a structured map of repository files, claims, metrics, linked datasets,
schemas, and next-step quality upgrades. See `docs/REPRODUCIBILITY.md` for validation commands.
