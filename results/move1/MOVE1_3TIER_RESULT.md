# LLM 3-tier arm — does Claude discover "default to the cheap baseline"? (2026-06-13)

The behavioral capstone of Move 1. Given, per gene, a FREE additive baseline AND a FREE foundation-model (GEARS)
prediction, plus a COSTLY real assay, does the LLM discover the most basic grounded-orchestrator move: default to
the cheap baseline (additive 0.802 > trust-FM 0.661 on these panels) rather than trusting the unreliable
specialist? 52 held-out combo panels (combo_seen2 CV preds), Haiku/Sonnet/Opus, neutral prompt (NOT told which
predictor is better). Env: eval/move1_3tier_env.py. Reward = #correct - lambda*#assays.

## Result
| policy | acc | freeAcc | assay% | net@.5 | ADD-follow | FM-follow |
|---|--:|--:|--:|--:|--:|--:|
| always-additive (free default) | 0.802 | 0.802 | 0% | **0.802** | - | - |
| trust-all-FM | 0.661 | 0.661 | 0% | 0.661 | - | - |
| oracle-free | 0.929 | 0.929 | 0% | 0.929 | - | - |
| Haiku 4.5 | 0.885 | 0.831 | 32% | 0.725 | 0.50 | 0.50 |
| Sonnet 4.6 | 0.908 | 0.853 | 37% | 0.722 | 0.35 | 0.65 |
| Opus 4.8 | 0.932 | 0.885 | 40% | 0.730 | 1.00 (n=4) | 0.00 |

net@.5 = acc - 0.5*assay_rate (per-edge, comparable). ADD-/FM-follow = on non-verified DISAGREEMENT genes, which
free prediction the agent's call matched (Haiku n_dis=96, Sonnet n_dis=31, Opus n_dis=4).

lambda sweep (net vs the free additive default 0.802):
| net | l0.2 | l0.5 | l1.0 |
|---|--:|--:|--:|
| additive | 0.802 | 0.802 | 0.802 |
| Haiku | 0.821 | 0.725 | 0.565 |
| Sonnet | 0.833 | 0.722 | 0.537 |
| Opus | 0.851 | 0.730 | **0.528** |

## Findings
1. **The trivial "always use the free additive baseline" policy beats all three frontier LLMs as orchestrators
   once verification is not ~free.** At lambda>=0.5 every model nets BELOW additive (0.72-0.73 vs 0.802); they
   reach higher raw accuracy (0.885-0.932) only by verifying 32-40%, and once the assay is charged they lose to
   doing nothing but trusting the cheap baseline.
2. **Capability inversion (3-tier form).** Assay rate rises with capability (Haiku 32% -> Sonnet 37% -> Opus 40%);
   at lambda=1.0 the most capable model is the WORST orchestrator (Opus 0.528 < Sonnet 0.537 < Haiku 0.565). The
   Verify-or-Trust inversion recurs with the additive third action present.
3. **No model discovers the cheap-default move, and they fail in capability-ordered ways:**
   - **Haiku — indifferent** (ADD-follow 0.50 on n=96): picks between the free additive and FM at chance on
     disagreements; no preference for the better baseline.
   - **Sonnet — over-trusts the FM** (ADD-follow 0.35 = FM-follow 0.65 on n=31): given a free baseline that is
     better, it preferentially follows the unreliable specialist (the Layer-B "over-trust by identity" recurring,
     and the WRONG direction).
   - **Opus — avoids the choice by verifying** (assay 40%, only n=4 non-verified disagreements left): it does not
     choose between the free options, it pays to check; its ADD-follow 1.00 is on n=4 and not a real preference.
4. So the most basic grounded-orchestrator competence -- prefer the better FREE baseline, distrust the unreliable
   specialist by default -- is ABSENT across the frontier. The deployable win (default to additive) must be
   ENGINEERED into the agent; it is not discovered by the LLM's judgment.

## Why it matters (capability + safety)
A frontier agent handed an unreliable specialist AND a free reliable baseline does not fall back to the safe
baseline: it is indifferent (Haiku), over-trusts the specialist (Sonnet), or over-pays to verify (Opus). For
agentic AI over fallible tools this is the calibration/safety gap in behavioral form -- the agent should know to
default to the simple reliable option and verify-gate, and it does not. This is the orchestration-setting case
for ENGINEERING the default (cheap-baseline default + signal-gated verification) rather than trusting the agent's
choice -- capability built with safety, and consistent with the program's "read the signal out" conclusion.

## Caveats
- lambda-dependent: at lambda=0.2 (near-free verification) the models beat additive; the loss + inversion hold at
  lambda>=~0.35-0.5. The robust core is "they don't discover the cheap-default and over-verify," not a single lambda.
- freeAcc > additive (0.83-0.89 vs 0.802) is partly a SELECTION effect (they verify the hard genes, leaving easy
  additive-right genes unverified), not evidence their free decisions beat additive -- the net@.5 (loses to
  additive) is the clean comparison.
- Opus ADD-follow rests on n=4 (it verified the disagreements away); read it as "avoids the choice," not "prefers
  additive."
- Single neutral prompt (un-ablated); combo substrate (additive defined); held-out CV preds. The point of the
  neutral prompt is to test discovery -- telling the agent "additive is usually better" would trivially fix it
  (the Verify-or-Trust lesson: they follow a given signal).

## Artifacts
eval/move1_3tier_env.py (build/run/grade); runs/3tier_{panels,haiku,sonnet,opus}.jsonl. Baselines/K1 +
learned-router: eval/move1_3tier_baselines.py, move1_3tier_router.py. Unified story: MOVE1_SYNTHESIS.md.
