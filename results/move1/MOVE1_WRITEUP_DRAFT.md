# Where grounding lives: the per-input signal, not the model, the agent, or the RL

*Full draft (blog/preprint), 2026-06-14. A measured account of LLM-orchestrated perturbation foundation models,
across three layers. Posture: MEASURE, not settle (real data cannot separate faulty architecture from faulty
signal; Tejada-Lapuerta, Nat Genet 2025). Author: JangKeun Kim. Citations verified 2026-06-13/14.
Numbers are from results/move1/* and the Verify-or-Trust benchmark; see Methods.*

## TL;DR
We ask whether an LLM can ground causal biology by orchestrating a fallible perturbation foundation model (FM),
and we measure it across three layers: (A) does the FM beat simple baselines, (B) can the LLM orchestrate it, (C)
can the agent choose what experiment to run next. The same constraint binds all three: the per-input
competence/value signal, which is weak because perturbation effects are mean-dominated. Agents exploit a supplied
signal but cannot self-generate it; scaling the model, the agent, or naive RL does not substitute. The deployable
moves are unglamorous and safety-aligned: default to the cheap baseline, read the reliability signal out rather
than trusting the agent's judgment, and use a value-predictor to choose high-impact experiments. Crucially the bottleneck is engineerable, not
fundamental: a cheap baseline-disagreement feature lifts the competence signal from AUC ~0.70 to 0.88, enough to
drive near-oracle verification with no privileged information -- the central negative becomes constructive.

## 1. Three scales of one phenomenon
After AlphaGenome, the visible bottleneck in regulatory genomics moved from the model to the data: a model on the
public corpus widens the candidate space but does not measure how a variant behaves in an unseen disease cell
state. The same shape appears one level down (a model cannot exceed its verifier: a predictor fit to past
experiments can at best reproduce them; new causal signal needs a new intervention) and one level up (an LLM
handed a fallible specialist behaves like a faithful follower, not a discoverer). These are not three findings;
they are one phenomenon. Grounding is gated by access to new interventional signal and by the quality of the
per-input signal that says what is worth trusting or doing, not by model scale, agent capability, or RL.

## 2. Layer A -- does the foundation model ground causation?
Setup: edge-level three-state truth from held-out sceptre tests (POSITIVE / TESTED-NEGATIVE via TOST / UNTESTED);
a magnitude-robust discrimination metric (edge-AUROC of |predicted log2FC|, NOT Pearson-on-pseudobulk); the
field-standard observed-additive baseline (y_A + y_B); powered by leave-combo-out CV + perturbation-cluster
bootstrap. Substrates: GEARS on Norman (genetic combos), Arc STATE on Tahoe (drug).
- It is a REGIME MAP, not a flat negative. Aggregate: additive beats GEARS (held-out interpolation pool n=27,
  ADD-GEARS accuracy +0.113 [0.07,0.16]; AUROC additive 0.899 [0.87,0.93] vs GEARS 0.788 [0.75,0.83]). Additive
  is near-perfect on the additive majority (~75% of edges, AUROC 0.945) and is free.
- The FM wins only the non-additive (synergy) minority: GEARS beats additive on synergy edges (AUROC 0.815 vs
  0.608; held-out CV +0.142 [0.05,0.23]). On drugs, Arc STATE call-accuracy 0.732 is BELOW the no-change floor
  0.805. So the FM grounds the hard, narrow slice (interaction) and loses the easy majority to a free baseline.
- This reproduces the settled perturbation-FM negative (Ahlmann-Eltze, Huber & Anders, Nat Methods 2025) on our
  own held-out truth, under a calibrated metric, and answers the named conflicted rebuttal (Miller, Bo Wang et al.,
  bioRxiv 2025.10.20.683304): on an interpolation-proof split, additive still beats GEARS on the rank metric.

## 3. Layer B -- can the LLM orchestrate the fallible FM?
The Verify-or-Trust benchmark (github.com/jang1563): the LLM sees the FM's per-edge predictions and decides, per
edge, to TRUST or pay a costly real assay to VERIFY; reward = correct - lambda*assays.
- Capability inversion: more capable models verify MORE but not more accurately, so under non-trivial cost the
  frontier model is the WORST orchestrator (paired Opus-Haiku t=-4.7, Opus-Sonnet t=-6.5, p<1e-4).
- Allocation failure, not knowledge gap: the verify-decision discriminates FM-wrong edges at AUC ~0.57 (near
  chance); a gene-annotation database tool does not help.
- An external reliability signal fixes allocation: models follow it 94-99% and net scales with the SIGNAL's AUC,
  not the LLM's reasoning (faithful but uncritical; it parrots even a bad signal).
- RLVR negative: a small policy (Qwen2.5-7B, GRPO) converges to trust-everything (verify-AUC ~0.5), yet a linear
  probe on the cold LLM's hidden states recovers the signal at AUC 0.67. Present in representation, absent in
  policy (a knowing-doing gap).
- The verify-decision tracks gene-name familiarity, not FM reliability (anonymizing names raises false-trust
  27%->58%): the parroting control at the orchestration layer.

A 3-tier analysis adds the missing free action -- the additive baseline. The largest deployable, safety-aligned
move is to DEFAULT to the cheap baseline rather than the FM (always-additive 0.802 vs trust-FM 0.661 = +14 free
on those panels). Beating that for free is a precision wall (a learned add-vs-FM router captures +0.003, NS,
despite AUC 0.73 on disagreements). And given {additive, FM, verify} with a neutral prompt, NO frontier model
discovers the default-to-baseline move: all net below always-additive at lambda>=0.5, and they fail in
capability-ordered ways (Haiku indifferent, Sonnet over-trusts the FM, Opus over-verifies). The cheap default
must be engineered, not left to the agent.

## 4. Layer C -- can the agent choose what experiment to run next?
The one RL target not capped by a fixed answer key: select which perturbation to run; the experiment supplies new
truth. Offline on Replogle K562_essential (2003 perturbations, GO/co-expression features, kNN/ridge world model).
- Substrate has structure (GO features predict held-out effects: ridge AUROC 0.865 vs permutation-null 0.794),
  but the null is already ~0.78: the predictable signal is mostly the SHARED mean response; the perturbation-
  specific increment is thin (mean-dominated, again).
- For uniform map-reconstruction, informed selection LOSES to random (uncertainty/k-center worse; CI excludes 0
  in the negative direction), and the naive uncertainty is anti-calibrated (corr -0.15). A better-calibrated
  committee uncertainty (corr +0.13) still loses: uncertainty-sampling is the wrong objective for a representative
  target.
- For HIT-DISCOVERY (find the top-decile high-effect perturbations -- the real data-production question), informed
  selection (exploit a GO-predicted value) BEATS random ~2.7x (recall@budget 0.19 vs 0.07). The value lies in the
  goal and the value-predictor, not the selection policy.

## 4b. The competence signal is engineerable -- which closes a deployable loop
The bottleneck across all three layers is the per-input competence signal. Is it a wall? No. Predicting per-edge
FM-wrongness on Norman combos (cross-fit by perturbation), the magnitude+regime features prior work used cap at
AUC 0.737 (regime adds nothing) -- the ~0.70 "ceiling" was feature poverty. Adding the Layer-A
additive-disagreement feature (|FM - additive|, call-disagreement; inference-available, not leakage) lifts it to
AUC 0.881. Mechanism: the free additive baseline is usually right on combos (~0.82), so "the FM deviates from
additive" is a strong, cheap proxy for "the FM is wrong" -- the additive baseline is both the FM's competitor
(Layer A) and the key feature for when to trust it (Layer B), unified. Fed into signal-gated verification, the
buildable 0.88 signal drives NEAR-ORACLE allocation at low budget: net@lambda0.5 is trust-FM 0.728, random-verify
hurts, the 0.70 signal stays flat, and the 0.88 signal reaches 0.78 at a 20% verify-budget and matches the oracle
at 10% (0.772 vs 0.778; vRecall of FM-wrong 0.67 vs oracle 1.0). The central negative becomes constructive: the
signal that gates grounded orchestration is buildable, and a buildable signal yields near-oracle behavior.

## 5. The thesis (sharpened, scoped, stress-tested)
For perturbation-biology foundation models, the binding constraint on grounded/agentic use is the QUALITY of the
per-input COMPETENCE/VALUE signal -- weak because perturbation effects are MEAN-DOMINATED -- not model scale,
agent capability, or naive RL. Agents EXPLOIT a supplied signal (route / verify / select) but cannot SELF-GENERATE
it; and whether exploiting it helps is GOAL-DEPENDENT (it helps for hit-discovery, not for uniform mapping).

We stress-tested the unification for shared-artifact confounds: Layer A uses real GEARS plus the peer-reviewed
literature, Layer B uses real frontier LLMs plus GEARS/STATE, and only Layer C uses a kNN world model (whose
conclusion is robust to model class). The three are independent on components, so the triple-confirmation is not
an artifact of one weak shared model.

## 6. Deployable conclusions (capability built with safety)
- **The recipe, measured end to end:** (1) default to the free additive baseline; (2) competence-flag the edges
  where the FM deviates from it (a cheap inference-available classifier, AUC 0.88); (3) verify only the top ~10%
  flagged -> near-oracle accuracy at low cost, with no LLM self-estimation, no RL, and no privileged information.
- Default to the cheap baseline, not the specialist. The orchestrator's first competence is knowing NOT to trust
  an unreliable model by default; no frontier LLM does this unaided, so engineer it.
- Read the reliability signal out (an external classifier or a hidden-state probe) and gate verification on it,
  rather than relying on the agent's prompted or RL-trained choice.
- For "which experiment next," use a learned value-predictor for hit-search; do not use uncertainty sampling.
- The lever is the SFM's per-input uncertainty / a value-predictor, not better LLM reasoning or RL. Invest there.

## 7. What this is NOT (limitations, honest)
- MEASURE, not settle: real data cannot separate faulty architecture from faulty signal; the decisive cause-test
  (simulation with known causal ground truth) is out of scope.
- Scoped to perturbation-biology; contingent on mean-dominance (a real domain property, not a universal AI law).
- The "not RL" leg is the thinnest: one GRPO recipe plus the LLM-self-estimation deficit; experiment-selection
  RL was argued, not run.
- Offline experiment-selection is corpus-bound: it tests active-learning feasibility, not discovery. The genuine
  discovery test is an online closed loop (real or accurately-simulated experiments), which is resource-blocked.
- Substrates are the public-data-rich, commoditized regime (K562 / cancer lines); disease/patient cell-states,
  where the data-production gap is largest, are untested and likely harder.

## 8. Methods (brief)
Layer A: held-out sceptre three-state labels; edge-AUROC + cosine-PDS; observed-additive (y_A+y_B from measured
singles); leave-combo-out CV; perturbation-cluster bootstrap. Layer B: native tool-use agent, reward
correct-lambda*assays, held-out experiment as verifier; GRPO/RLCR for the RL arm; linear probe on hidden states.
Layer C: kNN/ridge world model in GO/co-expression feature space; permutation-null ceiling gate; AULC over a
learning curve; hit-search recall@budget. Code/results: results/move1/*, eval/move1_*.py, Verify-or-Trust repo.
