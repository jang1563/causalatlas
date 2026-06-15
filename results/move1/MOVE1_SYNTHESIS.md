# Move 1 synthesis — LLM-orchestrated foundation-model grounding (Layer A + Layer B)

*2026-06-13. The unified Move 1 result: a MEASURE (not a settlement) of how well an LLM grounds causal
biology by orchestrating a fallible single-cell foundation model (FM). Combines Layer A (does the FM beat the
simple baseline? regime map) with Layer B (can the LLM orchestrate it? = the Verify-or-Trust benchmark), and
states the novelty boundary narrowly against prior art (two literature-review passes, 2026-06-13).
Posture: MEASURE, not settle; real data cannot separate faulty-architecture from faulty-signal
(Tejada-Lapuerta, Nat Genet 2025), so this measures grounding behavior, not its cause.*

## Thesis (one line)
A perturbation FM grounds a NARROW slice of causal structure (non-additive interaction, generalizing to novel
combos) but loses in aggregate to a FREE additive baseline; and the LLM that orchestrates it cannot identify
that slice on its own, over-verifies under cost, follows an external regime signal uncritically, and cannot be
RL-trained to internalize it. Grounded orchestration is gated by exposing the FM's competitive regime, not by
LLM capability.

## Cross-layer thesis (stress-tested 2026-06-14; the writeup's central claim, sharpened + scoped)
Adversarial check of the UNIFICATION (not just the pieces): are the three layers a real independent triple-
confirmation or a shared-artifact? Verdict: SURVIVES (Layer A = real GEARS + literature; Layer B = real Claude +
GEARS/STATE; only exp-selection uses a kNN world model, and its conclusion is robust to model class), but must be
stated SHARPLY, not as a blanket "signal not policy":

  "For perturbation-biology foundation models, the binding constraint on grounded/agentic use is the QUALITY of
   the per-input COMPETENCE/VALUE signal -- weak because perturbation effects are MEAN-DOMINATED -- not model
   scale, agent capability, or naive RL. Agents can EXPLOIT a supplied signal (route / verify / select) but
   cannot SELF-GENERATE it; and whether exploiting it helps is GOAL-DEPENDENT (helps for hit-discovery, not for
   uniform mapping)."

Honest caveats for any writeup: (1) scoped to perturbation-biology (Tahoe drug adds some breadth; not a universal
AI law); (2) contingent on mean-dominance of perturbation effects (a real domain property shown across metrics,
but a contingency); (3) the "not RL" leg is the THINNEST -- one GRPO recipe (Qwen-7B) + the LLM-self-
estimation deficit; exp-selection RL was argued, not run; (4) NOT "policy never helps" -- informed selection
BEATS random for hit-discovery (2.7x), so the claim is goal-conditional.

## Layer A (SFM-vs-baseline; no LLM) — the regime map
Edge-level three-state truth (held-out sceptre), magnitude-robust discrimination metric (edge-AUROC of
|predicted log2FC|, NOT Pearson-on-pseudobulk), field-standard observed-additive baseline (y_A+y_B), powered
by leave-combo-out CV + perturbation-cluster bootstrap. (Combo data verified duplicate-free; singles deduped.)
- **Aggregate: additive beats GEARS** on Norman combos (held-out CV, gap -0.107 [-0.132,-0.083]); additive is
  near-perfect on the additive majority (~75% of edges, AUROC 0.945 vs GEARS 0.755) AND free.
- **Synergy regime: GEARS beats additive** on the non-additive subset (AUROC 0.749 vs 0.606; gap **+0.142
  [+0.050,+0.233]**, CI excludes 0), and it GENERALIZES to held-out novel combos (leave-combo-out CV, n=61
  perturbations; resolves the single-split underpowered +0.104 NS).
- Tahoe / Arc STATE (drug): STATE call-accuracy 0.732 < no-change floor 0.805; metric-flip (cosine-PDS 0.793
  flatters STATE, accuracy/AUROC do not).
- **Verdict:** not a flat negative. The FM grounds the HARD part (interaction) and loses on the EASY part
  (additivity) to a baseline built from oracle singles. A regime map.

## Layer B (LLM orchestrates the fallible FM) — the Verify-or-Trust benchmark
The LLM sees the FM's per-edge predictions and decides, per edge, TRUST vs pay a costly real assay to VERIFY;
reward = correct - lambda*assays; graded by held-out experiment. (Shipped: github.com/jang1563/verify-or-trust.)
- **Capability inversion** — more capable models verify MORE but not more accurately, so under non-trivial cost
  the frontier model (Opus) is the WORST orchestrator (paired Opus-Haiku t=-4.7, Opus-Sonnet t=-6.5, p<1e-4).
- **Allocation failure, not knowledge gap** — the verify-decision discriminates FM-wrong edges at AUC ~0.57
  (near chance); a gene-annotation DB tool does not help.
- **External reliability signal fixes it** — models follow it 94-99% and net scales with the SIGNAL's AUC, not
  the LLM's reasoning; faithful but UNCRITICAL (parrots even a bad signal).
- **RLVR negative** — GRPO/RLCR (Qwen2.5-7B) converges to trust-everything (verify-AUC ~0.5), yet a linear
  probe on the cold LLM's hidden states recovers the reliability signal at AUC 0.67. Signal present in the
  representation, absent from the policy.
- **Anonymization** — the verify-decision tracks gene-name familiarity, not FM reliability (anon raises
  false-trust 27%->58%): the name-vs-content (parroting) control at the orchestration layer.

## The connection (the original move)
Verify-or-Trust frames the choice as "trust the FM vs verify." **Layer A reveals a missing THIRD action: the
FREE additive baseline.** Since the FM beats additive only on synergy (~25%) and additive is better AND free
elsewhere, the truly optimal orchestration is **3-tier regime routing**: cheap/free baseline on additive
inputs -> specialist FM on its competitive (synergy) regime -> costly assay verification on the residual. This
- SHARPENS the over-verify finding: the LLM over-verifies a model that is often worse than a free baseline (the
  wasted budget is not just untargeted, it is spent where the FM should not be used at all);
- REFRAMES the LLM's grounding task as REGIME identification (synergy vs additive) it cannot self-estimate
  (AUC 0.57), cannot be handed-and-reasoned (it only follows), and cannot be RL-trained (knowing-doing gap).
Layer B already found "calibration pays decisively only when reliability is BIMODAL." Layer A IS that bimodal
structure, and names its true axis: additive-regime vs synergy-regime, not seen vs unseen.

## 3-tier build (2026-06-13) — the precision wall, the default-to-baseline win, the signal ceiling
Built + tested the 3-tier router (additive / FM / verify) on the held-out CV combos (eval/move1_3tier_baselines.py,
move1_3tier_router.py). Three findings refine the connection thesis:
1. **DEFAULT-TO-BASELINE WIN (deployable, safety-aligned).** always-additive 0.825 vs trust-all-FM 0.718:
   defaulting the orchestrator to the FREE additive baseline instead of the FM is **+10.7 accuracy for free**, the
   largest deployable win and the safety-aligned move (do not trust the unreliable specialist by default) — the
   orchestration-setting operationalization of "use simple baselines" (Ahlmann-Eltze). The orchestrator's first
   competence is knowing NOT to default to the FM.
2. **PRECISION WALL (beating additive-default for free is not deployable).** Routing between the two free options
   has real headroom (oracle-free max(add,fm)=0.921) but it is not capturable: oracle-regime routing only +0.030,
   the naive inference signal |fm_pred-add_pred| FAILS (0.794 < additive 0.825), and a cross-fit LEARNED router
   captures only 3% (+0.003 [-0.011,+0.018], NS) despite predicting which-is-right at AUC 0.733 on disagreement
   edges. Structural reason: the additive-wins disagreement class is ~2x the FM-wins class, so AUC 0.73 lacks the
   precision to net-gain. Free routing beyond additive-default is a precision wall.
3. **COMPETENCE-SIGNAL CEILING (the unifying bottleneck).** Across all layers the binding constraint is the
   per-input competence signal, plateauing at AUC ~0.7 (trust-head 0.70, hidden-state probe 0.67, router 0.73 on
   disagreements). NOT the LLM's reasoning (follows signals 94-99%), NOT the action set (3-tier = no free lunch),
   NOT RL (knowing-doing gap). The deployable lever is the PAID verify tier gated on the best competence signal;
   its quality is the ceiling.
**Deployable orchestrator (honest):** default to the cheap baseline (+10.7 free, safety-aligned), then verify-gate
the residual on the best competence signal. Beating that needs a signal above ~0.7 (open frontier: SFM
self-uncertainty, richer features) — not better LLM reasoning or RL.
4. **LLM 3-TIER ARM (behavioral capstone, MOVE1_3TIER_RESULT.md).** Given {additive, FM, verify} with a NEUTRAL
   prompt, NO frontier model discovers the default-to-baseline move: all three net BELOW always-additive at
   lambda>=0.5 (capability inverts at lambda=1.0, Opus worst 0.528), and they fail in capability-ordered ways --
   Haiku indifferent (ADD-follow 0.50), Sonnet over-trusts the FM (0.35), Opus avoids the choice by verifying
   (40% assay). The most basic grounded-orchestrator competence (prefer the free reliable baseline, distrust the
   specialist) is ABSENT across the frontier -> the cheap-default must be ENGINEERED, not left to the agent.

## Competence-signal is ENGINEERABLE (2026-06-14) — the ~0.7 "ceiling" was feature poverty
Dig into the #1 lever (eval/move1_competence_uq.py). Predicting per-edge FM-wrong on Norman combos (cross-fit by
perturbation): base |pred| 0.737, +regime 0.737 (regime adds nothing), **+additive-disagreement 0.881** (AUROC
+0.14, Brier .174->.120). The ~0.70 ceiling (trust-head .695, hidden-state probe .67) was a FEATURE-POVERTY
artifact, not fundamental. Mechanism (honest): the free additive baseline is usually right on combos (~0.82), so
"FM deviates from additive" ~= "FM is wrong" -- the competence signal is "disagreement with the free, usually-right
baseline" (inference-available; not leakage -- additive uses observed singles, not the held-out truth). Unifies
Layer A + B: the additive baseline is both the FM's competitor AND the key feature for WHEN to trust the FM.
Payoff: Verify-or-Trust showed net scales with signal AUC (sim 0.85 -> near-oracle); a real BUILDABLE 0.88 signal
should drive near-oracle orchestration, vs the 0.70 we measured with. The #1 lever is engineerable, not a wall.
Caveats: combo-specific (singles need a mean-baseline disagreement feature); theoretical ceiling < 1 (can't
perfectly predict errors); the mechanism is baseline-as-truth-proxy (a strong feature, not magic).

**CLOSE (deployable loop, eval/move1_competence_close.py):** feeding the BUILDABLE 0.88 signal into signal-gated
verification (LLM-free; Verify-or-Trust already showed the LLM follows a signal 94-99%) drives NEAR-ORACLE
allocation at low budget. net@lambda0.5: trust-FM 0.728; verify random HURTS (0.69-0.70); the 0.70 signal stays
~flat (0.72-0.73); the **0.88 signal lifts net to 0.78 (20% budget) and matches the oracle at 10% budget (0.772
vs 0.778)**. vRecall of FM-wrong @ base-rate budget: random 0.28 / 0.70 sig 0.47 / **0.88 sig 0.67** / oracle 1.0.
=> the project's central negative (LLM can't self-estimate; the signal is the bottleneck) becomes a CONSTRUCTIVE
positive: the signal is buildable and a buildable signal -> near-oracle. **Deployable grounded-orchestrator recipe
(measured end-to-end): (1) default to additive [+10.7 free]; (2) competence-flag where the FM deviates from
additive [0.88, inference-available]; (3) verify the top ~10% -> near-oracle accuracy at low cost. No LLM
self-estimation, no RL, no privileged info.** (Caveat: 0.88 is near-oracle on its confident picks, falls off at
high budget; mechanism = additive-as-proxy = the flip side of the FM's small marginal value.)

## Novelty boundary (narrow; stated against verified prior art)
NOT NOVEL (do not claim): three-tier cost cascades (Fanconi & van der Schaar, NeurIPS 2025, arXiv 2506.11887:
base model / costlier model / human oracle); per-input confidence cost routing incl. a free model on the easy
majority (FrugalML, NeurIPS 2020, arXiv 2006.07512); two-action defer/abstain (Mozannar & Sontag ICML 2020;
SelectiveNet ICML 2019); model-only multi-tier cascades (ABC TMLR 2025; GATEKEEPER NeurIPS 2025); per-input
competence estimation (META-DES ~2015; MetaOOD ICLR 2025); trust-vs-escalate LLM judging (Trust-or-Escalate,
arXiv 2407.18370); tool-call utility economics (Wu et al., arXiv 2605.00737); aggregate "DL < simple baselines"
+ the additive baseline definition (Ahlmann-Eltze, Huber & Anders, Nat Methods 2025); "GEARS predicts GI
subtypes incl. synergy for unseen combos" (Roohani et al., Nat Biotechnol 2024 — its own GI-score claim); the
"knowing-doing / knowledge-prediction gap" (arXiv 2509.23782, 2509.10625) and "RLVR does not exceed the base
model" (Yue et al., arXiv 2504.13837).

GENUINELY NOVEL (the conjunction, not found assembled in any surveyed prior work):
1. The third routing action is a FREE NON-MODEL baseline (additive y_A+y_B), not a cheaper model.
2. A costly REAL biological-assay ground-truth oracle as a purchasable runtime VERIFY action (prior cascades'
   most-expensive tier is a model or a human; ground truth is only a training reward).
3. The empirical grounding that the specialist FM beats the FREE baseline ONLY on a narrow, held-out-
   generalizing non-additive/synergy regime, making the orchestrator's job REGIME identification.
4. The LLM-side measurement that this regime is not self-estimable (AUC 0.57), is followed-not-reasoned, and is
   not RL-internalizable (an instance of the knowing-doing gap in agentic tool-orchestration over a bio FM).
The Layer-A<->B unification (Verify-or-Trust's "trust vs verify" is missing the free additive third action) is
the contribution; each ingredient has prior art, the assembly does not.

## Reviewer landmines (handle explicitly, do not over-claim)
- **Angle-5 tension (most exposed).** Ahlmann-Eltze report no-change BEATING DL on synergy (profile-L2 /
  interaction detection). Our synergy-positive is NOT a contradiction: different baseline (additive vs
  no-change), different unit (edge-level effect/no-effect AUROC vs profile-L2 magnitude), different task
  (does perturbing A change B vs reconstructing the interaction profile). State it precisely; our claim is
  edge-level discrimination vs the additive baseline, held-out. It RECONCILES GEARS's GI claim with the
  aggregate negative (synergy is where GEARS's GI-capture pays, and it is the minority). It cannot be
  attributed to prior literature; it is the most novel AND most contestable empirical leg.
- **Angle-3 (Verify-or-Trust novelty).** Trust-vs-defer (Trust-or-Escalate) and tool-call economics (Wu et al.)
  are close prior art on the decision skeleton. Do NOT claim "verify-vs-trust" or "3-tier routing" as novel.
  Novelty = the bio-specific conjunction above only.
- **Capability inversion is a frontier-model inversion, not a strict monotone trend** (Haiku~Sonnet tied).
- λ-dependence; single un-ablated prompt; one RL recipe/model-size; Tahoe edge-AUROC compressed (oracle ceiling
  0.577). GEARS is one (20-epoch) instance; the peer-reviewed literature backs the aggregate direction
  independently.

## Discoverer-thread framing
The FM is an ACCELERATOR: it grounds a narrow, generalizing regime (interaction) but is dominated overall by a
free interpolation baseline. The LLM is a faithful ROUTER, not a DISCOVERER: it cannot find the FM's competitive
regime, only follow it when handed, and RL does not teach it. The discoverer gap is precisely the unfilled slot:
autonomous regime identification (where the specialist beats the free baseline) is what neither the LLM nor RL
supplies. This is the verification half of the discovery loop, measured.

## Field connections (data-production war / quantitative genetics) + substrate caveat (2026-06-13)
Positioning against the post-AlphaGenome "data-production" thesis and quantitative genetics. (Citations VERIFIED
against primary sources 2026-06-13; DOIs inline.)

1. **SUBSTRATE-REGIME CAVEAT (honesty, add to all Layer-A limitations).** Our substrates -- GEARS/Norman (K562)
   and STATE/Tahoe (cancer lines) -- are exactly the public-data-rich, heavily-measured regime (ENCODE/K562 etc.)
   where models are commoditized. We measured in the EASY regime; grounding in disease/patient cell-states
   (developing brain, reactive astrocytes, diabetic beta cells), where public data is thin, is UNTESTED and likely
   WORSE (out-of-distribution). The data-production gap is precisely where grounding is hardest.
2. **LAYER A IN THE QUANTITATIVE-GENETICS FRAME (positioning for manuscript/release).** Layer A's axis -- additive
   null vs non-additive signal -- is the model-level echo of quantitative genetics: additive effects are captured
   by the additive baseline (and by additive PRS); the FM's only measured value is the NON-ADDITIVE
   (synergy/epistasis/dosage-nonlinearity) regime. Our combo substrate is Norman **CRISPRa = a gain-of-function
   screen**, the regime argued harder and more central than LoF (additive alleles, both-direction/dosage effects).
   The population-scale analog of our edge-level "does perturbing A change B" is regulators->programs->traits
   causal modelling (Ota et al., Nature 2026, 650:399-408, doi:10.1038/s41586-025-09866-3 -- Pritchard lab;
   LoF-burden gene-trait estimates + Perturb-seq regulatory connections -> a causal graph = the macro version of
   our task); the non-additive/asymmetric dosage point is Milind et al., Cell Genomics 2026, PMC12363730
   (~40% of dosage-response curves non-monotone); omnigenic background = Boyle, Li & Pritchard, Cell 2017,
   169:1177-1186, PMID 28622505. POSITION: the
   FM's value as capturing the non-additive structure that additive models (PRS, additive baseline) MISS -- the
   scientifically central slice -- not as a general perturbation predictor.
3. **THE COST TERM IS THE DATA-PRODUCTION COST.** The reward's lambda (correct - lambda*assays) is, economically,
   the real assay / data-production cost that the post-AlphaGenome "data-production war" centers. This makes the
   one structurally-unblocked RLVR slot -- experiment SELECTION / active learning (reward = information gain; the
   experiment supplies NEW ground truth, escaping cannot-exceed-the-verifier) -- the bridge to the real closed
   loop (map -> candidate -> validate -> update; template: Zhang et al., Science 2026, doi:10.1126/science.adw2156
   -- iGOF-Perturb-seq, ~1000 TFs in astrocytes, map -> Ferd3l -> AD-mouse rescue; CAS Shanghai). The
   data-production-frontier substrate that would actually test
   grounding: in vivo / disease-state GoF Perturb-seq, not cell lines.

## Public artifacts
Layer A: `results/move1/RESULT_layerA.md`, `layerA_norman.json`, `layerA_tahoe.json`,
`eval/move1_layerA{,_tahoe}.py`, `move1_gears_cv.py`, and `move1_synergy_cv.py`.
Layer B: the public Verify-or-Trust benchmark. Three-tier analysis:
`eval/move1_3tier_baselines.py`, `eval/move1_3tier_router.py`, and
`results/move1/MOVE1_3TIER_RESULT.md`. The measured result is that no tested frontier model discovers
default-to-baseline unaided; the cheap-default action must be engineered. Future work: a manuscript-scale
unification of the result, or the SFM-self-uncertainty frontier (can a real GEARS-UQ break the ~0.7
competence-signal ceiling and lift deployable orchestration?).
