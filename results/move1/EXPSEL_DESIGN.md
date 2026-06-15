# Experiment-selection RLVR — a self-improving closed loop (design, 2026-06-13)

The one RLVR target not structurally capped by a fixed answer key (see MOVE1_SYNTHESIS.md "cold RLVR" verdict):
do not train the model to PREDICT a causal answer; train a policy to SELECT which experiment to run next, and let
the experiment supply new ground truth. Offline-first on existing assets; honest about where it breaks. Posture:
MEASURE, not settle. Citations verified 2026-06-13.

## 0. Why this target (one paragraph)
predictor-RLVR is capped: the policy can at best reproduce the held-out answer corpus, and prior
verification-allocation training collapsed to trust-all. experiment-selection moves the reward from "match the
answer" to "choose an informative experiment";
the experiment, when run, produces signal that was not in the corpus. The ceiling moves from "the answer corpus"
to "the value/information function + the experiment space," and in a true online loop the SYSTEM (selector + lab)
exceeds any fixed corpus. The discoverer is the policy+lab system, not the policy alone.

## 1. Dataset review (newer data IS more suitable; verified availability)
Experiment-selection needs a LARGE pool of candidate experiments with known ground truth, ideally with structure
worth discovering. Norman (~236 perturbations) is too small for a selection problem. Ranked fit:

| dataset | pool | type | structure | where | fit |
|---|---|---|---|---|---|
| **Southard et al. 2025** (Nat Genet s41588-025-02284-1) | **1,836 TFs / 21,958 perts** | CRISPRa = **GoF**, regulators, 2 cell types | regulator->state | Zenodo 10.5281/ZENODO.15373940 (Norman lab) | **best**: scale + GoF + regulators, aligns with Ota regulators->programs->traits + the GoF emphasis; same lab as our Norman substrate |
| **Replogle 2022 gwps** | **9,867 perts**, ~2M cells | CRISPRi = LoF, single, K562 | genome-scale main effects | public/source data | **scale champion**: stand the loop up here first |
| **Jiang et al. 2025** (Nat Cell Biol 27:505) | 1,500+ perts x 6 lines x **5 contexts** | mixed, context cross | **context-dependence** | Zenodo (5 pathways) | **disease-frontier variant**: select pert x context = closest to the data-production frontier |
| Norman 2019 | ~131 combos | CRISPRa combos | **synergy/interaction** | local | interaction-discovery variant (small pool, but the Layer-A structure) |

RECOMMENDATION: start offline on **Replogle gwps** (largest pool, fastest offline gate), then **Southard 2025**
(GoF-regulator pool that matches the scientific framing; download from Zenodo), with **Jiang 2025** as the
context-dependence variant and **Norman** as the interaction variant. No brand-new data needed to start; the
upgrade path (Southard/Jiang) is verified-available and strictly better-aligned than Norman alone.

## 2. The loop (offline closed-loop simulation of experiment selection)
- **Pool P**: all perturbations (each with a HIDDEN real effect profile / sceptre or pseudobulk label).
- **Test set T**: fixed, never selectable; measures the world model's generalization.
- **Budget K** experiments per episode, K << |P|.

INNER LOOP (active learning; the world model self-improves):
  t=0: world model M_0 from a small seed (or a prior / pretrained FM).
  step t: (a) M_t predicts effects for all unrevealed perts + a per-pert UNCERTAINTY u_t (ensemble/MC-dropout);
          (b) selector picks a_t from the unrevealed pool; (c) "run" a_t = reveal its real effect;
          (d) M_{t+1} = retrain M_t on revealed u {a_t}.
  reward r_t = [test-score(M_{t+1}) - test-score(M_t)] on T  -  lambda  (lambda = the data-production cost).
  episode return = sum r_t = total generalization gained per budget = Area Under the Learning Curve (AULC).

OUTER LOOP (RLVR; the selector self-improves): train the selector policy pi(a | revealed data, M_t, u_t) by RL to
maximize AULC, vs classical active-learning heuristics. The RLVR question: does a learned/LLM selector beat the
heuristics?

## 3. Self-improvement, stated at three levels (do not conflate)
1. **Inner / world model** (well-defined): each acquired experiment improves M; its improved uncertainty improves
   the next selection -> a COMPOUNDING cycle (better selection -> better model -> better uncertainty -> ...). This
   is the self-improving engine.
2. **Outer / selector policy** (RLVR): the policy learns, across episodes, a transferable selection strategy. The
   policy-extraction question recurs here: does a selection reward avoid the trust-all collapse seen in
   verification allocation?
3. **Meta / the loop edits itself** (ASPIRATIONAL, risky): the agent refines the goal, the reward, or M's design.
   High reward-hacking risk, ill-defined objective. OUT OF SCOPE for v1; flagged so it is not smuggled in.

THE LOAD-BEARING CONNECTION: the engine is M_t's UNCERTAINTY u_t, which is exactly the per-input competence signal
the project found plateaus at AUC ~0.7. So the loop's efficiency is gated by the world-model's uncertainty quality.
This design is therefore a direct test of the project's central thesis -- is the bottleneck the selector (policy)
or the signal (uncertainty)? Predicted: the signal. The design measures whether RLVR/LLM-selection adds anything
over uncertainty-sampling-with-a-good-uncertainty-model, in the one regime where the loop yields new ground truth.

## 4. Baselines + pre-registered K1 gate (no LLM/RL until this passes)
Selectors: random; uncertainty-sampling (max u_t); BALD / expected info gain; diversity (k-center); FM-prior (use
a pretrained perturbation FM's predicted magnitude -- note Layer A says FM ~ additive, so this is a weak prior,
itself an interesting test); oracle (greedy: pick the pert that maximally improves test-score = cheating upper
bound).
K1 GATE: does the best informed selector reach a target test-score in MATERIALLY fewer experiments than random
(AULC gap, perturbation-cluster bootstrap CI excludes 0)? If not, the substrate has no exploitable selection
structure -> fix before any LLM/RL. (Same discipline as the 3-tier K1.)

## 5. Metrics
- Learning curve: test-score (edge-AUROC / held-out Pearson / PDS-cosine) vs # experiments.
- AULC per selector; experiments-to-reach-X% vs random.
- Selector value = (AULC_selector - AULC_random) / (AULC_oracle - AULC_random), in [0,1].

## 6. Cold limits (where it breaks)
1. OFFLINE = CORPUS-BOUND: the pool is a fixed, biased, already-run set; this learns a selection POLICY within
   known experiments, not open discovery. The genuine escape needs the ONLINE loop (real or accurately-simulated
   experiments) -- expensive (the data-production war's whole point).
2. UNCERTAINTY CEILING: the loop cannot beat M_t's uncertainty quality (~0.7 recurs).
3. REWARD GAMING: AULC / info-gain is gameable -- a selector can chase noisy, under-measured regions that look
   informative; report a gaming probe (does the policy over-select high-variance low-signal perts?).
4. NO FIXED TARGET AT THE FRONTIER: "reconstruct the map" is well-defined offline; open discovery has no fixed
   goal, so info-gain is only definable in the enumerable regime (the falsification-thesis limit).
5. RLVR MAY ADD NOTHING over classical AL unless the uncertainty model improves -- the design TESTS this; it does
   not assume a win. Honest expected outcome (from the project's evidence): classical AL + a good uncertainty
   model is strong; the LLM/RL selector's value, if any, comes from a better signal, not a better policy.
6. SIM-GROUNDED online variant (virtual cell as the experiment oracle, rbio1-style) inherits sim bias
   (cannot-exceed-the-simulator); "know when the sim lies" = the calibration problem again.

## 7. Build plan (baselines/K1 before LLM/RL)
- **Local pilot:** offline loop + world model + uncertainty + all baselines; K1 gate. No LLM/RL.
- **Replogle scale-up:** scale the loop; classical-AL learning curves; quantify the uncertainty-quality ceiling.
- **Southard 2025:** GoF-regulator selection pool; rerun K1 + curves.
- **LLM-selector arm:** does the LLM, given the partial map, select better than uncertainty sampling? (hypothesis: not unaided.)
- **RLVR-selector:** train pi; does it beat heuristics, or collapse to a trivial policy?
- **Online / simulated closed loop:** the real escape; needs a virtual-cell oracle or wet-lab partner.

## 7b. Replogle results (2026-06-14, Replogle K562_essential: 2003 perts x 2000 readout HVGs)
Ran the gated Replogle experiment. Features: GO (gene2go, 99% coverage, 50-d) + co-expr (control cells, 50-d); ESM2
DROPPED (saved file covers only 201 genes -- too thin). World model: kNN / ridge in feature space.

- **CEILING GATE:** GO PASSES (ridge AUROC real 0.865 vs perm-null 0.794, z=+16; knn 0.842 vs 0.778). co-expr
  WEAK (knn fails, ridge +0.004 negligible). KEY: the null AUROC is already ~0.78 -> most predictable signal is
  the SHARED/mean perturbation response; GO adds only a thin perturbation-SPECIFIC increment (~+0.07). Substrate
  has structure, but the exploitable specific headroom is mean-dominated (the perturbation-FM-negative, recurring).
- **LOOP, uniform map-reconstruction goal:** PRE-K1 PASS (M learns, random captures 79% of floor->ceiling), but
  **K1 FAIL** -- informed selection (uncertainty/kcenter) is SIGNIFICANTLY WORSE than random (gap -0.018, CI
  [-0.030,-0.005]); distance-uncertainty is ANTI-calibrated (corr(u,error) = -0.15).
- **CHECK (i) SIGNAL lever:** committee(bootstrap) uncertainty is weakly calibrated (+0.13, vs distance -0.10,
  disagree +0.02) but STILL loses to random (AULC 0.159 vs 0.232). => uncertainty-targeting is the WRONG
  objective for representative map-reconstruction (it chases hard outliers); a better signal does not rescue the
  uniform goal.
- **CHECK (ii) GOAL lever:** switch to HIT-SEARCH (find the top-decile high-effect perts). exploit (reveal highest
  GO-predicted effect-magnitude) BEATS random ~2.7x (recall@budget 0.19 vs 0.07; discovery-AULC +0.024). => the
  GOAL lever works: informed selection is valuable for hit-discovery, not for uniform mapping.

**CONCLUSION (experiment-selection, both levers measured):** informed experiment-selection's value is
GOAL-DEPENDENT. For uniform map-reconstruction, random is near-optimal and uncertainty-sampling HURTS (signal
calibration does not fix it). For HIT-DISCOVERY (which high-impact experiment to run next = the actual
data-production question), a simple value-predictor (GO->effect-magnitude) greedy-exploit beats random ~2.7x.
Deployable move: prioritize experiments by a learned value-predictor for hit-search; do NOT use uncertainty
sampling. RLVR target should be hit-search (not mapping); RL over greedy-exploit is expected marginal (the
value-predictor is the lever, not the policy -- the project thesis, recurring). Southard GoF, LLM-selector,
and RL-selector follow-ups remain optional; the decisive two-lever result is in hand.

## 7c. Controlled-simulator results (2026-06-14)
Built a controlled GRN simulator with KNOWN interaction ground truth + a feature-predictability knob beta
(eval/move1_p5_sim.py, move1_p5_policy.py). Discovery target = recover the interaction graph by selecting which
pair-experiments to run; the experiment supplies new truth (escapes cannot-exceed-verifier offline).

- **Machinery-K1 (sim validated as a FAIR testbed):** clean phase transition (3-seed). beta<=0.2: no learnable
  structure (ceiling ~0.5) -> collapse. beta=0.4: structure exists (ceiling 0.79) but simple heuristics tie/lose
  random (~0.09 headroom unclaimed). beta>=0.6: informed (uncertainty 0.92) beats random (0.87). So the sim is
  not rigged either way, and it turns the mean-dominance caveat into a measured curve (selection helps only when
  structure is feature-predictable; the real-bio thin-signal regime = low beta = collapse, reproducing Layer C).
- **KILL-TEST (learned vs heuristic, headroom regime):** an optimized parametric acquisition policy (CEM/random-
  search over uncertainty+exploit+diversity+density, train/test-seed split) does NOT beat the simple uncertainty
  heuristic: beta=0.4 learned 0.609 vs uncertainty 0.614 (-0.005); beta=0.6 0.900 vs 0.888 (+0.012, within noise).
  At beta=0.4 the large ceiling headroom (0.192) is unclaimed by ANY selection -> it is a DATA-QUANTITY gap (need
  more experiments), not a selection-ORDER gap. The thesis's weakest leg ("policy/RL is not the lever") SURVIVES
  its strongest controlled falsification: even where discovery is possible and headroom is large, a learned policy
  ~ a simple heuristic; the binding constraint is the signal/data (the ceiling), not the policy.
- Caveats: parametric policy not full deep-RL (but includes the main signals, and the beta=0.4 headroom is
  quantity-not-order so deep-RL cannot close it either); the in-script rand~ baseline was a zero-weight degenerate
  bug (use the K1 random ~0.696).

NET: the discovery-loop's lever is the experiment/data + the competence signal, NOT the selection policy or
RL. Combined with the verification-allocation RL result, "not RL/policy" is now confirmed across a real RL run AND a controlled
mechanism kill-test. The genuine discoverer payoff still requires the ONLINE loop (real/sim experiments supply
new truth); offline, even perfect selection cannot exceed what the data supports.

## 8. So what
If P0/P1 show informed selection beats random (likely) but the LLM/RL selector does NOT beat uncertainty-sampling
(predicted), the honest result is the project's thesis again, sharpened: the lever is the world-model's per-input
uncertainty, not the selection policy or RL -- and the discoverer loop is real only when an actual experiment (not
a corpus replay) supplies new ground truth. That is a precise, deployable, safety-relevant conclusion for building
self-improving scientific agents: invest in calibrated specialist uncertainty + a real experiment loop; do not
expect RLVR over selection to manufacture discovery from a fixed corpus.
