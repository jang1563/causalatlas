# Move 1, Layer A -- does the perturbation SFM ground causation beyond simple baselines? (2026-06-13)

The decisive MEASURE of Move 1: a pure model-vs-baseline statistical comparison (NO LLM). Does the
single-cell perturbation foundation model beat the field-standard simple baselines on an
interpolation-proof held-out, under a magnitude-robust DISCRIMINATION metric? Two substrates: GEARS on
Norman genetic combos and Arc-STATE on Tahoe drug zeroshot. Skeptical posture: this
MEASURES grounding behavior; it does NOT settle architecture-vs-signal (real data cannot, Tejada-Lapuerta
Nat Genet 2025 / arXiv 2310.14935).

## Design locks (literature review 2026-06-13, primary sources verified)
- **Metric:** PRIMARY = edge-level AUROC of |predicted log2FC| (effect vs no-effect); magnitude-robust,
  maps to our three-state, and avoids Pearson-on-pseudobulk (only a VCC *secondary* metric; the contested
  "FM works" reading rests on it). SECONDARY = cosine-PDS (the VCC Perturbation Discrimination Score but
  with COSINE distance: VCC/cell-eval ships L1-PDS, which is rescale-gameable -- arXiv 2511.16954, Liu et
  al. Nov 2025, independent statisticians; cosine is invariant to global rescaling = our magnitude-trap fix).
- **Baselines (field-standard):** observed-additive `y_add = y_A + y_B - y_ctrl` from OBSERVED single
  effects (combos only) -- verbatim the convention in Ahlmann-Eltze, Huber & Anders (Nat Methods 2025,
  10.1038/s41592-025-02772-6) methods + code AND GEARS (Roohani et al., Nat Biotechnol 2024,
  s41587-023-01905-6). no-change (predict zero). mean (mean single effect per gene; the unseen-single
  baseline). For drugs (Tahoe) additive is undefined -> no-change is the clean simple baseline.
- **Split:** interpolation-proof = combos where >=1 partner is unseen-in-combo. Strict seen0 (neither gene
  seen in any training combo) is intrinsically tiny (Norman has only ~131 two-gene combos; GEARS's
  two-of-two-unseen class is the smallest by construction) so seen0 = n=4 perturbations is a STRUCTURAL
  ceiling, not a bug. Powered statement = the seen1+seen0 pool (n=27 perturbations).
- **Uncertainty:** PERTURBATION-cluster bootstrap (resample perturbations, not edges); 2000x (Norman),
  250x AUROC / 2000x acc (Tahoe). This is the honest treatment of the small strict split.

## Part A -- GEARS vs observed-additive on Norman combos (local, delta=0.25)

| stratum | nP | nE | GEARS acc | ADD acc | ADD-GEARS [95% CI] | GEARS AUROC | ADD AUROC |
|---|--:|--:|--:|--:|--:|--:|--:|
| combo_in_train | 35 | 842 | .745 | .829 | +.084 [.04,.13] | .836 | .898 |
| combo_seen1 | 23 | 581 | .716 | .818 | +.102 [.05,.15] | .787 | .899 |
| combo_seen0 (strict) | 4 | 57 | .614 | .842 | +.228 [.18,.26] | .819 [.65,.94] | .935 |
| **interp_pool (seen1+seen0)** | **27** | 638 | .707 | .820 | **+.113 [.07,.16]** | .788 [.75,.83] | .899 [.87,.93] |
| all_combos | 62 | 1480 | .728 | .825 | +.097 [.06,.13] | .815 | .898 |

Singles (additive undefined -> GEARS vs no-change & mean floor):

| stratum | nP | nE | GEARS acc | noChg | mean | GEARS AUROC | mean AUROC |
|---|--:|--:|--:|--:|--:|--:|--:|
| single_seen | 67 | 1889 | .782 | .630 | .677 | .814 | .703 |
| single_unseen | 20 | 606 | .734 | .589 | .658 | .754 [.70,.80] | .748 [.72,.77] |

(Singles deduplicated 2026-06-13: the upstream substrate doubled a subset of single_* edges — 1355 exact
(pert,gene) dups, ZERO in combos. Re-run on deduped singles: conclusion unchanged, slightly tighter. Combo
strata above are byte-identical pre/post dedup, confirming the headline is dedup-robust.)

cosine-PDS (secondary): on combos additive PDS=1.000 (observed-singles are oracle-ish), GEARS .92-.98,
mean .77-.88 -- PDS is SATURATED here (distinct perturbations make retrieval easy), confirming edge-AUROC
is the right primary. delta sweep on interp_pool: ADD-GEARS gap excludes 0 at delta 0.1/0.25/0.5
(+.060/.113/.085), converges only at delta 1.0 (+.027 [-.01,.06], both predict ~no_effect).

**Part A verdict:** additive beats GEARS on every combo stratum and the gap CI excludes 0; on the powered
interpolation-proof pool (n=27) ADD-GEARS = +.113 [.07,.16] on accuracy and additive AUROC .899 [.87,.93]
vs GEARS .788 [.75,.83] (near-separating CIs). On unseen singles GEARS only TIES the mean baseline on
discrimination (.754 vs .748, deduped; AUROC gap .006). NOTE (corrected by the Review section below): "additive beats GEARS" holds
IN AGGREGATE, but this is a REGIME MAP, not a flat negative -- GEARS carries non-additive/synergy signal
that additive structurally cannot, and loses overall only because additive is near-perfect on the
additive-edge majority.

## Part C -- Arc-STATE vs simple baselines on Tahoe zeroshot (1135 novel drugs, cell line C32)

| policy | call-acc@0.25 [95% CI] | edge-AUROC [95% CI] |
|---|--:|--:|
| **STATE** (its own q<.05 & |lfc|>=delta call) | **.732 [.728,.736]** | .439 [.436,.443] |
| no-change (floor) | **.805 [.800,.809]** | 0.500 (constant) |
| mean | .454 [.452,.456] | .423 [.420,.426] |
| (|real| oracle-magnitude ceiling) | -- | .577 [.573,.582] |

cosine-PDS: STATE .793, mean-profile .500.

**Part C verdict:** STATE's own best call (.732) is BELOW the no-change floor (.805), CIs separate -> STATE
fails to beat predicting "no drug effect" on novel drugs. Metric nuance: edge-AUROC is COMPRESSED on this
substrate (even oracle |real| only reaches .577, because the sceptre label is significance/equivalence-
based, not magnitude); STATE's magnitude AUROC .439 lands below random but the metric is weak here, so the
robust Tahoe findings are (1) accuracy-below-floor and (2) the PDS/accuracy DISSOCIATION: cosine-PDS .793
flatters STATE (it captures perturbation identity) while accuracy/AUROC say it does not ground per-edge
causal effects. The chosen metric flips the verdict -- the Move 1 RQ1 thesis, shown on one substrate.

## Unified finding
On BOTH substrates the perturbation FM fails to beat the appropriate simple baseline in the
novel-perturbation / interpolation regime: GEARS < observed-additive on novel gene combos; STATE <
no-change on novel drugs. This REPRODUCES the settled negative (Ahlmann-Eltze 2025; scPerturBench Nat
Methods 2025 s41592-025-02980-0; Wong/Hill/Moccia Bioinformatics 2025 PMC12202205; Bendidi et al. arXiv
2410.13956) on this repository's held-out sceptre ground truth, now POWERED (cluster-bootstrap) and under a
magnitude-robust discrimination metric.

This directly answers the literature-context question -- does the named conflicted rebuttal (Miller, **Bo
Wang** [scGPT senior author], et al., "DL Models DO Outperform Uninformative Baselines on Well-Calibrated
Metrics," bioRxiv 2025.10.20.683304: "rank-based metrics let DL surpass even additive in some cases")
survive on a true interpolation-proof split? On Norman it does NOT: additive still beats GEARS on the
calibrated rank metric (AUROC) on the interpolation pool. On Tahoe the verdict is metric-dependent. The
interested-party counter (Arc STATE, "first to consistently exceed linear methods") concedes prior FMs
lose and claims its win only for SEEN-perturbation cell-context generalization, NOT the novel-perturbation
regime measured here.

## Two layers
- **Layer A (this doc):** SFM vs simple baseline, statistical, no LLM. Anonymization is irrelevant here (no
  name-reader). Result: the FM does not beat baselines in aggregate on either substrate.
- **Layer B (Verify-or-Trust):** does an LLM's edge judgment beat its baselines and survive anonymization?
  That is where the anon gate lives. Layer A provides the FLOOR that reframes verification allocation: the LLM in the
  orchestration eval is calibrating verification over an SFM that is itself no better than additive /
  no-change -- so the smart policy is often to not trust the FM at all.

## Caveats / honesty
- observed-additive uses OBSERVED singles (an oracle-ish strong baseline) = the field convention; this is
  the baseline GEARS is supposed to beat, not a strawman.
- combo_seen0 = n=4 perturbations (structural). The headline rests on the seen1+seen0 POOL (n=27), where
  seen1 = combo with >=1 unseen-in-combo partner (genuinely interpolation-stressing).
- Tahoe edge-AUROC is compressed (oracle ceiling .577); lead the Tahoe story with accuracy-vs-floor + the
  PDS/accuracy metric flip, not the sub-0.5 AUROC alone.
- This MEASURES grounding; it does not separate faulty-architecture from faulty-signal (real data cannot).

## Review (2026-06-13, adversarial self-review): synergy regime map + corrections
Data-grounded review (eval/ checks) corrected one over-claim and hardened the rest.

INTEGRITY (clean): real_label is exactly {POSITIVE, TESTED_NEGATIVE} -- no UNTESTED leaked into the scored
truth; observed-additive verified correct by manual spot-check (e.g. MAP4K3: ADD -.516 ~ real -.527).

CORRECTION -- it is a REGIME MAP, not a flat negative. Splitting combo edges by non-additive residual
|real - additive| (75th pct = 0.33), paired cluster-bootstrap CI on GEARS_AUROC - ADD_AUROC:

| subset | n | nP | majority-class | GEARS AUROC | ADD AUROC | GEARS-ADD gap [95% CI] |
|---|--:|--:|--:|--:|--:|--:|
| additive edges (low residual) | 1110 | 62 | .586 | .781 | .945 | -.164 [-.195,-.133] |
| synergy edges (top 25% residual) | 370 | 61 | .908 | .815 | .608 | **+.206 [+.130,+.281]** |
| synergy AND held-out (seen1+seen0) | 153 | 26 | .889 | .779 | .675 | +.104 [-.027,+.219] (NS) |

additive wins aggregate because most edges ARE additive and additive is near-perfect there (AUROC .945);
GEARS carries genuine non-additive interaction signal additive structurally cannot (synergy AUROC .815 vs
.608, CI excludes 0). BUT on HELD-OUT synergy edges (single-split) the GEARS advantage shrank to +.104 and was
NOT significant (n=26 perts, underpowered) -- so whether GEARS's interaction signal GENERALIZES to novel combos
was UNRESOLVED at single-split scale. **RESOLVED by leave-combo-out CV below (now significant, +.142 [+.050,
+.233]).** Use AUROC not accuracy on the synergy subset (majority-class .908, base-rate-skewed).

NET corrected verdict: on the interpolation-proof regime the FM does not SIGNIFICANTLY beat the appropriate
simple baseline (additive wins the combo aggregate; GEARS's synergy advantage does not reach significance on
held-out synergy edges). But the FM is NOT signal-free -- regime map: additive owns the additive majority,
GEARS owns non-additive structure (generalization-unconfirmed).

OPEN caveats from the review:
- combo_seen0 = 4 perturbations: a 4-cluster bootstrap CI is NOT trustworthy; rely on the seen1+seen0 pool (n=27).
- GEARS is ONE instance (20 epochs). It has real signal (synergy AUROC .815; beats no-change on singles) so
  not broken, but not necessarily a maximally-tuned FM. The peer-reviewed literature backs the direction
  independently (Ahlmann-Eltze 2025 etc.), so the result does not rest on this GEARS alone.
- Tahoe "zeroshot" is trusted from the upstream pipeline labeling; if the 1135 drugs were in STATE's
  training the interpolation claim weakens (not verifiable without STATE's training manifest).
- Layer A and Layer B answer distinct questions: this document isolates SFM-vs-baseline behavior, while
  anonymization belongs to the LLM-mediated verification layer.
- SUBSTRATE-REGIME: both substrates (GEARS/Norman = K562; STATE/Tahoe = cancer lines) are the public-data-rich,
  commoditized regime (post-AlphaGenome data-production critique). This MEASURES grounding in the easy regime;
  disease/patient cell-states (where public data is thin) are untested and likely worse OOD -- the data-production
  gap is exactly where grounding is hardest. See MOVE1_SYNTHESIS.md "Field connections".

## Powering update (2026-06-13): leave-combo-out CV RESOLVES the held-out synergy question
One question remained: does GEARS's synergy advantage GENERALIZE to novel combos? (single-split
held-out synergy was +.104 [-.03,+.22], NS, n=26 perts). Ran a 5-fold leave-combo-out CV
(eval/move1_gears_cv.py; every combo predicted while held out; ALL held-out = combo_seen2 = novel combination
of seen singles; 62 held-out combos, 1477 decidable edges) and recomputed (eval/move1_synergy_cv.py, paired
perturbation-cluster bootstrap):

| subset | n | nP | maj-class | GEARS AUROC | ADD AUROC | GEARS-ADD gap [95% CI] |
|---|--:|--:|--:|--:|--:|--:|
| ALL held-out combos | 1477 | 62 | .67 | .790 | .897 | -.107 [-.132,-.083] |
| additive edges (|inter|<.33) | 1108 | 62 | .58 | .755 | .945 | -.190 [-.217,-.162] |
| SYNERGY edges (|inter|>=.33) | 369 | 61 | .91 | .749 | .606 | **+.142 [+.050,+.233]** |

RESOLVED: on the powered held-out set, GEARS beats additive on synergy edges +.142 [+.050,+.233], CI EXCLUDES 0
(the single-split NS is gone). GEARS's non-additive interaction signal DOES generalize to novel combos. The
advantage is smaller than the trained+heldout mix (+.206 -> +.142, so some was train-reliant) but real and
significant held-out. Regime map confirmed on held-out: additive owns the additive majority (-.190), GEARS owns
the synergy minority (+.142), additive wins aggregate (-.107). (GEARS log2FC has a few EPS-log-ratio outliers
~ -17; AUROC is rank-based so unaffected.)

**FINAL Layer-A verdict (Norman), corrected + powered:** NOT a flat negative. The FM significantly captures
generalizable non-additive interaction structure that additive structurally cannot (+.142 held-out synergy, CI
excl 0) -- but LOSES in aggregate (-.107) because additive is near-perfect on the additive-dominated majority
(75% of edges, AUROC .945). The honest regime map, stronger and more interesting than
"additive wins": the FM grounds the hard part (interaction) and loses on the easy part (additivity) to a
baseline built from oracle singles.

## Artifacts / reproducibility
- `eval/move1_layerA.py` (Norman, local) -> `results/move1/layerA_norman.json` (~96s)
- `eval/move1_layerA_tahoe.py` (Tahoe / Arc-STATE comparison CSV) -> `results/move1/layerA_tahoe.json`
