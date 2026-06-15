#!/usr/bin/env python3
"""3-tier grounded orchestrator -- BASELINES / K1 gate (no LLM yet).

The deployable upgrade implied by Move 1: the orchestrator chooses, per edge, among THREE actions --
  additive baseline (FREE), trust the FM (FREE), or pay a real assay to VERIFY (cost lambda, always correct).
Because additive AND FM are both free, routing BETWEEN them is a free accuracy gain; verification is the paid
top tier. Layer A says additive owns the additive-edge majority and the FM owns the synergy minority, so a
regime-router should beat BOTH constant free policies for free -- that is the K1 gate this script tests before
we build the LLM arm.

DEPLOYABLE regime signal (no training, available at inference): |FM_pred - additive_pred| = the FM's own
predicted deviation from additivity. Route high-deviation edges to the FM (it claims an interaction there),
the rest to additive. We compare this to the ORACLE regime (route by the TRUE |real - additive|) as a ceiling.

Substrate: the held-out leave-combo-out CV predictions (cleanest interpolation set; all combo_seen2).
  python move1_3tier_baselines.py --cv /tmp/gears_cv_vs_real.csv --marginal results/gears_norman/labeled_marginal.csv
"""
import argparse, re, numpy as np, pandas as pd

DELTA = 0.25
SYN_THR = 0.33                      # synergy = |real - additive| >= 0.33 (the Layer-A 75th pct)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", default="/tmp/gears_cv_vs_real.csv")
    ap.add_argument("--marginal", default="results/gears_norman/labeled_marginal.csv")
    ap.add_argument("--lams", default="0.2,0.5,1.0")
    a = ap.parse_args()
    cv = pd.read_csv(a.cv).dropna(subset=["gears_log2FC"])
    m = pd.read_csv(a.marginal); m = m[np.isclose(m["delta"], DELTA)]
    single = {(r.target, r.response_id): r.log2FC for r in m[m.kind == "single"].itertuples()}
    creal  = {(r.target, r.response_id): r.log2FC for r in m[m.kind == "combo"].itertuples()}
    def addlfc(p, gn):
        ps = re.split(r"[+_]", p); v = [single.get((x, gn)) for x in ps]
        return float(np.sum(v)) if len(ps) > 1 and all(x is not None for x in v) else np.nan
    cv["add_lfc"]  = [addlfc(p, gn) for p, gn in zip(cv.perturbation, cv.gene)]
    cv["real_lfc"] = [creal.get((p, gn), np.nan) for p, gn in zip(cv.perturbation, cv.gene)]
    cv = cv.dropna(subset=["add_lfc", "real_lfc"]).copy()
    real_eff = (cv.real_call == "effect").to_numpy()
    fm_correct  = ((cv.gears_log2FC.abs() >= DELTA).to_numpy() == real_eff)
    add_correct = ((cv.add_lfc.abs()     >= DELTA).to_numpy() == real_eff)
    true_syn = (cv.real_lfc - cv.add_lfc).abs().to_numpy() >= SYN_THR        # oracle regime
    pred_dev = (cv.gears_log2FC - cv.add_lfc).abs().to_numpy()               # DEPLOYABLE signal (inference-available)
    perts = cv.perturbation.to_numpy(); N = len(cv)
    print(f"3-tier baselines on held-out CV: {N} edges / {cv.perturbation.nunique()} combos | "
          f"synergy frac (true) {true_syn.mean():.2f}")

    def route(use_fm_mask):                                                  # free routing add<->fm; returns per-edge correct
        return np.where(use_fm_mask, fm_correct, add_correct)
    # deployable: route top-q by predicted deviation to FM (q = true synergy frac, a natural operating point)
    q = true_syn.mean()
    thr = np.quantile(pred_dev, 1 - q)
    pred_route = pred_dev >= thr
    policies = {
        "always-additive":      (add_correct,                 np.zeros(N, bool)),
        "trust-all-FM":         (fm_correct,                  np.zeros(N, bool)),
        "always-verify":        (np.ones(N, bool),            np.ones(N, bool)),
        "oracle-free (max add,fm)": (add_correct | fm_correct, np.zeros(N, bool)),
        "regime-router ORACLE": (route(true_syn),             np.zeros(N, bool)),
        "regime-router PRED |fm-add|": (route(pred_route),    np.zeros(N, bool)),
        "oracle-3tier":         (np.ones(N, bool),            ~(add_correct | fm_correct)),
    }
    lams = [float(x) for x in a.lams.split(",")]
    print(f"\n{'policy':30s} {'acc':>7s} {'assay%':>7s} | " + " ".join(f"net@l{l:<4}" for l in lams))
    rows = {}
    for name, (corr, assay) in policies.items():
        acc = corr.mean(); ar = assay.mean()
        nets = {l: acc - l*ar for l in lams}
        rows[name] = (acc, ar, nets)
        print(f"{name:30s} {acc:>7.3f} {ar:>7.0%} | " + " ".join(f"{nets[l]:>8.3f}" for l in lams))

    # K1 gate: does regime routing beat the BEST constant free policy, for free? cluster-bootstrap by perturbation
    best_const = max(rows["always-additive"][0], rows["trust-all-FM"][0])
    def boot_gap(corr_a, corr_b, B=2000, seed=0):
        rng = np.random.default_rng(seed); P = np.unique(perts)
        idx = {p: np.where(perts == p)[0] for p in P}
        pt = corr_a.mean() - corr_b.mean(); vals = []
        for _ in range(B):
            take = np.concatenate([idx[p] for p in rng.choice(P, len(P), replace=True)])
            vals.append(corr_a[take].mean() - corr_b[take].mean())
        return pt, np.percentile(vals, 2.5), np.percentile(vals, 97.5)
    best_const_corr = add_correct if rows["always-additive"][0] >= rows["trust-all-FM"][0] else fm_correct
    print(f"\n=== K1 GATE (free regime routing vs best constant free policy, cluster-boot by perturbation) ===")
    for name, corr in [("regime ORACLE", route(true_syn)), ("regime PRED |fm-add|", route(pred_route))]:
        pt, lo, hi = boot_gap(corr, best_const_corr)
        sig = "PASS (CI excl 0)" if lo > 0 else "fail"
        print(f"  {name:22s} acc {corr.mean():.3f} vs best-const {best_const:.3f} | gap {pt:+.3f} [{lo:+.3f},{hi:+.3f}] -> {sig}")
    print(f"  deployable PRED captures {(route(pred_route).mean()-best_const)/(route(true_syn).mean()-best_const+1e-9):.0%} of the ORACLE free-routing gain")
    print(f"\n(interpret: both add & FM are FREE, so regime routing is a free accuracy gain; verify is the paid tier."
          f" If PRED passes K1, the deployable 3-tier orchestrator needs no trained head -- just |fm_pred-add_pred|.)")

if __name__ == "__main__":
    main()
