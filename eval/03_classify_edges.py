#!/usr/bin/env python3
"""
CausalAtlas Phase 1 - Step 03: apply the locked refinement #1 rule to label edges.

tested-negative = an edge where an effect >= delta is statistically EXCLUDED
(equivalence / TOST), NOT merely q>=q*. Spec locked 2026-06-09.

Input : discovery_results.csv from Step 02 (cis_knockdown rows + trans rows).
Output: labeled_edges.csv  (one row per trans edge x delta, with label + tier)
        summary.csv         (counts of POSITIVE/TESTED_NEGATIVE/UNTESTED per delta/tier)

Rule per trans edge A->B (within powered subspace), gates G0-G2 then label:
  POSITIVE        : q < q*  AND |log2FC| >= delta  AND G1 knockdown-confirmed
                    (a significant trans effect from a regulator that was NOT
                     knockdown-confirmed is UNTESTED: A was not perturbed, so A->B
                     is unattributable. Added 2026-06-10 after cross-line analysis
                     found 27% of RPE1 positives came from non-kd-confirmed regulators.)
  TESTED_NEGATIVE : q >= q* AND |log2FC| + z*SE < delta            (effect >= delta excluded)
  UNTESTED        : otherwise (underpowered: CI cannot exclude delta)
Tiers for negatives: A = knockdown>=50% & 95% CI (z=1.96) & B well-detected;
                     B = knockdown>=30% & 90% CI (z=1.645).
SE on log2 scale: explicit column if present, else recovered as |log2FC|/qnorm(1-p/2).
"""
import argparse, sys
import numpy as np
import pandas as pd
from scipy.stats import norm

DELTAS = [0.1, 0.25, 0.5, 1.0]
DELTA_PRIMARY = 0.25
QSTAR = 0.05
KD_MIN_LFC = np.log2(0.7)   # >=30% knockdown -> log2FC <= -0.515
KD_A_LFC   = np.log2(0.5)   # >=50% knockdown -> log2FC <= -1.0
N_TRT_MIN  = 25
N_CNTRL_MIN_B   = 100       # detectability proxy (B expressed in controls)
N_CNTRL_MIN_A   = 200
Z95, Z90 = 1.959964, 1.644854

def log(*a): print("[classify]", *a, file=sys.stderr, flush=True)

def col(df, *names):
    for n in names:
        if n in df.columns: return n
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    import os; os.makedirs(a.outdir, exist_ok=True)

    df = pd.read_csv(a.discovery)
    c_lfc = col(df, "log_2_fold_change", "log2_fold_change", "log_2_fc")
    c_p   = col(df, "p_value", "pvalue", "p")
    c_q   = col(df, "significant")  # sceptre marks BH-significant; we also recompute q if p present
    c_tgt = col(df, "grna_target", "target"); c_resp = col(df, "response_id", "response")
    c_ntrt = col(df, "n_nonzero_trt", "n_nonzero_treatment")
    c_ncon = col(df, "n_nonzero_cntrl", "n_nonzero_control")
    c_se   = col(df, "se_log_2_fold_change", "se_log2_fold_change")
    c_seFC = col(df, "se_fold_change"); c_FC = col(df, "fold_change")
    c_type = col(df, "pair_type")
    for nm, c in [("log2FC", c_lfc), ("p_value", c_p), ("target", c_tgt), ("response", c_resp)]:
        if c is None: sys.exit(f"missing required column for {nm} in {a.discovery}")

    # --- SE on log2 scale (prefer explicit; else from se_fold_change; else from test stat) ---
    p = df[c_p].clip(lower=1e-300).values
    lfc = df[c_lfc].values
    if c_se:
        se = df[c_se].values
        se_src = "explicit"
    elif c_seFC and c_FC:
        fc = df[c_FC].replace(0, np.nan).values
        se = df[c_seFC].values / (fc * np.log(2))
        se_src = "from se_fold_change"
    else:
        z = norm.isf(p / 2.0)               # two-sided test stat magnitude
        z = np.where(z <= 0, np.nan, z)
        se = np.abs(lfc) / z                # SE recovered from effect and p
        se_src = "recovered from p_value"
    df["_se_log2"] = se
    log("SE source:", se_src)

    # BH q-value from p, computed WITHIN each pair-type family (cis knockdown QC vs trans discovery).
    # Pooling them runs BH over a mixture where the ~1.9k ultra-significant cis tests (p~1e-25) shrink
    # ranks and make trans q anti-conservative (fix 2026-06-11). NaN-safe (untested pairs have NaN p).
    def _bh(praw):
        valid = ~np.isnan(praw)
        qq = np.full(len(praw), np.nan)
        if valid.any():
            pv = np.clip(praw[valid], 1e-300, 1.0)
            order = np.argsort(pv); ranks = np.empty_like(order); ranks[order] = np.arange(1, len(pv)+1)
            qv = pv * len(pv) / ranks
            qv = np.minimum.accumulate(qv[order][::-1])[::-1]   # over sorted; map back
            qvo = np.empty_like(qv); qvo[order] = qv
            qq[valid] = np.clip(qvo, 0, 1)
        return qq
    if c_type and df[c_type].notna().any():
        fam = df[c_type].astype(str)
    else:
        fam = pd.Series(np.where(df[c_tgt].astype(str) == df[c_resp].astype(str), "cis", "trans"),
                        index=df.index)
    df["_q"] = np.nan
    for g in pd.unique(fam):
        m = (fam == g).values
        df.loc[m, "_q"] = _bh(df.loc[m, c_p].values.astype(float))

    # --- G1 knockdown QC from cis rows ---
    if c_type and (df[c_type] == "cis_knockdown").any():
        cis = df[df[c_type] == "cis_knockdown"]
    else:
        cis = df[df[c_tgt].astype(str) == df[c_resp].astype(str)]
    kd = cis.set_index(c_tgt)[c_lfc].to_dict()
    kd_q = cis.set_index(c_tgt)["_q"].to_dict()
    kd_se = cis.set_index(c_tgt)["_se_log2"].to_dict()
    def kd_ok(t, lvl):
        # CI-based G1 (2026-06-11): require the UPPER 90% CI of the (negative) cis knockdown to be below the
        # threshold, i.e. the knockdown is CONFIDENTLY >= the target level -- robust to the noisy point estimate
        # that made the TN/UNTESTED split swing ~19% near the boundary (N6 sensitivity). Falls back to point if SE NaN.
        l = kd.get(t, np.nan); s = kd_se.get(t, np.nan)
        if np.isnan(l):
            return False
        ub = l + Z90 * s if not np.isnan(s) else l
        return (ub <= lvl) and (kd_q.get(t, 1.0) < QSTAR)

    # --- classify trans edges ---
    trans = df[df[c_resp].astype(str) != df[c_tgt].astype(str)].copy() if not c_type \
            else df[df[c_type] == "trans"].copy()
    rows = []
    for _, r in trans.iterrows():
        t, b = r[c_tgt], r[c_resp]
        l, s, qv = r[c_lfc], r["_se_log2"], r["_q"]
        ntrt = r[c_ntrt] if c_ntrt else np.nan
        ncon = r[c_ncon] if c_ncon else np.nan
        # gates
        g2_min = (np.isnan(ntrt) or ntrt >= N_TRT_MIN) and (np.isnan(ncon) or ncon >= N_CNTRL_MIN_B)
        g2_A   = (np.isnan(ncon) or ncon >= N_CNTRL_MIN_A)
        kd30, kd50 = kd_ok(t, KD_MIN_LFC), kd_ok(t, KD_A_LFC)
        for d in DELTAS:
            if qv < QSTAR and abs(l) >= d and kd30:
                label, tier = "POSITIVE", ""
            elif np.isnan(s):
                label, tier = "UNTESTED", ""
            else:
                ub90 = abs(l) + Z90 * s
                ub95 = abs(l) + Z95 * s
                # TESTED_NEGATIVE is defined purely by CI-exclusion + gates, INDEPENDENT of q
                # (a significant small effect whose CI excludes delta is the canonical tested-negative;
                # requiring q>=QSTAR here re-imported the NHST logic the equivalence test rejects -- fixed 2026-06-11)
                eligible = kd30 and g2_min
                if eligible and ub90 < d:
                    label = "TESTED_NEGATIVE"
                    tier = "A" if (kd50 and g2_A and ub95 < d) else "B"
                else:
                    label, tier = "UNTESTED", ""
            rows.append(dict(grna_target=t, response_id=b, delta=d, label=label, tier=tier,
                             log2FC=l, se_log2=s, q=qv, n_trt=ntrt, n_cntrl=ncon,
                             kd_log2FC=kd.get(t, np.nan), kd_ge30=kd30, kd_ge50=kd50))
    out = pd.DataFrame(rows)
    out.to_csv(f"{a.outdir}/labeled_edges.csv", index=False)

    # summary
    summ = (out.groupby(["delta", "label", "tier"]).size().reset_index(name="n"))
    summ.to_csv(f"{a.outdir}/summary.csv", index=False)
    log("trans edges:", trans.shape[0], "| knockdown-confirmed targets:",
        sum(kd_ok(t, KD_MIN_LFC) for t in set(trans[c_tgt])))
    for d in DELTAS:
        sub = out[out.delta == d]
        tn = (sub.label == "TESTED_NEGATIVE").sum()
        pos = (sub.label == "POSITIVE").sum(); un = (sub.label == "UNTESTED").sum()
        star = " <- primary" if d == DELTA_PRIMARY else ""
        log(f"  delta={d}: POSITIVE={pos} TESTED_NEGATIVE={tn} UNTESTED={un}{star}")

if __name__ == "__main__":
    main()
