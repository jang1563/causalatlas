#!/usr/bin/env python3
"""
CausalAtlas Phase 2 - Step 03: genetic-interaction (GI) edge classifier for Norman 2019.

Two three-state ground-truth axes from one sceptre discovery table (02_sceptre_norman.R):

  MARGINAL effect  (target T -> gene C, T in {single, combo}):  vs ZERO
    Phase-1 logic with the CRISPRa sign flip. POSITIVE if a >=delta effect is shown & T is
    activation-confirmed; TESTED_NEGATIVE if a >=delta effect is EXCLUDED (|lfc|+z*SE<delta) & gates;
    else UNTESTED.

  INTERACTION / epistasis  (combo A+B -> gene C):  vs the ADDITIVE prediction
    residual  d = lfc(A+B->C) - [lfc(A->C) + lfc(B->C)],  SE_d = sqrt(SE_AB^2 + SE_A^2 + SE_B^2)
    POSITIVE_INTERACTION (epistatic) : |d| >= delta AND |d| - z*SE_d > 0   (combo deviates from additive)
    TESTED_NEGATIVE_INTERACTION      : |d| + z*SE_d < delta                 (epistasis >= delta EXCLUDED = additive)
    UNTESTED                         : otherwise, or any of the 3 components underpowered/missing
  This additive-null tested-negative-vs-untested is the stricter interaction axis.

G1 is an ACTIVATION gate (CRISPRa): a single A is activation-confirmed when the cis self-edge A->A
goes confidently UP -- lower CI (lfc - z*SE) >= +level AND q<q*. A combo requires BOTH singles confirmed.

Usage:
  python 03_gi_classify.py --discovery <results>/discovery_results.csv --outdir <results>
"""
import argparse, os, sys
import numpy as np
import pandas as pd
from scipy.stats import norm

DELTAS = [0.1, 0.25, 0.5, 1.0]
DELTA_PRIMARY = 0.25
QSTAR = 0.05
ACT_MIN_LFC = np.log2(1.5)   # activation-confirmed: >=1.5x up (+0.585), mirrors KD-30% magnitude
ACT_A_LFC   = np.log2(2.0)   # tier A: >=2x up (+1.0)
N_TRT_MIN = 25
N_CNTRL_MIN_B = 100
N_CNTRL_MIN_A = 200
Z95, Z90 = 1.959964, 1.644854

def log(*a): print("[gi-classify]", *a, file=sys.stderr, flush=True)

def col(df, *names):
    for n in names:
        if n in df.columns: return n
    return None

def bh(praw):
    valid = ~np.isnan(praw)
    qq = np.full(len(praw), np.nan)
    if valid.any():
        pv = np.clip(praw[valid], 1e-300, 1.0)
        order = np.argsort(pv); ranks = np.empty_like(order); ranks[order] = np.arange(1, len(pv)+1)
        qv = pv * len(pv) / ranks
        qv = np.minimum.accumulate(qv[order][::-1])[::-1]
        qvo = np.empty_like(qv); qvo[order] = qv
        qq[valid] = np.clip(qvo, 0, 1)
    return qq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    df = pd.read_csv(a.discovery)
    c_lfc = col(df, "log_2_fold_change", "log2_fold_change")
    c_p   = col(df, "p_value", "pvalue", "p")
    c_tgt = col(df, "grna_target", "target")
    c_resp= col(df, "response_id", "response")
    c_ntrt= col(df, "n_nonzero_trt", "n_nonzero_treatment")
    c_ncon= col(df, "n_nonzero_cntrl", "n_nonzero_control")
    c_se  = col(df, "se_log_2_fold_change", "se_log2_fold_change")
    c_seFC= col(df, "se_fold_change")            # sceptre DOES emit this (review M1 fix)
    c_FC  = col(df, "fold_change")
    c_type= col(df, "pair_type")
    for nm, c in [("log2FC", c_lfc), ("p_value", c_p), ("target", c_tgt), ("response", c_resp)]:
        if c is None: sys.exit(f"missing column for {nm} in {a.discovery}")

    # SE on log2 scale. Prefer an explicit log2 SE; else convert sceptre's se_fold_change by the delta method
    # (se_log2 = se_FC / (FC * ln2)) -- this is sceptre's REAL effect-size SE and is 2-8x smaller than the
    # p-recovered Wald approximation, which also misattributes the conditional-randomization tail to variance.
    # Only fall back to p-recovery if neither is present.
    p = df[c_p].clip(lower=1e-300).values
    lfc = df[c_lfc].values
    if c_se:
        se = df[c_se].values; se_src = "explicit log2"
    elif c_seFC and c_FC:
        fc = df[c_FC].replace(0, np.nan).values
        se = df[c_seFC].values / (fc * np.log(2)); se_src = "delta-method from se_fold_change"
    else:
        z = norm.isf(p / 2.0); z = np.where(z <= 0, np.nan, z)
        se = np.abs(lfc) / z; se_src = "recovered from p (FALLBACK)"
    df["_se"] = se
    log("SE source:", se_src)

    # per-family BH q
    fam = df[c_type].astype(str) if c_type else pd.Series("all", index=df.index)
    df["_q"] = np.nan
    for g in pd.unique(fam):
        m = (fam == g).values
        df.loc[m, "_q"] = bh(df.loc[m, c_p].values.astype(float))

    # lookup: (target, response) -> row stats
    key = lambda t, r: (str(t), str(r))
    L = {key(r[c_tgt], r[c_resp]): (r[c_lfc], r["_se"], r["_q"],
                                    r[c_ntrt] if c_ntrt else np.nan,
                                    r[c_ncon] if c_ncon else np.nan)
         for _, r in df.iterrows()}

    # G1 activation gate from cis_activation rows (single A -> A, CRISPRa => UP)
    if c_type and (df[c_type] == "cis_activation").any():
        cis = df[df[c_type] == "cis_activation"]
    else:
        cis = df[df[c_tgt].astype(str) == df[c_resp].astype(str)]
    act_l  = cis.set_index(c_tgt)[c_lfc].to_dict()
    act_se = cis.set_index(c_tgt)["_se"].to_dict()
    act_q  = cis.set_index(c_tgt)["_q"].to_dict()
    def act_ok(gene, lvl):
        l = act_l.get(gene, np.nan); s = act_se.get(gene, np.nan)
        if np.isnan(l): return False
        lb = l - Z90 * s if not np.isnan(s) else l       # lower CI confidently above +lvl
        return (lb >= lvl) and (act_q.get(gene, 1.0) < QSTAR)

    def g2(ntrt, ncon, lvl_con):
        return (np.isnan(ntrt) or ntrt >= N_TRT_MIN) and (np.isnan(ncon) or ncon >= lvl_con)

    singles = set(str(t) for t in df[c_tgt].unique() if "+" not in str(t) and str(t) != "non-targeting")
    combos  = [str(t) for t in df[c_tgt].unique() if "+" in str(t)]

    # ---------- MARGINAL three-state (T -> C vs zero) ----------
    mrows = []
    targ_rows = df[df[c_tgt].astype(str) != "non-targeting"]
    for _, r in targ_rows.iterrows():
        T, C = str(r[c_tgt]), str(r[c_resp])
        l, s, qv = r[c_lfc], r["_se"], r["_q"]
        ntrt = r[c_ntrt] if c_ntrt else np.nan
        ncon = r[c_ncon] if c_ncon else np.nan
        if "+" in T:
            parts = T.split("+"); confirmed = all(act_ok(x, ACT_MIN_LFC) for x in parts)
            conf50 = all(act_ok(x, ACT_A_LFC) for x in parts)
        else:
            confirmed = act_ok(T, ACT_MIN_LFC); conf50 = act_ok(T, ACT_A_LFC)
        g2_min = g2(ntrt, ncon, N_CNTRL_MIN_B); g2_A = g2(ntrt, ncon, N_CNTRL_MIN_A)
        for d in DELTAS:
            if not np.isnan(qv) and qv < QSTAR and abs(l) >= d and confirmed:
                label, tier = "POSITIVE", ""
            elif np.isnan(s) or not confirmed or not g2_min:
                label, tier = "UNTESTED", ""
            elif abs(l) + Z90 * s < d:
                label = "TESTED_NEGATIVE"; tier = "A" if (conf50 and g2_A and abs(l) + Z95 * s < d) else "B"
            else:
                label, tier = "UNTESTED", ""
            mrows.append(dict(target=T, response_id=C, kind="combo" if "+" in T else "single",
                              delta=d, label=label, tier=tier, log2FC=l, se=s, q=qv,
                              n_trt=ntrt, n_cntrl=ncon, activation_confirmed=confirmed))
    mout = pd.DataFrame(mrows)
    mout.to_csv(f"{a.outdir}/labeled_marginal.csv", index=False)

    # ---------- INTERACTION three-state (combo A+B -> C vs additive) ----------
    irows = []
    for ab in combos:
        parts = ab.split("+")
        if len(parts) != 2: continue
        A, B = parts
        confirmed = act_ok(A, ACT_MIN_LFC) and act_ok(B, ACT_MIN_LFC)
        # response genes tested for the combo
        cs = [str(r[c_resp]) for _, r in df[df[c_tgt].astype(str) == ab].iterrows()]
        for C in cs:
            ab_s = L.get(key(ab, C)); a_s = L.get(key(A, C)); b_s = L.get(key(B, C))
            if ab_s is None or a_s is None or b_s is None:
                # a component not measured -> interaction untested
                for d in DELTAS:
                    irows.append(dict(combo=ab, A=A, B=B, response_id=C, delta=d, label="UNTESTED",
                                      tier="", delta_resid=np.nan, se_resid=np.nan,
                                      lfc_AB=np.nan, lfc_A=np.nan, lfc_B=np.nan,
                                      reason="component_missing", activation_confirmed=confirmed))
                continue
            (eAB, sAB, qAB, ntAB, ncAB) = ab_s
            (eA, sA, _, ntA, _) = a_s
            (eB, sB, _, ntB, _) = b_s
            resid = eAB - (eA + eB)
            se_r = np.sqrt(np.nansum([sAB**2, sA**2, sB**2])) if not any(np.isnan([sAB, sA, sB])) else np.nan
            powered = (g2(ntAB, ncAB, N_CNTRL_MIN_B) and (np.isnan(ntA) or ntA >= N_TRT_MIN)
                       and (np.isnan(ntB) or ntB >= N_TRT_MIN))
            for d in DELTAS:
                if np.isnan(se_r) or not confirmed or not powered:
                    label, tier, reason = "UNTESTED", "", "underpowered" if confirmed else "not_activation_confirmed"
                elif abs(resid) >= d and abs(resid) - Z90 * se_r > 0:
                    label, tier, reason = "POSITIVE_INTERACTION", ("A" if abs(resid) - Z95 * se_r > 0 else "B"), "epistatic"
                elif abs(resid) + Z90 * se_r < d:
                    label, tier, reason = "TESTED_NEGATIVE_INTERACTION", ("A" if abs(resid) + Z95 * se_r < d else "B"), "additive"
                else:
                    label, tier, reason = "UNTESTED", "", "ci_straddles_delta"
                irows.append(dict(combo=ab, A=A, B=B, response_id=C, delta=d, label=label, tier=tier,
                                  delta_resid=resid, se_resid=se_r, lfc_AB=eAB, lfc_A=eA, lfc_B=eB,
                                  reason=reason, activation_confirmed=confirmed))
    iout = pd.DataFrame(irows)
    iout.to_csv(f"{a.outdir}/labeled_interaction.csv", index=False)

    # summaries
    for name, out, lab_col in [("marginal", mout, "label"), ("interaction", iout, "label")]:
        if out.empty: continue
        summ = out.groupby(["delta", lab_col]).size().reset_index(name="n")
        summ.to_csv(f"{a.outdir}/summary_{name}.csv", index=False)
    log(f"activation-confirmed singles: {sum(act_ok(s, ACT_MIN_LFC) for s in singles)}/{len(singles)}")
    for d in [DELTA_PRIMARY]:
        m = mout[mout.delta == d]; i = iout[iout.delta == d]
        log(f"delta={d} MARGINAL: POS={int((m.label=='POSITIVE').sum())} "
            f"TN={int((m.label=='TESTED_NEGATIVE').sum())} UN={int((m.label=='UNTESTED').sum())}")
        if not i.empty:
            log(f"delta={d} INTERACTION: POS_INT={int((i.label=='POSITIVE_INTERACTION').sum())} "
                f"TN_INT={int((i.label=='TESTED_NEGATIVE_INTERACTION').sum())} UN={int((i.label=='UNTESTED').sum())}")

if __name__ == "__main__":
    main()
