#!/usr/bin/env python3
"""Experiment-selection P1 -- the two decisive checks the loop's NEGATIVE demands (multi-angle review angles 1,3):
the naive loop found informed selection WORSE than random with an ANTI-calibrated distance-uncertainty (-0.15).
Before concluding, test the two levers the project thesis names:

  CHECK (i)  SIGNAL: is a BETTER-calibrated uncertainty available, and does it flip K1?
             compare distance / neighbor-disagreement / committee(bootstrap-kNN) uncertainties by
             corr(u, error); then greedy-select by the best-calibrated u vs random (AULC on a held-out test).
  CHECK (ii) GOAL: switch from uniform map-reconstruction to HIT-SEARCH (find the top-effect perturbations).
             exploit (reveal highest GO-predicted effect-magnitude) vs random; recall of true top-decile @ budget.

  python move1_expsel_checks.py --effects results/expsel/effects.npy --features results/expsel/F_go.npy
"""
import argparse, numpy as np
from numpy.linalg import norm
EPS = 1e-8

def knn_pred(featz, eff, revealed, query, k=10):
    A = featz[revealed]; B = featz[query]
    D = np.sqrt(np.maximum(((B[:, None, :] - A[None, :, :]) ** 2).sum(-1), 0))   # q x R
    pred = np.zeros((len(query), eff.shape[1])); nbr = []
    kk = min(k, len(revealed))
    for i in range(len(query)):
        o = np.argsort(D[i])[:kk]; d = D[i, o]; w = 1.0 / (d + EPS); w /= w.sum()
        pred[i] = (w[:, None] * eff[np.asarray(revealed)[o]]).sum(0); nbr.append((np.asarray(revealed)[o], d))
    return pred, nbr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--effects", default="results/expsel/effects.npy")
    ap.add_argument("--features", default="results/expsel/F_go.npy")
    ap.add_argument("--k", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=200); ap.add_argument("--committee", type=int, default=8)
    a = ap.parse_args()
    eff = np.load(a.effects).astype(float); feat = np.load(a.features).astype(float)
    featz = (feat - feat.mean(0)) / (feat.std(0) + EPS)
    rng = np.random.default_rng(a.seed); P = eff.shape[0]
    idx = rng.permutation(P); nte = P // 4; test = idx[:nte]; poolall = idx[nte:]
    mag = norm(eff, axis=1)                                            # effect magnitude per pert

    # ---------- CHECK (i): uncertainty calibration ----------
    R = list(rng.choice(poolall, size=a.budget, replace=False))       # a mid-budget revealed set
    pred, nbr = knn_pred(featz, eff, R, test, a.k)
    err = 1 - (pred * eff[test]).sum(1) / (norm(pred, axis=1) * norm(eff[test], axis=1) + EPS)   # per-test error
    # u schemes for the TEST perts given R
    u_dist = np.array([d.mean() for (_, d) in nbr])
    u_disag = np.array([mag[ni].std() for (ni, _) in nbr])            # neighbor magnitude disagreement
    # committee: B bootstrap-kNN predicted magnitudes -> std
    comm = np.zeros((a.committee, len(test)))
    for b in range(a.committee):
        Rb = list(np.random.default_rng(10 + b).choice(R, size=len(R), replace=True))
        pb, _ = knn_pred(featz, eff, Rb, test, a.k); comm[b] = norm(pb, axis=1)
    u_comm = comm.std(0)
    print("CHECK (i) uncertainty CALIBRATION  corr(u, error) on held-out test (>0 = usable):")
    for nm, u in [("distance", u_dist), ("neighbor-disagree", u_disag), ("committee(bootstrap)", u_comm)]:
        print(f"   {nm:22s} corr = {np.corrcoef(u, err)[0,1]:+.3f}")
    # greedy AULC for the best-calibrated (committee) vs random
    def greedy(uscheme, steps=a.budget, every=5):
        rev = list(rng.choice(poolall, size=5, replace=False)); pool = [p for p in poolall if p not in rev]
        xs, cs = [], []
        for s in range(steps):
            if not pool: break
            if uscheme == "random":
                pick = int(np.random.default_rng(500 + s).choice(pool))
            else:                                                     # committee uncertainty over pool
                cm = np.zeros((a.committee, len(pool)))
                for b in range(a.committee):
                    Rb = list(np.random.default_rng(b).choice(rev, size=len(rev), replace=True)) if len(rev) > 2 else rev
                    pb, _ = knn_pred(featz, eff, Rb, pool, a.k); cm[b] = norm(pb, axis=1)
                pick = int(pool[np.argmax(cm.std(0))])
            rev.append(pick); pool.remove(pick)
            if s % every == 0 or s == steps - 1:
                pr, _ = knn_pred(featz, eff, rev, test, a.k)
                cs.append(float(np.mean((pr * eff[test]).sum(1) / (norm(pr, axis=1) * norm(eff[test], axis=1) + EPS))))
                xs.append(len(rev))
        return np.array(xs), np.array(cs)
    xr, cr = greedy("random"); xc, cc = greedy("committee")
    ar = np.trapz(cr, xr) / (xr[-1] - xr[0] + EPS); ac = np.trapz(cc, xc) / (xc[-1] - xc[0] + EPS)
    print(f"   K1 (committee-uncertainty vs random): AULC committee {ac:.3f} vs random {ar:.3f}  gap {ac-ar:+.3f}"
          f"  -> {'committee BEATS random' if ac > ar else 'still <= random'}")

    # ---------- CHECK (ii): hit-search goal ----------
    top = set(np.argsort(-mag)[:max(1, P // 10)].tolist())            # true top-decile by effect magnitude
    def search(mode, steps=a.budget):
        rev = list(rng.choice(poolall, size=5, replace=False)); pool = [p for p in poolall if p not in rev]
        found = []
        for s in range(steps):
            if not pool: break
            if mode == "random":
                pick = int(np.random.default_rng(700 + s).choice(pool))
            else:                                                     # exploit: highest GO-predicted magnitude
                pr, _ = knn_pred(featz, eff, rev, pool, a.k); pick = int(pool[np.argmax(norm(pr, axis=1))])
            rev.append(pick); pool.remove(pick)
            found.append(len([p for p in rev if p in top]) / len(top))
        return np.array(found)
    fr = search("random"); fe = search("exploit")
    print(f"\nCHECK (ii) HIT-SEARCH (recall of true top-decile high-effect perts @ budget {a.budget}):")
    print(f"   random  recall@budget {fr[-1]:.2f}   exploit recall@budget {fe[-1]:.2f}")
    print(f"   discovery-AULC: random {np.mean(fr):.3f}  exploit {np.mean(fe):.3f}  gap {np.mean(fe)-np.mean(fr):+.3f}"
          f"  -> {'exploit BEATS random (informed search works for non-uniform goal)' if np.mean(fe) > np.mean(fr) else 'exploit <= random'}")

if __name__ == "__main__":
    main()
