#!/usr/bin/env python3
"""P5 kill-test -- does a LEARNED acquisition policy beat the best heuristic in the HEADROOM regime?

The machinery-K1 (move1_p5_sim.py) validated the simulator: informed selection beats random only when the
interaction structure is feature-predictable (beta>=0.6); at beta=0.4 structure exists (ceiling ~0.79) but
simple heuristics FAIL (<= random), leaving ~0.09 headroom unclaimed. THAT is where "does a smarter policy/RL
help" is genuinely open. Here we learn a parametric acquisition policy and test it there.

Policy: score(candidate) = w . [uncertainty, predicted-prob, diversity, density] (each rank-normalized over the
pool); pick argmax. Weights w optimized by random-search on TRAIN seeds (maximize AULC), evaluated on HELD-OUT
seeds vs the best heuristic (uncertainty) and random. Train/test seed separation avoids overfitting the policy.
This is a confound-free form of policy-learning (a full deep-RL that lost could be merely under-tuned).

Kill-test verdict: in the headroom regime, learned-policy > uncertainty (and -> ceiling) means policy/RL IS a
lever where heuristics fail; learned-policy ~ uncertainty means policy is not the lever even with headroom.

  python move1_p5_policy.py --betas 0.4,0.6
"""
import argparse, sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
def log(*a): print("[p5pol]", *a, file=sys.stderr, flush=True)
EPS = 1e-9

def make_world(N=60, d=16, nclust=6, beta=0.6, q_pos=0.15, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((nclust, d)) * 2.0
    clu = rng.integers(0, nclust, N); feats = centers[clu] + rng.standard_normal((N, d)) * 0.6
    pairs = [(g, h) for g in range(N) for h in range(g + 1, N)]; P = len(pairs)
    fg = feats[[p[0] for p in pairs]]; fh = feats[[p[1] for p in pairs]]
    s = -((fg - fh) ** 2).sum(1); s = (s - s.mean()) / (s.std() + EPS)
    strength = beta * s + (1 - beta) * rng.standard_normal(P)
    label = (strength >= np.quantile(strength, 1 - q_pos)).astype(int)
    X = np.column_stack([np.abs(fg - fh), fg + fh])
    return X, label, P

def fit_predict(X, revealed, label):
    yr = label[revealed]
    if len(np.unique(yr)) < 2: return np.full(X.shape[0], yr.mean() if len(yr) else 0.5)
    return LogisticRegression(max_iter=500).fit(X[revealed], yr).predict_proba(X)[:, 1]

def rank01(v):
    r = np.argsort(np.argsort(v)).astype(float); return r / (len(v) - 1 + EPS)

def run_policy(w, X, label, test, budget, seed, density, seed_n=20, every=5):
    rng = np.random.default_rng(seed)
    pool = list(np.setdiff1d(np.arange(len(label)), test))
    revealed = list(rng.choice(pool, size=seed_n, replace=False)); pool = [i for i in pool if i not in revealed]
    mind = np.full(len(label), np.inf)                                   # running min-dist to revealed (diversity)
    for r in revealed: mind = np.minimum(mind, ((X - X[r]) ** 2).sum(1))
    xs, aucs = [], []
    for step in range(budget):
        if not pool: break
        p = fit_predict(X, revealed, label); pa = np.asarray(pool)
        feats = np.column_stack([rank01(1 - np.abs(2 * p[pa] - 1)),      # uncertainty
                                 rank01(p[pa]),                          # exploit (predicted prob)
                                 rank01(mind[pa]),                       # diversity (far from revealed)
                                 rank01(-density[pa])])                  # density (representative = low mean-dist)
        pick = int(pa[np.argmax(feats @ w)])
        revealed.append(pick); pool.remove(pick); mind = np.minimum(mind, ((X - X[pick]) ** 2).sum(1))
        if step % every == 0 or step == budget - 1 or not pool:
            pt = fit_predict(X, revealed, label)
            if len(np.unique(label[test])) == 2: xs.append(len(revealed)); aucs.append(roc_auc_score(label[test], pt[test]))
    return float(np.trapz(aucs, xs) / (xs[-1] - xs[0] + EPS)) if len(xs) > 1 else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--betas", default="0.4,0.6"); ap.add_argument("--budget", type=int, default=110)
    ap.add_argument("--ntrain", type=int, default=3); ap.add_argument("--ntest", type=int, default=3)
    ap.add_argument("--nweights", type=int, default=40); ap.add_argument("--N", type=int, default=60)
    a = ap.parse_args()
    UNC = np.array([1.0, 0, 0, 0]); RND = None
    for beta in [float(b) for b in a.betas.split(",")]:
        # precompute per-seed worlds + density + ceiling
        worlds = {}
        for sd in range(a.ntrain + a.ntest):
            X, label, P = make_world(N=a.N, beta=beta, seed=sd)
            dens = np.array([((X - X[i]) ** 2).sum(1).mean() for i in range(0, P, 1)]) if P < 2500 else np.zeros(P)
            test = np.random.default_rng(100 + sd).choice(P, size=P // 4, replace=False)
            nontest = np.setdiff1d(np.arange(P), test)
            cp = fit_predict(X, list(nontest), label)
            ceil = roc_auc_score(label[test], cp[test]) if len(np.unique(label[test])) == 2 else np.nan
            worlds[sd] = (X, label, test, dens, ceil)
        train = list(range(a.ntrain)); testseeds = list(range(a.ntrain, a.ntrain + a.ntest))
        # random-search weights on TRAIN seeds
        wrng = np.random.default_rng(7); best_w, best_s = UNC, -1
        cand = [UNC] + [wrng.dirichlet([1, 1, 1, 1]) for _ in range(a.nweights)]
        for w in cand:
            s = np.mean([run_policy(w, *worlds[sd][:3], a.budget, sd, worlds[sd][3]) for sd in train])
            if s > best_s: best_s, best_w = s, w
        # evaluate on TEST seeds: learned policy vs uncertainty vs random vs ceiling
        def ev(w, seeds): return np.mean([run_policy(w, *worlds[sd][:3], a.budget, sd, worlds[sd][3]) for sd in seeds])
        lp = ev(best_w, testseeds); un = ev(UNC, testseeds)
        rnd = np.mean([run_policy(np.array([0,0,0,0.]), *worlds[sd][:3], a.budget, sd+999, worlds[sd][3]) for sd in testseeds])  # ~random via zero weights+seed shuffle
        ceil = np.nanmean([worlds[sd][4] for sd in testseeds])
        gap = lp - un
        print(f"beta={beta}: ceiling {ceil:.3f} | uncertainty {un:.3f} | LEARNED-POLICY {lp:.3f} | (rand~ {rnd:.3f})")
        print(f"   best_w (unc,exploit,divers,density) = [{','.join(f'{x:.2f}' for x in best_w)}]")
        print(f"   learned - uncertainty = {gap:+.3f}  -> {'POLICY HELPS (beats heuristic)' if gap > 0.02 else 'policy ~ heuristic (NOT the lever)'};"
              f"  remaining ceiling gap {ceil-lp:+.3f}")
    print("\n(kill-test: policy>>uncertainty & ->ceiling = policy is a lever where heuristics fail; policy~uncertainty = not the lever even with headroom.)")

if __name__ == "__main__":
    main()
