#!/usr/bin/env python3
"""P5 -- controlled closed-loop discovery on a simulator with KNOWN causal ground truth.

The decisive mechanism test the real-data layers could not do (real data is corpus-bound + mean-dominated + no
known truth). Discovery target = recover the genetic-INTERACTION graph (which gene PAIRS are epistatic) by
SELECTING which pair-experiments to run; the experiment supplies new ground truth (escapes cannot-exceed-verifier).
This file = the SIMULATOR + the closed loop + classical selectors + the machinery-K1 (does informed selection
recover known structure faster than random?), swept over beta = feature-predictability of the interactions
(the controlled analog of the real-data thin/mean-dominated regime). RL/learned-policy arm is added only after K1.

Pre-registered predictions: high beta (structure feature-predictable) -> informed (uncertainty/greedy) BEATS
random; low beta (idiosyncratic, real-bio-like) -> all collapse to random. RL-vs-greedy = the open kill-test (later).

Honesty: sim generative process != agent world model (agent must learn); oracle reported as ceiling; result is a
beta CURVE not a single point. Tests the ALGORITHM/loop, NOT real biology.

  python move1_p5_sim.py            # beta sweep, machinery-K1
"""
import argparse, sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
def log(*a): print("[p5]", *a, file=sys.stderr, flush=True)
EPS = 1e-9

def make_world(N=60, d=16, nclust=6, beta=0.6, q_pos=0.15, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((nclust, d)) * 2.0
    clu = rng.integers(0, nclust, N)
    feats = centers[clu] + rng.standard_normal((N, d)) * 0.6          # gene latent features (clustered = pathways)
    pairs = [(g, h) for g in range(N) for h in range(g + 1, N)]
    P = len(pairs)
    fg = feats[[p[0] for p in pairs]]; fh = feats[[p[1] for p in pairs]]
    # predictable interaction signal: strong when the two genes are close in feature space (same pathway)
    s = -((fg - fh) ** 2).sum(1)
    s = (s - s.mean()) / (s.std() + EPS)
    idio = rng.standard_normal(P)
    strength = beta * s + (1 - beta) * idio                          # beta = feature-predictable fraction
    label = (strength >= np.quantile(strength, 1 - q_pos)).astype(int)  # top q_pos = interacting (epistatic)
    pair_feats = np.column_stack([np.abs(fg - fh), fg + fh])          # INFERENCE features (no cluster id given)
    return pair_feats, label, P

def fit_predict(X, revealed, label):
    yr = label[revealed]
    if len(np.unique(yr)) < 2:
        p = np.full(X.shape[0], yr.mean() if len(yr) else 0.5); return p
    m = LogisticRegression(max_iter=500, C=1.0).fit(X[revealed], yr)
    return m.predict_proba(X)[:, 1]

def run(selector, X, label, test, budget, seed=0, seed_n=20, every=5):
    rng = np.random.default_rng({"random": 1, "uncertainty": 2, "greedy": 3, "oracle": 4}[selector] + seed)
    pool = list(np.setdiff1d(np.arange(len(label)), test))
    revealed = list(rng.choice(pool, size=seed_n, replace=False)); pool = [i for i in pool if i not in revealed]
    xs, aucs = [], []
    for step in range(budget):
        if not pool: break
        p = fit_predict(X, revealed, label)
        if selector == "random":   pick = int(rng.choice(pool))
        elif selector == "greedy": pick = pool[int(np.argmax(p[pool]))]
        elif selector == "uncertainty": pick = pool[int(np.argmin(np.abs(p[pool] - 0.5)))]
        elif selector == "oracle": pick = pool[int(np.argmax(label[pool] + 1e-3 * rng.standard_normal(len(pool))))]
        revealed.append(pick); pool.remove(pick)
        if step % every == 0 or step == budget - 1 or not pool:
            pt = fit_predict(X, revealed, label)
            if len(np.unique(label[test])) == 2:
                xs.append(len(revealed)); aucs.append(roc_auc_score(label[test], pt[test]))
    return np.array(xs), np.array(aucs)

def aulc(xs, ys): return float(np.trapz(ys, xs) / (xs[-1] - xs[0] + EPS)) if len(xs) > 1 else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=60); ap.add_argument("--budget", type=int, default=180)
    ap.add_argument("--betas", default="0.0,0.2,0.4,0.6,0.8"); ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    betas = [float(b) for b in a.betas.split(",")]
    print(f"{'beta':>5s} | {'ceiling':>7s} {'random':>7s} {'uncert':>7s} {'greedy':>7s} | informed-vs-random  VERDICT  (avg of {a.seeds} seeds)")
    for beta in betas:
        rows = {k: [] for k in ["ceiling", "random", "uncertainty", "greedy"]}
        for sd in range(a.seeds):
            X, label, P = make_world(N=a.N, beta=beta, seed=sd)
            test = np.random.default_rng(100 + sd).choice(P, size=P // 4, replace=False)
            nontest = np.setdiff1d(np.arange(P), test)
            cp = fit_predict(X, list(nontest), label)                # FULL-DATA ceiling (train on all non-test)
            rows["ceiling"].append(roc_auc_score(label[test], cp[test]) if len(np.unique(label[test])) == 2 else np.nan)
            for sel in ["random", "uncertainty", "greedy"]:
                xs, ys = run(sel, X, label, test, a.budget, seed=sd); rows[sel].append(aulc(xs, ys))
        A = {k: float(np.nanmean(v)) for k, v in rows.items()}
        best_inf = max(A["uncertainty"], A["greedy"]); gap = best_inf - A["random"]
        structure = A["ceiling"] > 0.60                              # is there learnable structure at all?
        if not structure:
            verdict = f"no structure (ceiling {A['ceiling']:.2f}) -> collapse (expected at low beta)"
        else:
            verdict = (f"+{gap:.3f} informed-vs-random  {'INFORMED WINS' if gap > 0.02 else 'ties/loses random'}")
        print(f"{beta:>5.2f} | {A['ceiling']:>7.3f} {A['random']:>7.3f} {A['uncertainty']:>7.3f} {A['greedy']:>7.3f} | {verdict}")
    print("\n(pre-registered: high beta (discoverable) -> informed beats random; low beta -> collapse. A clean"
          "\n phase transition validates the sim as a FAIR testbed -> proceed to the RL/learned-policy kill-test.)")

if __name__ == "__main__":
    main()
