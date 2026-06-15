#!/usr/bin/env python3
"""Experiment-selection loop -- P0: offline active-learning machinery + PRE-K1 + K1 gate (no LLM/RL).

The loop is active learning in PERTURBATION-FEATURE space: a world model M predicts an unrevealed perturbation's
effect profile from gene features (ESM2/GO), trained (lazily, kNN) on revealed (perturbation, effect) pairs.
Selection = which perturbation to reveal next; revealing improves M; better M -> better uncertainty -> better
selection (the compounding self-improvement cycle).

Two gates IN ORDER (project discipline; the first is the real make-or-break and may be a NEGATIVE):
  PRE-K1: does M improve at all as data is revealed under RANDOM selection? Flat => feature->effect not
          predictive (the perturbation-FM-negative) => experiment-selection is moot here. Report and stop.
  K1:     does an INFORMED selector (uncertainty / k-center) reach a target faster than random? AULC gap with a
          test-resample bootstrap CI, normalized by the oracle (fraction of achievable headroom captured).
Honesty probes: uncertainty CALIBRATION corr(u, error); (gaming probe left for real data).

  smoke:  python move1_expsel.py --synthetic structured   (expect: M learns, smart > random, calib>0)
          python move1_expsel.py --synthetic null         (expect: M flat, K1 fails -- loop is honest)
  real:   python move1_expsel.py --effects E.npy --features F.npy --names names.txt
"""
import argparse, sys, numpy as np
from numpy.linalg import norm
from sklearn.metrics import roc_auc_score
def log(*a): print("[expsel]", *a, file=sys.stderr, flush=True)
EPS = 1e-8
SEL_SEED = {"random": 11, "uncertainty": 22, "kcenter": 33, "oracle": 44}

class KNNWorld:
    def __init__(self, feats, effects, k=10):
        self.E = effects; self.k = k
        Fz = (feats - feats.mean(0)) / (feats.std(0) + EPS)
        n = Fz.shape[0]; self.D = np.empty((n, n), np.float32)
        for i in range(0, n, 512):
            self.D[i:i+512] = np.sqrt(np.maximum(((Fz[i:i+512, None, :] - Fz[None, :, :]) ** 2).sum(-1), 0))
    def predict(self, revealed, query):
        revealed = np.asarray(revealed); query = np.asarray(query)
        Dq = self.D[np.ix_(query, revealed)]; kk = min(self.k, len(revealed))
        preds = np.zeros((len(query), self.E.shape[1])); unc = np.zeros(len(query))
        for i in range(len(query)):
            order = np.argsort(Dq[i])[:kk]; d = Dq[i, order]
            w = 1.0 / (d + EPS); w /= w.sum()
            preds[i] = (w[:, None] * self.E[revealed[order]]).sum(0); unc[i] = d.mean()
        return preds, unc

def cos_vec(P, T):                                          # per-row cosine between pred P and true T
    return (P * T).sum(1) / (norm(P, axis=1) * norm(T, axis=1) + EPS)

def score_perpert(world, revealed, test, delta=0.25):      # returns per-pert cosine array + mean AUROC
    pred, _ = world.predict(revealed, test); true = world.E[test]
    c = cos_vec(pred, true)
    aucs = []
    for i in range(len(test)):
        y = (np.abs(true[i]) >= delta).astype(int)
        if 0 < y.sum() < len(y): aucs.append(roc_auc_score(y, np.abs(pred[i])))
    return c, (float(np.mean(aucs)) if aucs else float("nan"))

# ---------------- ceiling gate (decisive PRE-K1): can features predict held-out effects vs a permutation null? ----------------
def predict_M(model, tr_feat, tr_eff, te_feat, k=10):
    if model == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=10.0).fit(tr_feat, tr_eff).predict(te_feat)
    mu, sd = tr_feat.mean(0), tr_feat.std(0) + EPS
    A = (tr_feat - mu) / sd; B = (te_feat - mu) / sd
    D = np.sqrt(np.maximum(((B[:, None, :] - A[None, :, :]) ** 2).sum(-1), 0))
    pred = np.zeros((B.shape[0], tr_eff.shape[1]))
    for i in range(B.shape[0]):
        o = np.argsort(D[i])[:k]; w = 1.0 / (D[i, o] + EPS); w /= w.sum(); pred[i] = (w[:, None] * tr_eff[o]).sum(0)
    return pred

def score_pred(pred, true, delta=0.25):
    c = float(np.mean(cos_vec(pred, true))); aucs = []
    for i in range(len(true)):
        y = (np.abs(true[i]) >= delta).astype(int)
        if 0 < y.sum() < len(y): aucs.append(roc_auc_score(y, np.abs(pred[i])))
    return c, (float(np.mean(aucs)) if aucs else float("nan"))

def ceiling(eff, feat, model, n_perm=10, test_frac=0.25, seed=0, k=10):
    rng = np.random.default_rng(seed); P = eff.shape[0]
    idx = rng.permutation(P); nte = int(test_frac * P); te, tr = idx[:nte], idx[nte:]
    rc, ra = score_pred(predict_M(model, feat[tr], eff[tr], feat[te], k), eff[te])
    ncs, nas = [], []
    for p in range(n_perm):
        fp = feat[np.random.default_rng(100 + p).permutation(P)]      # break gene<->feature correspondence
        nc, na = score_pred(predict_M(model, fp[tr], eff[tr], fp[te], k), eff[te]); ncs.append(nc); nas.append(na)
    return rc, ra, (np.mean(ncs), np.std(ncs) + EPS), (np.nanmean(nas), np.nanstd(nas) + EPS)

def select(name, world, revealed, pool, rng, oracle_test=None, oracle_cand=20):
    pool = np.asarray(pool)
    if name == "random": return int(rng.choice(pool))
    if name == "uncertainty": _, u = world.predict(revealed, pool); return int(pool[np.argmax(u)])
    if name == "kcenter":
        Dpr = world.D[np.ix_(pool, np.asarray(revealed))]; return int(pool[np.argmax(Dpr.min(1))])
    if name == "oracle":
        cand = rng.choice(pool, size=min(len(pool), oracle_cand), replace=False); best, bi = -2.0, int(cand[0])
        for c in cand:
            s, _ = score_perpert(world, list(revealed) + [int(c)], oracle_test); m = float(s.mean())
            if m > best: best, bi = m, int(c)
        return bi
    raise ValueError(name)

def run_selector(world, test, name, budget, seed_set, score_every=3, oracle_cand=20):
    rng = np.random.default_rng(SEL_SEED[name])
    revealed = list(seed_set); pool = list(np.setdiff1d(np.setdiff1d(np.arange(world.E.shape[0]), test), revealed))
    xs, C, AU = [], [], []
    step = 0
    while pool and step < budget:
        a = select(name, world, revealed, pool, rng, oracle_test=test if name == "oracle" else None, oracle_cand=oracle_cand)
        revealed.append(a); pool.remove(a); step += 1
        if step % score_every == 0 or step == budget or not pool:
            c, au = score_perpert(world, revealed, test); xs.append(len(revealed)); C.append(c); AU.append(au)
    return np.array(xs), np.array(C), np.array(AU)         # C: (steps x |test|) per-pert cosines

def aulc_from(xs, per_pert_mat, sub):                       # AULC of mean-over-subset cosine curve
    return float(np.trapz(per_pert_mat[:, sub].mean(1), xs) / (xs[-1] - xs[0] + EPS))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", choices=["structured", "null"])
    ap.add_argument("--effects"); ap.add_argument("--features"); ap.add_argument("--names")
    ap.add_argument("--k", type=int, default=10); ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--test-frac", type=float, default=0.25); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selectors", default="random,uncertainty,kcenter,oracle")
    ap.add_argument("--oracle-cand", type=int, default=20)
    ap.add_argument("--ceiling", action="store_true", help="run only the decisive ceiling gate (features vs permutation null)")
    ap.add_argument("--model", default="knn", choices=["knn", "ridge"])
    ap.add_argument("--n-perm", type=int, default=10)
    a = ap.parse_args()
    if a.synthetic:
        rng0 = np.random.default_rng(a.seed); P, F, G = 400, 32, 60
        feats = rng0.standard_normal((P, F))
        if a.synthetic == "structured":
            W = rng0.standard_normal((F, G)); eff = np.tanh(feats @ W) + 0.3 * rng0.standard_normal((P, G))
        else:
            eff = rng0.standard_normal((P, G))
        names = [f"p{i}" for i in range(P)]; log(f"synthetic {a.synthetic}: effects {eff.shape}")
    else:
        eff = np.load(a.effects); feats = np.load(a.features); names = [l.strip() for l in open(a.names)]
        log(f"real: effects {eff.shape}, features {feats.shape}")
    if a.ceiling:
        rc, ra, (nc, ncs), (na, nas) = ceiling(eff, feats, a.model, n_perm=a.n_perm, test_frac=a.test_frac, seed=a.seed, k=a.k)
        log(f"\nCEILING gate  model={a.model}  features={a.features or a.synthetic}")
        log(f"  cosine: real {rc:.3f} vs null {nc:.3f}+-{ncs:.3f}  z={(rc-nc)/ncs:+.1f}")
        log(f"  AUROC : real {ra:.3f} vs null {na:.3f}+-{nas:.3f}  z={(ra-na)/nas:+.1f}")
        gate = ((rc - nc) / ncs > 2) and (ra > 0.55)
        log(f"  GATE {'PASS: features predict held-out effects (selection has a substrate)' if gate else 'FAIL: ~= null -> experiment-selection MOOT (honest negative)'}")
        return
    rng = np.random.default_rng(a.seed)
    test = rng.choice(eff.shape[0], size=int(a.test_frac * eff.shape[0]), replace=False)
    world = KNNWorld(feats, eff, k=a.k)
    budget = min(a.budget, eff.shape[0] - len(test) - 5)
    pool_all = np.setdiff1d(np.arange(eff.shape[0]), test)
    seed_set = list(np.random.default_rng(99 + a.seed).choice(pool_all, size=5, replace=False))

    sels = [s.strip() for s in a.selectors.split(",")]
    curves = {n: run_selector(world, test, n, budget, seed_set, oracle_cand=a.oracle_cand) for n in sels}

    # PRE-K1 (the real make-or-break): does the ALL-DATA ceiling clear an absolute bar? (else features can't predict)
    ceil_c, _ = score_perpert(world, list(pool_all), test); floor_c, _ = score_perpert(world, list(seed_set), test)
    ceil_c, floor_c = float(ceil_c.mean()), float(floor_c.mean())
    rand_end = float(curves["random"][1][-1].mean())
    ceiling_ok = ceil_c > 0.10                              # all-data M predicts held-out perts non-trivially
    gap = ceil_c - floor_c
    rise = (rand_end - floor_c) / gap if gap > 0.02 else float("nan")
    learns = ceiling_ok and (rise == rise) and rise > 0.2
    log(f"\nPRE-K1 (random-selection learning, profile cosine):")
    log(f"  seed-floor {floor_c:.3f} -> random@budget {rand_end:.3f} -> all-data ceiling {ceil_c:.3f}")
    if not ceiling_ok:
        log(f"  all-data ceiling {ceil_c:.3f} <= 0.10 => features do NOT predict effects => M CANNOT LEARN")
        log(f"  => experiment-selection is MOOT on this substrate (honest negative)")
    else:
        log(f"  M captured {rise:.0%} of (ceiling-floor) => {'PASS: M learns' if learns else 'weak learning'}")
    mid = list(seed_set) + list(np.random.default_rng(7).choice([p for p in pool_all if p not in seed_set], size=min(budget//2, len(pool_all)-len(seed_set)), replace=False))
    pred, u = world.predict(mid, test); err = 1 - cos_vec(pred, world.E[test])
    calib = float(np.corrcoef(u, err)[0, 1]) if len(test) > 2 else float("nan")
    log(f"  uncertainty calibration corr(u, error) = {calib:+.2f}  (>0 = u tracks error)")

    # K1: informed vs random (AULC on profile cosine), test-resample bootstrap; normalize by oracle if present
    full = np.arange(len(test))
    A = {n: aulc_from(curves[n][0], curves[n][1], full) for n in curves}
    log(f"\nK1 (AULC cosine): " + " ".join(f"{n}={A[n]:.3f}" for n in sels))
    head = (A["oracle"] - A["random"]) if "oracle" in A else None
    for n in [s for s in ["uncertainty", "kcenter"] if s in A]:
        frac = f" = {(A[n]-A['random'])/(head+EPS):+.0%} of oracle headroom" if head else ""
        log(f"  {n}: gap vs random {A[n]-A['random']:+.3f}{frac}")
    lo = hi = float("nan")
    if "uncertainty" in A:
        brng = np.random.default_rng(123); boot = []
        xu, Cu, _ = curves["uncertainty"]; xr, Cr, _ = curves["random"]
        for _ in range(1000):
            sub = brng.choice(full, size=len(full), replace=True)
            boot.append(aulc_from(xu, Cu, sub) - aulc_from(xr, Cr, sub))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        log(f"  uncertainty-random AULC gap 95% CI [{lo:+.3f},{hi:+.3f}] -> K1 {'PASS (CI excl 0)' if lo > 0 else 'fail'}")
    log(f"\nVERDICT: PRE-K1 {'PASS (M learns)' if learns else 'FLAT/weak (selection moot)'} | calib {calib:+.2f} | "
        f"K1 {'PASS' if (lo==lo and lo>0) else 'fail/NA'}")

if __name__ == "__main__":
    main()
