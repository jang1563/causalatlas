#!/usr/bin/env python3
"""Move 1 powering -- leave-combo-out K-fold CV of GEARS on Norman.

Question being powered: does GEARS's non-additive (synergy) signal GENERALIZE to NOVEL combos? The single-split
result left the held-out synergy comparison underpowered (n=26 perturbations, GEARS-additive AUROC gap +.104
[-.03,+.22], NS). Here every combo becomes a held-out prediction:

Design: K folds over the ~62 two-gene combos. Each fold trains GEARS on (ALL singles + the other K-1/K combos)
and predicts the held-out 1/K combos. Because all singles are always in train, every held-out combo is the
**combo_seen2** regime (both constituent singles seen, only the COMBINATION novel) = the cleanest, fairest
additive-vs-FM interpolation regime (additive also uses the observed singles; the question is whether GEARS
adds interaction value on a novel pairing). Pooled across folds, ALL combos get a held-out prediction ->
maximizes the held-out synergy subset (Norman's ~131-combo ceiling notwithstanding).

Run on a GPU node in the gears env (data load/process is cached after the first GEARS run):
  PYTHONNOUSERSITE=1 ~/envs/gears/bin/python move1_gears_cv.py \
    --data-dir ~/CausalAtlas/gears_data \
    --real-labels ~/CausalAtlas/results/norman_gi_full/labeled_marginal.csv \
    --outdir ~/CausalAtlas/results/gears_cv --k 5 --epochs 20 --delta 0.25
"""
import argparse, os, sys, pickle, numpy as np, pandas as pd
def log(*a): print("[cv]", *a, file=sys.stderr, flush=True)

def parts_of(c): return [x for x in c.replace("+ctrl", "").split("+") if x and x != "ctrl"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--real-labels", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--delta", type=float, default=0.25)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    split_dir = os.path.join(a.outdir, "splits"); os.makedirs(split_dir, exist_ok=True)
    import torch
    from gears import PertData, GEARS
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log("device:", dev)

    pert_data = PertData(a.data_dir)
    pert_data.load(data_name="norman")
    adata = pert_data.adata
    conds = [c for c in adata.obs["condition"].unique() if c != "ctrl"]
    combos  = [c for c in conds if len(parts_of(c)) == 2]
    singles = [c for c in conds if len(parts_of(c)) == 1]
    log(f"{len(combos)} combos, {len(singles)} singles, {len(conds)} non-ctrl conditions")

    genes = list(adata.var["gene_name"]) if "gene_name" in adata.var else list(adata.var_names)
    g2i = {g: i for i, g in enumerate(genes)}
    ctrl_mean = np.asarray(adata[adata.obs["condition"] == "ctrl"].X.mean(axis=0)).ravel()

    rng = np.random.default_rng(a.seed)
    order = list(combos); rng.shuffle(order)
    folds = [list(f) for f in np.array_split(np.array(order, dtype=object), a.k)]
    EPS = 1e-2
    allrows = []
    for fi, testc in enumerate(folds):
        testset = set(testc)
        trainc = singles + [c for c in combos if c not in testset]
        nval = max(3, len(trainc) // 10)                              # ~10% held for GEARS early-stop/val
        val = list(np.random.default_rng(1000 + fi).choice(np.array(trainc, dtype=object), size=nval, replace=False))
        valset = set(val)
        train_final = [c for c in trainc if c not in valset]
        split_dict = {"train": train_final + ["ctrl"], "val": val, "test": testc}
        sp = os.path.join(split_dir, f"fold{fi}.pkl")
        with open(sp, "wb") as f: pickle.dump(split_dict, f)
        log(f"--- fold {fi+1}/{a.k}: train {len(train_final)} / val {len(val)} / test(combo) {len(testc)} ---")
        try:
            pert_data.prepare_split(split="custom", split_dict_path=sp)
            pert_data.get_dataloader(batch_size=32, test_batch_size=128)
            model = GEARS(pert_data, device=dev)
            model.model_initialize(hidden_size=a.hidden)
            model.train(epochs=a.epochs)
        except Exception as e:
            log(f"  fold {fi} TRAIN FAIL: {str(e)[:160]}"); continue
        ok = 0
        for cond in testc:
            p = parts_of(cond)
            try:
                pv = np.asarray(list(model.predict([p]).values())[0]).ravel()
            except Exception as e:
                log(f"   predict fail {cond}: {str(e)[:80]}"); continue
            lfc = np.log2((pv + EPS) / (ctrl_mean + EPS))
            combo = "+".join(p)
            for b, i in g2i.items():
                if i < len(lfc):
                    allrows.append((combo, b, float(lfc[i]), fi))
            ok += 1
        log(f"  fold {fi+1} predicted {ok}/{len(testc)} held-out combos")
        # incremental save
        pd.DataFrame(allrows, columns=["perturbation","gene","gears_log2FC","fold"]).to_csv(
            os.path.join(a.outdir, "gears_cv_heldout.csv"), index=False)

    pred = pd.DataFrame(allrows, columns=["perturbation","gene","gears_log2FC","fold"])
    pred["regime"] = "combo_seen2_cv"
    pred["gears_call"] = np.where(pred["gears_log2FC"].abs() >= a.delta, "effect", "no_effect")
    real = pd.read_csv(a.real_labels)
    real = real[np.isclose(real["delta"], a.delta)][["target","response_id","label"]].rename(
        columns={"target":"perturbation","response_id":"gene","label":"real_label"})
    out = pred.merge(real, on=["perturbation","gene"], how="inner")
    dec = out[out.real_label.isin(["POSITIVE","TESTED_NEGATIVE"])].copy()
    dec["real_call"] = np.where(dec.real_label == "POSITIVE", "effect", "no_effect")
    dec["gears_correct"] = dec.gears_call == dec.real_call
    dec.to_csv(os.path.join(a.outdir, "gears_cv_vs_real.csv"), index=False)
    log(f"DONE: {dec['perturbation'].nunique()} held-out combos, {len(dec)} decidable edges")
    log(f"  GEARS held-out combo_seen2 accuracy: {dec['gears_correct'].mean():.3f}")
    log(f"  -> {a.outdir}/gears_cv_vs_real.csv  (recompute synergy AUROC gap vs additive locally)")

if __name__ == "__main__":
    main()
