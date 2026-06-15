#!/usr/bin/env python3
"""
CausalAtlas Phase 3 - Step 06: GEARS perturbation predictions on Norman 2019, with seen/unseen REGIMES.

Produces the SFM-prediction substrate for the calibrated-orchestration eval: per (perturbation P, gene B), the
GEARS predicted effect + an effect/no-effect call, the held-out REAL three-state label (our sceptre ground
truth), and the RELIABILITY REGIME (GEARS combo_seen0/1/2 = neither/one/both singles seen in training; single).
GEARS is the canonical fallible perturbation SFM: it degrades on unseen combos (combo_seen0), which is exactly
the regime where the orchestrating LLM should HEDGE.

GEARS predicts a post-perturbation mean EXPRESSION profile; we derive predicted log2FC vs the predicted control
and threshold at delta to get GEARS's effect call, comparable to our three-state ground truth.

Run on a GPU node (scu-gpu). Usage:
  module load anaconda3/2023.09-3
  python 06_gears_norman.py --outdir ~/CausalAtlas/results/gears_norman \
      --real-labels ~/CausalAtlas/results/norman_gi_full/labeled_marginal.csv --epochs 20 --delta 0.25
"""
import argparse, os, sys
import numpy as np
import pandas as pd

def log(*a): print("[gears]", *a, file=sys.stderr, flush=True)

DELTA = 0.25

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--data-dir", default=os.path.expanduser("~/CausalAtlas/gears_data"))
    ap.add_argument("--real-labels", required=True, help="our labeled_marginal.csv (target,response_id,delta,label)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--delta", type=float, default=0.25)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    import torch
    from gears import PertData, GEARS
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log("device:", dev)

    # --- load Norman + a combo OOD split (GEARS ships the Norman 2019 dataset) ---
    pert_data = PertData(a.data_dir)
    pert_data.load(data_name="norman")
    # combo_seen0: test combos where NEITHER gene was seen as a single in train (hardest / most unreliable)
    pert_data.prepare_split(split="combo_seen0", seed=a.seed)
    pert_data.get_dataloader(batch_size=32, test_batch_size=128)

    model = GEARS(pert_data, device=dev)
    model.model_initialize(hidden_size=a.hidden)
    log(f"training GEARS {a.epochs} epochs ...")
    model.train(epochs=a.epochs)
    model.save_model(os.path.join(a.outdir, "gears_model"))

    # --- gene space + control mean ---
    adata = pert_data.adata
    genes = list(adata.var["gene_name"]) if "gene_name" in adata.var else list(adata.var_names)
    g2i = {g: i for i, g in enumerate(genes)}
    ctrl = adata[adata.obs["condition"] == "ctrl"].X
    ctrl_mean = np.asarray(ctrl.mean(axis=0)).ravel()

    # --- regime per test condition from the split ---
    set2cond = pert_data.set2conditions  # {'train':[...], 'test':[...], 'val':[...]}
    test_conds = [c for c in set2cond.get("test", []) if c != "ctrl"]
    log(f"test conditions (combo_seen0): {len(test_conds)}")

    def regime(cond):
        genes_in = [x for x in cond.replace("+ctrl", "").split("+") if x and x != "ctrl"]
        train_singles = set()
        for c in set2cond.get("train", []):
            parts = [x for x in c.replace("+ctrl", "").split("+") if x and x != "ctrl"]
            if len(parts) == 1:
                train_singles.add(parts[0])
        if len(genes_in) == 1:
            return "single_unseen" if genes_in[0] not in train_singles else "single_seen"
        seen = sum(1 for g in genes_in if g in train_singles)
        return f"combo_seen{seen}"

    # --- predict each test condition -> predicted log2FC per gene ---
    pred_pairs = []
    EPS = 1e-2
    for cond in test_conds:
        parts = [x for x in cond.replace("+ctrl", "").split("+") if x and x != "ctrl"]
        try:
            pred = model.predict([parts])          # {cond_key: pred_expr_vector}
            pv = np.asarray(list(pred.values())[0]).ravel()
        except Exception as e:
            log(f"  predict fail {cond}: {str(e)[:60]}"); continue
        lfc = np.log2((pv + EPS) / (ctrl_mean + EPS))
        reg = regime(cond)
        combo = "+".join(parts)
        for b, i in g2i.items():
            if i < len(lfc):
                pred_pairs.append((combo, b, float(lfc[i]), reg))
    pred = pd.DataFrame(pred_pairs, columns=["perturbation", "gene", "gears_log2FC", "regime"])
    pred["gears_call"] = np.where(pred["gears_log2FC"].abs() >= a.delta, "effect", "no_effect")
    log(f"gears predictions: {len(pred)} (pert,gene) rows; regimes={pred.regime.value_counts().to_dict()}")

    # --- join our held-out REAL three-state label ---
    real = pd.read_csv(a.real_labels)
    real = real[np.isclose(real["delta"], a.delta)][["target", "response_id", "label"]]
    real = real.rename(columns={"target": "perturbation", "response_id": "gene", "label": "real_label"})
    out = pred.merge(real, on=["perturbation", "gene"], how="inner")
    # gears correctness vs the real label (real POSITIVE -> effect; TESTED_NEGATIVE -> no_effect; UNTESTED dropped)
    dec = out[out.real_label.isin(["POSITIVE", "TESTED_NEGATIVE"])].copy()
    dec["real_call"] = np.where(dec.real_label == "POSITIVE", "effect", "no_effect")
    dec["gears_correct"] = (dec.gears_call == dec.real_call)
    out.to_csv(os.path.join(a.outdir, "gears_predictions.csv"), index=False)
    dec.to_csv(os.path.join(a.outdir, "gears_vs_real.csv"), index=False)
    log("GEARS reliability by regime (accuracy vs held-out real call):")
    print(dec.groupby("regime")["gears_correct"].agg(["mean", "size"]).to_string(), file=sys.stderr)
    log(f"done -> {a.outdir}")

if __name__ == "__main__":
    main()
