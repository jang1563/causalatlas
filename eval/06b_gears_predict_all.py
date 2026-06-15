#!/usr/bin/env python3
"""
CausalAtlas Phase 3 - Step 06b: expand the GEARS reliability substrate (predict-ONLY, no retrain).

The combo_seen0 test split gave only ~11 decidable unseen-combo edges (linchpin too thin). Here we load the
SAVED GEARS model and predict ALL Norman perturbations (singles + combos), tag each by regime relative to the
TRAINING set (in_train = reliable; unseen single; combo_seen0/1/2), and join our held-out three-state labels.
This yields a much larger per-edge "GEARS correct vs wrong" pool for the calibrated-orchestration eval, with a
proper reliability gradient.

Run in the GEARS env on a GPU node (predict is fast):
  PYTHONNOUSERSITE=1 ~/envs/gears/bin/python 06b_gears_predict_all.py \
      --model ~/CausalAtlas/results/gears_norman/gears_model \
      --data-dir ~/CausalAtlas/gears_data \
      --real-labels ~/CausalAtlas/results/norman_gi_full/labeled_marginal.csv \
      --outdir ~/CausalAtlas/results/gears_norman --delta 0.25
"""
import argparse, os, sys
import numpy as np, pandas as pd
def log(*a): print("[gears06b]", *a, file=sys.stderr, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--real-labels", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--delta", type=float, default=0.25)
    a = ap.parse_args()
    import torch
    from gears import PertData, GEARS
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    pert_data = PertData(a.data_dir); pert_data.load(data_name="norman")
    pert_data.prepare_split(split="combo_seen0", seed=1)
    pert_data.get_dataloader(batch_size=32, test_batch_size=128)
    model = GEARS(pert_data, device=dev)
    model.load_pretrained(a.model)
    log("loaded saved GEARS model on", dev)

    adata = pert_data.adata
    genes = list(adata.var["gene_name"]) if "gene_name" in adata.var else list(adata.var_names)
    g2i = {g: i for i, g in enumerate(genes)}
    ctrl_mean = np.asarray(adata[adata.obs.condition == "ctrl"].X.mean(axis=0)).ravel()

    # training singles for regime tagging
    set2c = pert_data.set2conditions
    def parts_of(c): return [x for x in c.replace("+ctrl", "").split("+") if x and x != "ctrl"]
    train_conds = set(set2c.get("train", []))
    train_singles = {parts_of(c)[0] for c in train_conds if len(parts_of(c)) == 1}
    def regime(cond):
        p = parts_of(cond)
        if len(p) == 1:
            return "single_seen" if p[0] in train_singles else "single_unseen"
        if cond in train_conds:
            return "combo_in_train"
        return f"combo_seen{sum(1 for g in p if g in train_singles)}"

    # predict EVERY non-control condition present in the data
    all_conds = [c for c in adata.obs.condition.unique() if c != "ctrl"]
    log(f"predicting {len(all_conds)} conditions ...")
    rows, EPS = [], 1e-2
    for n, cond in enumerate(all_conds):
        p = parts_of(cond)
        if not p:
            continue
        try:
            pv = np.asarray(list(model.predict([p]).values())[0]).ravel()
        except Exception:
            continue
        lfc = np.log2((pv + EPS) / (ctrl_mean + EPS)); reg = regime(cond); combo = "+".join(p)
        for b, i in g2i.items():
            if i < len(lfc):
                rows.append((combo, b, float(lfc[i]), reg))
        if (n + 1) % 50 == 0:
            log(f"  {n+1}/{len(all_conds)}")
    pred = pd.DataFrame(rows, columns=["perturbation", "gene", "gears_log2FC", "regime"])
    pred["gears_call"] = np.where(pred.gears_log2FC.abs() >= a.delta, "effect", "no_effect")

    real = pd.read_csv(a.real_labels)
    real = real[np.isclose(real["delta"], a.delta)][["target", "response_id", "label"]].rename(
        columns={"target": "perturbation", "response_id": "gene", "label": "real_label"})
    out = pred.merge(real, on=["perturbation", "gene"], how="inner")
    dec = out[out.real_label.isin(["POSITIVE", "TESTED_NEGATIVE"])].copy()
    dec["real_call"] = np.where(dec.real_label == "POSITIVE", "effect", "no_effect")
    dec["gears_correct"] = dec.gears_call == dec.real_call
    dec.to_csv(os.path.join(a.outdir, "gears_vs_real_all.csv"), index=False)
    log("expanded GEARS reliability by regime:")
    print(dec.groupby("regime")["gears_correct"].agg(["mean", "size"]).to_string(), file=sys.stderr)
    log(f"done -> {a.outdir}/gears_vs_real_all.csv ({len(dec)} decidable edges)")

if __name__ == "__main__":
    main()
