#!/usr/bin/env python3
"""
CausalAtlas synthesis figures. Saves PNG + PDF to results/figures/.
Numbers are the locked per-substrate results for Replogle, Norman, and Tahoe.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
os.makedirs(OUT, exist_ok=True)
MODELS = ["Haiku 4.5", "Sonnet 4.6", "Opus 4.8"]
C = {"data": "#2c7fb8", "lit": "#d95f0e", "chance": "#999999",
     "haiku": "#bdc9e1", "sonnet": "#74a9cf", "opus": "#0570b0"}

# ---- Fig 1: data-content grounding (unaided data-reading vs recall vs chance), per substrate ----
# data-reading = raw_no_rule accuracy averaged over the 3 models; literature/baseline = chance (~33%).
subs = ["Phase 1\nReplogle genetic\n(marginal, sceptre)",
        "Phase 2a\nNorman GI\n(epistasis, sceptre)",
        "Phase 2b\nTahoe drug\n(cross-context, DESeq2)"]
data_read = [0.63, 0.65, 0.82]     # raw_no_rule mean acc (P1 62-64%; Norman 63/65/67; Tahoe 82/78/86)
literature = [0.33, 0.33, 0.33]
baseline = [0.33, 0.33, 0.33]
x = np.arange(len(subs)); w = 0.26
fig, ax = plt.subplots(figsize=(8.2, 4.6))
ax.bar(x - w, data_read, w, label="reads the data (raw, no rule)", color=C["data"])
ax.bar(x, literature, w, label="literature / recall only", color=C["lit"])
ax.bar(x + w, baseline, w, label="no data (chance)", color=C["chance"])
ax.axhline(1/3, ls="--", lw=0.8, color="#555", zorder=0)
ax.set_xticks(x); ax.set_xticklabels(subs, fontsize=8.5)
ax.set_ylabel("3-way accuracy"); ax.set_ylim(0, 1)
ax.set_title("Data-content grounding generalizes across substrate\n(reading effect-size data beats recall on every substrate)", fontsize=11)
for i, v in enumerate(data_read):
    ax.text(i - w, v + 0.02, f"{v:.0%}", ha="center", fontsize=9, fontweight="bold")
ax.legend(fontsize=8.5, loc="upper left"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_grounding.png", dpi=200); fig.savefig(f"{OUT}/fig1_grounding.pdf")

# ---- Fig 2: the GI additive-vs-untested discrimination COLLAPSES to chance when power-matched on SE ----
# raw_no_rule, unconditional, Wilson 95% CI. As-built items are ~SE-separable (ceiling 95.5%); power-matched
# items force residual-reading. The "ladder" is an SE-thresholding artifact.
asbuilt = {"acc": [0.47, 0.67, 0.61], "lo": [0.41, 0.60, 0.55], "hi": [0.54, 0.73, 0.68]}
pmatch = {"acc": [0.37, 0.55, 0.41], "lo": [0.31, 0.48, 0.35], "hi": [0.44, 0.62, 0.48]}
fig, ax = plt.subplots(figsize=(8.2, 4.9))
xm = np.arange(3)
for d, lab, c, o in [(asbuilt, "as-built items (SE-separable, ceiling 95.5%)", "#74a9cf", -0.07),
                     (pmatch, "power-matched on SE (forces residual-reading)", "#d7301f", 0.07)]:
    yerr = [[d["acc"][i] - d["lo"][i] for i in range(3)], [d["hi"][i] - d["acc"][i] for i in range(3)]]
    ax.errorbar(xm + o, d["acc"], yerr=yerr, fmt="o-", color=c, lw=2, ms=9, capsize=4, label=lab)
ax.axhline(0.5, ls="--", lw=1.0, color="#333"); ax.text(2.08, 0.5, "chance", fontsize=8.5, color="#333", va="center")
ax.set_xticks(xm); ax.set_xticklabels(MODELS); ax.set_ylim(0.25, 0.8)
ax.set_ylabel("GI additive-vs-untested accuracy (raw, no rule)")
ax.set_title("The hard discrimination is SE-thresholding, not residual-reading\n"
             "power-matching on SE collapses every model to chance (Norman GI, single draw + 95% CI)",
             fontsize=10.5)
ax.legend(fontsize=8.5, loc="upper right"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_hard_discrimination.png", dpi=200); fig.savefig(f"{OUT}/fig2_hard_discrimination.pdf")

# ---- Fig 3: rule-following premium (raw WITH rule vs raw NO rule), per model, Tahoe + Norman ----
# shows the explicit rule mostly helps execute; the unaided gap is the genuine reading.
fig, ax = plt.subplots(figsize=(7.6, 4.4))
arms = ["raw\n(+rule)", "raw\n(no rule)", "literature"]
norman = {"Haiku 4.5": [0.90, 0.63, 0.33], "Sonnet 4.6": [0.97, 0.65, 0.33], "Opus 4.8": [0.97, 0.67, 0.33]}
xa = np.arange(len(arms)); w2 = 0.26
for i, (m, ys) in enumerate(norman.items()):
    ax.bar(xa + (i - 1) * w2, ys, w2, label=m, color=[C["haiku"], C["sonnet"], C["opus"]][i])
ax.set_xticks(xa); ax.set_xticklabels(arms); ax.set_ylim(0, 1)
ax.set_ylabel("3-way accuracy (Norman GI)")
ax.set_title("Rule-following premium vs genuine reading (Norman GI)\nrule lifts all to ~ceiling; unaided reading is the real signal", fontsize=10.5)
ax.legend(fontsize=8.5); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_rule_premium.png", dpi=200); fig.savefig(f"{OUT}/fig3_rule_premium.pdf")

print("wrote figures to", os.path.abspath(OUT))
for f in sorted(os.listdir(OUT)):
    print("  ", f)
