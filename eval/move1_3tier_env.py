#!/usr/bin/env python3
"""LLM 3-tier grounded-orchestrator arm.

Tests the most basic grounded-orchestrator competence: given a FREE additive baseline AND a FREE foundation-model
(GEARS) prediction per gene, plus a COSTLY real assay, does the LLM DISCOVER that it should default to the cheap
baseline (additive 0.825 > trust-FM 0.718) rather than trusting the foundation model? (Hypothesis from Layer B:
it does not -- it over-trusts the FM / over-verifies.)

Per gene the agent sees BOTH free predictions (neutrally, not told which is better) and may run_de (cost lambda,
returns truth). It submits effect/no_effect per gene. Reward = #correct - lambda*#assays.

KEY DIAGNOSTIC: on non-verified genes where the two predictions DISAGREE, what fraction does the agent's call
match the ADDITIVE baseline vs the FM? additive-following > 0.5 = discovered the cheap-default; <= 0.5 = over-trusts FM.

  build+run:  python move1_3tier_env.py --model claude-haiku-4-5-20251001 --limit 40 --lam 0.5 --out runs/3tier_haiku.jsonl
  grade:      python move1_3tier_env.py --grade --panels runs/3tier_panels.jsonl runs/3tier_*.jsonl
"""
import argparse, json, os, sys, re, time
from pathlib import Path
import numpy as np, pandas as pd

DELTA = 0.25
PANELS_DEFAULT = "runs/3tier_panels.jsonl"

# ---------------- panel construction ----------------
def build_panels(cv_path, marginal_path, out, panel_size=20, min_disagree=6, seed=0):
    cv = pd.read_csv(cv_path).dropna(subset=["gears_log2FC"])
    m = pd.read_csv(marginal_path); m = m[np.isclose(m["delta"], DELTA)]
    single = {(r.target, r.response_id): r.log2FC for r in m[m.kind == "single"].itertuples()}
    def addlfc(p, gn):
        ps = re.split(r"[+_]", p); v = [single.get((x, gn)) for x in ps]
        return float(np.sum(v)) if len(ps) > 1 and all(x is not None for x in v) else np.nan
    cv["add_lfc"] = [addlfc(p, gn) for p, gn in zip(cv.perturbation, cv.gene)]
    cv = cv.dropna(subset=["add_lfc"]).copy()
    call = lambda x: "effect" if abs(x) >= DELTA else "no_effect"
    rng = np.random.default_rng(seed); panels = []
    for pert, g in cv.groupby("perturbation"):
        g = g.copy()
        g["fm_call"] = g.gears_log2FC.map(call); g["add_call"] = g.add_lfc.map(call)
        g["disagree"] = g.fm_call != g.add_call
        dis = g[g.disagree]; agr = g[~g.disagree]
        take = pd.concat([dis.sample(min(len(dis), max(min_disagree, panel_size//2)), random_state=seed),
                          agr.sample(min(len(agr), panel_size), random_state=seed)]).drop_duplicates("gene").head(panel_size)
        if len(take) < 8 or take.disagree.sum() < 3:
            continue
        genes = [{"gene": r.gene, "fm_log2FC": float(r.gears_log2FC), "fm_call": r.fm_call,
                  "add_log2FC": float(r.add_lfc), "add_call": r.add_call, "real_call": r.real_call}
                 for r in take.itertuples()]
        panels.append({"perturbation": pert, "genes": genes})
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        for p in panels: f.write(json.dumps(p) + "\n")
    print(f"[panels] {len(panels)} combo panels -> {out} (median {int(np.median([len(p['genes']) for p in panels]))} genes, "
          f"{int(np.median([sum(g['fm_call']!=g['add_call'] for g in p['genes']) for p in panels]))} disagree)", file=sys.stderr)
    return panels

# ---------------- agent ----------------
SYSTEM = """You are a computational biologist deciding, for each gene in a panel, whether a genetic perturbation
changes its expression (effect) or not (no_effect). For EACH gene you are given two FREE predictions that may be
wrong: an ADDITIVE baseline (the sum of the two single-gene perturbation effects) and a FOUNDATION MODEL (GEARS).
You may also run a real differential-expression assay on specific genes to get the TRUTH, but it COSTS {LAM} per
gene. Scoring: +1 per correct final call, -{LAM} per gene assayed. Maximize NET. Decide per gene whether to rely
on one of the free predictions or to pay for the assay, then submit a call for every gene. Return tool calls only."""

def tools_spec():
    return [
        {"name": "run_de", "description": "Run the real DE assay on one or more genes; returns the true call. COSTS lambda per gene.",
         "input_schema": {"type": "object", "properties": {"genes": {"type": "array", "items": {"type": "string"}}}, "required": ["genes"]}},
        {"name": "submit", "description": "Submit final effect/no_effect call for ALL panel genes.",
         "input_schema": {"type": "object", "properties": {"calls": {"type": "array", "items": {"type": "object",
            "properties": {"gene": {"type": "string"}, "call": {"type": "string", "enum": ["effect", "no_effect"]}},
            "required": ["gene", "call"]}}}, "required": ["calls"]}},
    ]

def run_episode(client, model, panel, lam, max_turns=10):
    import anthropic  # noqa
    genes = {g["gene"]: g for g in panel["genes"]}
    lines = [f"Perturbation: {panel['perturbation']} (genetic, K562). Panel of {len(genes)} genes; for each: "
             f"ADDITIVE baseline and FOUNDATION MODEL predictions."]
    for g in panel["genes"]:
        lines.append(f"  {g['gene']}: additive={g['add_log2FC']:+.2f} ({g['add_call']}) | "
                     f"FM={g['fm_log2FC']:+.2f} ({g['fm_call']})")
    user = "\n".join(lines) + f"\n\nAssay (cost {lam}/gene) only where worth it, then submit a call for all {len(genes)} genes."
    msgs = [{"role": "user", "content": user}]
    verified = set(); submitted = None; nt = 0
    for _ in range(max_turns):
        nt += 1
        try:
            r = client.messages.create(model=model, max_tokens=3000, system=SYSTEM.replace("{LAM}", str(lam)),
                                       messages=msgs, tools=tools_spec())
        except Exception as e:
            time.sleep(3)
            try:
                r = client.messages.create(model=model, max_tokens=3000, system=SYSTEM.replace("{LAM}", str(lam)),
                                           messages=msgs, tools=tools_spec())
            except Exception as e2:
                return {"error": str(e2)[:150], "perturbation": panel["perturbation"]}
        msgs.append({"role": "assistant", "content": r.content})
        tube = [b for b in r.content if b.type == "tool_use"]
        if not tube:
            msgs.append({"role": "user", "content": "Continue: assay if useful, then submit all genes."}); continue
        results = []
        for b in tube:
            if b.name == "run_de":
                out = []
                for gn in b.input.get("genes", []):
                    if gn in genes: verified.add(gn); out.append(f"{gn}: {genes[gn]['real_call']}")
                    else: out.append(f"{gn}: NOT in panel")
                results.append((b.id, "\n".join(out) or "no genes"))
            elif b.name == "submit":
                submitted = b.input.get("calls", []); results.append((b.id, "submitted"))
        msgs.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": i, "content": c} for i, c in results]})
        if submitted is not None: break
    calls = {c.get("gene"): c.get("call") for c in (submitted or []) if c.get("gene") in genes}
    return {"model": model, "perturbation": panel["perturbation"], "n": len(genes),
            "verified": sorted(verified), "calls": calls}

# ---------------- grading ----------------
def grade(panel_path, episode_paths):
    panels = {p["perturbation"]: p for p in (json.loads(l) for l in open(panel_path) if l.strip())}
    print(f"\n{'model':18s} {'acc':>6s} {'freeAcc':>7s} {'assay%':>7s} {'net@.5':>7s} | {'ADD-follow':>10s} {'FM-follow':>9s}  (free=non-verified genes)")
    # baselines (constant, no assay) over the same panels
    allg = [g for p in panels.values() for g in p["genes"]]
    addacc = np.mean([g["add_call"] == g["real_call"] for g in allg])
    fmacc  = np.mean([g["fm_call"] == g["real_call"] for g in allg])
    orac   = np.mean([(g["add_call"] == g["real_call"]) or (g["fm_call"] == g["real_call"]) for g in allg])
    print(f"{'always-additive':18s} {addacc:>6.3f} {addacc:>7.3f} {'0%':>7s} {addacc:>7.3f} | {'-':>10s} {'-':>9s}")
    print(f"{'trust-all-FM':18s} {fmacc:>6.3f} {fmacc:>7.3f} {'0%':>7s} {fmacc:>7.3f} | {'-':>10s} {'-':>9s}")
    print(f"{'oracle-free':18s} {orac:>6.3f} {orac:>7.3f} {'0%':>7s} {orac:>7.3f} | {'-':>10s} {'-':>9s}")
    for ep in episode_paths:
        recs = [json.loads(l) for l in open(ep) if l.strip()]
        recs = [r for r in recs if "calls" in r and r["perturbation"] in panels]
        ncorr = ntot = nass = 0; nfc = nft = 0; add_follow = fm_follow = ndis = 0
        for r in recs:
            P = panels[r["perturbation"]]; ver = set(r["verified"])
            for g in P["genes"]:
                gn = g["gene"]; ntot += 1
                if gn in ver: nass += 1; ncorr += 1; continue  # verified -> correct
                call = r["calls"].get(gn)
                if call is None: continue
                nft += 1                                         # non-verified, called
                if call == g["real_call"]: ncorr += 1; nfc += 1
                if g["add_call"] != g["fm_call"]:               # disagreement, non-verified
                    ndis += 1
                    if call == g["add_call"]: add_follow += 1
                    elif call == g["fm_call"]: fm_follow += 1
        acc = ncorr / ntot; ar = nass / ntot; net = acc - 0.5 * ar; free_acc = nfc / nft if nft else float("nan")
        af = add_follow / ndis if ndis else float("nan"); ff = fm_follow / ndis if ndis else float("nan")
        name = ep.split("/")[-1].replace("3tier_", "").replace(".jsonl", "")
        print(f"{name:18s} {acc:>6.3f} {free_acc:>7.3f} {ar:>6.0%} {net:>7.3f} | {af:>10.2f} {ff:>9.2f}  (n_dis={ndis})")
    print("\n(net@.5 = acc - 0.5*assay_rate, per-edge, comparable to baselines. freeAcc = accuracy on the agent's"
          "\n NON-verified calls = its FREE-decision quality; beating additive's freeAcc requires defaulting to additive."
          "\n ADD-follow > 0.5 = discovered the cheap-default; <= 0.5 = over-trusts the FM / no preference.)")

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grade", action="store_true"); ap.add_argument("eps", nargs="*")
    ap.add_argument("--panels", default=PANELS_DEFAULT)
    ap.add_argument("--cv", default="results/move1/gears_cv_vs_real.csv")
    ap.add_argument("--marginal", default="results/gears_norman/labeled_marginal.csv")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001"); ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--out"); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.grade:
        grade(a.panels, a.eps); return
    if not os.path.exists(a.panels):
        build_panels(a.cv, a.marginal, a.panels)
    panels = [json.loads(l) for l in open(a.panels) if l.strip()]
    if a.limit: panels = panels[:a.limit]
    out = a.out or f"runs/3tier_{a.model.split('-')[1]}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):
        for l in open(out):
            try: done.add(json.loads(l).get("perturbation"))
            except: pass
    todo = [p for p in panels if p["perturbation"] not in done]
    import anthropic
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    from concurrent.futures import ThreadPoolExecutor
    import threading
    lock = threading.Lock(); fh = open(out, "a"); n = [0]
    print(f"[3tier] {len(todo)} panels x {a.model} -> {out}", file=sys.stderr)
    def work(p):
        rec = run_episode(client, a.model, p, a.lam)
        with lock:
            fh.write(json.dumps(rec) + "\n"); fh.flush(); n[0] += 1
            print(f"  {n[0]}/{len(todo)} {p['perturbation']} ass={len(rec.get('verified',[]))}", file=sys.stderr)
    with ThreadPoolExecutor(max_workers=a.workers) as ex: list(ex.map(work, todo))
    fh.close()

if __name__ == "__main__":
    main()
