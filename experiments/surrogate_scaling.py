"""Assessment v2, largest gap: scale the surrogate/direct/random comparison to
300 and 3,000 additional calls, to test whether the coverage null result is
budget-limited (as the 30-call noise-injection conclusion asserts) and to locate
any crossover where coverage separates.

Writes results/expanded/surrogate_scaling_{history,summary}.csv.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))
import torch
torch.set_num_threads(1)
import np_encoder_patch  # noqa: F401
from run_learned_surrogate import load_encoder, add_initial_columns, run_one

OUT = ROOT / "results" / "expanded"

# reps per budget (fewer at large budgets for tractability)
BUDGETS = {30: 6, 300: 4, 3000: 3}


def stratified_initial(corpus, rep):
    rng = np.random.default_rng(1100 + rep)
    idx = []
    for band in ["low", "medium", "high"]:
        pool = np.where(corpus.density_band.values == band)[0]
        idx.extend(rng.choice(pool, 40, replace=False).tolist())
    return corpus.iloc[idx].copy()


def main():
    ae, lo, hi = load_encoder()
    corpus = add_initial_columns(pd.read_csv(OUT / "single_behavior_corpus_latent.csv"))
    corpus = corpus[corpus.boundary == "periodic"].reset_index(drop=True)
    all_hist = []
    for calls, nrep in BUDGETS.items():
        for rep in range(nrep):
            initial = stratified_initial(corpus, rep)
            for method in ["surrogate", "direct", "random"]:
                _, h, _ = run_one(rep, method, initial, ae, lo, hi, calls=calls)
                h["budget"] = calls
                all_hist.append(h)
                print(f"budget={calls} rep={rep} {method}: final cov "
                      f"{h.coverage.iloc[-1]:.3f}", flush=True)
    hist = pd.concat(all_hist, ignore_index=True)
    hist.to_csv(OUT / "surrogate_scaling_history.csv", index=False)
    # final coverage per (budget, method)
    final = hist.sort_values("true_calls").groupby(["budget", "replicate", "method"]).tail(1)
    summ = final.groupby(["budget", "method"]).agg(
        coverage_mean=("coverage", "mean"), coverage_sd=("coverage", "std"),
        quality_mean=("mean_quality", "mean"), n=("coverage", "size")).reset_index()
    summ.to_csv(OUT / "surrogate_scaling_summary.csv", index=False)
    print("=== summary ===")
    print(summ.to_string(index=False))
    # crossover: coverage gap (surrogate - best of direct/random) per budget
    piv = summ.pivot_table(index="budget", columns="method", values="coverage_mean")
    piv["surrogate_minus_best_baseline"] = piv["surrogate"] - piv[["direct", "random"]].max(axis=1)
    piv.to_csv(OUT / "surrogate_scaling_gap.csv")
    print(piv.round(4).to_string())
    json.dump({"summary": summ.to_dict("records"),
               "gap_by_budget": {int(b): float(g) for b, g in piv["surrogate_minus_best_baseline"].items()}},
              open(OUT / "surrogate_scaling_headline.json", "w"), indent=2)


if __name__ == "__main__":
    main()
