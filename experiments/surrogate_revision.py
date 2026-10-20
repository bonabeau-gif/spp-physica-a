"""Assessment #2: strengthen the surrogate-illumination result.

Part A - replication: rerun the surrogate / direct / random matched-budget
comparison with more independent replicates so the small coverage differences
carry error bars instead of resting on two repetitions.

Part B - mechanism: inject controlled Gaussian noise of magnitude eps into the
learned archive coordinates used by the (otherwise oracle) direct illuminator,
and measure how true coverage degrades as a function of eps. This directly tests
the paper's claim that error in the learned coordinates - not in stability - is
what limits surrogate-assisted coverage. If coverage falls sharply as eps passes
through the GP's own coordinate error (0.1-0.2), the mechanism is confirmed.
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
import np_encoder_patch  # noqa: F401  (numpy forward pass; torch<->numpy2 bridge is broken)
from run_learned_surrogate import (load_encoder, add_initial_columns, run_one,
                                   eval_batch, generate_direct, DIMS)
from expanded_common import archive_cells

OUT = ROOT / "results" / "expanded"
ACTIVE = None


def stratified_initial(corpus, rep):
    rng = np.random.default_rng(1100 + rep)
    idx = []
    for band in ["low", "medium", "high"]:
        pool = np.where(corpus.density_band.values == band)[0]
        idx.extend(rng.choice(pool, 40, replace=False).tolist())
    return corpus.iloc[idx].copy()


def true_coverage(df, xcol="learned_0", ycol="learned_1"):
    cells = archive_cells(df[[xcol, ycol]].to_numpy(), df["temporal_stability"].to_numpy(), DIMS)
    return len(cells) / (DIMS[0] * DIMS[1])


def run_direct_noisy(rep, initial, ae, lo, hi, eps, calls=30, batch=10):
    """Direct illumination whose selection and accounting use noisy archive
    coordinates a = z_true + N(0, eps); returns final TRUE and perceived coverage."""
    rng = np.random.default_rng(9000 + rep * 131 + int(eps * 1000))
    df = initial.copy().reset_index(drop=True)
    df["a0"] = df["learned_0"]; df["a1"] = df["learned_1"]     # initial verified => no error
    for rd in range(1, calls // batch + 1):
        # parent selection and empty-cell logic operate on perceived coords a0,a1
        tmp = df.rename(columns={"learned_0": "_z0", "learned_1": "_z1",
                                 "a0": "learned_0", "a1": "learned_1"})
        Unew = generate_direct(rng, tmp, batch)
        nd, _ = eval_batch(Unew, [910000 + rep * 10000 + rd * 100 + i for i in range(batch)], ae, lo, hi)
        nd["a0"] = np.clip(nd["learned_0"] + rng.normal(0, eps, len(nd)), 0, 1)
        nd["a1"] = np.clip(nd["learned_1"] + rng.normal(0, eps, len(nd)), 0, 1)
        df = pd.concat([df, nd], ignore_index=True)
    return {"rep": rep, "eps": eps,
            "true_coverage": true_coverage(df, "learned_0", "learned_1"),
            "perceived_coverage": true_coverage(df, "a0", "a1"),
            "mean_quality": float(np.mean([v[1] for v in archive_cells(
                df[["learned_0", "learned_1"]].to_numpy(), df["temporal_stability"].to_numpy(), DIMS).values()]))}


def main():
    ae, lo, hi = load_encoder()
    corpus = add_initial_columns(pd.read_csv(OUT / "single_behavior_corpus_latent.csv"))
    corpus = corpus[corpus.boundary == "periodic"].reset_index(drop=True)

    # -------- Part A: replicated three-method comparison --------
    N_REP = 6
    all_hist = []
    for rep in range(N_REP):
        initial = stratified_initial(corpus, rep)
        for method in ["surrogate", "direct", "random"]:
            _, h, _ = run_one(rep, method, initial, ae, lo, hi, calls=30)
            all_hist.append(h)
    hist = pd.concat(all_hist, ignore_index=True)
    final = hist.sort_values("true_calls").groupby(["replicate", "method"]).tail(1)
    summ = final.groupby("method").agg(
        coverage_mean=("coverage", "mean"), coverage_sd=("coverage", "std"),
        elites_mean=("num_elites", "mean"), elites_sd=("num_elites", "std"),
        quality_mean=("mean_quality", "mean"), quality_sd=("mean_quality", "std"),
        n=("coverage", "size")).reset_index()
    hist.to_csv(OUT / "surrogate_replicated_history.csv", index=False)
    summ.to_csv(OUT / "surrogate_replicated_summary.csv", index=False)
    print("=== Part A: replicated comparison ===")
    print(summ.to_string(index=False), flush=True)

    # -------- Part B: coordinate-noise injection --------
    eps_grid = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    rows = []
    for eps in eps_grid:
        for rep in range(6):
            initial = stratified_initial(corpus, rep)
            rows.append(run_direct_noisy(rep, initial, ae, lo, hi, eps))
            print(rows[-1], flush=True)
    nd = pd.DataFrame(rows)
    nd.to_csv(OUT / "surrogate_noise_injection.csv", index=False)
    agg = nd.groupby("eps").agg(true_cov_mean=("true_coverage", "mean"),
                                true_cov_sd=("true_coverage", "std"),
                                perceived_cov_mean=("perceived_coverage", "mean"),
                                quality_mean=("mean_quality", "mean")).reset_index()
    agg.to_csv(OUT / "surrogate_noise_injection_summary.csv", index=False)
    print("=== Part B: noise injection ===")
    print(agg.to_string(index=False))
    base = float(agg[agg.eps == 0.0].true_cov_mean.iloc[0])
    with open(OUT / "surrogate_revision_headline.json", "w") as f:
        json.dump({"three_method": summ.to_dict("records"),
                   "noise_injection": agg.to_dict("records"),
                   "coverage_at_eps0": base,
                   "coverage_at_eps0.15": float(agg[np.isclose(agg.eps, 0.15)].true_cov_mean.iloc[0]),
                   "relative_drop_at_0.15": float(1 - agg[np.isclose(agg.eps, 0.15)].true_cov_mean.iloc[0] / base)}, f, indent=2)


if __name__ == "__main__":
    main()
