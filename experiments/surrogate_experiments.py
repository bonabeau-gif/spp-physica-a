"""Section 6 surrogate experiments E1 (3000-call top-up), E2 (1000-call crossover),
with E4 wall-clock instrumentation and E5 calibration logging.

Reimplements run_one with per-round timers (fit_gps refit, acquisition/generate, eval)
and saves per-(budget, rep, method) results incrementally so the long GP-heavy runs are
resumable. Deterministic seeds match run_learned_surrogate exactly, so reps 0-2 at 3000
reproduce the existing summary (code-version check).

Usage: surrogate_experiments.py <budget> <rep_lo> <rep_hi> [methods]
  e.g. E2: surrogate_experiments.py 1000 0 6
       E1: surrogate_experiments.py 3000 3 6
       code-check: surrogate_experiments.py 30 0 1 surrogate
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
import np_encoder_patch  # noqa: F401
from run_learned_surrogate import (load_encoder, add_initial_columns, fit_gps,
                                    generate_surrogate, generate_direct, eval_batch,
                                    current_cells, DIMS)

OUT = ROOT / "results" / "expanded" / "surrogate_exp"
OUT.mkdir(parents=True, exist_ok=True)
BATCH = 10
# NOTE: sys.argv is parsed inside main(), not at module level, so spawned workers that
# re-import this module (macOS spawn) do not fail on a missing argv.


def stratified_initial(corpus, rep):
    rng = np.random.default_rng(1100 + rep)
    idx = []
    for band in ["low", "medium", "high"]:
        pool = np.where(corpus.density_band.values == band)[0]
        idx.extend(rng.choice(pool, 40, replace=False).tolist())
    return corpus.iloc[idx].copy()


def run_instrumented(rep, method, initial, ae, lo, hi, calls):
    rng = np.random.default_rng(7000 + rep * 101 + {"surrogate": 1, "direct": 2, "random": 3}[method])
    df = initial.copy().reset_index(drop=True); df["source"] = "initial"; df["round"] = 0
    hist, preds, timing = [], [], []
    for rd in range(1, calls // BATCH + 1):
        t_fit = t_gen = 0.0; ntrain = len(df)
        if method == "surrogate":
            U = df[[f"u_{i}" for i in range(9)]].to_numpy()
            Y = df[["learned_0", "learned_1", "temporal_stability"]].to_numpy()
            t0 = time.perf_counter(); models = fit_gps(U, Y); t_fit = time.perf_counter() - t0
            t0 = time.perf_counter(); Unew, mu, sd = generate_surrogate(rng, df, models, BATCH); t_gen = time.perf_counter() - t0
        elif method == "direct":
            t0 = time.perf_counter(); Unew = generate_direct(rng, df, BATCH); t_gen = time.perf_counter() - t0; mu = sd = None
        else:
            Unew = rng.random((BATCH, 9)); mu = sd = None
        t0 = time.perf_counter()
        nd, _ = eval_batch(Unew, [800000 + rep * 10000 + rd * 100 + i for i in range(BATCH)], ae, lo, hi)
        t_eval = time.perf_counter() - t0
        nd["source"] = method; nd["round"] = rd
        if mu is not None:
            for i in range(BATCH):
                preds.append({"replicate": rep, "round": rd, "ntrain": ntrain,
                              "actual_z0": nd.iloc[i].learned_0, "actual_z1": nd.iloc[i].learned_1,
                              "actual_q": nd.iloc[i].temporal_stability,
                              "pred_z0": mu[i, 0], "pred_z1": mu[i, 1], "pred_q": mu[i, 2],
                              "sd_z0": sd[i, 0], "sd_z1": sd[i, 1], "sd_q": sd[i, 2]})
        df = pd.concat([df, nd], ignore_index=True)
        cells = current_cells(df)
        hist.append({"budget": calls, "replicate": rep, "method": method, "true_calls": rd * BATCH,
                     "coverage": len(cells) / (DIMS[0] * DIMS[1]), "num_elites": len(cells),
                     "mean_quality": float(np.mean([v[1] for v in cells.values()]))})
        # E4: per-round wall-clock, with mean particle count of this batch and its density
        timing.append({"budget": calls, "replicate": rep, "method": method, "round": rd, "ntrain": ntrain,
                       "t_fit": t_fit, "t_gen": t_gen, "t_eval": t_eval,
                       "n_particles_mean": float(nd["n"].mean()) if "n" in nd else np.nan,
                       "rho_mean": float(nd["rho"].mean()) if "rho" in nd else np.nan})
        if rd % 10 == 0 or rd == calls // BATCH:
            print(f"{method} rep{rep} rd{rd}/{calls//BATCH} cov={hist[-1]['coverage']:.3f} "
                  f"t_fit={t_fit:.2f} t_eval={t_eval:.2f} ntrain={ntrain}", flush=True)
    return hist, preds, timing


def main():
    budget = int(sys.argv[1]); rep_lo = int(sys.argv[2]); rep_hi = int(sys.argv[3])
    methods = sys.argv[4].split(",") if len(sys.argv) > 4 else ["surrogate", "direct", "random"]
    ae, lo, hi = load_encoder()
    corpus = add_initial_columns(pd.read_csv(ROOT / "results/expanded/single_behavior_corpus_latent.csv"))
    corpus = corpus[corpus.boundary == "periodic"].reset_index(drop=True)
    for rep in range(rep_lo, rep_hi):
        initial = stratified_initial(corpus, rep)
        for method in methods:
            tag = f"b{budget}_rep{rep}_{method}"
            hf = OUT / f"hist_{tag}.csv"
            if hf.exists():
                print(f"skip {tag} (exists)", flush=True); continue
            t0 = time.time()
            hist, preds, timing = run_instrumented(rep, method, initial, ae, lo, hi, budget)
            pd.DataFrame(hist).to_csv(hf, index=False)
            if preds:
                pd.DataFrame(preds).to_csv(OUT / f"preds_{tag}.csv", index=False)
            pd.DataFrame(timing).to_csv(OUT / f"timing_{tag}.csv", index=False)
            print(f"DONE {tag} final_cov={hist[-1]['coverage']:.4f} wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
