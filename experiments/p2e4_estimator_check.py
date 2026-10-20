"""P2-E4 rigor: reproduce-then-extend the manuscript's noise-cut claim (Paper 2 par. 128/132:
"susceptibility peak stays near 0.50 across sizes -> transition, not crossover").

The manuscript used the ACROSS-SEED variance of the window-averaged order parameter,
chi_seedmean = N*Var_s(<phi>_seed), at short windows (220/440 steps) over N up to ~317.
Here we compute BOTH that estimator and the standard pooled fluctuation susceptibility
chi_pooled = N*Var_{t,s}(phi), at short windows 220 and 440, over N up to 1024 (20 seeds,
cell-list). Question: does the manuscript's estimator reproduce "~0.50, no drift" at its
own protocol, and does extending N / the estimator choice reveal a drift?

Pairs with the equilibrated confirmatory run (5000 steps) which already shows chi_pooled
peak drifting 0.61 -> 0.45.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, json, math
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from spp_unified.parameters import SingleParams, SimulationConfig
from spp_unified.model import simulate_single
from expanded_common import run_parallel

OUT = ROOT / "results" / "boundary"
RHO, R0 = 0.15, 0.1
DENS = RHO * (math.sqrt(3) / 6) / R0 ** 2
TARGET_N = [64, 128, 256, 512, 1024]
ETA = [0.20, 0.35, 0.425, 0.50, 0.575, 0.65, 0.725, 0.80, 0.90]
N_SEEDS = 20
STEPS = 440           # single run; windows [110,220] ("220") and [220,440] ("440") extracted


def s_for_N(n):
    return math.sqrt(n / DENS)


def _task(t):
    N, eta, seed = t
    p = SingleParams(rho=RHO, v0=0.05, inertia=0.3, interaction_range_ratio=5.0,
                     force_strength=0.0, adhesion=0.0, alignment_range_ratio=8.0,
                     alignment_strength=3.5, noise=float(eta))
    cfg = SimulationConfig(S=s_for_N(N), r0=R0, steps=STEPS, sample_start=0,
                           sample_every=2, boundary="periodic", seed=int(seed),
                           use_cell_list=True)
    r = simulate_single(p, cfg)
    phi = np.asarray(r.timeseries["phi"], float)          # sampled every 2 steps from t=0
    t_axis = np.arange(len(phi)) * 2
    w220 = phi[(t_axis >= 110) & (t_axis < 220)]
    w440 = phi[(t_axis >= 220) & (t_axis < 440)]
    return {"N": int(r.summary["n"]), "eta": float(eta), "seed": int(seed),
            "m220": float(w220.mean()), "s2_220": float((w220**2).sum()), "c220": int(w220.size),
            "s1_220": float(w220.sum()),
            "m440": float(w440.mean()), "s2_440": float((w440**2).sum()), "c440": int(w440.size),
            "s1_440": float(w440.sum())}


def peak_eta(sub, col):
    e = sub.eta.to_numpy(); c = sub[col].to_numpy()
    i = int(c.argmax())
    if 0 < i < len(c) - 1:
        x, y = e[i-1:i+2], c[i-1:i+2]
        d = (x[0]-x[1])*(x[0]-x[2])*(x[1]-x[2])
        A = (x[2]*(y[1]-y[0])+x[1]*(y[0]-y[2])+x[0]*(y[2]-y[1]))/d
        B = (x[2]**2*(y[0]-y[1])+x[1]**2*(y[2]-y[0])+x[0]**2*(y[1]-y[2]))/d
        if A < 0:
            return float(np.clip(-B/(2*A), x[0], x[2]))
    return float(e[i])


def main():
    tasks = [(N, round(float(e), 4), s) for N in TARGET_N for e in ETA for s in range(N_SEEDS)]
    print(f"Estimator check: {len(tasks)} runs x {STEPS} steps, {N_SEEDS} seeds, cell list", flush=True)
    res = run_parallel(_task, tasks, max_workers=6, desc="p2e4-estcheck")
    df = pd.DataFrame(res)
    rows = []
    for (N, eta), g in df.groupby(["N", "eta"]):
        rec = {"N": int(N), "eta": float(eta)}
        for w in ("220", "440"):
            cnt = g[f"c{w}"].sum()
            m1 = g[f"s1_{w}"].sum() / cnt; m2 = g[f"s2_{w}"].sum() / cnt
            rec[f"chi_pooled_{w}"] = int(N) * max(m2 - m1*m1, 0.0)
            rec[f"chi_seedmean_{w}"] = int(N) * float(g[f"m{w}"].to_numpy().var())
        rows.append(rec)
    grid = pd.DataFrame(rows).sort_values(["N", "eta"])
    grid.to_csv(OUT / "P2E4_estimator_check_grid.csv", index=False)

    Ns = sorted(grid.N.unique())
    out = {"config": {"N": TARGET_N, "eta": ETA, "seeds": N_SEEDS, "windows": [220, 440]},
           "eta_star": {}}
    print("\n=== eta*(N) peak location by estimator x window ===")
    for col in ("chi_seedmean_220", "chi_seedmean_440", "chi_pooled_220", "chi_pooled_440"):
        es = {int(N): round(peak_eta(grid[grid.N == N].sort_values("eta"), col), 4) for N in Ns}
        out["eta_star"][col] = es
        spread = max(es.values()) - min(es.values())
        tag = "MANUSCRIPT estimator" if "seedmean" in col else "pooled (standard)"
        print(f"  {col:22} [{tag:20}]: " + " ".join(f"N{n}={v:.3f}" for n, v in es.items())
              + f"   spread={spread:.3f}")
    out["equilibrated_pooled_note"] = "confirmatory 5000-step run: chi_pooled peak 0.61->0.45 (spread 0.16)"
    json.dump(out, open(OUT / "P2E4_estimator_check.json", "w"), indent=2)
    print("\nwrote P2E4_estimator_check_{grid.csv,.json}")


if __name__ == "__main__":
    main()
