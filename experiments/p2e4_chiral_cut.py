"""P2-E4 reciprocal-to-chiral cut: finite-size stability of the non-reciprocal rotation onset.

Manuscript (Paper 2 par.112/137) locates the reciprocal-to-chiral onset by the across-seed
susceptibility of the rotation rate |omega| at a fixed coupling. Same finite-size question as
the noise cut: does the onset LOCATION (susceptibility peak in the antisymmetric alignment
coupling ja) stabilize as N grows? Vary ja from the reciprocal point (0) through the onset,
around the strongest chiral rule (chirality_decay.chiral_rule), at five sizes via box S at
fixed density. Order parameter |omega| is one scalar per run, so the natural estimator is the
across-seed susceptibility chi = N*Var_s(|omega|). Cell-list accelerated, equilibrated runs.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, json, math, dataclasses
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from spp_unified.parameters import SimulationConfig
from spp_unified.model import simulate_two_species
from chirality_decay import chiral_rule
from expanded_common import run_parallel

OUT = ROOT / "results" / "boundary"; OUT.mkdir(parents=True, exist_ok=True)
R0 = 0.1
P0, _ = chiral_rule()
DENS = P0.base.rho * (math.sqrt(3) / 6) / R0 ** 2
TARGET_N = [64, 128, 256, 512, 1024]
JA = [0.0, -0.25, -0.5, -0.75, -1.0, -1.25, -1.5, -1.75, -2.0]   # reciprocal(0) -> beyond rule
N_SEEDS = 10
# |omega| is a FINITE-HORIZON observable (net mean-heading rotation over the window; it
# decays at long horizons, see manuscript par.104-107). Measure at the manuscript's horizon
# (chirality_ndep_v2.py: 800 steps, burn 400) for comparability across sizes.
STEPS, BURN = 800, 400


def s_for_N(n):
    return math.sqrt(n / DENS)


def _task(t):
    N, ja, seed = t
    p = dataclasses.replace(P0, alignment_antisymmetric=float(ja))
    cfg = SimulationConfig(S=s_for_N(N), r0=R0, steps=STEPS, sample_start=BURN,
                           sample_every=2, boundary="periodic", seed=int(seed),
                           use_cell_list=True)
    r = simulate_two_species(p, cfg)
    return {"N": int(r.summary["n"]), "ja": float(ja), "seed": int(seed),
            "abs_omega": float(r.summary["abs_omega"]), "chirality": float(r.summary["chirality"]),
            "phi": float(r.summary["phi"])}


def peak_x(x, y):
    x = np.asarray(x); y = np.asarray(y); i = int(y.argmax())
    if 0 < i < len(y) - 1:
        xs, ys = x[i-1:i+2], y[i-1:i+2]
        d = (xs[0]-xs[1])*(xs[0]-xs[2])*(xs[1]-xs[2])
        A = (xs[2]*(ys[1]-ys[0])+xs[1]*(ys[0]-ys[2])+xs[0]*(ys[2]-ys[1]))/d
        B = (xs[2]**2*(ys[0]-ys[1])+xs[1]**2*(ys[2]-ys[0])+xs[0]**2*(ys[1]-ys[2]))/d
        if A < 0:
            return float(np.clip(-B/(2*A), xs.min(), xs.max()))
    return float(x[i])


def main():
    tasks = [(N, round(float(j), 4), s) for N in TARGET_N for j in JA for s in range(N_SEEDS)]
    print(f"Chiral cut: {len(tasks)} runs x {STEPS} steps (burn {BURN}), cell list", flush=True)
    res = run_parallel(_task, tasks, max_workers=6, desc="p2e4-chiral")
    df = pd.DataFrame(res)
    df.to_csv(OUT / "P2E4_chiral_cut_perseed.csv", index=False)

    rows = []
    for (N, ja), g in df.groupby(["N", "ja"]):
        om = g.abs_omega.to_numpy()
        rows.append({"N": int(N), "ja": float(ja), "abs_omega_mean": float(om.mean()),
                     "abs_omega_sd": float(om.std()), "chi": int(N) * float(om.var()),
                     "chirality_mean": float(g.chirality.mean())})
    grid = pd.DataFrame(rows).sort_values(["N", "ja"])
    grid.to_csv(OUT / "P2E4_chiral_cut_grid.csv", index=False)

    Ns = sorted(grid.N.unique())
    head = {"config": {"N": TARGET_N, "ja": JA, "seeds": N_SEEDS, "steps": STEPS,
                       "rule_ja": P0.alignment_antisymmetric}, "by_size": {}}
    for N in Ns:
        sub = grid[grid.N == N].sort_values("ja")
        head["by_size"][int(N)] = {
            "ja_star_chi": peak_x(sub.ja.to_numpy(), sub.chi.to_numpy()),
            "chi_max": float(sub.chi.max()),
            "ja": sub.ja.tolist(),
            "abs_omega_mean": [round(float(x), 4) for x in sub.abs_omega_mean.tolist()],
            "chi": [round(float(x), 4) for x in sub.chi.tolist()]}
    head["ja_star_by_size"] = {int(N): round(head["by_size"][int(N)]["ja_star_chi"], 4) for N in Ns}
    sv = list(head["ja_star_by_size"].values())
    head["ja_star_spread"] = float(max(sv) - min(sv))
    json.dump(head, open(OUT / "P2E4_chiral_cut_headline.json", "w"), indent=2)

    print("\n=== onset ja* (|omega| susceptibility peak) by size ===")
    for N in Ns:
        b = head["by_size"][int(N)]
        print(f"  N={N:5d}: ja*={b['ja_star_chi']:.3f}  chi_max={b['chi_max']:.4f}  "
              f"|omega|(rule ja=-1.5)={b['abs_omega_mean'][JA.index(-1.5)]:.3f}")
    print(f"\nja* spread across sizes: {head['ja_star_spread']:.3f}")
    print("|omega|_mean at rule (ja=-1.5) by N:",
          {N: head['by_size'][N]['abs_omega_mean'][JA.index(-1.5)] for N in Ns})
    print("\nwrote P2E4_chiral_cut_{perseed,grid}.csv + _headline.json")


if __name__ == "__main__":
    main()
