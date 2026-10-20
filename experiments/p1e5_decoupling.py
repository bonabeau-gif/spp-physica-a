"""P1-E5 (revised, tractable): decouple density, N, box size, and boundary condition.

Which apparent density effects are physical density effects, finite-N effects, or wall
effects? Parameterization: number density rho_box = N/S^2 = rho*(sqrt3/6)/r0^2 depends only
on rho (not S), so rho SETS density and S SETS N at fixed density. For each of 5 single-
species behaviors (from objective champions), a space-filling ~17-condition design over
(rho, N, boundary) -- NOT a full factorial -- with 6 seeds, cell-list accelerated. A
hierarchical/OLS fit (companion script) then separates the rho, logN, and boundary effects.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, math
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from spp_unified.parameters import SingleParams, SimulationConfig
from spp_unified.model import simulate_single
from expanded_common import run_parallel

OUT = ROOT / "results" / "boundary"; OUT.mkdir(parents=True, exist_ok=True)
R0 = 0.1
DENSFAC = (math.sqrt(3) / 6) / R0 ** 2          # N = rho * DENSFAC * S^2
RULES = ["aligned_flock", "dense_cluster", "density_waves", "hexatic_crystal", "active_turbulence"]
PARAM_COLS = ["v0", "inertia", "interaction_range_ratio", "force_strength", "adhesion",
              "alignment_range_ratio", "alignment_strength", "noise"]
N_SEEDS = 6
STEPS, BURN = 1500, 500


def s_for(rho, N):
    return math.sqrt(N / (rho * DENSFAC))


def design():
    """~17 unique (rho, N, boundary) conditions: density-at-fixed-N, N-at-fixed-density,
    corners, and reflecting-wall replicates to identify the boundary effect."""
    conds = set()
    for d in (0.05, 0.10, 0.15, 0.20, 0.40):        # density axis at N=256
        conds.add((d, 256, "periodic"))
    for N in (64, 128, 256, 512, 1024):             # N axis at rho=0.10
        conds.add((0.10, N, "periodic"))
    for d in (0.05, 0.40):                          # corners
        for N in (64, 512):
            conds.add((d, N, "periodic"))
    for (d, N) in [(0.10, 256), (0.20, 256), (0.10, 64), (0.10, 512)]:   # wall replicates
        conds.add((d, N, "reflecting"))
    return sorted(conds)


def load_rules():
    c = pd.read_csv(ROOT / "results/expanded/objective_champions_periodic.csv").set_index("objective")
    return {r: {col: float(c.loc[r, col]) for col in PARAM_COLS} for r in RULES}


def _task(t):
    rule, base, rho, N, boundary, seed = t
    p = SingleParams(rho=float(rho), **base)
    cfg = SimulationConfig(S=s_for(rho, N), r0=R0, steps=STEPS, sample_start=BURN,
                           sample_every=2, boundary=boundary, seed=int(seed), use_cell_list=True)
    r = simulate_single(p, cfg)
    s = r.summary
    return {"rule": rule, "rho": float(rho), "N_target": int(N), "boundary": boundary,
            "seed": int(seed), "N": int(s["n"]),
            "phi": float(s["phi"]), "nematic": float(s["nematic"]), "milling": float(s["milling"]),
            "clustering": float(s["clustering"]), "temporal_stability": float(s["temporal_stability"])}


def main():
    rules = load_rules()
    conds = design()
    tasks = [(r, rules[r], d, N, b, sd) for r in RULES for (d, N, b) in conds for sd in range(N_SEEDS)]
    print(f"P1-E5: {len(RULES)} rules x {len(conds)} conditions x {N_SEEDS} seeds = {len(tasks)} runs "
          f"({STEPS} steps, cell list)", flush=True)
    res = run_parallel(_task, tasks, max_workers=6, desc="p1e5")
    df = pd.DataFrame(res)
    df.to_csv(OUT / "P1E5_decoupling_perseed.csv", index=False)
    print("wrote P1E5_decoupling_perseed.csv  (", len(df), "rows )")
    # quick per-rule condition means for a first look
    agg = df.groupby(["rule", "rho", "N_target", "boundary"]).agg(
        phi=("phi", "mean"), clustering=("clustering", "mean"), milling=("milling", "mean"),
        N=("N", "first")).reset_index()
    agg.to_csv(OUT / "P1E5_decoupling_condmeans.csv", index=False)
    print("wrote P1E5_decoupling_condmeans.csv")


if __name__ == "__main__":
    main()
