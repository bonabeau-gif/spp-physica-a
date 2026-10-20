"""P2-E4 prep: measure the actual equilibration (relaxation) time of phi(t) at
representative (N, eta) points, to set a defensible burn-in for the confirmatory noise cut.
Stage A showed 400 steps is under-equilibrated at large N; this records the full transient
(cell-list accelerated) so the burn-in is measured, not guessed.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, json, math
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from spp_unified.parameters import SingleParams, SimulationConfig
from spp_unified.model import simulate_single
from expanded_common import run_parallel

RHO, R0 = 0.15, 0.1
DENS = RHO * (math.sqrt(3) / 6) / R0 ** 2
STEPS = 5000
EVERY = 25


def s_for_N(n):
    return math.sqrt(n / DENS)


def _task(t):
    N, eta, seed = t
    p = SingleParams(rho=RHO, v0=0.05, inertia=0.3, interaction_range_ratio=5.0,
                     force_strength=0.0, adhesion=0.0, alignment_range_ratio=8.0,
                     alignment_strength=3.5, noise=float(eta))
    cfg = SimulationConfig(S=s_for_N(N), r0=R0, steps=STEPS, sample_start=0,
                           sample_every=EVERY, boundary="periodic", seed=int(seed),
                           use_cell_list=True)
    r = simulate_single(p, cfg)
    phi = np.asarray(r.timeseries["phi"])
    return {"N": int(r.summary["n"]), "eta": float(eta), "seed": int(seed),
            "phi_t": phi.tolist()}


def main():
    pts = [(1024, 0.35), (1024, 0.575), (1024, 0.725), (256, 0.575), (64, 0.575)]
    tasks = [(N, e, s) for (N, e) in pts for s in (0, 1)]
    res = run_parallel(_task, tasks, max_workers=6, desc="equil-probe")
    times = np.arange(0, STEPS, EVERY)
    marks = [0, 250, 500, 1000, 2000, 3000, 4000, 4975]
    idx = [int(np.argmin(np.abs(times - m))) for m in marks]
    print(f"phi(t) at t = {marks}  (steps)\n")
    agg = {}
    for r in res:
        key = (r["N"], r["eta"])
        agg.setdefault(key, []).append(np.asarray(r["phi_t"]))
    for (N, eta), arrs in sorted(agg.items()):
        m = np.mean(arrs, axis=0)
        # equilibration time: first t where running mean of the 2nd-half-window is within
        # 0.02 of the final-1000-step mean and stays there
        final = float(np.mean([a[times >= STEPS - 1000].mean() for a in arrs]))
        run_stab = None
        for k, t in enumerate(times):
            if abs(m[k] - final) < 0.02 and np.all(np.abs(m[k:] - final) < 0.05):
                run_stab = int(t); break
        vals = "  ".join(f"{m[i]:.3f}" for i in idx)
        print(f"N={N:5d} eta={eta:.3f}: {vals}   final~{final:.3f}  equil~{run_stab} steps")
    json.dump({"steps": STEPS, "every": EVERY,
               "series": {f"{r['N']}_{r['eta']}_{r['seed']}": r["phi_t"] for r in res}},
              open(ROOT / "results/boundary/P2E4_equilibration_probe.json", "w"))
    print("\nwrote P2E4_equilibration_probe.json")


if __name__ == "__main__":
    main()
