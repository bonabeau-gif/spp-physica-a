"""P2-E4 (revised, tractable) — noise cut, Stage A: coarse localization.

Tests whether the noise-boundary location stabilizes as N increases and whether Binder
curves show a stable crossing. Deliberately coarse: 5 sizes x 9 noise values x 6 matched
seeds = 270 initial 400-step trajectories. tau_int is recorded per trajectory so we know
whether 400 steps is adequate (window >= 20 tau_int) or the point is under-resolved and
must be extended in Stage C. Precision comes later (Stage B refinement, Stage C long runs).

N is scaled via box size S at fixed density (rho, r0): N = round(rho * (sqrt3/6) * (S/r0)^2).
Observables per (N, eta): phi_mean, susceptibility = N*Var_s(phi), Binder cumulant, and the
per-seed phi distribution (bimodality check). See revised_scaling_experiments_plan.md.
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
from boundary.protocol import tau_int
from expanded_common import run_parallel

OUT = ROOT / "results" / "boundary"; OUT.mkdir(parents=True, exist_ok=True)

RHO, R0 = 0.15, 0.1
DENS = RHO * (math.sqrt(3) / 6) / R0 ** 2                 # N = DENS * S^2
TARGET_N = [64, 128, 256, 512, 1024]
ETA = [0.20, 0.35, 0.425, 0.50, 0.575, 0.65, 0.725, 0.80, 0.90]
N_SEEDS = 6
STEPS = 400
SAMPLE_EVERY = 2
TAU_MULT = 20.0                                          # adequacy: window >= 20 tau_int


def s_for_N(n):
    return math.sqrt(n / DENS)


def _task(t):
    """One 400-step trajectory. t = (Ntarget, S, eta, seed)."""
    Ntarget, S, eta, seed = t
    p = SingleParams(rho=RHO, v0=0.05, inertia=0.3, interaction_range_ratio=5.0,
                     force_strength=0.0, adhesion=0.0, alignment_range_ratio=8.0,
                     alignment_strength=3.5, noise=float(eta))
    cfg = SimulationConfig(S=S, r0=R0, steps=STEPS, sample_start=STEPS // 2,
                           sample_every=SAMPLE_EVERY, boundary="periodic", seed=int(seed))
    r = simulate_single(p, cfg)
    tau = float("nan")
    s = r.timeseries.get("phi") if hasattr(r, "timeseries") and r.timeseries else None
    if s is not None and len(s) > 3:
        tau = float(tau_int(np.asarray(s)) * SAMPLE_EVERY)
    return {"Ntarget": Ntarget, "S": S, "eta": float(eta), "seed": int(seed),
            "N": int(r.summary["n"]), "phi": float(r.summary["phi"]),
            "abs_omega": float(r.summary.get("abs_omega", 0.0)), "tau_phi": tau}


def main():
    tasks = [(n, s_for_N(n), round(float(e), 4), sd)
             for n in TARGET_N for e in ETA for sd in range(N_SEEDS)]
    print(f"Stage A: {len(tasks)} trajectories ({len(TARGET_N)} sizes x {len(ETA)} eta x {N_SEEDS} seeds), "
          f"{STEPS} steps each", flush=True)
    res = run_parallel(_task, tasks, max_workers=6, desc="p2e4-noise-stageA")
    df = pd.DataFrame(res)
    df.to_csv(OUT / "P2E4_noise_stageA_perseed.csv", index=False)

    rows = []
    for (Nt, eta), g in df.groupby(["Ntarget", "eta"]):
        phi = g.phi.to_numpy()
        O2 = float((phi ** 2).mean()); O4 = float((phi ** 4).mean())
        binder = 1.0 - O4 / (3.0 * O2 * O2) if O2 > 1e-12 else float("nan")
        N = int(g.N.iloc[0])
        tau_med = float(np.nanmedian(g.tau_phi))
        rows.append({"Ntarget": int(Nt), "N": N, "eta": float(eta),
                     "phi_mean": float(phi.mean()), "phi_var": float(phi.var()),
                     "susceptibility": N * float(phi.var()), "binder": binder,
                     "tau_phi_median": tau_med,
                     "adequate": bool((STEPS - STEPS // 2) >= TAU_MULT * tau_med) if tau_med == tau_med else False})
    grid = pd.DataFrame(rows).sort_values(["N", "eta"])
    grid.to_csv(OUT / "P2E4_noise_stageA_grid.csv", index=False)

    # headline: susceptibility peak (eta*) per size + Binder curve + adjacent-size crossings
    headline = {"config": {"target_N": TARGET_N, "eta_grid": ETA, "n_seeds": N_SEEDS,
                           "steps": STEPS, "rho": RHO, "r0": R0}, "by_size": {}}
    for N in sorted(grid.N.unique()):
        sub = grid[grid.N == N].sort_values("eta")
        i = int(sub.susceptibility.values.argmax())
        headline["by_size"][int(N)] = {
            "eta_star": float(sub.eta.values[i]), "chi_max": float(sub.susceptibility.values[i]),
            "binder_at_etastar": float(sub.binder.values[i]),
            "eta": sub.eta.tolist(), "susceptibility": sub.susceptibility.round(4).tolist(),
            "binder": [round(float(x), 4) for x in sub.binder.tolist()],
            "phi_mean": [round(float(x), 4) for x in sub.phi_mean.tolist()],
            "tau_median": [round(float(x), 1) for x in sub.tau_phi_median.tolist()],
            "n_underresolved": int((~sub.adequate).sum())}
    Ns = sorted(grid.N.unique())
    headline["eta_star_by_size"] = {int(N): headline["by_size"][int(N)]["eta_star"] for N in Ns}
    headline["eta_star_spread"] = float(max(headline["eta_star_by_size"].values()) -
                                        min(headline["eta_star_by_size"].values()))
    # adjacent-size Binder crossings (linear interpolation of the difference)
    cross = {}
    for a, b in zip(Ns[:-1], Ns[1:]):
        ga = grid[grid.N == a].sort_values("eta"); gb = grid[grid.N == b].sort_values("eta")
        d = gb.binder.to_numpy() - ga.binder.to_numpy(); e = ga.eta.to_numpy()
        sign = np.sign(d); idx = np.where(np.diff(sign) != 0)[0]
        xs = []
        for k in idx:
            if d[k + 1] != d[k]:
                xs.append(float(e[k] - d[k] * (e[k + 1] - e[k]) / (d[k + 1] - d[k])))
        cross[f"{a}-{b}"] = xs
    headline["binder_crossings_adjacent"] = cross
    json.dump(headline, open(OUT / "P2E4_noise_stageA_headline.json", "w"), indent=2)

    print("\n=== eta* (susceptibility peak) by size ===")
    for N in Ns:
        b = headline["by_size"][int(N)]
        print(f"  N={N:5d}: eta*={b['eta_star']:.3f}  chi_max={b['chi_max']:.3f}  "
              f"Binder(eta*)={b['binder_at_etastar']:.3f}  under-resolved={b['n_underresolved']}/{len(ETA)}")
    print(f"\neta* spread across sizes: {headline['eta_star_spread']:.3f}  (stable if < ~0.075, one grid step)")
    print("Binder crossings (adjacent sizes):", json.dumps(cross))
    print("\nwrote P2E4_noise_stageA_{perseed,grid}.csv + _headline.json")


if __name__ == "__main__":
    main()
