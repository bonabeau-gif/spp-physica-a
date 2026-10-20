"""P2-E4 noise cut, confirmatory (equilibrated, cell-list accelerated).

Replaces the under-equilibrated 400-step Stage A screen. Per the equilibration probe the
grid transient clears by ~250 steps but slow critical fluctuations need a long sampling
window, so: 5000 steps, burn-in 500, sample every 5 (900 samples/seed). Susceptibility is
the STANDARD order-parameter fluctuation chi = N*(<phi^2> - <phi>^2) pooled over time and
seeds (window-robust), with the Binder cumulant from the same pooled moments. Also reports
the across-seed variance of per-seed means (self-averaging / bimodality) and adequacy
(window >= 30*tau_int, and pooled-mean drift between window halves < 0.02).

Tests: (1) does eta*(N) (susceptibility peak) stabilize; (2) do Binder curves cross;
these are finite-size STABILITY questions, not exponent/thermodynamic-limit claims.
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
DENS = RHO * (math.sqrt(3) / 6) / R0 ** 2
TARGET_N = [64, 128, 256, 512, 1024]
ETA = [0.20, 0.35, 0.425, 0.50, 0.575, 0.65, 0.725, 0.80, 0.90]
N_SEEDS = 6
STEPS, BURN, EVERY = 5000, 500, 5
TAU_MULT = 30.0


def s_for_N(n):
    return math.sqrt(n / DENS)


def _task(t):
    N, eta, seed = t
    p = SingleParams(rho=RHO, v0=0.05, inertia=0.3, interaction_range_ratio=5.0,
                     force_strength=0.0, adhesion=0.0, alignment_range_ratio=8.0,
                     alignment_strength=3.5, noise=float(eta))
    cfg = SimulationConfig(S=s_for_N(N), r0=R0, steps=STEPS, sample_start=BURN,
                           sample_every=EVERY, boundary="periodic", seed=int(seed),
                           use_cell_list=True)
    r = simulate_single(p, cfg)
    phi = np.asarray(r.timeseries["phi"], float)
    h = len(phi) // 2
    return {"N": int(r.summary["n"]), "eta": float(eta), "seed": int(seed),
            "count": int(phi.size), "s1": float(phi.sum()), "s2": float((phi**2).sum()),
            "s4": float((phi**4).sum()), "seed_mean": float(phi.mean()),
            "tau": float(tau_int(phi) * EVERY),
            "drift": float(abs(phi[:h].mean() - phi[h:].mean()))}


def main():
    tasks = [(N, round(float(e), 4), s) for N in TARGET_N for e in ETA for s in range(N_SEEDS)]
    print(f"Confirmatory noise cut: {len(tasks)} runs x {STEPS} steps (burn {BURN}), cell list", flush=True)
    res = run_parallel(_task, tasks, max_workers=6, desc="p2e4-confirm")
    df = pd.DataFrame(res)
    df.to_csv(OUT / "P2E4_noise_confirm_perseed.csv", index=False)

    rows = []
    for (N, eta), g in df.groupby(["N", "eta"]):
        cnt = g["count"].sum()
        m1 = g.s1.sum() / cnt; m2 = g.s2.sum() / cnt; m4 = g.s4.sum() / cnt
        var = max(m2 - m1 * m1, 0.0)
        binder = 1.0 - m4 / (3.0 * m2 * m2) if m2 > 1e-12 else float("nan")
        seed_means = g.seed_mean.to_numpy()
        rows.append({"N": int(N), "eta": float(eta), "phi_mean": m1,
                     "chi": int(N) * var, "binder": binder,
                     "var_seedmeans": float(seed_means.var()),
                     "tau_median": float(np.nanmedian(g.tau)),
                     "drift_max": float(g.drift.max()),
                     "adequate": bool(((STEPS - BURN) >= TAU_MULT * np.nanmedian(g.tau))
                                      and g.drift.max() < 0.02)})
    grid = pd.DataFrame(rows).sort_values(["N", "eta"])
    grid.to_csv(OUT / "P2E4_noise_confirm_grid.csv", index=False)

    Ns = sorted(grid.N.unique())
    head = {"config": {"target_N": TARGET_N, "eta_grid": ETA, "n_seeds": N_SEEDS,
                       "steps": STEPS, "burn": BURN, "chi_def": "N*Var_pooled(phi)"},
            "by_size": {}}
    for N in Ns:
        sub = grid[grid.N == N].sort_values("eta")
        i = int(sub.chi.values.argmax())
        head["by_size"][int(N)] = {
            "eta_star": float(sub.eta.values[i]), "chi_max": float(sub.chi.values[i]),
            "eta": sub.eta.tolist(), "chi": sub.chi.round(4).tolist(),
            "binder": [round(float(x), 4) for x in sub.binder.tolist()],
            "phi_mean": [round(float(x), 4) for x in sub.phi_mean.tolist()],
            "tau_median": [round(float(x), 1) for x in sub.tau_median.tolist()],
            "n_underresolved": int((~sub.adequate).sum())}
    head["eta_star_by_size"] = {int(N): head["by_size"][int(N)]["eta_star"] for N in Ns}
    sv = list(head["eta_star_by_size"].values())
    head["eta_star_spread"] = float(max(sv) - min(sv))
    # self-averaging: Var_s(seed means) vs N off-transition (0.35, 0.90) and near (0.575)
    def var_series(eta):
        return {int(N): float(grid[(grid.N == N) & (np.isclose(grid.eta, eta))].var_seedmeans.iloc[0])
                for N in Ns}
    head["self_averaging"] = {"ordered_eta0.35": var_series(0.35),
                              "near_eta0.575": var_series(0.575),
                              "disordered_eta0.90": var_series(0.90)}
    # adjacent-size Binder crossings
    cross = {}
    for a, b in zip(Ns[:-1], Ns[1:]):
        ga = grid[grid.N == a].sort_values("eta"); gb = grid[grid.N == b].sort_values("eta")
        d = gb.binder.to_numpy() - ga.binder.to_numpy(); e = ga.eta.to_numpy()
        s = np.sign(d); xs = []
        for k in np.where(np.diff(s) != 0)[0]:
            if d[k + 1] != d[k]:
                xs.append(round(float(e[k] - d[k] * (e[k+1]-e[k]) / (d[k+1]-d[k])), 4))
        cross[f"{a}-{b}"] = xs
    head["binder_crossings_adjacent"] = cross
    json.dump(head, open(OUT / "P2E4_noise_confirm_headline.json", "w"), indent=2)

    print("\n=== eta* (susceptibility peak) by size, equilibrated ===")
    for N in Ns:
        b = head["by_size"][int(N)]
        print(f"  N={N:5d}: eta*={b['eta_star']:.3f}  chi_max={b['chi_max']:.2f}  "
              f"under-resolved={b['n_underresolved']}/{len(ETA)}")
    print(f"\neta* spread across sizes: {head['eta_star_spread']:.3f}  (size-stable if <= 0.075)")
    print("self-averaging Var_s(seed means) vs N:")
    for k, v in head["self_averaging"].items():
        print(f"  {k}: " + "  ".join(f"N{n}={x:.4f}" for n, x in v.items()))
    print("Binder crossings (adjacent):", json.dumps(cross))
    print("\nwrote P2E4_noise_confirm_{perseed,grid}.csv + _headline.json")


if __name__ == "__main__":
    main()
