"""Assessment v2 problem #2: rule out the equilibration confound in the
finite-size noise cut. Three checks:
  (a) repeat the cut at two trajectory lengths; show eta* and the chi_max
      exponent are stable (not a relaxation-time artifact);
  (b) show across-seed Var_s(phi) DECREASES with N away from the transition
      (self-averaging off-transition), vs growing at the peak;
  (c) save the per-seed phi distribution at eta* for each N (bimodality would
      indicate a first-order / coexistence signature, cf. Gregoire-Chate).
Wider N range (29..~277).
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
from spp_unified.parameters import SingleParams, SimulationConfig
from spp_unified.model import simulate_single
from expanded_common import run_parallel

OUT = ROOT / "results" / "expanded"
S_LIST = [2.6, 3.6, 5.0, 6.6, 8.6]      # rho=0.15 -> N ~ 29, 56, 108, 189, 317
N_SEEDS = 48


def _task(t):
    eta, S, steps, nseed = t
    p = SingleParams(rho=0.15, v0=0.05, inertia=0.3, interaction_range_ratio=5.0,
                     force_strength=0.0, adhesion=0.0, alignment_range_ratio=8.0,
                     alignment_strength=3.5, noise=float(eta))
    phis = []
    n = None
    for s in range(nseed):
        cfg = SimulationConfig(S=S, r0=0.1, steps=steps, sample_start=steps // 2,
                               sample_every=2, boundary="periodic", seed=int(s))
        r = simulate_single(p, cfg); phis.append(r.summary["phi"]); n = r.summary["n"]
    phis = np.asarray(phis)
    O2 = float((phis ** 2).mean()); O4 = float((phis ** 4).mean())
    binder = 1.0 - O4 / (3.0 * O2 * O2) if O2 > 1e-12 else 0.0
    return {"eta": float(eta), "S": S, "steps": steps, "N": int(n),
            "phi_mean": float(phis.mean()), "phi_var": float(phis.var()),
            "susceptibility": int(n) * float(phis.var()), "binder": float(binder),
            "phis": phis.tolist()}


def run_grid(etas, steps):
    tasks = [(round(float(e), 4), S, steps, N_SEEDS) for S in S_LIST for e in etas]
    res = run_parallel(_task, tasks, max_workers=6, desc=f"fs2 steps={steps}")
    return res


def main():
    # full grid at reference length
    etas_full = np.round(np.linspace(0.20, 0.95, 16), 4)
    r220 = run_grid(etas_full, 220)
    # peak region at a longer length
    etas_peak = np.round(np.linspace(0.45, 0.75, 7), 4)
    r440 = run_grid(etas_peak, 440)

    rows = []
    perseed = []
    for r in r220 + r440:
        rows.append({k: v for k, v in r.items() if k != "phis"})
        if r["steps"] == 220:
            for i, ph in enumerate(r["phis"]):
                perseed.append({"N": r["N"], "eta": r["eta"], "seed": i, "phi": ph})
    grid = pd.DataFrame(rows).sort_values(["steps", "N", "eta"])
    grid.to_csv(OUT / "finite_size_v2_grid.csv", index=False)
    pd.DataFrame(perseed).to_csv(OUT / "finite_size_v2_perseed.csv", index=False)

    # analysis
    def chi_exponent(sub):
        Ns = sorted(sub.N.unique())
        cmax = [sub[sub.N == n].susceptibility.max() for n in Ns]
        peak = [float(sub[sub.N == n].sort_values("susceptibility").eta.iloc[-1]) for n in Ns]
        a = float(np.polyfit(np.log(Ns), np.log(cmax), 1)[0])
        return {"sizes": [int(x) for x in Ns], "chi_max": [float(x) for x in cmax],
                "eta_star_by_N": peak, "eta_star_largest": peak[-1], "chi_exponent": a}

    g220 = grid[grid.steps == 220]; g440 = grid[grid.steps == 440]
    ex220 = chi_exponent(g220)
    # for 440, restrict 220 to same eta window for a fair exponent comparison
    ex440 = chi_exponent(g440)

    # self-averaging: Var_s vs N at eta*~0.60 and off-transition (0.30, 0.90)
    def var_by_N(eta_target):
        sub = g220.iloc[(g220.eta - eta_target).abs().groupby(g220.N).idxmin()]
        return {int(r.N): float(r.phi_var) for _, r in sub.iterrows()}
    var_peak = var_by_N(0.60); var_ord = var_by_N(0.30); var_dis = var_by_N(0.90)

    ps = pd.DataFrame(perseed)
    # per-seed phi distribution at the grid eta closest to 0.60, per N
    eta_star_grid = float(ps.eta.unique()[np.argmin(np.abs(ps.eta.unique() - 0.60))])
    bim = {}
    for n in sorted(ps.N.unique()):
        v = ps[(ps.N == n) & (np.isclose(ps.eta, eta_star_grid))].phi.to_numpy()
        if len(v):
            bim[int(n)] = {"mean": float(v.mean()), "sd": float(v.std()),
                           "min": float(v.min()), "max": float(v.max())}

    head = {
        "eta_star_220": ex220["eta_star_largest"], "chi_exponent_220": ex220["chi_exponent"],
        "eta_star_440": ex440["eta_star_largest"], "chi_exponent_440": ex440["chi_exponent"],
        "var_at_peak_by_N": var_peak, "var_ordered_eta0.30_by_N": var_ord,
        "var_disordered_eta0.90_by_N": var_dis,
        "phi_dist_at_peak_by_N": bim,
        "n_range": [int(min(ps.N)), int(max(ps.N))],
    }
    json.dump(head, open(OUT / "finite_size_v2_headline.json", "w"), indent=2)
    print(json.dumps(head, indent=2))


if __name__ == "__main__":
    main()
