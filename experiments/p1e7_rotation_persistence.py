"""P1-E7 (revised, tractable): finite-size persistence of rotation, scaling arms only.

Question: is persistent global rotation confined to the non-reciprocal family within the
tested model and bounds? Three arms -- the strongest non-reciprocal chiral rule, its MATCHED
reciprocal control (antisymmetric couplings zeroed), and the best INDEPENDENT reciprocal rule
(max-chirality rule from the reciprocal archive) -- evaluated at N in {64..1024} (box S at
fixed per-arm density), 12 seeds, at the manuscript's 800-step horizon (|omega| is a finite-
horizon observable; see chiral cut). Rotation rate |omega|, phase-locking (rotation_r2), and
polarization (phi) are reported SEPARATELY. Acceptance: an exclusivity claim requires the
best reciprocal distribution to stay separated from the non-reciprocal one across sizes.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, json, math
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from spp_unified.parameters import ACTIVE_SINGLE_NAMES, SingleParams, TwoSpeciesParams, SimulationConfig
from spp_unified.model import simulate_two_species
from expanded_common import run_parallel

OUT = ROOT / "results" / "boundary"; OUT.mkdir(parents=True, exist_ok=True)
R0 = 0.1
DENSFAC = (math.sqrt(3) / 6) / R0 ** 2
TARGET_N = [64, 128, 256, 512, 1024]
N_SEEDS = 12
STEPS, BURN = 800, 400


def _rule_from_row(row):
    # reciprocal-archive rows omit the antisymmetric columns (0 by construction)
    return TwoSpeciesParams(
        base=SingleParams.from_active_vector(np.array([row[n] for n in ACTIVE_SINGLE_NAMES])),
        fraction_B=float(row.get("fraction_B", 0.5)),
        alignment_symmetric=float(row.get("alignment_symmetric", 1.0)),
        alignment_antisymmetric=float(row.get("alignment_antisymmetric", 0.0)),
        force_symmetric=float(row.get("force_symmetric", 0.0)),
        force_antisymmetric=float(row.get("force_antisymmetric", 0.0)))


def build_arms():
    non = pd.read_csv(ROOT / "results/multi_qd_nonreciprocal.csv")
    rn = non.loc[non.chirality.idxmax()]
    p_non = _rule_from_row(rn)
    p_ctrl = replace(p_non, alignment_antisymmetric=0.0, force_antisymmetric=0.0)
    rec = pd.read_csv(ROOT / "results/multi_qd_reciprocal.csv")
    rr = rec.loc[rec.chirality.idxmax()]
    p_rec = _rule_from_row(rr)
    return [("nonreciprocal", p_non, float(rn.rho)),
            ("matched_reciprocal_control", p_ctrl, float(rn.rho)),
            ("best_reciprocal", p_rec, float(rr.rho))]


ARMS = build_arms()
ARMMAP = {a[0]: (a[1], a[2]) for a in ARMS}


def _task(t):
    arm, N, seed = t
    p, rho = ARMMAP[arm]
    S = math.sqrt(N / (rho * DENSFAC))
    cfg = SimulationConfig(S=S, r0=R0, steps=STEPS, sample_start=BURN, sample_every=2,
                           boundary="periodic", seed=int(seed), use_cell_list=True)
    r = simulate_two_species(p, cfg); s = r.summary
    return {"arm": arm, "N_target": int(N), "seed": int(seed), "N": int(s["n"]),
            "abs_omega": float(s["abs_omega"]), "rotation_r2": float(s["rotation_r2"]),
            "phi": float(s["phi"]), "chirality": float(s["chirality"])}


def main():
    for name, p, rho in ARMS:
        print(f"arm {name}: rho={rho:.3f} align_antisym={p.alignment_antisymmetric:.3f} "
              f"force_antisym={p.force_antisymmetric:.3f}", flush=True)
    tasks = [(a, N, s) for a in ARMMAP for N in TARGET_N for s in range(N_SEEDS)]
    print(f"P1-E7: {len(tasks)} runs ({len(ARMMAP)} arms x {len(TARGET_N)} N x {N_SEEDS} seeds), "
          f"{STEPS} steps, cell list", flush=True)
    res = run_parallel(_task, tasks, max_workers=6, desc="p1e7")
    df = pd.DataFrame(res)
    df.to_csv(OUT / "P1E7_rotation_perseed.csv", index=False)

    rows = []
    for (arm, N), g in df.groupby(["arm", "N_target"]):
        om = g.abs_omega.to_numpy()
        rows.append({"arm": arm, "N": int(g.N.iloc[0]),
                     "abs_omega_median": float(np.median(om)), "abs_omega_mean": float(om.mean()),
                     "abs_omega_q25": float(np.percentile(om, 25)), "abs_omega_q75": float(np.percentile(om, 75)),
                     "rotation_r2_median": float(np.median(g.rotation_r2)),
                     "phi_median": float(np.median(g.phi))})
    grid = pd.DataFrame(rows).sort_values(["arm", "N"])
    grid.to_csv(OUT / "P1E7_rotation_grid.csv", index=False)

    Ns = sorted(grid.N.unique())
    head = {"config": {"N": TARGET_N, "seeds": N_SEEDS, "steps": STEPS}, "by_arm": {}, "separation": {}}
    for arm in ARMMAP:
        s = grid[grid.arm == arm].sort_values("N")
        head["by_arm"][arm] = {"N": s.N.tolist(),
                               "abs_omega_median": [round(x, 4) for x in s.abs_omega_median],
                               "rotation_r2_median": [round(x, 3) for x in s.rotation_r2_median],
                               "phi_median": [round(x, 3) for x in s.phi_median]}
    # separation: non-reciprocal median omega vs the max over reciprocal arms, per N
    for N in Ns:
        nonr = float(grid[(grid.arm == "nonreciprocal") & (grid.N == N)].abs_omega_median.iloc[0])
        recs = [float(grid[(grid.arm == a) & (grid.N == N)].abs_omega_median.iloc[0])
                for a in ("matched_reciprocal_control", "best_reciprocal")]
        head["separation"][int(N)] = {"nonrecip_median": round(nonr, 4),
                                      "recip_max_median": round(max(recs), 4),
                                      "ratio": round(nonr / max(max(recs), 1e-6), 1)}
    json.dump(head, open(OUT / "P1E7_rotation_headline.json", "w"), indent=2)

    print("\n=== |omega| median by arm x N ===")
    for arm in ARMMAP:
        a = head["by_arm"][arm]
        print(f"  {arm:26}: " + "  ".join(f"N{n}={v:.3f}" for n, v in zip(a['N'], a['abs_omega_median'])))
    print("\n=== non-reciprocal vs reciprocal separation (median |omega|) ===")
    for N, s in head["separation"].items():
        print(f"  N={N:5d}: nonrecip={s['nonrecip_median']:.3f}  recip(best)={s['recip_max_median']:.3f}  "
              f"ratio={s['ratio']}x")
    print("\nwrote P1E7_rotation_{perseed,grid}.csv + _headline.json")


if __name__ == "__main__":
    main()
