"""Assessment #10: diagnose the chirality decay at long horizons.

(a) Run the strongest non-reciprocal chiral rule for a long trajectory and plot
    windowed omega(t), R^2(t), phi(t) separately -- this decides whether it is a
    long-lived rotating regime or a slow transient.
(b) N-dependence of |omega|: rerun the same rule at several box sizes (fixed
    density, so N scales with S^2) to test whether persistent rotation is a
    finite-size effect.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spp_unified.parameters import ACTIVE_SINGLE_NAMES, SingleParams, TwoSpeciesParams, SimulationConfig
from spp_unified.model import simulate_two_species
from spp_unified.metrics import heading_rotation

OUT = ROOT / "results" / "expanded"
OUT.mkdir(parents=True, exist_ok=True)


def chiral_rule():
    non = pd.read_csv(ROOT / "results" / "multi_qd_nonreciprocal.csv")
    row = non.loc[non.chirality.idxmax()]
    b = SingleParams.from_active_vector(np.array([row[n] for n in ACTIVE_SINGLE_NAMES]))
    return TwoSpeciesParams(base=b, fraction_B=row.fraction_B,
                            alignment_symmetric=row.alignment_symmetric,
                            alignment_antisymmetric=row.alignment_antisymmetric,
                            force_symmetric=row.force_symmetric,
                            force_antisymmetric=row.force_antisymmetric), row


def windowed_timeseries(p, steps=2000, seeds=range(10), win=60, hop=12, S=4.0):
    """Return per-window omega(t), R2(t), phi(t) averaged over seeds."""
    per_seed = []
    centers = None
    for sd in seeds:
        cfg = SimulationConfig(S=S, steps=steps, sample_start=0, sample_every=2,
                               boundary="periodic", seed=int(sd))
        r = simulate_two_species(p, cfg)
        theta = r.timeseries["theta"]   # (T, N)
        phi = r.timeseries["phi"]       # (T,)
        T = len(phi)
        oms, r2s, phs, cts = [], [], [], []
        for start in range(0, T - win, hop):
            sl = slice(start, start + win)
            rot = heading_rotation(theta[sl], phi[sl])
            # convert per-sample slope to per-step omega
            oms.append(abs(rot["omega"]) / (cfg.sample_every * cfg.dt))
            r2s.append(rot["rotation_r2"])
            phs.append(float(np.mean(phi[sl])))
            cts.append((start + win / 2) * cfg.sample_every)
        per_seed.append((np.array(cts), np.array(oms), np.array(r2s), np.array(phs)))
        centers = per_seed[-1][0]
    A = np.stack([s[1] for s in per_seed]); R = np.stack([s[2] for s in per_seed]); P = np.stack([s[3] for s in per_seed])
    df = pd.DataFrame({
        "step": centers,
        "abs_omega_mean": A.mean(0), "abs_omega_sd": A.std(0),
        "r2_mean": R.mean(0), "r2_sd": R.std(0),
        "phi_mean": P.mean(0), "phi_sd": P.std(0),
    })
    return df


def n_dependence(p, rho, seeds=range(16), steps=800):
    rows = []
    for S in [2.6, 3.2, 4.0, 5.0, 6.0, 7.2]:
        oms = []
        n_val = None
        for sd in seeds:
            cfg = SimulationConfig(S=S, steps=steps, sample_start=steps // 2, sample_every=2,
                                   boundary="periodic", seed=int(sd))
            r = simulate_two_species(p, cfg)
            oms.append(abs(r.summary["abs_omega"]))
            n_val = r.summary["n"]
        oms = np.array(oms)
        rows.append({"S": S, "N": int(n_val), "abs_omega_mean": float(oms.mean()),
                     "abs_omega_sd": float(oms.std()), "abs_omega_median": float(np.median(oms))})
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def main():
    p, row = chiral_rule()
    print(f"chiral rule: rho={row.rho:.3f} baseline N~{int(round(row.rho*SimulationConfig(S=4.0).n_max()))}", flush=True)
    ts = windowed_timeseries(p, steps=2000, seeds=range(12))
    ts.to_csv(OUT / "chirality_decay_timeseries.csv", index=False)
    print("timeseries written", ts.shape, flush=True)
    nd = n_dependence(p, row.rho)
    nd.to_csv(OUT / "chirality_decay_ndependence.csv", index=False)
    print("N-dependence written", flush=True)


if __name__ == "__main__":
    main()
