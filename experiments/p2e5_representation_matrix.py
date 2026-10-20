"""P2-E5: representation-dependence matrix (mostly cached post-processing).

Which boundaries are visible under different descriptors, and which are invariant across
representations? The key economy: multiple representations are computed from the SAME
trajectories, so simulator cost = number of parameter points, not number of representations.
Along two single-species cuts (noise, density) we run trajectories once and compute three
representations per run -- classical physics observables, the 51-feature invariant
descriptor, and its PCA -- then locate each representation's sensitivity peak. Agreement of
peak locations => representation-invariant boundary; offsets => representation-specific.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
from boundary import params as P
from spp_unified.parameters import SingleParams, SimulationConfig
from spp_unified.model import simulate_single
from spp_unified.learned import behavior_descriptor
from sklearn.decomposition import PCA
from expanded_common import run_parallel

OUT = ROOT / "results" / "boundary"
SP = P.get_space("single")
BASE = dict(rho=0.15, v0=0.05, inertia=0.3, interaction_range_ratio=5.0, force_strength=0.0,
            adhesion=0.0, alignment_range_ratio=8.0, alignment_strength=3.5, noise=0.5)
CLASSICAL = ["phi", "nematic", "milling", "clustering", "drift"]
N_SEEDS = 4
CUTS = {"noise": ("noise", np.round(np.linspace(0.20, 0.90, 11), 3)),
        "density": ("rho", np.round(np.linspace(0.05, 0.45, 11), 3))}


def _task(t):
    cut, param, val, seed = t
    kw = dict(BASE); kw[param] = float(val)
    p = SingleParams(**kw)
    cfg = SimulationConfig(S=SP.S, r0=SP.r0, steps=800, sample_start=400, sample_every=2,
                           boundary="periodic", seed=int(seed), use_cell_list=True)
    r = simulate_single(p, cfg, record_states=True, state_every=SP.state_every)
    psi = np.asarray(behavior_descriptor(r.states, cfg, nbins=SP.nbins, max_frames=SP.max_frames), float)
    classical = np.array([float(r.summary[k]) for k in CLASSICAL])
    return {"cut": cut, "val": float(val), "seed": int(seed),
            "classical": classical.tolist(), "psi": psi.tolist()}


def zscore(M):
    s = M.std(0); s[s < 1e-9] = 1.0
    return (M - M.mean(0)) / s


def sensitivity_curve(point_means, vals):
    """Standardized finite-difference rate ||dR/dparam|| between adjacent cut points."""
    Z = zscore(point_means)
    rate = np.full(len(vals), np.nan)
    for i in range(1, len(vals) - 1):
        rate[i] = np.linalg.norm(Z[i + 1] - Z[i - 1]) / (vals[i + 1] - vals[i - 1])
    return rate


def peak_of(rate, vals):
    v = rate.copy(); v[np.isnan(v)] = -np.inf
    i = int(v.argmax())
    med = np.nanmedian(rate)
    return float(vals[i]), float(rate[i] / med if med > 0 else np.nan)


def main():
    tasks = [(cut, param, v, s) for cut, (param, vals) in CUTS.items()
             for v in vals for s in range(N_SEEDS)]
    print(f"P2-E5: {len(tasks)} runs ({len(CUTS)} cuts x 11 points x {N_SEEDS} seeds), shared trajectories")
    res = run_parallel(_task, tasks, max_workers=6, desc="p2e5")

    out = {"cuts": {}}
    for cut, (param, vals) in CUTS.items():
        rc = [r for r in res if r["cut"] == cut]
        # per-point seed means for each representation
        cls_pt, psi_pt = [], []
        for v in vals:
            rv = [r for r in rc if abs(r["val"] - v) < 1e-6]
            cls_pt.append(np.mean([r["classical"] for r in rv], 0))
            psi_pt.append(np.mean([r["psi"] for r in rv], 0))
        cls_pt = np.array(cls_pt); psi_pt = np.array(psi_pt)
        pca = PCA(n_components=8).fit(zscore(psi_pt))
        pca_pt = pca.transform(zscore(psi_pt))
        reps = {"classical(5)": cls_pt, "descriptor51": psi_pt, "PCA8": pca_pt}
        cutrec = {"param": param, "vals": vals.tolist(), "representations": {}}
        for name, M in reps.items():
            rate = sensitivity_curve(M, vals)
            loc, sharp = peak_of(rate, vals)
            cutrec["representations"][name] = {"peak_loc": loc, "peak_sharpness": round(sharp, 2),
                                               "rate": [None if np.isnan(x) else round(float(x), 3) for x in rate]}
        locs = [cutrec["representations"][n]["peak_loc"] for n in reps]
        cutrec["peak_spread"] = round(max(locs) - min(locs), 3)
        out["cuts"][cut] = cutrec

    json.dump(out, open(OUT / "P2E5_representation_matrix.json", "w"), indent=2)
    print("\n=== representation-dependence: sensitivity-peak location per cut x representation ===")
    for cut, rec in out["cuts"].items():
        print(f"\n  {cut} cut (vary {rec['param']}):")
        for name, r in rec["representations"].items():
            print(f"    {name:14}: peak at {rec['param']}={r['peak_loc']:.3f}  (sharpness {r['peak_sharpness']}x median)")
        print(f"    -> peak spread across representations: {rec['peak_spread']:.3f}  "
              f"({'INVARIANT' if rec['peak_spread'] <= 0.08 else 'representation-DEPENDENT'})")
    print("\nwrote P2E5_representation_matrix.json")


if __name__ == "__main__":
    main()
