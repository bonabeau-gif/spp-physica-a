"""P2-E3: continuation robustness + curvature-adaptive stepping.

Reuses the P2-E2 synthetic harness (known analytic surfaces; import applies the monkeypatch).
Part A: for each benchmark, trace with the FIXED-step predictor vs the new CURVATURE-ADAPTIVE
predictor (continue_boundary curvature_adaptive=True), measuring Hausdorff-to-truth. This
directly tests whether the O(ds^2/r) curved-surface drift found in P2-E2 (~0.14 on a sphere)
is reduced. Part B: reproducibility -- trace each of two surfaces from several starts and
under a Latin-hypercube of numerical settings (ds0, b_min, eps_pred, refresh), and report the
spread of Hausdorff-to-truth and trace-to-trace agreement.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "experiments"))
import p2e2_benchmark as BM
from boundary.continuation import continue_boundary

OUT = ROOT / "results" / "boundary"


def localize_start(sp, proto, b, rng):
    p_on = b.onpoints(1, rng)[0]
    nrm = b.gn(p_on)
    u0 = np.clip(p_on - 0.2 * nrm, 0, 1); u1 = np.clip(p_on + 0.2 * nrm, 0, 1)
    loc = BM.localize_segment(sp, u0, u1, proto, n_coarse=11, delta=0.05, n_side_refine=8,
                              b_min=BM.B_MIN, cache=None)
    return loc


def hausdorff(tr, b):
    if not tr.points:
        return float("nan"), 0
    return float(np.max([abs(b.g(p)) for p in tr.points])), len(tr.points)


def main():
    sp = BM.synth_space(); proto = BM.Protocol(space="synth", n_steps=1)
    benches = BM.make_benches()
    results = {"partA_fixed_vs_adaptive": {}, "partB_reproducibility": {}}

    # ---- Part A: fixed vs curvature-adaptive, Hausdorff-to-truth ----
    print("=== Part A: fixed vs curvature-adaptive predictor (Hausdorff-to-truth) ===")
    for name in ["hyperplane", "intersecting", "hypersphere", "shell", "bounded_patch"]:
        b = benches[name]; BM.fit_standardizer(b)
        rng = np.random.default_rng(int.from_bytes(name.encode("utf-8"), "little") % 2**31)
        loc = localize_start(sp, proto, b, rng)
        u_star = loc.u_star; nrm = loc.accept["rep"].normal
        tr_fix = continue_boundary(sp, u_star, nrm, proto, b_min=BM.B_MIN, ds0=0.04,
                                   max_steps=18, cache=None, curvature_adaptive=False)
        tr_ad = continue_boundary(sp, u_star, nrm, proto, b_min=BM.B_MIN, ds0=0.06, ds_min=0.008,
                                  max_steps=30, cache=None, curvature_adaptive=True,
                                  eps_pred=0.004, refresh_angle_deg=5.0)
        hf, nf = hausdorff(tr_fix, b); ha, na = hausdorff(tr_ad, b)
        geom = b.geom
        results["partA_fixed_vs_adaptive"][name] = {
            "geom": geom, "hausdorff_fixed": hf, "n_fixed": nf,
            "hausdorff_adaptive": ha, "n_adaptive": na,
            "improvement": round((hf - ha) / hf, 3) if hf and hf == hf else None}
        print(f"  {name:14} [{geom:>14}]: fixed={hf:.3f}(n={nf})  adaptive={ha:.3f}(n={na})  "
              f"{'->' if ha<hf else '=='} {100*(hf-ha)/hf:+.0f}%" if hf else f"  {name}: n/a")

    # ---- Part B: reproducibility across starts x numerical settings ----
    print("\n=== Part B: reproducibility (Hausdorff-to-truth spread across settings/starts) ===")
    # 9-point Latin-hypercube over (ds0, b_min, eps_pred, refresh_every)
    rlh = np.random.default_rng(20)
    L = 9
    grid = {"ds0": np.linspace(0.03, 0.07, L), "b_min": np.linspace(2.6, 3.4, L),
            "eps_pred": np.linspace(0.003, 0.006, L), "refresh_every": np.array([3, 4, 5] * 3)}
    for k in grid:
        if k != "refresh_every":
            grid[k] = grid[k][rlh.permutation(L)]
    for name in ["hyperplane", "hypersphere"]:
        b = benches[name]; BM.fit_standardizer(b)
        H = []
        for start in range(3):
            rng = np.random.default_rng(1000 + start)
            loc = localize_start(sp, proto, b, rng)
            for i in range(L):
                tr = continue_boundary(sp, loc.u_star, loc.accept["rep"].normal, proto,
                                       b_min=float(grid["b_min"][i]), ds0=float(grid["ds0"][i]),
                                       ds_min=0.008, max_steps=26, cache=None,
                                       curvature_adaptive=True, eps_pred=float(grid["eps_pred"][i]),
                                       refresh_every=int(grid["refresh_every"][i]))
                h, n = hausdorff(tr, b)
                if n >= 3:
                    H.append(h)
        H = np.array(H)
        results["partB_reproducibility"][name] = {
            "geom": b.geom, "n_traces": int(len(H)),
            "hausdorff_median": float(np.median(H)), "hausdorff_max": float(H.max()),
            "hausdorff_iqr": float(np.percentile(H, 75) - np.percentile(H, 25))}
        print(f"  {name:14} [{b.geom:>14}]: {len(H)} traces  Hausdorff median={np.median(H):.3f} "
              f"IQR={np.percentile(H,75)-np.percentile(H,25):.3f} max={H.max():.3f}")

    json.dump(results, open(OUT / "P2E3_continuation_robustness.json", "w"), indent=2, default=float)
    print("\nwrote P2E3_continuation_robustness.json")


if __name__ == "__main__":
    main()
