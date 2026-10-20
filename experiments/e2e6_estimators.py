"""E6 + E2: reference-value resolution and descriptor-vs-susceptibility estimator offset.

Six 1-D cuts. On each, both the order-parameter susceptibility peak (reference) and the
descriptor-sensitivity ridge peak are located by parabolic interpolation around a lightly
smoothed maximum, with bootstrap CIs over the common seed set. The signed offset
Delta = lambda*_descriptor - lambda*_susceptibility (normalized by the cut's parameter
range) is reported per cut plus mean/spread. E6's reference intervals are the
susceptibility peaks on the eta (C1) and J_a (C6) cuts at finer grids.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from spp_unified.parameters import SingleParams, ACTIVE_SINGLE_NAMES
from boundary import params as P
from boundary import config as CFG
from boundary.protocol import Protocol
from boundary.cache import RunCache
from boundary.adapters import run_many, seeds_for, standardize, run_adequate

OUT = ROOT / "results" / "boundary"
RNG = np.random.default_rng(20260727)


def single_phys(rho, v0, inertia, r1, f0, B, ra, alpha, eta):
    d = dict(rho=rho, v0=v0, eps=inertia, r1_ratio=r1, f0=f0, B=B, ra_ratio=ra, alpha=alpha, eta=eta)
    return d


def chiral_two_phys():
    non = pd.read_csv(ROOT / "results" / "multi_qd_nonreciprocal.csv")
    row = non.loc[non.chirality.idxmax()]
    b = SingleParams.from_active_vector(np.array([row[n] for n in ACTIVE_SINGLE_NAMES], float))
    d = {P.PLAN_TO_CODE[n]: getattr(b, P.PLAN_TO_CODE[n]) for n in P.PARAM_ORDER_SINGLE}
    d.update({"fraction_B": float(row.fraction_B), "alignment_symmetric": float(row.alignment_symmetric),
              "alignment_antisymmetric": 0.0, "force_symmetric": float(row.force_symmetric),
              "force_antisymmetric": 0.0})
    # relabel to plan names
    return {n: d[P.PLAN_TO_CODE[n]] for n in P.PARAM_ORDER_TWO}


CUTS = {
    "C1_eta_vicsek": {"space": "single", "scan": "eta", "order": "phi",
                      "grid": np.linspace(0.30, 0.85, 21),
                      "base": single_phys(0.15, 0.05, 0.3, 5.0, 0.0, 0.0, 8.0, 3.5, 0.0)},
    "C2_eta_denseflock": {"space": "single", "scan": "eta", "order": "phi",
                          "grid": np.linspace(0.20, 0.80, 21),
                          "base": single_phys(0.55, 0.05, 1.0, 6.0, 1.0, 0.2, 7.0, 4.0, 0.0)},
    "C3_eta_clustered": {"space": "single", "scan": "eta", "order": "phi",
                         "grid": np.linspace(0.20, 0.80, 21),
                         "base": single_phys(0.40, 0.03, 0.5, 8.0, 3.0, 0.6, 4.0, 1.0, 0.0)},
    # rho cuts held near the order-disorder boundary (eta=0.55) so varying rho crosses it
    "C4_rho_vicsek": {"space": "single", "scan": "rho", "order": "phi",
                      "grid": np.linspace(0.05, 0.70, 21),
                      "base": single_phys(0.15, 0.05, 0.3, 5.0, 0.0, 0.0, 8.0, 3.5, 0.55)},
    "C5_rho_altalpha": {"space": "single", "scan": "rho", "order": "phi",
                        "grid": np.linspace(0.05, 0.70, 21),
                        "base": single_phys(0.15, 0.05, 0.3, 5.0, 0.0, 0.0, 8.0, 2.0, 0.55)},
    "C6_Ja_chiral": {"space": "two", "scan": "J_a", "order": "abs_omega",
                     "grid": np.linspace(1.0, 1.8, 25), "base": chiral_two_phys()},
}


def to_unit(sp, base_plan, scan, val):
    """Unit vector for the cut. The scan axis is left UNCLIPPED so a validation cut can
    extend past the search box (e.g. J_a to 1.8, physically valid); all other axes are
    clipped into the box."""
    d = dict(base_plan); d[scan] = float(val)
    phys = np.array([d[n] for n in sp.param_order])
    u = P.to_unit(phys, sp)
    ax = sp.param_order.index(scan)
    uc = P.clip_unit(u); uc[ax] = u[ax]
    return uc


def locate_peak(grid, values, smooth_frac=0.10):
    g = np.asarray(grid, float); v = np.asarray(values, float)
    ksig = smooth_frac * (g[-1] - g[0])
    W = np.exp(-0.5 * ((g[:, None] - g[None, :]) / ksig) ** 2)
    sm = (W * v[None, :]).sum(1) / W.sum(1)
    k = int(np.clip(np.argmax(sm[1:-1]) + 1, 1, len(g) - 2))
    x = g[k - 1:k + 2]; y = sm[k - 1:k + 2]
    c = np.polyfit(x, y, 2)
    return float(np.clip(-c[1] / (2 * c[0]), g[0], g[-1])) if c[0] < 0 else float(g[k])


def run_cut(sp, proto, cut, cache):
    scan, grid, order = cut["scan"], cut["grid"], cut["order"]
    G = len(grid)
    order_vals = np.zeros((G, CFG.N_SEEDS_SUSC))       # per-point per-seed order parameter
    rate_plus = [None] * G; rate_minus = [None] * G     # per-point per-seed standardized psi
    ax = sp.param_order.index(scan); e = np.zeros(sp.d); e[ax] = 1.0
    delta = 0.05
    n_desc = 16                                    # denoise the descriptor rate vs the plan's 8
    for i, val in enumerate(grid):
        u = to_unit(sp, cut["base"], scan, val)
        seeds = seeds_for(u, proto.hash(), CFG.N_SEEDS_SUSC, offset=110000)
        res = run_many(sp, [(u, proto.n_steps, proto.boundary, s) for s in seeds], cache=cache)
        order_vals[i] = [r["summary"][order] for r in res]
        # descriptor: psi at u +/- delta*axis over n_desc seeds (scan axis unclipped)
        up = u + delta * e; dn = u - delta * e
        sp_seeds = seeds_for(up, proto.hash(), n_desc, offset=3000)
        sm_seeds = seeds_for(dn, proto.hash(), n_desc, offset=7000)
        rp = run_many(sp, [(up, proto.n_steps, proto.boundary, s) for s in sp_seeds], cache=cache)
        rm = run_many(sp, [(dn, proto.n_steps, proto.boundary, s) for s in sm_seeds], cache=cache)
        rate_plus[i] = np.array([standardize(r["psi"], sp) for r in rp])
        rate_minus[i] = np.array([standardize(r["psi"], sp) for r in rm])
    N = float(res[0]["n_particles"])

    # point estimates
    susc = N * np.nanvar(order_vals, axis=1)
    rate = np.array([np.linalg.norm(rate_plus[i].mean(0) - rate_minus[i].mean(0)) / (2 * delta)
                     for i in range(G)])
    susc_peak = locate_peak(grid, susc)
    desc_peak = locate_peak(grid, rate)

    # bootstrap peaks over the seed sets
    sp_boot, ds_boot = [], []
    for _ in range(CFG.N_BOOT):
        idx = RNG.integers(0, CFG.N_SEEDS_SUSC, CFG.N_SEEDS_SUSC)
        sp_boot.append(locate_peak(grid, N * np.nanvar(order_vals[:, idx], axis=1)))
        jd = RNG.integers(0, n_desc, n_desc)
        rb = np.array([np.linalg.norm(rate_plus[i][jd].mean(0) - rate_minus[i][jd].mean(0)) / (2 * delta)
                       for i in range(G)])
        ds_boot.append(locate_peak(grid, rb))
    prange = float(grid[-1] - grid[0])
    return {
        "susc_peak": susc_peak, "susc_ci": [float(np.percentile(sp_boot, 5)), float(np.percentile(sp_boot, 95))],
        "desc_peak": desc_peak, "desc_ci": [float(np.percentile(ds_boot, 5)), float(np.percentile(ds_boot, 95))],
        "offset": (desc_peak - susc_peak), "offset_norm": (desc_peak - susc_peak) / prange,
        "offset_norm_ci": [float(np.percentile((np.array(ds_boot) - np.array(sp_boot)) / prange, 5)),
                           float(np.percentile((np.array(ds_boot) - np.array(sp_boot)) / prange, 95))],
        "grid_range": [float(grid[0]), float(grid[-1])], "N": N,
        "susc_curve": list(zip([float(x) for x in grid], [float(x) for x in susc])),
        "rate_curve": list(zip([float(x) for x in grid], [float(x) for x in rate])),
    }


def main():
    cache = RunCache(OUT / "cache_two.pkl")
    cache_s = RunCache(OUT / "cache_single.pkl")
    results = {"config_hash": CFG.CONFIG_HASH, "cuts": {}}
    t0 = time.time()
    for name, cut in CUTS.items():
        sp = P.get_space(cut["space"])
        ca = cache if cut["space"] == "two" else cache_s
        # Run length: verify adequacy at the grid midpoint (near the boundary, where tau
        # is largest) but cap escalation to bound cost; report tau and the length used.
        base_proto = Protocol(space=cut["space"], n_steps=360, max_steps=440)
        mid_u = to_unit(sp, cut["base"], cut["scan"], cut["grid"][len(cut["grid"]) // 2])
        _, tau, under = run_adequate(sp, mid_u, base_proto, keys=("phi", "milling"), cache=ca)
        proto = base_proto
        r = run_cut(sp, proto, cut, ca)
        r["n_steps"] = proto.n_steps; r["tau_mid"] = tau; r["under_resolved_mid"] = bool(under)
        results["cuts"][name] = r
        ca.flush()
        print(f"{name}: susc*={r['susc_peak']:.3f} CI{[round(x,3) for x in r['susc_ci']]} "
              f"desc*={r['desc_peak']:.3f} CI{[round(x,3) for x in r['desc_ci']]} "
              f"offset_norm={r['offset_norm']:+.3f} CI{[round(x,3) for x in r['offset_norm_ci']]} "
              f"(steps={proto.n_steps})", flush=True)
    offs = np.array([results["cuts"][c]["offset_norm"] for c in CUTS])
    results["offset_mean"] = float(offs.mean()); results["offset_std"] = float(offs.std())
    results["all_negative"] = bool((offs < 0).all())
    results["mean_mag"] = float(np.abs(offs).mean())
    results["cache_hits_two"] = cache.hits; results["cache_misses_two"] = cache.misses
    json.dump(results, open(OUT / "E2E6_estimators.json", "w"), indent=2)
    print(f"\noffset mean={offs.mean():+.3f} std={offs.std():.3f} all_neg={results['all_negative']} "
          f"mean_mag={results['mean_mag']:.3f}; wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
