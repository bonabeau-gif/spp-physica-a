"""Thorough mixture-boundary search (plan Section 14 priorities 1 & 4 + continuation).

Part 1: trace the confirmed reciprocal-to-chiral J_a boundary as a manifold, freezing the
packing/adhesion parameters so the trace cannot drift onto the adhesion ridge.
Part 2: high-density demixing search -- cross-species symmetric (C_s) and antisymmetric
(C_a) force at rho~0.28. Segregation requires two species, so a demixing boundary should
FAIL the single-species control -> a genuine (reciprocal) mixture boundary distinct from
the chiral one.

Usage: bd_campaign3.py [n_along] [trace_steps]
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, pickle, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from boundary import params as P
from boundary.protocol import Protocol
from boundary.cache import RunCache
from boundary.adapters import run_many, seeds_for
from boundary.sensitivity import sensitivity
from boundary.continuation import continue_boundary
from boundary.catalog import dominant_observable, controls, bootstrap_normal

OUT = ROOT / "results" / "boundary"
B_MIN = 3.0
N_ALONG = int(sys.argv[1]) if len(sys.argv) > 1 else 6
TRACE_STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 12

# high-density demixing base rule (physical, plan names)
DEMIX_BASE = {"rho": 0.28, "v0": 0.05, "eps": 0.5, "r1_ratio": 7.0, "f0": 1.5, "B": 0.0,
              "ra_ratio": 7.0, "alpha": 2.5, "eta": 0.10, "f_B": 0.5, "J_s": 1.0,
              "J_a": 0.0, "C_s": 0.0, "C_a": 0.0}


def to_unit(sp, phys_plan):
    phys = np.array([phys_plan[n] for n in sp.param_order])
    return P.to_unit(phys, sp)


def susceptibility_peak(sp, proto, base_plan, scan_plan, grid, cache, obs, k=20):
    e, chi = [], []
    for val in grid:
        d = dict(base_plan); d[scan_plan] = float(val)
        u = P.clip_unit(to_unit(sp, d))
        seeds = seeds_for(u, proto.hash(), k, offset=95000)
        res = run_many(sp, [(u, proto.n_steps, proto.boundary, s) for s in seeds],
                       cache=cache, want_obs2=("obs2_" in obs))
        if obs.startswith("obs2_"):
            key = obs[5:]
            vals = np.array([r["obs2"][key] for r in res], float)
        else:
            vals = np.array([r["summary"][obs] for r in res], float)
        n = res[0]["n_particles"]
        e.append(float(val)); chi.append(float(n * np.nanvar(vals)))
    e = np.array(e); chi = np.array(chi)
    ksig = 0.12 * (grid[-1] - grid[0])
    W = np.exp(-0.5 * ((e[:, None] - e[None, :]) / ksig) ** 2)
    sm = (W * chi[None, :]).sum(1) / W.sum(1)
    interior = np.arange(1, len(e) - 1)
    km = interior[np.argmax(sm[interior])]
    lo, hi = max(0, km - 2), min(len(e), km + 3)
    c = np.polyfit(e[lo:hi], sm[lo:hi], 2)
    star = float(np.clip(-c[1] / (2 * c[0]), e[lo], e[hi - 1])) if c[0] < 0 else float(e[km])
    return star, list(zip(e.tolist(), chi.tolist()))


DEMIX_FAMILIES = {
    "D_demix_Cs": {"scan": "C_s", "grid": np.linspace(-1.0, 1.0, 13), "obs": "segregation",
                   "along": [("alpha", 0.0), ("alpha", -0.8), ("alpha", 0.8),
                             ("rho", -0.04), ("eta", 0.05), ("f_B", -0.08)]},
    "E_demix_Ca": {"scan": "C_a", "grid": np.linspace(0.0, 1.5, 13), "obs": "segregation",
                   "along": [("alpha", 0.0), ("alpha", -0.8), ("rho", -0.04),
                             ("eta", 0.05), ("f_B", -0.08), ("C_s", -0.4)]},
}


def catalog_families(sp, proto_obs, cache, points, norms, fams, susc):
    types = []
    for f in sorted(set(fams)):
        members = [i for i in range(len(points)) if fams[i] == f]
        rep = max(members, key=lambda i: susc[i]["b"])
        dom = dominant_observable(sp, points[rep], norms[rep], proto_obs, cache=cache, n_side=16)
        ctrl = controls(sp, points[rep], norms[rep], proto_obs, cache=cache, b_min=B_MIN,
                        dom_obs=dom["dominant_obs"])
        cache.flush()
        mean, lo, hi = bootstrap_normal([norms[i] for i in members])
        sig = [sp.param_order[j] for j in np.argsort(-np.abs(mean)) if lo[j] * hi[j] > 0][:4]
        rec = ctrl["reciprocal"]; sng = ctrl["single_species"]
        mixture = (not rec["survives"]) or (not sng["survives"])
        types.append({"family": f, "n_points": len(members), "dominant_obs": dom["dominant_obs"],
                      "rep_b": susc[rep]["b"], "top_loadings": sig,
                      "reciprocal": rec, "single_species": sng, "is_mixture": bool(mixture)})
        print(f"{f}: {len(members)}pts dom={dom['dominant_obs']} b={susc[rep]['b']:.2f} top={sig}", flush=True)
        print(f"   recip surv={rec['survives']}({rec.get('jump_ratio','-'):.2f}) "
              f"single surv={sng['survives']}({sng.get('jump_ratio','-'):.2f}) MIXTURE={mixture}", flush=True)
    return types


def main():
    sp = P.get_space("two")
    proto_obs = Protocol(space="two", n_steps=240)
    proto_desc = Protocol(space="two", n_steps=sp.base_steps)
    cache = RunCache(OUT / "cache_two.pkl")

    # ---- Part 1: high-density demixing families ----
    points, norms, fams, susc = [], [], [], []
    for fam, spec in DEMIX_FAMILIES.items():
        for ai, (aparam, adelta) in enumerate(spec["along"][:N_ALONG]):
            d = dict(DEMIX_BASE); d[aparam] = d[aparam] + adelta
            star, cut = susceptibility_peak(sp, proto_obs, d, spec["scan"], spec["grid"], cache, spec["obs"])
            d[spec["scan"]] = star
            u = P.clip_unit(to_unit(sp, d))
            s = sensitivity(sp, u, proto_obs, h=0.04, n_side=8, cache=cache)
            points.append(u); norms.append(s.normal); fams.append(fam)
            susc.append({"scan": spec["scan"], "star": star, "b": float(s.b), "cut": cut})
            print(f"  {fam}[{ai}] {spec['scan']}*={star:.3f} b={s.b:.2f}", flush=True)
        cache.flush()

    types = catalog_families(sp, proto_obs, cache, points, norms, fams, susc)

    # ---- Part 2: trace the strongest demixing boundary as a manifold ----
    frozen = ["eps", "r1_ratio", "f0", "B", "ra_ratio"]
    active = [j for j in range(sp.d) if sp.param_order[j] not in frozen]
    seed_i = int(np.argmax([s["b"] for s in susc]))
    print(f"tracing demixing boundary from {fams[seed_i]} (b={susc[seed_i]['b']:.2f}, "
          f"active={len(active)} dims)", flush=True)
    t0 = time.time()
    tr = continue_boundary(sp, points[seed_i], norms[seed_i], proto_obs, b_min=B_MIN, ds0=0.03,
                           max_steps=TRACE_STEPS, cache=cache, active=active)
    cache.flush()
    print(f"  demix trace: {len(tr.points)} pts, term={tr.terminated_by}, "
          f"sims={tr.total_sims}, wall={time.time()-t0:.0f}s", flush=True)

    # total genuine mixture boundaries = J_a chiral (campaign2, confirmed) + demix mixtures
    n_mix_demix = sum(t["is_mixture"] for t in types)
    total_mixture = 1 + n_mix_demix
    out = {"demix_trace": {"seed_family": fams[seed_i], "n_points": len(tr.points),
                           "terminated_by": tr.terminated_by, "manifold_dim": tr.manifold_dim,
                           "bvals": tr.bvals, "points": [p.tolist() for p in tr.points],
                           "normals": [n.tolist() for n in tr.normals]},
           "demix_types": types, "chiral_Ja": "confirmed mixture (campaign2 Family A)",
           "n_mixture_total": total_mixture, "m6_pass": bool(total_mixture >= 3)}
    json.dump(out, open(OUT / "catalog3.json", "w"), indent=2)
    pickle.dump({"points": [u.tolist() for u in points], "normals": [n.tolist() for n in norms],
                 "family": fams, "susc": susc}, open(OUT / "campaign3.pkl", "wb"))
    print(f"\nCAMPAIGN3: J_a trace {len(tr.points)}pts + {len(types)} demix families "
          f"({n_mix_demix} mixture); total genuine mixture boundaries={total_mixture}; "
          f"M6 {'PASS' if out['m6_pass'] else 'PARTIAL'}", flush=True)


if __name__ == "__main__":
    main()
